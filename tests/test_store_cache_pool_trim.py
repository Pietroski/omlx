# SPDX-License-Identifier: Apache-2.0
"""store_cache allocator-pool trim + cooperative abort guards.

Boundary snapshots carry full cache state at strictly growing token counts,
so their transient buffers all have distinct sizes: freed ones land in the
MLX allocator pool and are never reused (the next snapshot is bigger) nor
returned to the OS. Over a long-context store (dozens of snapshots) the pool
grows quadratically with context length while the inference thread's
_sync_and_clear_cache is locked out by _mx_buffer_access_lock for the whole
store. store_cache therefore trims the pool from the worker itself between
blocks (test 1), and honours a cooperative ``should_abort`` callable so
engine teardown can cancel a long store instead of fatal-exiting on the
teardown watchdog (tests 2-3).

The store harness mirrors test_prefix_cache_cachelist_mixed.py:
production-shaped CacheList(KVCache, ArraysCache) layer dicts round-tripped
through a real hot-cache-only PagedSSDCacheManager.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import omlx.cache.prefix_cache as prefix_cache_module
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.cache.type_registry import CacheTypeRegistry

try:
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

BLOCK_SIZE = 4
NUM_LAYERS = 1
CONV_CHANNELS = (16, 16, 32, 32)


class MockModel:
    def __init__(self, num_layers: int = NUM_LAYERS):
        self._num_layers = num_layers
        self.layers = [MagicMock() for _ in range(num_layers)]

    @property
    def args(self):
        a = MagicMock()
        a.num_hidden_layers = self._num_layers
        return a


class _TrimSpyMX:
    """Proxy over mlx.core that spies on the pool-trim entry points.

    ``get_cache_memory`` reports an over-ceiling pool so the trim branch
    always fires; ``clear_cache`` counts invocations and delegates.
    Everything else falls through to the real module.
    """

    def __init__(self, real):
        self._real = real
        self.clear_calls = 0

    def get_cache_memory(self):
        return prefix_cache_module._STORE_POOL_TRIM_BYTES + 1

    def clear_cache(self):
        self.clear_calls += 1
        self._real.clear_cache()

    def __getattr__(self, name):
        return getattr(self._real, name)


def _make_cache(tmp_path):
    paged_cache = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=100,
        model_name="test-model",
        initial_blocks=100,
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / "ssd_cache",
        max_size_bytes=100 * 1024**2,
        hot_cache_max_bytes=10 * 1024**2,
        hot_cache_only=True,
        expected_model_name="test-model",
    )
    cache = BlockAwarePrefixCache(
        model=MockModel(),
        paged_cache_manager=paged_cache,
        paged_ssd_cache_manager=ssd,
    )
    return cache


def _position_kv(seq_len):
    pos = mx.arange(seq_len, dtype=mx.float32).reshape(1, 1, seq_len, 1)
    keys = mx.broadcast_to(pos, (1, 2, seq_len, 8))
    values = keys + 1000.0
    return mx.contiguous(keys), mx.contiguous(values)


def _build_mixed_cachelist(seq_len):
    kv = KVCache()
    keys, values = _position_kv(seq_len)
    kv.update_and_fetch(keys, values)

    arrays = ArraysCache(size=4)
    for i, channels in enumerate(CONV_CHANNELS):
        arrays[i] = mx.full((1, 3, channels), seq_len + i / 10.0, dtype=mx.float32)

    cache_list = CacheList(kv, arrays)
    mx.eval([t for t in [keys, values] + list(arrays.cache) if t is not None])
    return cache_list


def _layer_dict(cache_list):
    handler = CacheTypeRegistry.get_handler_by_class_name("CacheList")
    state_dict = handler.extract_state(cache_list)
    return {
        "state": list(state_dict["sub_states"]),
        "meta_state": (
            list(state_dict["sub_class_names"]),
            list(state_dict["sub_meta_states"]),
        ),
        "class_name": "CacheList",
        "cache_type": "CacheList",
    }


def _cache_data(seq_len):
    return [_layer_dict(_build_mixed_cachelist(seq_len))]


def _boundary_snapshots(num_blocks):
    return {
        BLOCK_SIZE * (i + 1): _cache_data(BLOCK_SIZE * (i + 1))
        for i in range(num_blocks)
    }


# ---------------------------------------------------------------------------
# 1) Pool trim fires between snapshot-consuming blocks
# ---------------------------------------------------------------------------


def test_pool_trimmed_between_snapshot_blocks(tmp_path, monkeypatch):
    """An over-ceiling pool is cleared once per stored block."""
    cache = _make_cache(tmp_path)
    spy = _TrimSpyMX(mx)
    monkeypatch.setattr(prefix_cache_module, "mx", spy)

    num_blocks = 5
    tokens = list(range(num_blocks * BLOCK_SIZE))
    table = cache.store_cache(
        "req-trim",
        tokens,
        _cache_data(len(tokens)),
        boundary_snapshots=_boundary_snapshots(num_blocks),
    )

    assert table is not None
    assert table.num_tokens == num_blocks * BLOCK_SIZE
    # One trim opportunity per fully stored block; with the pool reported
    # over-ceiling every check must fire.
    assert spy.clear_calls == num_blocks


def test_pool_not_trimmed_when_under_ceiling(tmp_path, monkeypatch):
    """Under the ceiling the trim must stay dormant (no pointless clears)."""
    cache = _make_cache(tmp_path)
    spy = _TrimSpyMX(mx)
    spy.get_cache_memory = lambda: 0  # pool always under ceiling
    monkeypatch.setattr(prefix_cache_module, "mx", spy)

    num_blocks = 3
    tokens = list(range(num_blocks * BLOCK_SIZE))
    table = cache.store_cache(
        "req-cool",
        tokens,
        _cache_data(len(tokens)),
        boundary_snapshots=_boundary_snapshots(num_blocks),
    )

    assert table is not None
    assert spy.clear_calls == 0


# ---------------------------------------------------------------------------
# 2) Cooperative abort stops the store at a block boundary
# ---------------------------------------------------------------------------


def test_should_abort_stops_store_cleanly(tmp_path):
    """Abort after the first block: stored prefix stays valid, rest skipped."""
    cache = _make_cache(tmp_path)

    calls = {"n": 0}

    def abort_after_first() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    num_blocks = 4
    tokens = list(range(num_blocks * BLOCK_SIZE))
    table = cache.store_cache(
        "req-abort",
        tokens,
        _cache_data(len(tokens)),
        boundary_snapshots=_boundary_snapshots(num_blocks),
        should_abort=abort_after_first,
    )

    assert table is not None
    # Exactly one block survived; the abort landed before block 2.
    assert table.num_tokens == BLOCK_SIZE
    assert len(table.block_ids) == 1


def test_should_abort_true_from_start_stores_nothing(tmp_path):
    """An abort signal raised before the store begins persists zero blocks."""
    cache = _make_cache(tmp_path)

    num_blocks = 3
    tokens = list(range(num_blocks * BLOCK_SIZE))
    table = cache.store_cache(
        "req-abort-early",
        tokens,
        _cache_data(len(tokens)),
        boundary_snapshots=_boundary_snapshots(num_blocks),
        should_abort=lambda: True,
    )

    assert table is not None
    assert table.num_tokens == 0
    assert len(table.block_ids) == 0


def test_no_abort_callable_stores_everything(tmp_path):
    """Default path (no should_abort) is unchanged."""
    cache = _make_cache(tmp_path)

    num_blocks = 3
    tokens = list(range(num_blocks * BLOCK_SIZE))
    table = cache.store_cache(
        "req-noabort",
        tokens,
        _cache_data(len(tokens)),
        boundary_snapshots=_boundary_snapshots(num_blocks),
    )

    assert table is not None
    assert table.num_tokens == num_blocks * BLOCK_SIZE
    assert len(table.block_ids) == num_blocks
