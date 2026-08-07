# SPDX-License-Identifier: Apache-2.0
"""Per-member CacheList block storage (``__cache_list_pm__``) guards.

Mixed CacheList layers (inkling-style ``CacheList(KVCache, ArraysCache)``)
previously stored the FULL cumulative state of every member in every block —
quadratic in context length on the allocator pool and the SSD (issue #2546).
Per-member storage slices the sliceable KV member per block and keeps only
the boundary snapshot's small non-sliceable state, restoring linear cost.

Guards here:
1. Stored blocks are per-block sized (KV member holds BLOCK_SIZE tokens,
   not the cumulative prefix) and round-trip positionally.
2. Legacy cumulative blocks still restore with last-block semantics.
3. A chain mixing legacy and per-member blocks is rejected, not corrupted.
4. Member-filtered snapshots (blanked KV member) still store correctly.

Harness mirrors test_prefix_cache_cachelist_mixed.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import omlx.cache.prefix_cache as prefix_cache_module
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache, cachelist_pm_member_plan
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
    return cache, ssd


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


def _layer_dict(cache_list, blank_kv=False):
    handler = CacheTypeRegistry.get_handler_by_class_name("CacheList")
    state_dict = handler.extract_state(cache_list)
    state = list(state_dict["sub_states"])
    if blank_kv:
        # Mirrors Scheduler._extract_snapshot_cache_states: sliceable
        # members blanked, boundary members kept.
        state[0] = ()
    return {
        "state": state,
        "meta_state": (
            list(state_dict["sub_class_names"]),
            list(state_dict["sub_meta_states"]),
        ),
        "class_name": "CacheList",
        "cache_type": "CacheList",
    }


def _cache_data(seq_len, blank_kv=False):
    return [_layer_dict(_build_mixed_cachelist(seq_len), blank_kv=blank_kv)]


def _boundary_snapshots(num_blocks, blank_kv=False):
    return {
        BLOCK_SIZE * (i + 1): _cache_data(BLOCK_SIZE * (i + 1), blank_kv=blank_kv)
        for i in range(num_blocks)
    }


def _store_blocks(cache, num_blocks, request_id="req-pm", blank_kv=False):
    tokens = list(range(num_blocks * BLOCK_SIZE))
    return cache.store_cache(
        request_id,
        tokens,
        _cache_data(len(tokens)),
        boundary_snapshots=_boundary_snapshots(num_blocks, blank_kv=blank_kv),
    )


def _assert_restored(result, expected_seq_len):
    assert result is not None
    restored = result[0]
    assert type(restored).__name__ == "CacheList"
    kv = list(restored.caches)[0]
    keys = kv.state[0]
    assert keys.shape[2] == expected_seq_len
    expected_keys, _ = _position_kv(expected_seq_len)
    assert mx.max(mx.abs(keys - expected_keys)).item() == 0.0
    arrays = list(restored.caches)[1]
    for i, channels in enumerate(CONV_CHANNELS):
        slot = list(arrays.state)[i]
        assert tuple(slot.shape) == (1, 3, channels)
        assert mx.max(mx.abs(slot - (expected_seq_len + i / 10.0))).item() == 0.0


def test_plan_helper_classification():
    cases = {
        "mixed kv+arrays": (["KVCache", "ArraysCache"], True),
        "kv only": (["KVCache", "KVCache"], False),
        "arrays only": (["ArraysCache"], False),
        "pooling member": (["KVCache", "PoolingCache"], False),
        "no names": ([], False),
    }
    live = _build_mixed_cachelist(BLOCK_SIZE)
    handler = CacheTypeRegistry.get_handler_by_class_name("CacheList")
    kv_state, arrays_state = handler.extract_state(live)["sub_states"]
    states_by_class = {
        "KVCache": kv_state,
        "ArraysCache": arrays_state,
        "PoolingCache": arrays_state,
    }
    for name, (classes, eligible) in cases.items():
        states = [states_by_class[c] for c in classes]
        plan = cachelist_pm_member_plan(classes, states)
        assert (plan is not None) == eligible, name
        if plan is not None:
            assert plan == ["slice", "boundary"]


def test_blocks_stored_per_member_sized(tmp_path):
    """The core assertion: block payloads hold per-block KV slices, not the
    cumulative prefix — and load re-tags them as per-member."""
    cache, ssd = _make_cache(tmp_path)
    num_blocks = 3
    table = _store_blocks(cache, num_blocks)
    assert table is not None
    assert len(table.block_ids) == num_blocks

    for idx, bid in enumerate(table.block_ids):
        block = cache.paged_cache.allocated_blocks[bid]
        payload, _meta = ssd.load_block_with_metadata(block.block_hash)
        assert payload is not None
        layer = payload[0]
        assert (
            isinstance(layer, tuple)
            and len(layer) == 2
            and layer[0] == "__cache_list_pm__"
        ), f"block {idx} not per-member tagged: {type(layer)}"
        subs = layer[1]
        kv_keys = subs[0][0]
        assert kv_keys.shape[2] == BLOCK_SIZE, (
            f"block {idx} KV member holds {kv_keys.shape[2]} tokens — "
            f"cumulative storage leaked back in"
        )


def test_pm_multiblock_roundtrip(tmp_path):
    cache, _ = _make_cache(tmp_path)
    table = _store_blocks(cache, num_blocks=3)
    _assert_restored(cache.reconstruct_cache(table), expected_seq_len=3 * BLOCK_SIZE)


def test_pm_partial_prefix_roundtrip(tmp_path):
    from omlx.cache.paged_cache import BlockTable

    cache, _ = _make_cache(tmp_path)
    table = _store_blocks(cache, num_blocks=3, request_id="req-part")
    for bid in table.block_ids[:2]:
        cache.paged_cache.allocated_blocks[bid].ref_count += 1
    partial = BlockTable(
        request_id="req-part-restore",
        block_ids=list(table.block_ids[:2]),
        num_tokens=2 * BLOCK_SIZE,
    )
    _assert_restored(cache.reconstruct_cache(partial), expected_seq_len=2 * BLOCK_SIZE)


def test_legacy_cumulative_blocks_still_restore(tmp_path, monkeypatch):
    """Blocks produced by the legacy cumulative path (pm plan ineligible)
    keep last-block restore semantics."""
    monkeypatch.setattr(
        prefix_cache_module, "cachelist_pm_member_plan", lambda *a, **k: None
    )
    cache, ssd = _make_cache(tmp_path)
    table = _store_blocks(cache, num_blocks=3, request_id="req-legacy")
    assert table is not None

    block = cache.paged_cache.allocated_blocks[table.block_ids[0]]
    payload, _ = ssd.load_block_with_metadata(block.block_hash)
    assert isinstance(payload[0], list), "legacy blocks must stay untagged"

    _assert_restored(cache.reconstruct_cache(table), expected_seq_len=3 * BLOCK_SIZE)


def test_mixed_format_chain_rejected(tmp_path, monkeypatch):
    """Legacy blocks + per-member blocks in one chain must reject (miss),
    never concatenate cumulative KV into a duplicated sequence."""
    cache, _ = _make_cache(tmp_path)

    monkeypatch.setattr(
        prefix_cache_module, "cachelist_pm_member_plan", lambda *a, **k: None
    )
    table = _store_blocks(cache, num_blocks=2, request_id="req-mix")
    assert table is not None
    monkeypatch.undo()

    # Extend the same request with two more blocks — now stored per-member.
    tokens = list(range(4 * BLOCK_SIZE))
    table = cache.store_cache(
        "req-mix",
        tokens,
        _cache_data(len(tokens)),
        boundary_snapshots=_boundary_snapshots(4),
    )
    assert table is not None
    assert len(table.block_ids) == 4

    assert cache.reconstruct_cache(table) is None


def test_filtered_snapshots_store_correctly(tmp_path):
    """Snapshots with blanked KV members (as produced by
    _extract_snapshot_cache_states) still yield correct per-member blocks —
    KV comes from the live cache, conv state from the snapshot."""
    cache, _ = _make_cache(tmp_path)
    table = _store_blocks(cache, num_blocks=3, request_id="req-blank", blank_kv=True)
    assert table is not None
    _assert_restored(cache.reconstruct_cache(table), expected_seq_len=3 * BLOCK_SIZE)
