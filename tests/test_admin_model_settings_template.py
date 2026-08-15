"""Regression tests for admin model-settings UI gates."""

from pathlib import Path


def _model_settings_template() -> str:
    root = Path(__file__).resolve().parents[1]
    return (
        root / "omlx/admin/templates/dashboard/_modal_model_settings.html"
    ).read_text()


def _section(html: str, start_marker: str, end_marker: str) -> str:
    return html.split(start_marker, 1)[1].split(end_marker, 1)[0]


def test_lightning_mtp_and_turboquant_are_not_ui_mutexed():
    html = _model_settings_template()

    turboquant = _section(
        html,
        "<!-- TurboQuant KV Cache -->",
        "<!-- IndexCache (DSA models only) -->",
    )
    lightning_mtp = _section(
        html,
        "<!-- Lightning MTP (built-in MTP head speculative decoding) -->",
        "<!-- Experimental Features -->",
    )

    assert "modelSettings.mtp_enabled" not in turboquant
    assert "modelSettings.turboquant_kv_enabled" not in lightning_mtp


def test_vlm_mtp_still_conflicts_with_turboquant():
    html = _model_settings_template()
    vlm_mtp = _section(
        html,
        "<!-- VLM MTP",
        "<!-- Performance",
    )

    assert "modelSettings.turboquant_kv_enabled" in vlm_mtp


def test_reasoning_effort_is_freeform_input_with_suggestions():
    """The effort entry must be free-form: models disagree on the vocabulary
    (Qwen3.8 "xhigh", Inkling numeric 0.1-0.99), so a fixed <select> silently
    degrades unknown values to its first option on panel reload."""
    html = _model_settings_template()

    # The effort input block is the SECOND template with this marker (the
    # first renders the entry label); enable_thinking's <select> sits between.
    marker = "<template x-if=\"entry.type === 'reasoning_effort'\">"
    section = html.split(marker, 2)[2].split(
        "<template x-if=\"entry.type !== 'custom'\">", 1
    )[0]

    assert '<input type="text"' in section
    assert 'list="reasoning-effort-options"' in section
    assert "<select" not in section

    datalist = _section(
        html, '<datalist id="reasoning-effort-options">', "</datalist>"
    )
    order = ["low", "medium", "high", "xhigh", "max"]
    positions = [datalist.index(f'value="{value}"') for value in order]
    assert positions == sorted(positions)


def test_reasoning_effort_add_guard_covers_custom_entries():
    """Stored reasoning_effort reloads as a custom entry, so the add-dropdown
    guard must hide the typed option when either shape owns the key."""
    html = _model_settings_template()

    guard = (
        "e.type === 'reasoning_effort' || "
        "(e.type === 'custom' && e.key && e.key.trim() === 'reasoning_effort')"
    )
    assert guard in html
