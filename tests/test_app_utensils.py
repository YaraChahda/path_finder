"""Tests for app_utensils.py.

Streamlit, route_engine, and rdkit are mocked so the suite runs without
a Streamlit server context and without heavy optional dependencies.
"""

import sys
import os
import tempfile
from unittest.mock import MagicMock, patch

# Mock heavy dependencies before any local import
for mod in [
    "streamlit",
    "streamlit.components",
    "streamlit.components.v1",
    "aizynthfinder",
    "aizynthfinder.aizynthfinder",
    "rxn_insight",
    "rxn_insight.reaction",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

sys.modules["aizynthfinder.aizynthfinder"].AiZynthFinder = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "path_finder"))

import pytest

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False

MPL_SKIP = pytest.mark.skipif(not MPL_OK, reason="matplotlib not available")

try:
    import numpy as np
    NP_OK = True
except ImportError:
    NP_OK = False

from src.path_finder.app_utensils import (
    load_banner,
    strip_emoji,
    is_purification_step,
    build_clickable_scheme_html,
    build_score_table_html,
    COMPONENT_STYLE,
    CRITERIA_SCORE_DESC,
)

if MPL_OK and NP_OK:
    from src.path_finder.app_utensils import (
        hires_fig,
        make_ranking_chart,
        make_yield_chart,
        make_comparison_chart,
    )


# Shared test data
BENZENE = "c1ccccc1"
TOLUENE = "Cc1ccccc1"
ASPIRIN  = "CC(=O)Oc1ccccc1C(=O)O"


def _make_step(n, reactants, product, yield_pct=None, reaction_type="Alkylation"):
    return {
        "step_number":      n,
        "reactants_smiles": reactants,
        "product_smiles":   product,
        "yield_percent":    yield_pct,
        "reaction_type":    reaction_type,
        "source":           "dataset",
        "conditions":       {"temperature_C": 60, "solvent": "THF", "reagents": []},
        "fg_reactants":     [],
    }


# ---------------------------------------------------------------------------
# load_banner
# ---------------------------------------------------------------------------

def test_load_banner_missing_file():
    result = load_banner("/no/such/file.png")
    assert result == ""


def test_load_banner_existing_png(tmp_path):
    png = tmp_path / "test.png"
    # Minimal valid PNG header
    png.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    result = load_banner(str(png))
    assert result.startswith("data:image/png;base64,")


def test_load_banner_non_png_extension(tmp_path):
    jpg = tmp_path / "test.jpg"
    jpg.write_bytes(b"\xff\xd8\xff\xe0some_jpeg_data")
    result = load_banner(str(jpg))
    assert result.startswith("data:image/jpg;base64,")


def test_load_banner_returns_string():
    result = load_banner("/nonexistent.png")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# strip_emoji
# ---------------------------------------------------------------------------

def test_strip_emoji_no_emoji():
    assert strip_emoji("Hello World") == "Hello World"


def test_strip_emoji_removes_emoji():
    result = strip_emoji("Hello 🌍 World")
    assert "🌍" not in result
    assert "Hello" in result
    assert "World" in result


def test_strip_emoji_only_emoji():
    result = strip_emoji("🔬⚗️🧪")
    assert result.strip() == ""


def test_strip_emoji_strips_whitespace():
    result = strip_emoji("  hello  ")
    assert result == "hello"


def test_strip_emoji_mixed_emoji_and_text():
    result = strip_emoji("📊 Analysis")
    assert result.strip() == "Analysis"


def test_strip_emoji_returns_string():
    assert isinstance(strip_emoji("test"), str)


def test_strip_emoji_empty_string():
    assert strip_emoji("") == ""


def test_strip_emoji_unicode_non_emoji():
    result = strip_emoji("café résumé")
    assert result == "café résumé"


# ---------------------------------------------------------------------------
# is_purification_step
# ---------------------------------------------------------------------------

def test_is_purification_step_chromatography():
    step = {"reaction_type": "Column chromatography", "product_smiles": BENZENE,
            "reactants_smiles": [TOLUENE]}
    assert is_purification_step(step) is True


def test_is_purification_step_recryst():
    step = {"reaction_type": "Recrystallisation", "product_smiles": BENZENE,
            "reactants_smiles": [TOLUENE]}
    assert is_purification_step(step) is True


def test_is_purification_step_workup():
    step = {"reaction_type": "Workup procedure", "product_smiles": BENZENE,
            "reactants_smiles": [TOLUENE]}
    assert is_purification_step(step) is True


def test_is_purification_step_isolation():
    step = {"reaction_type": "Isolation", "product_smiles": BENZENE,
            "reactants_smiles": [TOLUENE]}
    assert is_purification_step(step) is True


def test_is_purification_step_purif():
    step = {"reaction_type": "Purification by flash", "product_smiles": BENZENE,
            "reactants_smiles": [TOLUENE]}
    assert is_purification_step(step) is True


def test_is_purification_step_identity_transform():
    step = {"reaction_type": "", "product_smiles": BENZENE,
            "reactants_smiles": [BENZENE, TOLUENE]}
    assert is_purification_step(step) is True


def test_is_purification_step_normal_reaction():
    step = {"reaction_type": "Suzuki coupling", "product_smiles": TOLUENE,
            "reactants_smiles": [BENZENE]}
    assert is_purification_step(step) is False


def test_is_purification_step_case_insensitive():
    step = {"reaction_type": "CHROMA separation", "product_smiles": BENZENE,
            "reactants_smiles": [TOLUENE]}
    assert is_purification_step(step) is True


def test_is_purification_step_missing_reaction_type():
    step = {"product_smiles": TOLUENE, "reactants_smiles": [BENZENE]}
    assert isinstance(is_purification_step(step), bool)


# ---------------------------------------------------------------------------
# build_clickable_scheme_html
# ---------------------------------------------------------------------------

def test_build_clickable_scheme_html_empty_steps():
    result = build_clickable_scheme_html([], "rid")
    assert isinstance(result, str)
    assert "No steps" in result


def test_build_clickable_scheme_html_returns_html():
    steps = [_make_step(1, [BENZENE], TOLUENE, 80.0)]
    result = build_clickable_scheme_html(steps, "test_route")
    assert isinstance(result, str)
    assert "<!DOCTYPE html>" in result or "<html" in result


def test_build_clickable_scheme_html_contains_route_id():
    steps = [_make_step(1, [BENZENE], TOLUENE)]
    result = build_clickable_scheme_html(steps, "myroute123")
    assert "myroute123" in result


def test_build_clickable_scheme_html_predicted_uses_orange():
    steps = [_make_step(1, [BENZENE], TOLUENE)]
    result = build_clickable_scheme_html(steps, "r1", is_predicted=True)
    assert "E65100" in result or "#E65100" in result


def test_build_clickable_scheme_html_validated_uses_navy():
    steps = [_make_step(1, [BENZENE], TOLUENE)]
    result = build_clickable_scheme_html(steps, "r1", is_predicted=False)
    assert "1a2e44" in result


def test_build_clickable_scheme_html_multiple_steps():
    steps = [
        _make_step(1, [BENZENE], TOLUENE, 80.0),
        _make_step(2, [TOLUENE], ASPIRIN, 65.0),
    ]
    result = build_clickable_scheme_html(steps, "r2")
    assert "Step 1" in result
    assert "Step 2" in result


def test_build_clickable_scheme_html_purification_step():
    steps = [_make_step(1, [BENZENE], BENZENE, reaction_type="Column chromatography")]
    result = build_clickable_scheme_html(steps, "rp")
    assert isinstance(result, str)


def test_build_clickable_scheme_html_contains_javascript():
    steps = [_make_step(1, [BENZENE], TOLUENE)]
    result = build_clickable_scheme_html(steps, "r1")
    assert "<script" in result


# ---------------------------------------------------------------------------
# build_score_table_html
# ---------------------------------------------------------------------------

def _make_details(criteria):
    return {
        c: {"raw": 0.75, "weight": 1 / len(criteria), "weighted": 0.75 / len(criteria),
            "excluded": False}
        for c in criteria
    }


def test_build_score_table_html_returns_string():
    criteria = ["steps", "yield", "atom_economy"]
    details  = _make_details(criteria)
    weights  = {"steps": 0.73, "yield": 0.18, "atom_economy": 0.09}
    result   = build_score_table_html(details, criteria, weights, "en")
    assert isinstance(result, str)


def test_build_score_table_html_contains_table():
    criteria = ["steps", "yield", "atom_economy"]
    details  = _make_details(criteria)
    weights  = {"steps": 0.73, "yield": 0.18, "atom_economy": 0.09}
    result   = build_score_table_html(details, criteria, weights, "en")
    assert "<table" in result


def test_build_score_table_html_excluded_criterion():
    criteria = ["steps", "yield", "atom_economy"]
    details  = _make_details(criteria)
    details["yield"] = {"raw": None, "weight": 0.0, "weighted": 0.0, "excluded": True}
    weights  = {"steps": 0.73, "yield": 0.0, "atom_economy": 0.09}
    result   = build_score_table_html(details, criteria, weights, "en")
    assert "excluded" in result


def test_build_score_table_html_french():
    criteria = ["steps", "yield", "atom_economy"]
    details  = _make_details(criteria)
    weights  = {"steps": 0.73, "yield": 0.18, "atom_economy": 0.09}
    result   = build_score_table_html(details, criteria, weights, "fr")
    assert isinstance(result, str)
    assert "<table" in result


def test_build_score_table_html_contains_component_style():
    criteria = ["steps", "yield", "atom_economy"]
    details  = _make_details(criteria)
    weights  = {"steps": 0.73, "yield": 0.18, "atom_economy": 0.09}
    result   = build_score_table_html(details, criteria, weights, "en")
    assert "score-table" in result


# ---------------------------------------------------------------------------
# hires_fig
# ---------------------------------------------------------------------------

@MPL_SKIP
def test_hires_fig_returns_fig_and_ax():
    fig, ax = hires_fig()
    assert fig is not None
    assert ax is not None
    plt.close(fig)


@MPL_SKIP
def test_hires_fig_custom_dpi():
    fig, ax = hires_fig(dpi=150)
    assert fig.get_dpi() == 150.0
    plt.close(fig)


@MPL_SKIP
def test_hires_fig_background_color():
    from path_finder.app_layout import FIG_BG
    fig, ax = hires_fig()
    assert fig.get_facecolor() is not None
    plt.close(fig)


@MPL_SKIP
def test_hires_fig_with_subplots():
    fig, axes = hires_fig(1, 2)
    assert len(axes) == 2
    plt.close(fig)


# ---------------------------------------------------------------------------
# make_ranking_chart
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (MPL_OK and NP_OK), reason="matplotlib/numpy not available")
def test_make_ranking_chart_returns_figure():
    results = [
        (0.85, {}, {"matched_route_name": "Route A"}),
        (0.70, {}, {"matched_route_name": "Route B"}),
    ]
    fig = make_ranking_chart(results, "Aspirin", "en")
    assert fig is not None
    plt.close(fig)


@pytest.mark.skipif(not (MPL_OK and NP_OK), reason="matplotlib/numpy not available")
def test_make_ranking_chart_single_route():
    results = [(0.9, {}, {"matched_route_name": "Only Route"})]
    fig = make_ranking_chart(results, "Morphine", "en")
    assert fig is not None
    plt.close(fig)


@pytest.mark.skipif(not (MPL_OK and NP_OK), reason="matplotlib/numpy not available")
def test_make_ranking_chart_french():
    results = [(0.9, {}, {"matched_route_name": "Route 1"})]
    fig = make_ranking_chart(results, "Morphine", "fr")
    assert fig is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# make_yield_chart
# ---------------------------------------------------------------------------

def _make_yield_steps():
    return [
        {"step_number": 1, "yield_percent": 80},
        {"step_number": 2, "yield_percent": None},
        {"step_number": 3, "yield_percent": 65},
    ]


@pytest.mark.skipif(not (MPL_OK and NP_OK), reason="matplotlib/numpy not available")
def test_make_yield_chart_returns_figure():
    fig = make_yield_chart(_make_yield_steps(), "en")
    assert fig is not None
    plt.close(fig)


@pytest.mark.skipif(not (MPL_OK and NP_OK), reason="matplotlib/numpy not available")
def test_make_yield_chart_empty_steps():
    fig = make_yield_chart([], "en")
    assert fig is not None
    plt.close(fig)


@pytest.mark.skipif(not (MPL_OK and NP_OK), reason="matplotlib/numpy not available")
def test_make_yield_chart_no_reported_yields():
    steps = [{"step_number": 1, "yield_percent": None}]
    fig = make_yield_chart(steps, "en")
    assert fig is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# make_comparison_chart
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (MPL_OK and NP_OK), reason="matplotlib/numpy not available")
def test_make_comparison_chart_returns_figure():
    criteria = ["steps", "yield", "atom_economy"]
    sel_results = [
        (0.8, {c: {"raw": 0.8} for c in criteria}, {"matched_route_name": "R1"}),
        (0.7, {c: {"raw": 0.7} for c in criteria}, {"matched_route_name": "R2"}),
    ]
    fig = make_comparison_chart(sel_results, criteria, "en")
    assert fig is not None
    plt.close(fig)


@pytest.mark.skipif(not (MPL_OK and NP_OK), reason="matplotlib/numpy not available")
def test_make_comparison_chart_single_route():
    criteria = ["steps", "yield", "atom_economy"]
    sel_results = [
        (0.8, {c: {"raw": 0.8} for c in criteria}, {"matched_route_name": "R1"}),
    ]
    fig = make_comparison_chart(sel_results, criteria, "fr")
    assert fig is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# COMPONENT_STYLE & CRITERIA_SCORE_DESC
# ---------------------------------------------------------------------------

def test_component_style_is_string():
    assert isinstance(COMPONENT_STYLE, str)


def test_component_style_contains_css():
    assert "<style>" in COMPONENT_STYLE


def test_criteria_score_desc_has_en_and_fr():
    assert "en" in CRITERIA_SCORE_DESC
    assert "fr" in CRITERIA_SCORE_DESC


def test_criteria_score_desc_has_all_criteria():
    for lang in ("en", "fr"):
        for key in ("steps", "yield", "atom_economy", "e_factor", "toxicity"):
            assert key in CRITERIA_SCORE_DESC[lang]


def test_criteria_score_desc_values_are_strings():
    for lang in ("en", "fr"):
        for k, v in CRITERIA_SCORE_DESC[lang].items():
            assert isinstance(v, str)

# --- build_why_ranked_html ---
from src.path_finder.app_utensils import build_why_ranked_html

def _why_details(criteria):
    return {c:{"raw":0.7,"weight":0.33,"weighted":0.23,"excluded":False} for c in criteria}

def test_why_ranked_steps_criterion():
    c = ["steps","yield","atom_economy"]
    result = build_why_ranked_html(1, 0.8, _why_details(c), c,
                                   {"steps":0.73,"yield":0.18,"atom_economy":0.09},
                                   [_make_step(1,[BENZENE],TOLUENE,75.0)], "en")
    assert "<div" in result

def test_why_ranked_yield_with_bottleneck():
    c = ["yield","steps","atom_economy"]
    result = build_why_ranked_html(1, 0.7, _why_details(c), c,
                                   {"yield":0.73,"steps":0.18,"atom_economy":0.09},
                                   [_make_step(1,[BENZENE],TOLUENE,65.0)], "en")
    assert "65" in result

def test_why_ranked_yield_no_data():
    c = ["yield","steps","atom_economy"]
    result = build_why_ranked_html(1, 0.5, _why_details(c), c,
                                   {"yield":0.73,"steps":0.18,"atom_economy":0.09},
                                   [], "en")
    assert isinstance(result, str)

def test_why_ranked_atom_economy():
    c = ["atom_economy","yield","steps"]
    result = build_why_ranked_html(1, 0.9, _why_details(c), c,
                                   {"atom_economy":0.73,"yield":0.18,"steps":0.09},
                                   [], "en")
    assert "atom economy" in result

def test_why_ranked_e_factor():
    c = ["e_factor","yield","steps"]
    result = build_why_ranked_html(1, 0.8, _why_details(c), c,
                                   {"e_factor":0.73,"yield":0.18,"steps":0.09},
                                   [], "en")
    assert "E-factor" in result

def test_why_ranked_toxicity():
    c = ["toxicity","yield","steps"]
    result = build_why_ranked_html(1, 0.8, _why_details(c), c,
                                   {"toxicity":0.73,"yield":0.18,"steps":0.09},
                                   [], "en")
    assert "safety" in result

def test_why_ranked_french():
    c = ["steps","yield","atom_economy"]
    result = build_why_ranked_html(2, 0.75, _why_details(c), c,
                                   {"steps":0.73,"yield":0.18,"atom_economy":0.09},
                                   [_make_step(1,[BENZENE],TOLUENE,80.0)], "fr")
    assert isinstance(result, str)

# --- load_dataset_cached / get_targets_cached ---
def test_load_dataset_cached_is_callable():
    from src.path_finder.app_utensils import load_dataset_cached
    assert callable(load_dataset_cached)

def test_get_targets_cached_is_callable():
    from src.path_finder.app_utensils import get_targets_cached
    assert callable(get_targets_cached)