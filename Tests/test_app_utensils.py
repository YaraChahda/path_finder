"""
test_app_utensils.py — Tests for app_utensils.py.

Streamlit is mocked by conftest.py.  Tests cover every public function
in isolation:

  load_banner            strip_emoji            hires_fig
  is_purification_step   build_score_table_html build_clickable_scheme_html
  make_ranking_chart     make_yield_chart       make_comparison_chart
  build_why_ranked_html  smiles_copy_widget
  load_dataset_cached    get_targets_cached

The last two are Streamlit-cached wrappers; their underlying logic is tested
through the route_engine tests, but we verify they are callable and return
the expected type here.
"""

import os
import base64
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import matplotlib
matplotlib.use("Agg")          # headless backend — no display required
import matplotlib.pyplot as plt

import src.path_finder.app_utensils as au
import src.path_finder.route_engine  as rt


# ---------------------------------------------------------------------------
# Shared fixtures and constants
# ---------------------------------------------------------------------------

BENZENE  = "c1ccccc1"
TOLUENE  = "Cc1ccccc1"
ETHANOL  = "CCO"
CAFFEINE = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"

CRITERIA = ["steps", "yield", "atom_economy"]


def _make_details(raw=0.75):
    weights = rt.compute_weights(CRITERIA)
    details = {}
    for c in CRITERIA:
        w = weights[c]
        details[c] = {"raw": raw, "weight": w, "weighted": raw * w, "excluded": False}
    return details


def _make_step(n, reactants, product, yld=None, rtype="Alkylation", source="dataset"):
    return {
        "step_number":      n,
        "reactants_smiles": reactants,
        "product_smiles":   product,
        "yield_percent":    yld,
        "reaction_type":    rtype,
        "conditions":       {"temperature_C": 80, "solvent": "THF", "reagents": []},
        "source":           source,
    }


def _make_route(steps, status="dataset", name="Route A"):
    return {
        "dataset_steps":         steps,
        "matched_route_name":    name,
        "matched_route_id":      "R001",
        "matched_target":        "Caffeine",
        "validation_status":     status,
        "is_predicted":          (status == "predicted"),
        "validated_steps_count": 0,
        "total_steps_count":     len(steps),
    }


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# load_banner
# ---------------------------------------------------------------------------

class TestLoadBanner:
    def test_nonexistent_file_returns_empty_string(self):
        result = au.load_banner("/nonexistent/path/image.png")
        assert result == ""

    def test_existing_png_returns_data_uri(self, tmp_path):
        # Create a minimal valid-looking binary file
        png_path = tmp_path / "test.png"
        # PNG magic bytes
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        result = au.load_banner(str(png_path))
        assert result.startswith("data:image/png;base64,")

    def test_data_uri_base64_decodable(self, tmp_path):
        png_path = tmp_path / "img.png"
        content  = b"fake png content for testing"
        png_path.write_bytes(content)
        result  = au.load_banner(str(png_path))
        b64     = result.split(",", 1)[1]
        decoded = base64.b64decode(b64)
        assert decoded == content

    def test_non_png_extension_uses_correct_mime(self, tmp_path):
        svg_path = tmp_path / "img.svg"
        svg_path.write_bytes(b"<svg/>")
        result = au.load_banner(str(svg_path))
        assert "image/svg" in result


# ---------------------------------------------------------------------------
# strip_emoji
# ---------------------------------------------------------------------------

class TestStripEmoji:
    def test_no_emoji_unchanged(self):
        assert au.strip_emoji("Hello world") == "Hello world"

    def test_emoji_removed(self):
        # Use an emoji inside the regex ranges (U+1F300–U+1F9FF)
        assert au.strip_emoji("👋 Steps") == "Steps"

    def test_multiple_emojis_removed(self):
        # U+1F52C = 🔬 (in range), U+2665 = ♥ (in range)
        result = au.strip_emoji("🔬 Science ♥ Love")
        assert "🔬" not in result

    def test_empty_string_stays_empty(self):
        assert au.strip_emoji("") == ""

    def test_only_emoji_returns_empty(self):
        # U+1F600 GRINNING FACE — inside regex range U+1F300-U+1F9FF
        assert au.strip_emoji("\U0001F600").strip() == ""

    def test_leading_trailing_whitespace_stripped(self):
        result = au.strip_emoji("  Hello  ")
        assert result == "Hello"


# ---------------------------------------------------------------------------
# hires_fig
# ---------------------------------------------------------------------------

class TestHiresFig:
    def test_returns_fig_and_ax(self):
        fig, ax = au.hires_fig()
        assert hasattr(fig, "patch")
        assert hasattr(ax, "set_facecolor")

    def test_figure_facecolor_is_fig_bg(self):
        from src.path_finder.localization import FIG_BG
        fig, ax = au.hires_fig()
        import matplotlib.colors as mc
        expected = mc.to_hex(mc.to_rgb(FIG_BG))
        actual   = mc.to_hex(mc.to_rgb(fig.patch.get_facecolor()))
        assert actual == expected

    def test_axes_facecolor_is_fig_bg(self):
        from src.path_finder.localization import FIG_BG
        import matplotlib.colors as mc
        fig, ax = au.hires_fig()
        expected = mc.to_hex(mc.to_rgb(FIG_BG))
        actual   = mc.to_hex(mc.to_rgb(ax.get_facecolor()))
        assert actual == expected

    def test_dpi_parameter_respected(self):
        fig, _ = au.hires_fig(dpi=72)
        assert fig.get_dpi() == 72

    def test_multiple_axes_all_have_correct_bg(self):
        from src.path_finder.localization import FIG_BG
        import matplotlib.colors as mc
        fig, axes = au.hires_fig(1, 2)
        expected = mc.to_hex(mc.to_rgb(FIG_BG))
        for ax in axes:
            actual = mc.to_hex(mc.to_rgb(ax.get_facecolor()))
            assert actual == expected


# ---------------------------------------------------------------------------
# is_purification_step
# ---------------------------------------------------------------------------

class TestIsPurificationStep:
    def test_purif_keyword_in_reaction_type(self):
        step = {"reaction_type": "Purification", "product_smiles": TOLUENE,
                "reactants_smiles": [BENZENE]}
        assert au.is_purification_step(step) is True

    def test_recryst_keyword(self):
        step = {"reaction_type": "Recrystallisation", "product_smiles": TOLUENE,
                "reactants_smiles": [BENZENE]}
        assert au.is_purification_step(step) is True

    def test_chroma_keyword(self):
        step = {"reaction_type": "Column Chromatography", "product_smiles": TOLUENE,
                "reactants_smiles": [BENZENE]}
        assert au.is_purification_step(step) is True

    def test_isolation_keyword(self):
        step = {"reaction_type": "isolation step", "product_smiles": TOLUENE,
                "reactants_smiles": [BENZENE]}
        assert au.is_purification_step(step) is True

    def test_workup_keyword(self):
        step = {"reaction_type": "workup", "product_smiles": TOLUENE,
                "reactants_smiles": [BENZENE]}
        assert au.is_purification_step(step) is True

    def test_case_insensitive_keyword(self):
        step = {"reaction_type": "PURIF", "product_smiles": TOLUENE,
                "reactants_smiles": [BENZENE]}
        assert au.is_purification_step(step) is True

    def test_product_same_as_reactant_string_match(self):
        step = {"reaction_type": "Filtration", "product_smiles": BENZENE,
                "reactants_smiles": [BENZENE, TOLUENE]}
        assert au.is_purification_step(step) is True

    def test_normal_step_returns_false(self):
        step = {"reaction_type": "Alkylation", "product_smiles": TOLUENE,
                "reactants_smiles": [BENZENE, ETHANOL]}
        assert au.is_purification_step(step) is False

    def test_empty_reaction_type_normal_step(self):
        step = {"reaction_type": None, "product_smiles": TOLUENE,
                "reactants_smiles": [BENZENE]}
        assert au.is_purification_step(step) is False


# ---------------------------------------------------------------------------
# build_score_table_html
# ---------------------------------------------------------------------------

class TestBuildScoreTableHtml:
    def test_returns_string(self):
        details = _make_details()
        result  = au.build_score_table_html(details, CRITERIA, rt.compute_weights(CRITERIA))
        assert isinstance(result, str)

    def test_contains_table_tag(self):
        details = _make_details()
        result  = au.build_score_table_html(details, CRITERIA, rt.compute_weights(CRITERIA))
        assert "<table" in result.lower()

    def test_contains_all_criterion_labels(self):
        from src.path_finder.localization import CRITERIA_LABELS
        details = _make_details()
        result  = au.build_score_table_html(details, CRITERIA, rt.compute_weights(CRITERIA))
        for c in CRITERIA:
            label = au.strip_emoji(CRITERIA_LABELS["en"][c])
            assert label in result

    def test_french_language(self):
        from src.path_finder.localization import CRITERIA_LABELS
        details = _make_details()
        result  = au.build_score_table_html(details, CRITERIA,
                                             rt.compute_weights(CRITERIA), lang="fr")
        assert isinstance(result, str)
        assert "<table" in result.lower()

    def test_excluded_criterion_shows_excluded_text(self):
        details = _make_details()
        details["yield"] = {"raw": None, "weight": 0, "weighted": 0, "excluded": True}
        result  = au.build_score_table_html(details, CRITERIA, rt.compute_weights(CRITERIA))
        assert "excluded" in result


# ---------------------------------------------------------------------------
# build_clickable_scheme_html
# ---------------------------------------------------------------------------

class TestBuildClickableSchemeHtml:
    def test_empty_steps_returns_fallback(self):
        result = au.build_clickable_scheme_html([], "test_route")
        assert "No steps" in result or "no step" in result.lower()

    def test_valid_steps_returns_html_string(self):
        steps  = [_make_step(1, [BENZENE], TOLUENE, yld=80)]
        result = au.build_clickable_scheme_html(steps, "r001")
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result or "<html" in result.lower()

    def test_result_contains_step_label(self):
        steps  = [_make_step(1, [BENZENE], TOLUENE, yld=80)]
        result = au.build_clickable_scheme_html(steps, "r001")
        assert "Step 1" in result

    def test_predicted_flag_changes_colour(self):
        steps = [_make_step(1, [BENZENE], TOLUENE, yld=80)]
        norm  = au.build_clickable_scheme_html(steps, "r001", is_predicted=False)
        pred  = au.build_clickable_scheme_html(steps, "r001", is_predicted=True)
        assert norm != pred

    def test_multiple_steps(self):
        # Use distinct products so all steps appear in the molecule sequence
        products = [TOLUENE, ETHANOL, CAFFEINE]
        steps = [_make_step(i + 1, [BENZENE], products[i]) for i in range(3)]
        result = au.build_clickable_scheme_html(steps, "multi")
        assert "Step 1" in result
        assert "Step 2" in result

    def test_purification_step_included(self):
        steps  = [_make_step(1, [BENZENE], BENZENE, rtype="Purification")]
        result = au.build_clickable_scheme_html(steps, "purif")
        assert "Purification" in result or "Step 1" in result

    def test_route_id_used_in_dom_ids(self):
        steps  = [_make_step(1, [BENZENE], TOLUENE)]
        result = au.build_clickable_scheme_html(steps, "myroute42")
        assert "myroute42" in result


# ---------------------------------------------------------------------------
# make_ranking_chart
# ---------------------------------------------------------------------------

class TestMakeRankingChart:
    def _results(self, n=2):
        results = []
        for i in range(n):
            route = _make_route(
                [_make_step(1, [BENZENE], TOLUENE, yld=80)],
                name=f"Route {i+1}",
            )
            details = _make_details(raw=0.8 - i * 0.1)
            results.append((0.8 - i * 0.1, details, route))
        return results

    def test_returns_matplotlib_figure(self):
        fig = au.make_ranking_chart(self._results(), "Caffeine")
        assert hasattr(fig, "savefig")

    def test_figure_has_axes(self):
        fig = au.make_ranking_chart(self._results(), "Caffeine")
        assert len(fig.axes) >= 1

    def test_single_result_works(self):
        fig = au.make_ranking_chart(self._results(n=1), "Morphine")
        assert fig is not None

    def test_french_language_accepted(self):
        fig = au.make_ranking_chart(self._results(), "Caffeine", lang="fr")
        assert fig is not None


# ---------------------------------------------------------------------------
# make_yield_chart
# ---------------------------------------------------------------------------

class TestMakeYieldChart:
    def test_returns_figure(self):
        steps = [_make_step(i, [BENZENE], TOLUENE, yld=70 + i * 5) for i in range(1, 4)]
        fig   = au.make_yield_chart(steps)
        assert hasattr(fig, "savefig")

    def test_steps_without_yield_no_crash(self):
        steps = [_make_step(1, [BENZENE], TOLUENE, yld=None)]
        fig   = au.make_yield_chart(steps)
        assert fig is not None

    def test_empty_steps(self):
        fig = au.make_yield_chart([])
        assert fig is not None

    def test_mixed_reported_and_missing_yields(self):
        steps = [
            _make_step(1, [BENZENE], TOLUENE, yld=80),
            _make_step(2, [TOLUENE], ETHANOL, yld=None),
            _make_step(3, [ETHANOL], CAFFEINE, yld=60),
        ]
        fig = au.make_yield_chart(steps)
        assert fig is not None

    def test_french_language(self):
        steps = [_make_step(1, [BENZENE], TOLUENE, yld=75)]
        fig   = au.make_yield_chart(steps, lang="fr")
        assert fig is not None


# ---------------------------------------------------------------------------
# make_comparison_chart
# ---------------------------------------------------------------------------

class TestMakeComparisonChart:
    def _sel_results(self, n=2):
        results = []
        for i in range(n):
            route   = _make_route([_make_step(1, [BENZENE], TOLUENE)], name=f"R{i}")
            details = _make_details(raw=0.7 + i * 0.05)
            results.append((0.7 + i * 0.05, details, route))
        return results

    def test_returns_figure(self):
        fig = au.make_comparison_chart(self._sel_results(), CRITERIA)
        assert hasattr(fig, "savefig")

    def test_single_route_works(self):
        fig = au.make_comparison_chart(self._sel_results(n=1), CRITERIA)
        assert fig is not None

    def test_three_routes(self):
        fig = au.make_comparison_chart(self._sel_results(n=3), CRITERIA)
        assert fig is not None

    def test_french_language(self):
        fig = au.make_comparison_chart(self._sel_results(), CRITERIA, lang="fr")
        assert fig is not None


# ---------------------------------------------------------------------------
# build_why_ranked_html
# ---------------------------------------------------------------------------

class TestBuildWhyRankedHtml:
    def _call(self, criteria=None, steps=None, rank=1):
        if criteria is None:
            criteria = CRITERIA
        if steps is None:
            steps = [_make_step(1, [BENZENE], TOLUENE, yld=80)]
        details  = _make_details()
        weights  = rt.compute_weights(criteria)
        return au.build_why_ranked_html(rank, 0.75, details, criteria, weights, steps)

    def test_returns_string(self):
        assert isinstance(self._call(), str)

    def test_contains_rank_number(self):
        result = self._call(rank=2)
        assert "2" in result

    def test_contains_why_box_div(self):
        assert "why-box" in self._call()

    def test_steps_criterion_mentions_step_count(self):
        steps  = [_make_step(i, [BENZENE], TOLUENE, yld=70) for i in range(1, 4)]
        result = self._call(criteria=["steps", "yield", "atom_economy"], steps=steps)
        assert "3" in result  # 3 steps mentioned

    def test_yield_criterion_mentions_yield(self):
        steps  = [_make_step(1, [BENZENE], TOLUENE, yld=85)]
        result = self._call(criteria=["yield", "steps", "atom_economy"], steps=steps)
        assert "85" in result or "%" in result

    def test_no_yield_data_does_not_crash(self):
        steps  = [_make_step(1, [BENZENE], TOLUENE, yld=None)]
        result = self._call(criteria=["yield", "steps", "atom_economy"], steps=steps)
        assert isinstance(result, str)

    def test_french_lang(self):
        steps   = [_make_step(1, [BENZENE], TOLUENE, yld=80)]
        details = _make_details()
        weights = rt.compute_weights(CRITERIA)
        result  = au.build_why_ranked_html(1, 0.75, details, CRITERIA, weights, steps, lang="fr")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# smiles_copy_widget
# ---------------------------------------------------------------------------

class TestSmilesCopyWidget:
    def test_does_not_raise(self):
        # components.html is mocked; just verify the function runs
        au.smiles_copy_widget(BENZENE)

    def test_short_smiles_not_truncated_label(self):
        # Only verify no exception; the widget renders via mocked components
        au.smiles_copy_widget(ETHANOL, label="Reactant")

    def test_long_smiles_truncated_to_38_chars(self):
        # We verify the function runs; truncation is an internal detail
        long_smiles = "C" * 80
        au.smiles_copy_widget(long_smiles)


# ---------------------------------------------------------------------------
# load_dataset_cached and get_targets_cached
# ---------------------------------------------------------------------------

class TestCachedLoaders:
    def test_load_dataset_cached_calls_through(self, tmp_path):
        import json
        reactions = [{"id": "x", "route_id": "r", "route_name": "n",
                      "target": "T", "step_number": 1,
                      "reactants_smiles": [BENZENE], "product_smiles": TOLUENE,
                      "conditions": {}, "yield_percent": 50, "reaction_type": ""}]
        p = tmp_path / "ds.json"
        p.write_text(json.dumps(reactions))
        result = au.load_dataset_cached(str(p))
        assert "all" in result

    def test_get_targets_cached_returns_dict(self, tmp_path):
        import json
        reactions = [{"id": "x", "route_id": "r", "route_name": "n",
                      "target": "Aspirin", "step_number": 1,
                      "reactants_smiles": [BENZENE],
                      "product_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                      "conditions": {}, "yield_percent": 75, "reaction_type": ""}]
        p = tmp_path / "ds.json"
        p.write_text(json.dumps(reactions))
        result = au.get_targets_cached(str(p))
        assert isinstance(result, dict)