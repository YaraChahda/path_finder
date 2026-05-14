"""
Tests for src/path_finder/app_utensils.py

Covers the pure / non-Streamlit functions:
  strip_emoji, is_purification_step, load_banner,
  hires_fig, build_clickable_scheme_html, build_score_table_html,
  make_ranking_chart, make_yield_chart, make_comparison_chart,
  build_why_ranked_html.

Functions that call st.* at invocation time (display_route_card,
smiles_copy_widget, load_dataset_cached, get_targets_cached) are not
exercised here because they require a running Streamlit server.
"""

import sys
import os
import base64
import json
import tempfile

# conftest.py already stubs aizynthfinder; we only need to add src to path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "path_finder"))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI / headless tests
import matplotlib.pyplot as plt
import pytest

# Import app_utensils; this pulls in streamlit (installed), matplotlib, numpy
import src.path_finder.app_utensils as au


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(reactants, product, rtype="", y=None, cond=None, src="dataset"):
    return {
        "reactants_smiles": reactants,
        "product_smiles":   product,
        "reaction_type":    rtype,
        "yield_percent":    y,
        "conditions":       cond or {},
        "source":           src,
        "step_number":      1,
        "fg_reactants":     [],
    }


def _result(score, route_name, steps, status="dataset"):
    details = {
        "steps":        {"raw": 0.5, "weight": 0.73, "weighted": 0.365, "excluded": False},
        "yield":        {"raw": 0.8, "weight": 0.18, "weighted": 0.144, "excluded": False},
        "atom_economy": {"raw": 0.7, "weight": 0.09, "weighted": 0.063, "excluded": False},
        "_all_scores":  {"steps": 0.5, "yield": 0.8, "atom_economy": 0.7,
                         "e_factor": 0.6, "toxicity": 0.5},
    }
    route = {
        "matched_route_name": route_name,
        "dataset_steps":      steps,
        "validation_status":  status,
        "is_predicted":       (status == "predicted"),
    }
    return (score, details, route)


BENZENE = "c1ccccc1"
ETHANOL = "CCO"
ACETIC  = "CC(=O)O"


# ===========================================================================
# strip_emoji
# ===========================================================================

class TestStripEmoji:
    def test_plain_text_unchanged(self):
        assert au.strip_emoji("Hello World") == "Hello World"

    def test_emoji_removed(self):
        assert au.strip_emoji("🔍 Route Search") == "Route Search"

    def test_multiple_emoji_removed(self):
        result = au.strip_emoji("📊 Score ⚗️ Economy")
        assert "📊" not in result
        assert "⚗️" not in result
        assert "Score" in result
        assert "Economy" in result

    def test_leading_trailing_whitespace_stripped(self):
        result = au.strip_emoji("  🔍 Search  ")
        assert result == "Search"

    def test_empty_string_returns_empty(self):
        assert au.strip_emoji("") == ""

    def test_only_emoji_returns_empty(self):
        assert au.strip_emoji("🔬") == ""

    def test_text_with_no_emoji_unchanged(self):
        text = "Yield (%) — step 1"
        assert au.strip_emoji(text) == text


# ===========================================================================
# is_purification_step
# ===========================================================================

class TestIsPurificationStep:
    def test_purification_keyword_in_reaction_type(self):
        step = _step([BENZENE], ETHANOL, rtype="purification")
        assert au.is_purification_step(step) is True

    def test_recrystallisation_keyword(self):
        step = _step([BENZENE], ETHANOL, rtype="Recrystallisation")
        assert au.is_purification_step(step) is True

    def test_chromatography_keyword(self):
        step = _step([BENZENE], ETHANOL, rtype="Column chromatography")
        assert au.is_purification_step(step) is True

    def test_isolation_keyword(self):
        step = _step([BENZENE], ETHANOL, rtype="isolation workup")
        assert au.is_purification_step(step) is True

    def test_workup_keyword(self):
        step = _step([BENZENE], ETHANOL, rtype="workup")
        assert au.is_purification_step(step) is True

    def test_normal_reaction_not_purification(self):
        step = _step([BENZENE], ETHANOL, rtype="esterification")
        assert au.is_purification_step(step) is False

    def test_identity_transform_is_purification(self):
        # product == one of the reactants → purification/workup step
        step = _step([BENZENE, ETHANOL], BENZENE, rtype="")
        assert au.is_purification_step(step) is True

    def test_no_reaction_type_no_identity_is_not_purif(self):
        step = _step([BENZENE], ETHANOL, rtype="")
        assert au.is_purification_step(step) is False

    def test_empty_step_dict_returns_false(self):
        assert au.is_purification_step({}) is False


# ===========================================================================
# load_banner
# ===========================================================================

class TestLoadBanner:
    def test_missing_file_returns_empty_string(self):
        assert au.load_banner("/nonexistent/path/image.png") == ""

    def test_valid_file_returns_data_uri(self, tmp_path):
        # Write a tiny 1×1 PNG (valid PNG magic header)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        img_file = tmp_path / "test.png"
        img_file.write_bytes(png_bytes)
        result = au.load_banner(str(img_file))
        assert result.startswith("data:image/png;base64,")

    def test_data_uri_base64_is_valid(self, tmp_path):
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        result = au.load_banner(str(img_file))
        b64_part = result.split(",", 1)[1]
        base64.b64decode(b64_part)  # must not raise

    def test_jpg_mime_type(self, tmp_path):
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
        result = au.load_banner(str(img_file))
        assert result.startswith("data:image/jpg;base64,")


# ===========================================================================
# hires_fig
# ===========================================================================

class TestHiresFig:
    def teardown_method(self):
        plt.close("all")

    def test_returns_fig_and_ax(self):
        fig, ax = au.hires_fig()
        assert hasattr(fig, "savefig")
        assert hasattr(ax, "plot")

    def test_dpi_is_180(self):
        fig, _ = au.hires_fig()
        assert fig.dpi == 180

    def test_custom_dpi(self):
        fig, _ = au.hires_fig(dpi=96)
        assert fig.dpi == 96

    def test_multiple_axes_all_get_background(self):
        fig, axes = au.hires_fig(1, 2)
        from localization import FIG_BG
        for ax in axes:
            assert ax.get_facecolor() is not None


# ===========================================================================
# build_clickable_scheme_html
# ===========================================================================

class TestBuildClickableSchemeHtml:
    def test_empty_steps_returns_fallback(self):
        result = au.build_clickable_scheme_html([], "r1")
        assert "No steps" in result

    def test_returns_html_string(self):
        steps = [_step([BENZENE], ETHANOL, rtype="esterification", y=80)]
        result = au.build_clickable_scheme_html(steps, "route_test")
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result

    def test_contains_route_id(self):
        steps = [_step([BENZENE], ETHANOL)]
        result = au.build_clickable_scheme_html(steps, "myroute")
        assert "myroute" in result

    def test_predicted_route_uses_orange(self):
        steps = [_step([BENZENE], ETHANOL)]
        result = au.build_clickable_scheme_html(steps, "r1", is_predicted=True)
        assert "#E65100" in result  # orange colour for predicted routes

    def test_non_predicted_uses_navy(self):
        steps = [_step([BENZENE], ETHANOL)]
        result = au.build_clickable_scheme_html(steps, "r1", is_predicted=False)
        assert "#1a2e44" in result  # navy colour

    def test_purification_step_handled(self):
        # Identity transform → purification
        steps = [_step([BENZENE], BENZENE, rtype="")]
        result = au.build_clickable_scheme_html(steps, "r_purif")
        assert "Purification" in result

    def test_multi_step_scheme(self):
        s1 = _step([BENZENE], ETHANOL, rtype="reduction", y=85)
        s2 = _step([ETHANOL], ACETIC,  rtype="oxidation",  y=70)
        s1["step_number"] = 1
        s2["step_number"] = 2
        result = au.build_clickable_scheme_html([s1, s2], "multi")
        assert "Step 1" in result
        assert "Step 2" in result


# ===========================================================================
# build_score_table_html
# ===========================================================================

class TestBuildScoreTableHtml:
    CRITERIA = ["steps", "yield", "atom_economy"]
    WEIGHTS  = {"steps": 0.7347, "yield": 0.1837, "atom_economy": 0.0816}

    def _details(self, excluded_yield=False):
        d = {
            "steps":        {"raw": 0.5,  "weight": 0.7347, "weighted": 0.367, "excluded": False},
            "yield":        {"raw": None if excluded_yield else 0.8,
                             "weight": 0.0 if excluded_yield else 0.1837,
                             "weighted": 0.0 if excluded_yield else 0.147,
                             "excluded": excluded_yield},
            "atom_economy": {"raw": 0.7,  "weight": 0.0816, "weighted": 0.057, "excluded": False},
        }
        return d

    def test_returns_html_string(self):
        result = au.build_score_table_html(
            self._details(), self.CRITERIA, self.WEIGHTS
        )
        assert isinstance(result, str) and "<table" in result

    def test_contains_criterion_labels(self):
        result = au.build_score_table_html(
            self._details(), self.CRITERIA, self.WEIGHTS
        )
        assert "Steps" in result
        assert "Yield" in result

    def test_excluded_yield_shows_excluded_text(self):
        result = au.build_score_table_html(
            self._details(excluded_yield=True), self.CRITERIA, self.WEIGHTS
        )
        assert "excluded" in result.lower()

    def test_french_lang(self):
        result = au.build_score_table_html(
            self._details(), self.CRITERIA, self.WEIGHTS, lang="fr"
        )
        # French label for steps criterion
        assert "Étapes" in result

    def test_scores_are_formatted(self):
        result = au.build_score_table_html(
            self._details(), self.CRITERIA, self.WEIGHTS
        )
        assert "0.500" in result  # raw score for steps


# ===========================================================================
# make_ranking_chart
# ===========================================================================

class TestMakeRankingChart:
    def teardown_method(self):
        plt.close("all")

    def _make_results(self, n=3):
        steps = [_step([BENZENE], ETHANOL)]
        return [_result(1.0 - i * 0.1, f"Route {i+1}", steps) for i in range(n)]

    def test_returns_figure(self):
        results = self._make_results(2)
        fig = au.make_ranking_chart(results, "Morphine")
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_figure_has_one_axes(self):
        results = self._make_results(2)
        fig = au.make_ranking_chart(results, "Morphine")
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_target_name_in_title(self):
        results = self._make_results(2)
        fig = au.make_ranking_chart(results, "Aspirin")
        title = fig.axes[0].get_title()
        assert "Aspirin" in title
        plt.close(fig)

    def test_single_result(self):
        results = self._make_results(1)
        fig = au.make_ranking_chart(results, "Target")
        assert fig is not None
        plt.close(fig)


# ===========================================================================
# make_yield_chart
# ===========================================================================

class TestMakeYieldChart:
    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self):
        steps = [
            {"yield_percent": 80, "step_number": 1},
            {"yield_percent": 60, "step_number": 2},
        ]
        fig = au.make_yield_chart(steps)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_steps_without_yield_do_not_crash(self):
        steps = [
            {"yield_percent": None, "step_number": 1},
            {"yield_percent": 70,   "step_number": 2},
        ]
        fig = au.make_yield_chart(steps)
        assert fig is not None
        plt.close(fig)

    def test_all_missing_yields_no_crash(self):
        steps = [{"yield_percent": None}] * 3
        fig = au.make_yield_chart(steps)
        assert fig is not None
        plt.close(fig)

    def test_french_lang(self):
        steps = [{"yield_percent": 80, "step_number": 1}]
        fig = au.make_yield_chart(steps, lang="fr")
        assert fig is not None
        plt.close(fig)


# ===========================================================================
# make_comparison_chart
# ===========================================================================

class TestMakeComparisonChart:
    def teardown_method(self):
        plt.close("all")

    CRITERIA = ["steps", "yield", "atom_economy"]

    def _make_results(self, n=2):
        steps = [_step([BENZENE], ETHANOL)]
        return [_result(1.0 - i * 0.1, f"Route {i+1}", steps) for i in range(n)]

    def test_returns_figure(self):
        results = self._make_results(2)
        fig = au.make_comparison_chart(results, self.CRITERIA)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_single_route_no_crash(self):
        results = self._make_results(1)
        fig = au.make_comparison_chart(results, self.CRITERIA)
        assert fig is not None
        plt.close(fig)

    def test_yticks_match_criteria_count(self):
        results = self._make_results(2)
        fig = au.make_comparison_chart(results, self.CRITERIA)
        ax = fig.axes[0]
        assert len(ax.get_yticks()) == len(self.CRITERIA)
        plt.close(fig)

    def test_french_lang(self):
        results = self._make_results(2)
        fig = au.make_comparison_chart(results, self.CRITERIA, lang="fr")
        assert fig is not None
        plt.close(fig)


# ===========================================================================
# build_why_ranked_html
# ===========================================================================

class TestBuildWhyRankedHtml:
    CRITERIA = ["steps", "yield", "atom_economy"]
    WEIGHTS  = {"steps": 0.7347, "yield": 0.1837, "atom_economy": 0.0816}

    def _details(self):
        return {
            "steps":        {"raw": 0.5,  "weight": 0.73, "weighted": 0.365, "excluded": False},
            "yield":        {"raw": 0.8,  "weight": 0.18, "weighted": 0.144, "excluded": False},
            "atom_economy": {"raw": 0.7,  "weight": 0.09, "weighted": 0.063, "excluded": False},
        }

    def _steps(self, y=80):
        return [_step([BENZENE], ETHANOL, y=y)]

    def test_returns_html_string(self):
        result = au.build_why_ranked_html(
            1, 0.85, self._details(), self.CRITERIA, self.WEIGHTS, self._steps()
        )
        assert isinstance(result, str) and "<div" in result

    def test_contains_rank(self):
        result = au.build_why_ranked_html(
            2, 0.75, self._details(), self.CRITERIA, self.WEIGHTS, self._steps()
        )
        assert "#2" in result

    def test_steps_bullet_contains_count(self):
        # "steps" is first criterion so it generates a bullet about step count
        steps = self._steps()
        result = au.build_why_ranked_html(
            1, 0.85, self._details(), ["steps", "yield", "atom_economy"],
            self.WEIGHTS, steps
        )
        assert str(len(steps)) in result

    def test_yield_bullet_contains_bottleneck(self):
        steps = [_step([BENZENE], ETHANOL, y=73)]
        result = au.build_why_ranked_html(
            1, 0.85, self._details(), ["yield", "steps", "atom_economy"],
            self.WEIGHTS, steps
        )
        assert "73" in result

    def test_no_yield_data_does_not_crash(self):
        steps = [_step([BENZENE], ETHANOL, y=None)]
        result = au.build_why_ranked_html(
            1, 0.85, self._details(), ["yield", "steps", "atom_economy"],
            self.WEIGHTS, steps
        )
        assert isinstance(result, str)

    def test_french_lang(self):
        result = au.build_why_ranked_html(
            1, 0.85, self._details(), self.CRITERIA, self.WEIGHTS,
            self._steps(), lang="fr"
        )
        assert isinstance(result, str) and len(result) > 0

    def test_atom_economy_bullet_contains_percentage(self):
        result = au.build_why_ranked_html(
            1, 0.85, self._details(),
            ["atom_economy", "steps", "yield"],
            self.WEIGHTS, self._steps()
        )
        assert "%" in result

    def test_suffix_text_present(self):
        from localization import LANG
        result = au.build_why_ranked_html(
            1, 0.85, self._details(), self.CRITERIA, self.WEIGHTS, self._steps()
        )
        # The suffix explains the weighted combination
        suffix = LANG["en"]["why_suffix"]
        assert suffix[:30] in result