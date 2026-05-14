"""
Tests for src/path_finder/localization.py

Covers: LANG, CRITERIA_LABELS, PALETTE, FIG_BG
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "path_finder"))

from src.path_finder.localization import LANG, CRITERIA_LABELS, PALETTE, FIG_BG


# ---------------------------------------------------------------------------
# LANG structure
# ---------------------------------------------------------------------------

class TestLangStructure:
    def test_lang_has_english_and_french(self):
        assert "en" in LANG
        assert "fr" in LANG

    def test_english_and_french_have_same_keys(self):
        assert set(LANG["en"].keys()) == set(LANG["fr"].keys())

    def test_all_values_are_strings(self):
        for lang_code, strings in LANG.items():
            for key, val in strings.items():
                assert isinstance(val, str), (
                    f"LANG[{lang_code!r}][{key!r}] is not a string: {val!r}"
                )

    def test_no_empty_strings(self):
        for lang_code, strings in LANG.items():
            for key, val in strings.items():
                assert val.strip(), (
                    f"LANG[{lang_code!r}][{key!r}] is empty or whitespace-only"
                )


class TestLangKeyPresence:
    REQUIRED_KEYS = [
        "page_title", "page_caption", "tab_search", "tab_analysis",
        "tab_dataset", "tab_help", "run_btn", "welcome",
        "no_routes", "n_found", "chart_title", "score_axis",
        "metric_score", "metric_steps", "metric_bottleneck",
        "smiles_invalid", "smiles_valid",
        "sec_dataset", "sec_validated", "sec_predicted",
        "badge_dataset", "badge_validated", "badge_partial", "badge_predicted",
        "why_prefix", "why_suffix",
        "partial_badge", "partial_tip",
    ]

    def test_required_keys_present_in_english(self):
        for key in self.REQUIRED_KEYS:
            assert key in LANG["en"], f"Missing key in LANG['en']: {key!r}"

    def test_required_keys_present_in_french(self):
        for key in self.REQUIRED_KEYS:
            assert key in LANG["fr"], f"Missing key in LANG['fr']: {key!r}"


class TestLangFormatStrings:
    def test_n_found_has_n_placeholder(self):
        assert "{n}" in LANG["en"]["n_found"]
        assert "{n}" in LANG["fr"]["n_found"]

    def test_chart_title_has_target_placeholder(self):
        assert "{target}" in LANG["en"]["chart_title"]
        assert "{target}" in LANG["fr"]["chart_title"]

    def test_why_prefix_has_r_placeholder(self):
        assert "{r}" in LANG["en"]["why_prefix"]
        assert "{r}" in LANG["fr"]["why_prefix"]

    def test_partial_badge_placeholders(self):
        badge_en = LANG["en"]["partial_badge"]
        assert "{v}" in badge_en
        assert "{t}" in badge_en

    def test_format_n_found_interpolates_correctly(self):
        result = LANG["en"]["n_found"].format(n=5)
        assert "5" in result

    def test_format_chart_title_interpolates_correctly(self):
        result = LANG["en"]["chart_title"].format(target="Morphine")
        assert "Morphine" in result


# ---------------------------------------------------------------------------
# CRITERIA_LABELS
# ---------------------------------------------------------------------------

class TestCriteriaLabels:
    EXPECTED_CRITERIA = {"steps", "yield", "atom_economy", "e_factor", "toxicity"}

    def test_has_english_and_french(self):
        assert "en" in CRITERIA_LABELS
        assert "fr" in CRITERIA_LABELS

    def test_english_has_all_criteria(self):
        assert set(CRITERIA_LABELS["en"].keys()) == self.EXPECTED_CRITERIA

    def test_french_has_all_criteria(self):
        assert set(CRITERIA_LABELS["fr"].keys()) == self.EXPECTED_CRITERIA

    def test_all_labels_are_non_empty_strings(self):
        for lang_code, labels in CRITERIA_LABELS.items():
            for key, val in labels.items():
                assert isinstance(val, str) and val.strip(), (
                    f"CRITERIA_LABELS[{lang_code!r}][{key!r}] is empty"
                )

    def test_english_and_french_have_same_keys(self):
        assert set(CRITERIA_LABELS["en"].keys()) == set(CRITERIA_LABELS["fr"].keys())


# ---------------------------------------------------------------------------
# PALETTE
# ---------------------------------------------------------------------------

class TestPalette:
    def test_palette_is_list(self):
        assert isinstance(PALETTE, list)

    def test_palette_has_five_colours(self):
        assert len(PALETTE) == 5

    def test_all_entries_are_hex_strings(self):
        for colour in PALETTE:
            assert isinstance(colour, str), f"Not a string: {colour!r}"
            assert colour.startswith("#"), f"Not a hex colour: {colour!r}"
            assert len(colour) == 7, f"Unexpected hex length: {colour!r}"

    def test_all_hex_digits_valid(self):
        import re
        pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
        for colour in PALETTE:
            assert pattern.match(colour), f"Invalid hex colour: {colour!r}"


# ---------------------------------------------------------------------------
# FIG_BG
# ---------------------------------------------------------------------------

class TestFigBg:
    def test_fig_bg_is_string(self):
        assert isinstance(FIG_BG, str)

    def test_fig_bg_is_hex_colour(self):
        import re
        assert re.match(r"^#[0-9a-fA-F]{6}$", FIG_BG), (
            f"FIG_BG is not a valid 6-digit hex colour: {FIG_BG!r}"
        )