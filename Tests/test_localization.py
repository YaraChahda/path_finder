"""
test_localization.py — Tests for localization.py.

localization.py is pure Python with no external runtime dependencies, so all
tests run unconditionally.  We verify structural correctness (all required
keys present, types match) and a sample of string/value content.
"""

import pytest
from src.path_finder.localization import LANG, CRITERIA_LABELS, PALETTE, FIG_BG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUPPORTED_LANGS   = ("en", "fr")
CRITERIA_KEYS     = ("steps", "yield", "atom_economy", "e_factor", "toxicity")

# A representative subset of keys every language block must contain.
REQUIRED_LANG_KEYS = [
    "page_title", "page_caption", "sidebar_title", "tab_search",
    "tab_analysis", "tab_dataset", "tab_help", "run_btn",
    "metric_score", "metric_steps", "metric_bottleneck", "metric_avg",
    "score_th_crit", "score_th_raw", "score_th_weight", "score_th_contrib",
    "why_steps", "why_yield", "why_prefix", "why_suffix",
    "n_found", "chart_title", "footer",
]


# ---------------------------------------------------------------------------
# LANG structure
# ---------------------------------------------------------------------------

class TestLang:
    def test_supported_languages_present(self):
        for lang in SUPPORTED_LANGS:
            assert lang in LANG, f"Missing language block: {lang!r}"

    @pytest.mark.parametrize("lang", SUPPORTED_LANGS)
    def test_required_keys_present(self, lang):
        block = LANG[lang]
        for key in REQUIRED_LANG_KEYS:
            assert key in block, f"LANG[{lang!r}] missing key {key!r}"

    @pytest.mark.parametrize("lang", SUPPORTED_LANGS)
    def test_all_values_are_strings(self, lang):
        for key, val in LANG[lang].items():
            assert isinstance(val, str), (
                f"LANG[{lang!r}][{key!r}] should be str, got {type(val).__name__}"
            )

    @pytest.mark.parametrize("lang", SUPPORTED_LANGS)
    def test_no_empty_strings(self, lang):
        for key, val in LANG[lang].items():
            assert val.strip(), f"LANG[{lang!r}][{key!r}] is empty"

    def test_english_page_title_contains_synthesis(self):
        assert "Synthesis" in LANG["en"]["page_title"] or "Route" in LANG["en"]["page_title"]

    def test_french_page_title_is_different_from_english(self):
        assert LANG["fr"]["page_title"] != LANG["en"]["page_title"]

    def test_why_steps_format_placeholder(self):
        # The {n} placeholder must be present for .format(n=...) to work.
        assert "{n}" in LANG["en"]["why_steps"]
        assert "{n}" in LANG["fr"]["why_steps"]

    def test_why_yield_format_placeholder(self):
        assert "{y}" in LANG["en"]["why_yield"]
        assert "{y}" in LANG["fr"]["why_yield"]

    def test_n_found_format_placeholder(self):
        assert "{n}" in LANG["en"]["n_found"]
        assert LANG["en"]["n_found"].format(n=3)  # should not raise

    def test_chart_title_format_placeholder(self):
        assert "{target}" in LANG["en"]["chart_title"]
        assert LANG["en"]["chart_title"].format(target="Galanthamine")  # no error

    def test_partial_badge_format_placeholders(self):
        assert "{v}" in LANG["en"]["partial_badge"]
        assert "{t}" in LANG["en"]["partial_badge"]

    @pytest.mark.parametrize("lang", SUPPORTED_LANGS)
    def test_why_prefix_rank_placeholder(self, lang):
        assert "{r}" in LANG[lang]["why_prefix"]
        # Must be usable
        LANG[lang]["why_prefix"].format(r=1)


# ---------------------------------------------------------------------------
# CRITERIA_LABELS structure
# ---------------------------------------------------------------------------

class TestCriteriaLabels:
    def test_languages_present(self):
        for lang in SUPPORTED_LANGS:
            assert lang in CRITERIA_LABELS

    @pytest.mark.parametrize("lang", SUPPORTED_LANGS)
    def test_all_criteria_keys_present(self, lang):
        block = CRITERIA_LABELS[lang]
        for crit in CRITERIA_KEYS:
            assert crit in block, f"CRITERIA_LABELS[{lang!r}] missing {crit!r}"

    @pytest.mark.parametrize("lang", SUPPORTED_LANGS)
    def test_all_label_values_are_non_empty_strings(self, lang):
        for crit, label in CRITERIA_LABELS[lang].items():
            assert isinstance(label, str) and label.strip(), (
                f"CRITERIA_LABELS[{lang!r}][{crit!r}] must be a non-empty string"
            )

    def test_english_and_french_steps_differ(self):
        # Not strictly required but good as a sanity check.
        en = CRITERIA_LABELS["en"]["steps"]
        fr = CRITERIA_LABELS["fr"]["steps"]
        assert en != fr, "English and French labels for 'steps' should differ"


# ---------------------------------------------------------------------------
# PALETTE
# ---------------------------------------------------------------------------

class TestPalette:
    def test_palette_is_list(self):
        assert isinstance(PALETTE, list)

    def test_palette_is_non_empty(self):
        assert len(PALETTE) >= 3

    def test_palette_colours_are_hex_strings(self):
        import re
        hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for colour in PALETTE:
            assert isinstance(colour, str)
            assert hex_re.match(colour), f"{colour!r} is not a valid hex colour"

    def test_palette_has_distinct_colours(self):
        assert len(set(PALETTE)) == len(PALETTE), "PALETTE should have no duplicates"


# ---------------------------------------------------------------------------
# FIG_BG
# ---------------------------------------------------------------------------

class TestFigBg:
    def test_fig_bg_is_string(self):
        assert isinstance(FIG_BG, str)

    def test_fig_bg_is_valid_hex(self):
        import re
        assert re.match(r"^#[0-9A-Fa-f]{6}$", FIG_BG), (
            f"FIG_BG {FIG_BG!r} is not a valid 6-digit hex colour"
        )