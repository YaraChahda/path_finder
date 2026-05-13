"""
Tests for path_finder.localization

Verifies that the LANG, CRITERIA_LABELS, PALETTE, and FIG_BG constants
are well-formed and self-consistent across supported languages.
No Streamlit or RDKit dependency — pure Python.
"""

import pytest
from path_finder.localization import LANG, CRITERIA_LABELS, PALETTE, FIG_BG

SUPPORTED_LANGS = ["en", "fr"]

# All string keys that must be present in every language entry.
REQUIRED_LANG_KEYS = [
    "page_title", "page_caption", "sidebar_title", "files_section",
    "ds_label", "tox_label", "cfg_label", "rxni_label", "generic_label",
    "topn_label", "naiz_label", "weights_exp",
    "tab_search", "tab_analysis", "tab_dataset", "tab_help",
    "target_section", "mode_pre", "mode_custom",
    "mol_label", "smiles_label", "smiles_ph", "smiles_invalid", "smiles_valid",
    "criteria_section", "criteria_caption", "c1_label", "c2_label", "c3_label",
    "run_btn", "welcome",
    "loading_ds", "loading_aiz", "loading_rxni",
    "search_ok", "err_file", "err_param", "err_other",
    "no_routes", "n_found",
    "chart_title", "score_axis",
    "metric_score", "metric_steps", "metric_bottleneck", "metric_avg",
    "contrib_title",
    "score_th_crit", "score_th_raw", "score_th_weight", "score_th_contrib",
    "why_best_title", "flow_title", "steps_title",
    "yield_ok", "yield_na", "yield_predicted",
    "cond_lbl", "smi_exp", "react_lbl", "prod_lbl",
    "compare_title", "sel_routes", "radar_title", "no_analysis",
    "ds_not_found", "filter_lbl", "filter_all",
    "total_rxn", "dist_routes", "tgt_mols",
    "col_route", "col_target", "col_steps", "col_cumyield", "col_missing", "col_avg",
    "detail_sel", "help_title", "footer", "mod_err", "searching",
    "sec_dataset", "sec_validated", "sec_predicted",
    "cap_dataset", "cap_validated", "cap_predicted",
    "badge_dataset", "badge_validated", "badge_partial", "badge_predicted",
    "why_steps", "why_yield", "why_avg_yield", "why_top_score",
    "why_balanced", "why_prefix", "why_suffix",
    "help_ds", "help_tox", "help_cfg", "help_rxni", "help_generic",
    "help_topn", "help_naiz",
    "partial_badge", "partial_tip",
]

CRITERIA_KEYS = ["steps", "yield", "atom_economy", "e_factor", "toxicity"]


# ===========================================================================
# LANG structure
# ===========================================================================

class TestLang:
    def test_supported_languages_present(self):
        for lang in SUPPORTED_LANGS:
            assert lang in LANG, f"Language '{lang}' missing from LANG"

    @pytest.mark.parametrize("lang", SUPPORTED_LANGS)
    def test_all_required_keys_present(self, lang):
        missing = [k for k in REQUIRED_LANG_KEYS if k not in LANG[lang]]
        assert missing == [], f"Missing keys in LANG['{lang}']: {missing}"

    @pytest.mark.parametrize("lang", SUPPORTED_LANGS)
    def test_all_values_are_strings(self, lang):
        for key, value in LANG[lang].items():
            assert isinstance(value, str), (
                f"LANG['{lang}']['{key}'] is {type(value).__name__}, expected str"
            )

    @pytest.mark.parametrize("lang", SUPPORTED_LANGS)
    def test_no_empty_values(self, lang):
        empty = [k for k, v in LANG[lang].items() if v == ""]
        assert empty == [], f"Empty string values in LANG['{lang}']: {empty}"

    def test_en_and_fr_have_same_keys(self):
        assert set(LANG["en"].keys()) == set(LANG["fr"].keys()), (
            "Key sets differ between 'en' and 'fr'"
        )

    @pytest.mark.parametrize("lang", SUPPORTED_LANGS)
    def test_n_found_contains_placeholder(self, lang):
        assert "{n}" in LANG[lang]["n_found"]

    @pytest.mark.parametrize("lang", SUPPORTED_LANGS)
    def test_chart_title_contains_placeholder(self, lang):
        assert "{target}" in LANG[lang]["chart_title"]

    @pytest.mark.parametrize("lang", SUPPORTED_LANGS)
    def test_partial_badge_placeholders(self, lang):
        assert "{v}" in LANG[lang]["partial_badge"]
        assert "{t}" in LANG[lang]["partial_badge"]

    @pytest.mark.parametrize("lang", SUPPORTED_LANGS)
    def test_why_steps_contains_placeholder(self, lang):
        assert "{n}" in LANG[lang]["why_steps"]

    @pytest.mark.parametrize("lang", SUPPORTED_LANGS)
    def test_why_prefix_contains_rank_placeholder(self, lang):
        assert "{r}" in LANG[lang]["why_prefix"]


# ===========================================================================
# CRITERIA_LABELS structure
# ===========================================================================

class TestCriteriaLabels:
    def test_supported_languages_present(self):
        for lang in SUPPORTED_LANGS:
            assert lang in CRITERIA_LABELS

    @pytest.mark.parametrize("lang", SUPPORTED_LANGS)
    def test_all_criteria_keys_present(self, lang):
        missing = [k for k in CRITERIA_KEYS if k not in CRITERIA_LABELS[lang]]
        assert missing == [], f"Missing criteria in CRITERIA_LABELS['{lang}']: {missing}"

    @pytest.mark.parametrize("lang", SUPPORTED_LANGS)
    def test_all_values_are_non_empty_strings(self, lang):
        for key, value in CRITERIA_LABELS[lang].items():
            assert isinstance(value, str) and value, (
                f"CRITERIA_LABELS['{lang}']['{key}'] is empty or not a string"
            )

    def test_en_and_fr_have_same_criteria_keys(self):
        assert set(CRITERIA_LABELS["en"].keys()) == set(CRITERIA_LABELS["fr"].keys())


# ===========================================================================
# PALETTE
# ===========================================================================

class TestPalette:
    def test_is_list(self):
        assert isinstance(PALETTE, list)

    def test_not_empty(self):
        assert len(PALETTE) > 0

    def test_all_hex_colours(self):
        import re
        hex_re = re.compile(r"^#[0-9a-fA-F]{6}$")
        for colour in PALETTE:
            assert hex_re.match(colour), f"Not a valid hex colour: {colour!r}"

    def test_all_unique(self):
        assert len(PALETTE) == len(set(PALETTE)), "Duplicate colours in PALETTE"


# ===========================================================================
# FIG_BG
# ===========================================================================

class TestFigBg:
    def test_is_string(self):
        assert isinstance(FIG_BG, str)

    def test_is_hex_colour(self):
        import re
        assert re.match(r"^#[0-9a-fA-F]{6}$", FIG_BG), (
            f"FIG_BG is not a valid hex colour: {FIG_BG!r}"
        )