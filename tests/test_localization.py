"""Tests for localization.py — pure-Python module, no heavy dependencies."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "path_finder"))

import pytest
from src.path_finder.localization import LANG, CRITERIA_LABELS, PALETTE, FIG_BG


# ---------------------------------------------------------------------------
# LANG structure
# ---------------------------------------------------------------------------

REQUIRED_KEYS = [
    "page_title", "page_caption", "sidebar_title", "tab_search", "tab_analysis",
    "tab_dataset", "tab_help", "run_btn", "search_ok", "err_file", "no_routes",
    "n_found", "chart_title", "score_axis", "metric_score", "metric_steps",
    "metric_bottleneck", "metric_avg", "contrib_title", "score_th_crit",
    "score_th_raw", "score_th_weight", "score_th_contrib", "footer",
    "why_steps", "why_yield", "why_prefix", "why_suffix",
]


def test_lang_has_english_key():
    assert "en" in LANG


def test_lang_has_french_key():
    assert "fr" in LANG


def test_lang_no_extra_languages():
    assert set(LANG.keys()) == {"en", "fr"}


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_lang_english_has_required_key(key):
    assert key in LANG["en"], f"Missing key '{key}' in LANG['en']"


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_lang_french_has_required_key(key):
    assert key in LANG["fr"], f"Missing key '{key}' in LANG['fr']"


def test_lang_english_strings_are_str():
    for k, v in LANG["en"].items():
        assert isinstance(v, str), f"LANG['en']['{k}'] is not a string"


def test_lang_french_strings_are_str():
    for k, v in LANG["fr"].items():
        assert isinstance(v, str), f"LANG['fr']['{k}'] is not a string"


def test_lang_same_keys_both_languages():
    assert set(LANG["en"].keys()) == set(LANG["fr"].keys())


def test_lang_page_title_non_empty():
    assert LANG["en"]["page_title"].strip()
    assert LANG["fr"]["page_title"].strip()


def test_lang_why_steps_has_placeholder():
    assert "{n}" in LANG["en"]["why_steps"]
    assert "{n}" in LANG["fr"]["why_steps"]


def test_lang_why_yield_has_placeholder():
    assert "{y}" in LANG["en"]["why_yield"]
    assert "{y}" in LANG["fr"]["why_yield"]


def test_lang_n_found_has_placeholder():
    assert "{n}" in LANG["en"]["n_found"]
    assert "{n}" in LANG["fr"]["n_found"]


def test_lang_chart_title_has_placeholder():
    assert "{target}" in LANG["en"]["chart_title"]
    assert "{target}" in LANG["fr"]["chart_title"]


# ---------------------------------------------------------------------------
# CRITERIA_LABELS
# ---------------------------------------------------------------------------

CRITERIA_KEYS = ["steps", "yield", "atom_economy", "e_factor", "toxicity"]


def test_criteria_labels_has_english():
    assert "en" in CRITERIA_LABELS


def test_criteria_labels_has_french():
    assert "fr" in CRITERIA_LABELS


@pytest.mark.parametrize("key", CRITERIA_KEYS)
def test_criteria_labels_english_has_key(key):
    assert key in CRITERIA_LABELS["en"]


@pytest.mark.parametrize("key", CRITERIA_KEYS)
def test_criteria_labels_french_has_key(key):
    assert key in CRITERIA_LABELS["fr"]


def test_criteria_labels_values_are_strings():
    for lang in ("en", "fr"):
        for k, v in CRITERIA_LABELS[lang].items():
            assert isinstance(v, str), f"CRITERIA_LABELS['{lang}']['{k}'] is not a string"


def test_criteria_labels_same_keys():
    assert set(CRITERIA_LABELS["en"].keys()) == set(CRITERIA_LABELS["fr"].keys())


# ---------------------------------------------------------------------------
# PALETTE
# ---------------------------------------------------------------------------

def test_palette_is_list():
    assert isinstance(PALETTE, list)


def test_palette_non_empty():
    assert len(PALETTE) > 0


def test_palette_entries_are_hex_strings():
    for color in PALETTE:
        assert isinstance(color, str)
        assert color.startswith("#"), f"'{color}' does not start with '#'"
        assert len(color) == 7, f"'{color}' is not a valid 6-digit hex color"


def test_palette_hex_values_valid():
    for color in PALETTE:
        int(color[1:], 16)  # raises ValueError if not valid hex


# ---------------------------------------------------------------------------
# FIG_BG
# ---------------------------------------------------------------------------

def test_fig_bg_is_string():
    assert isinstance(FIG_BG, str)


def test_fig_bg_is_hex_color():
    assert FIG_BG.startswith("#")
    assert len(FIG_BG) == 7
    int(FIG_BG[1:], 16)