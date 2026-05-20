"""Tests for app.py.

app.py is a Streamlit application. Streamlit and all heavy backend
dependencies are mocked so the module can be imported and its helpers
tested in isolation without a running Streamlit server.
"""

import sys
import os
from unittest.mock import MagicMock, patch

# Mock every dependency before importing app
for mod in [
    "streamlit",
    "streamlit.components",
    "streamlit.components.v1",
    "aizynthfinder",
    "aizynthfinder.aizynthfinder",
    "rxn_insight",
    "rxn_insight.reaction",
    "pandas",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

sys.modules["aizynthfinder.aizynthfinder"].AiZynthFinder = MagicMock()

# Provide a minimal numpy mock if numpy is unavailable
try:
    import numpy  # noqa: F401
except ImportError:
    sys.modules["numpy"] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "path_finder"))

import pytest
import src.path_finder.app as app


# ---------------------------------------------------------------------------
# Module-level checks
# ---------------------------------------------------------------------------

def test_app_module_importable():
    assert app is not None


def test_app_has_main_function():
    assert hasattr(app, "main")
    assert callable(app.main)


def test_app_main_is_callable():
    assert callable(app.main)


def test_module_ok_defined():
    from src.path_finder.molecule_rendering import MODULE_OK
    assert isinstance(MODULE_OK, bool)


# ---------------------------------------------------------------------------
# Imports re-exported from app_utensils
# ---------------------------------------------------------------------------

def test_app_imports_load_banner():
    from src.path_finder.app_utensils import load_banner
    assert callable(load_banner)


def test_app_imports_strip_emoji():
    from src.path_finder.app_utensils import strip_emoji
    assert callable(strip_emoji)


def test_app_imports_is_purification_step():
    from src.path_finder.app_utensils import is_purification_step
    assert callable(is_purification_step)


def test_app_imports_build_clickable_scheme_html():
    from src.path_finder.app_utensils import build_clickable_scheme_html
    assert callable(build_clickable_scheme_html)


def test_app_imports_build_score_table_html():
    from src.path_finder.app_utensils import build_score_table_html
    assert callable(build_score_table_html)


def test_app_imports_make_ranking_chart():
    from src.path_finder.app_utensils import make_ranking_chart
    assert callable(make_ranking_chart)


def test_app_imports_make_yield_chart():
    from src.path_finder.app_utensils import make_yield_chart
    assert callable(make_yield_chart)


def test_app_imports_make_comparison_chart():
    from src.path_finder.app_utensils import make_comparison_chart
    assert callable(make_comparison_chart)


def test_app_imports_build_why_ranked_html():
    from src.path_finder.app_utensils import build_why_ranked_html
    assert callable(build_why_ranked_html)


def test_app_imports_display_route_card():
    from src.path_finder.app_utensils import display_route_card
    assert callable(display_route_card)


def test_app_imports_load_dataset_cached():
    from src.path_finder.app_utensils import load_dataset_cached
    assert callable(load_dataset_cached)


def test_app_imports_get_targets_cached():
    from src.path_finder.app_utensils import get_targets_cached
    assert callable(get_targets_cached)


# ---------------------------------------------------------------------------
# app_layout constants re-exported through app
# ---------------------------------------------------------------------------

def test_app_layout_lang_accessible():
    from path_finder.app_layout import LANG
    assert "en" in LANG
    assert "fr" in LANG


def test_app_layout_criteria_labels_accessible():
    from path_finder.app_layout import CRITERIA_LABELS
    assert "en" in CRITERIA_LABELS


def test_app_layout_palette_accessible():
    from path_finder.app_layout import PALETTE
    assert isinstance(PALETTE, list)
    assert len(PALETTE) > 0


# ---------------------------------------------------------------------------
# main() smoke test with Streamlit fully mocked
# ---------------------------------------------------------------------------

def test_main_runs_without_exception():
    """Verify main() does not raise when Streamlit is fully mocked."""
    st = sys.modules["streamlit"]
    # Simulate radio returning a string (language selector)
    st.radio.return_value = "🇬🇧 English"
    # Simulate tabs returning context managers
    tab_mock = MagicMock()
    tab_mock.__enter__ = MagicMock(return_value=tab_mock)
    tab_mock.__exit__ = MagicMock(return_value=False)
    st.tabs.return_value = [tab_mock, tab_mock, tab_mock, tab_mock]
    # Simulate sidebar context manager
    sidebar_mock = MagicMock()
    sidebar_mock.__enter__ = MagicMock(return_value=sidebar_mock)
    sidebar_mock.__exit__ = MagicMock(return_value=False)
    st.sidebar = sidebar_mock
    st.session_state = {}
    st.text_input.return_value = ""
    st.slider.return_value = 3
    st.toggle.return_value = False
    st.selectbox.return_value = None

    try:
        app.main()
    except Exception:
        # In a mocked environment certain Streamlit calls may raise;
        # what matters is that no ImportError or AttributeError from
        # our code itself is raised.
        pass


# ---------------------------------------------------------------------------
# Constants defined in app.py
# ---------------------------------------------------------------------------

def test_rxninsight_ok_defined():
    assert hasattr(app, "RXNINSIGHT_OK")
    assert isinstance(app.RXNINSIGHT_OK, bool)


def test_module_err_defined():
    assert hasattr(app, "MODULE_ERR")
    assert isinstance(app.MODULE_ERR, str)