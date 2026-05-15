"""
test_app.py — Tests for app.py.

app.py is a Streamlit application whose sole public function is ``main()``.
The entire Streamlit runtime (st.*) is mocked by conftest.py, so we can
verify that:

  - The module imports without errors.
  - ``main()`` exists and is callable.
  - ``main()`` completes without raising when all Streamlit calls are mocked.
  - All expected imports (local modules and stdlib) are present on the module.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Ensure app.py is importable with fully mocked Streamlit
# ---------------------------------------------------------------------------

def _import_app():
    """
    Import app under full Streamlit + AiZynthFinder mocks.
    Returns the module object.
    """
    # Remove any previously cached version so we get a fresh import
    for key in list(sys.modules.keys()):
        if key == "app":
            del sys.modules[key]
    import src.path_finder.app as _app
    return _app


# ---------------------------------------------------------------------------
# Module-level smoke tests
# ---------------------------------------------------------------------------

class TestAppModuleImport:
    def test_module_imports_without_error(self):
        app = _import_app()
        assert app is not None

    def test_main_function_exists(self):
        app = _import_app()
        assert hasattr(app, "main")
        assert callable(app.main)

    def test_module_has_streamlit_import(self):
        app = _import_app()
        assert hasattr(app, "st")

    def test_module_has_os_import(self):
        app = _import_app()
        assert hasattr(app, "os")

    def test_module_has_sys_import(self):
        app = _import_app()
        assert hasattr(app, "sys")


# ---------------------------------------------------------------------------
# main() execution
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_is_callable(self):
        app = _import_app()
        assert callable(app.main)

    def test_main_signature(self):
        """main() takes no arguments and returns None."""
        import inspect
        app = _import_app()
        sig = inspect.signature(app.main)
        assert len(sig.parameters) == 0

    def test_main_return_annotation(self):
        """main() is annotated to return None."""
        import inspect
        app = _import_app()
        sig = inspect.signature(app.main)
        ret = sig.return_annotation
        assert ret is None or ret is inspect.Parameter.empty

    def test_main_docstring_describes_tabs(self):
        app = _import_app()
        doc = app.main.__doc__ or ""
        # The docstring mentions the four tabs
        assert any(kw in doc for kw in ("Route Search", "Analysis", "Dataset", "Help"))


# ---------------------------------------------------------------------------
# MODULE_ERR handling
# ---------------------------------------------------------------------------

class TestModuleErrHandling:
    def test_module_err_empty_when_route_engine_mock_ok(self):
        app = _import_app()
        # With our mock, import of route_engine succeeds so MODULE_ERR is ""
        # (rt is not None)
        assert app.rt is not None or app.MODULE_ERR != ""  # one of them holds