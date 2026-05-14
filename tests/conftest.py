"""
conftest.py — shared pytest fixtures and import mocks.

aizynthfinder is an optional heavy dependency not installed in the test
environment.  A lightweight stub is injected into sys.modules before any
test module imports route_engine, so the import never raises ModuleNotFoundError.
"""

import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub aizynthfinder so route_engine can be imported without the real package
# ---------------------------------------------------------------------------
_aiz_stub = types.ModuleType("aizynthfinder")
_aiz_aiz_stub = types.ModuleType("aizynthfinder.aizynthfinder")
_aiz_aiz_stub.AiZynthFinder = MagicMock()

sys.modules.setdefault("aizynthfinder", _aiz_stub)
sys.modules.setdefault("aizynthfinder.aizynthfinder", _aiz_aiz_stub)

# ---------------------------------------------------------------------------
# Ensure src/path_finder is on sys.path so all modules can be imported
# ---------------------------------------------------------------------------
import os
_here = os.path.dirname(__file__)
_src  = os.path.normpath(os.path.join(_here, "..", "src", "path_finder"))
if _src not in sys.path:
    sys.path.insert(0, _src)