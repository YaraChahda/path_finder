"""
conftest.py — shared pytest configuration for path_finder tests.

Sets up sys.path so that src/path_finder modules are importable with flat
imports (e.g. ``import route_engine``), and installs mock modules for
heavyweight dependencies (aizynthfinder, streamlit, rxn_insight) so tests
can run without a full AiZynthFinder installation or a running Streamlit
server.
"""

import os
import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# sys.path — make src/path_finder importable
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
_SRC_DIR   = os.path.join(_REPO_ROOT, "src", "path_finder")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---------------------------------------------------------------------------
# Mock aizynthfinder (requires model files; never run in unit tests)
# ---------------------------------------------------------------------------
_aiz_root = types.ModuleType("aizynthfinder")
_aiz_sub  = types.ModuleType("aizynthfinder.aizynthfinder")
_aiz_sub.AiZynthFinder = MagicMock()
sys.modules.setdefault("aizynthfinder",              _aiz_root)
sys.modules.setdefault("aizynthfinder.aizynthfinder", _aiz_sub)

# ---------------------------------------------------------------------------
# Mock rxn_insight (optional dependency)
# ---------------------------------------------------------------------------
_rxni_root          = types.ModuleType("rxn_insight")
_rxni_reaction      = types.ModuleType("rxn_insight.reaction")
_rxni_reaction.Reaction = MagicMock()
sys.modules.setdefault("rxn_insight",          _rxni_root)
sys.modules.setdefault("rxn_insight.reaction", _rxni_reaction)

# ---------------------------------------------------------------------------
# Mock streamlit
# st.cache_data is used as a decorator: @st.cache_data(show_spinner=False)
# The mock must return a pass-through decorator so the wrapped functions
# remain callable with their original signatures.
# ---------------------------------------------------------------------------
_st = MagicMock()
_st.cache_data.return_value = lambda f: f   # @st.cache_data(...) → identity

sys.modules.setdefault("streamlit",              _st)
sys.modules.setdefault("streamlit.components",   MagicMock())
sys.modules.setdefault("streamlit.components.v1", MagicMock())