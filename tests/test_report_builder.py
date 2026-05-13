"""
Tests for path_finder.report_builder

The only public function is build_route_report_pdf().  We call it with
minimal but valid route dicts and verify the output is a well-formed PDF.

Import note
-----------
report_builder.py contains `import route_engine as fi` (a bare absolute
import).  When the package is installed via `pip install -e .` that name
is not on sys.path, so we temporarily insert the package source directory
before importing the module.
"""

import io
import os
import sys
import pytest

# ---------------------------------------------------------------------------
# Path fix — allow `import route_engine as fi` inside report_builder.py
# ---------------------------------------------------------------------------
_SRC_PKG = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "src", "path_finder")
)
if _SRC_PKG not in sys.path:
    sys.path.insert(0, _SRC_PKG)

# ---------------------------------------------------------------------------
# Optional dep guards — skip the whole module if PIL or RDKit are absent
# ---------------------------------------------------------------------------
rdkit = pytest.importorskip("rdkit", reason="RDKit not installed")
PIL   = pytest.importorskip("PIL",   reason="Pillow not installed")

from path_finder import report_builder  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
PHENOL  = "Oc1ccccc1"
ACETIC  = "CC(=O)O"


def _make_step(step_number, reactants, product, yield_pct=None,
               source="dataset", reaction_type="Esterification"):
    return {
        "step_number":      step_number,
        "reactants_smiles": reactants,
        "product_smiles":   product,
        "yield_percent":    yield_pct,
        "source":           source,
        "reaction_type":    reaction_type,
        "conditions": {
            "temperature_C": 80,
            "solvent":       "AcOH",
            "reagents":      [],
            "apparatus":     None,
        },
    }


def _make_route(steps, validation_status="dataset"):
    return {
        "matched_route_name":    "Test route",
        "matched_target":        "Aspirin",
        "validation_status":     validation_status,
        "validated_steps_count": len(steps),
        "total_steps_count":     len(steps),
        "dataset_steps":         steps,
        "is_predicted":          validation_status == "predicted",
        "is_validated":          validation_status in ("validated", "partial"),
    }


def _make_details(criteria):
    """Build a minimal details dict as returned by rank_weighted()."""
    return {
        c: {"raw": 0.75, "weight": 1 / len(criteria), "weighted": 0.25, "excluded": False}
        for c in criteria
    }


def _is_valid_pdf(data: bytes) -> bool:
    """Return True if data starts with the PDF magic bytes."""
    return data[:4] == b"%PDF"


# ===========================================================================
# build_route_report_pdf
# ===========================================================================

class TestBuildRouteReportPdf:
    def test_returns_bytes(self):
        steps  = [_make_step(1, [PHENOL, ACETIC], ASPIRIN, yield_pct=85.0)]
        route  = _make_route(steps)
        result = report_builder.build_route_report_pdf(
            0.75, _make_details(["steps", "yield", "atom_economy"]), route,
            ["steps", "yield", "atom_economy"]
        )
        assert isinstance(result, bytes)

    def test_output_is_valid_pdf(self):
        steps  = [_make_step(1, [PHENOL, ACETIC], ASPIRIN, yield_pct=85.0)]
        route  = _make_route(steps)
        result = report_builder.build_route_report_pdf(
            0.75, _make_details(["steps", "yield", "atom_economy"]), route,
            ["steps", "yield", "atom_economy"]
        )
        assert _is_valid_pdf(result), "Output is not a valid PDF"

    def test_zero_steps_produces_pdf(self):
        route  = _make_route([])
        result = report_builder.build_route_report_pdf(
            0.5, _make_details(["steps", "yield", "atom_economy"]), route,
            ["steps", "yield", "atom_economy"]
        )
        assert _is_valid_pdf(result)

    def test_multiple_steps_produces_pdf(self):
        steps = [
            _make_step(i, [PHENOL], ASPIRIN, yield_pct=80.0)
            for i in range(1, 7)
        ]
        route  = _make_route(steps)
        result = report_builder.build_route_report_pdf(
            0.6, _make_details(["steps", "yield", "atom_economy"]), route,
            ["steps", "yield", "atom_economy"]
        )
        assert _is_valid_pdf(result)

    def test_predicted_route_produces_pdf(self):
        steps = [
            _make_step(1, [PHENOL], ASPIRIN, yield_pct=None,
                       source="rxn-insight", reaction_type="Predicted")
        ]
        route  = _make_route(steps, validation_status="predicted")
        details = {
            "steps":        {"raw": 0.5,  "weight": 0.7, "weighted": 0.35, "excluded": False},
            "yield":        {"raw": None, "weight": 0.0, "weighted": 0.0,  "excluded": True},
            "atom_economy": {"raw": 0.6,  "weight": 0.3, "weighted": 0.18, "excluded": False},
            "_all_scores":  {"steps": 0.5, "yield": 1.0, "atom_economy": 0.6,
                             "e_factor": 0.7, "toxicity": 0.5},
        }
        result = report_builder.build_route_report_pdf(
            0.53, details, route, ["steps", "yield", "atom_economy"]
        )
        assert _is_valid_pdf(result)

    def test_pdf_non_empty(self):
        steps  = [_make_step(1, [PHENOL, ACETIC], ASPIRIN, yield_pct=90.0)]
        route  = _make_route(steps)
        result = report_builder.build_route_report_pdf(
            0.8, _make_details(["steps", "yield", "atom_economy"]), route,
            ["steps", "yield", "atom_economy"]
        )
        assert len(result) > 1000  # a real PDF is always > 1 KB

    def test_excluded_criterion_handled(self):
        steps = [_make_step(1, [PHENOL], ASPIRIN, yield_pct=None, source="rxn-insight")]
        route = _make_route(steps, validation_status="predicted")
        details = {
            "steps":        {"raw": 1.0,  "weight": 1.0, "weighted": 1.0,  "excluded": False},
            "yield":        {"raw": None, "weight": 0.0, "weighted": 0.0,  "excluded": True},
            "atom_economy": {"raw": 0.5,  "weight": 0.0, "weighted": 0.0,  "excluded": False},
        }
        result = report_builder.build_route_report_pdf(
            1.0, details, route, ["steps", "yield", "atom_economy"]
        )
        assert isinstance(result, bytes)
        assert _is_valid_pdf(result)

    def test_many_reactants_handled(self):
        # Step with 4 reactants — only 3 are shown; should not raise
        steps = [
            _make_step(1, [PHENOL, ACETIC, "CCO", "c1ccccc1"], ASPIRIN, yield_pct=70.0)
        ]
        route  = _make_route(steps)
        result = report_builder.build_route_report_pdf(
            0.7, _make_details(["steps", "yield", "atom_economy"]), route,
            ["steps", "yield", "atom_economy"]
        )
        assert _is_valid_pdf(result)

    def test_long_route_name_handled(self):
        steps  = [_make_step(1, [PHENOL], ASPIRIN, yield_pct=80.0)]
        route  = _make_route(steps)
        route["matched_route_name"] = "A" * 200  # very long name
        result = report_builder.build_route_report_pdf(
            0.6, _make_details(["steps", "yield", "atom_economy"]), route,
            ["steps", "yield", "atom_economy"]
        )
        assert _is_valid_pdf(result)

    def test_invalid_smiles_in_step_does_not_raise(self):
        steps = [_make_step(1, ["INVALID_SMI"], "ALSO_INVALID", yield_pct=50.0)]
        route  = _make_route(steps)
        result = report_builder.build_route_report_pdf(
            0.5, _make_details(["steps", "yield", "atom_economy"]), route,
            ["steps", "yield", "atom_economy"]
        )
        assert isinstance(result, bytes)