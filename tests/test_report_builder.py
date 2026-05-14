"""
Tests for src/path_finder/report_builder.py

Covers: build_route_report_pdf

The function returns raw PDF bytes (a multi-page A4 document rendered with
PIL).  Tests verify the output type, the PDF header signature, and that
various combinations of inputs (empty routes, predicted routes, multiple
steps) do not raise exceptions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "path_finder"))

import pytest

# conftest.py stubs aizynthfinder; report_builder imports route_engine which
# needs the stub.
from src.path_finder.report_builder import build_route_report_pdf

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

BENZENE = "c1ccccc1"
ETHANOL = "CCO"
ACETIC  = "CC(=O)O"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def _step(reactants, product, rtype="esterification", y=80, src="dataset", cond=None):
    return {
        "reactants_smiles": reactants,
        "product_smiles":   product,
        "reaction_type":    rtype,
        "yield_percent":    y,
        "source":           src,
        "step_number":      1,
        "conditions":       cond or {"temperature_C": 80, "solvent": "THF",
                                     "reagents": ["Et3N"], "apparatus": "reflux"},
        "fg_reactants":     ["alcohol"],
        "fg_products":      ["ester"],
        "by_products":      [],
    }


def _make_route(steps, status="dataset", name="Test Route"):
    return {
        "matched_route_name": name,
        "matched_target":     "TestTarget",
        "dataset_steps":      steps,
        "validation_status":  status,
        "is_predicted":       (status == "predicted"),
        "starting_materials": [steps[0]["reactants_smiles"][0]] if steps else [],
    }


def _make_details(criteria, excluded_yield=False):
    d = {}
    for i, c in enumerate(criteria):
        if c == "yield" and excluded_yield:
            d[c] = {"raw": None, "weight": 0.0, "weighted": 0.0, "excluded": True}
        else:
            d[c] = {"raw": 0.7, "weight": 0.7 / (i + 1), "weighted": 0.49, "excluded": False}
    d["_all_scores"] = {c: 0.7 for c in criteria}
    return d


CRITERIA = ["steps", "yield", "atom_economy"]


# ===========================================================================
# build_route_report_pdf
# ===========================================================================

class TestBuildRouteReportPdf:
    def _minimal_call(self, steps=None, status="dataset", criteria=None):
        if steps is None:
            steps = [_step([BENZENE], ETHANOL)]
        if criteria is None:
            criteria = CRITERIA
        route   = _make_route(steps, status=status)
        details = _make_details(criteria, excluded_yield=(status == "predicted"))
        return build_route_report_pdf(0.82, details, route, criteria)

    # --- Output type and PDF signature ----------------------------------------

    def test_returns_bytes(self):
        result = self._minimal_call()
        assert isinstance(result, bytes)

    def test_output_is_non_empty(self):
        result = self._minimal_call()
        assert len(result) > 0

    def test_output_starts_with_pdf_header(self):
        result = self._minimal_call()
        # PIL-generated PDFs start with the PDF magic bytes
        assert result[:4] == b"%PDF"

    # --- Robustness with varied inputs ----------------------------------------

    def test_empty_steps_does_not_raise(self):
        result = self._minimal_call(steps=[])
        assert isinstance(result, bytes)

    def test_predicted_route_does_not_raise(self):
        steps  = [_step([BENZENE], ETHANOL, src="rxn-insight", y=None)]
        result = self._minimal_call(steps=steps, status="predicted")
        assert isinstance(result, bytes)

    def test_multi_step_route(self):
        steps = [
            _step([BENZENE],  ETHANOL, rtype="reduction",   y=85, src="dataset"),
            _step([ETHANOL],  ACETIC,  rtype="oxidation",   y=72, src="dataset"),
            _step([ACETIC],   ASPIRIN, rtype="esterification", y=90, src="dataset"),
        ]
        result = self._minimal_call(steps=steps)
        assert isinstance(result, bytes) and len(result) > 0

    def test_missing_yield_does_not_raise(self):
        steps = [_step([BENZENE], ETHANOL, y=None)]
        result = self._minimal_call(steps=steps)
        assert isinstance(result, bytes)

    def test_invalid_smiles_does_not_raise(self):
        steps = [_step(["INVALID_SMILES"], "ALSO_INVALID")]
        result = self._minimal_call(steps=steps)
        assert isinstance(result, bytes)

    def test_four_step_route_spanning_multiple_pages(self):
        steps = [_step([BENZENE], ETHANOL, y=80 + i) for i in range(4)]
        result = self._minimal_call(steps=steps)
        assert isinstance(result, bytes)

    def test_score_zero_does_not_raise(self):
        route   = _make_route([_step([BENZENE], ETHANOL)])
        details = _make_details(CRITERIA)
        result  = build_route_report_pdf(0.0, details, route, CRITERIA)
        assert isinstance(result, bytes)

    def test_score_one_does_not_raise(self):
        route   = _make_route([_step([BENZENE], ETHANOL)])
        details = _make_details(CRITERIA)
        result  = build_route_report_pdf(1.0, details, route, CRITERIA)
        assert isinstance(result, bytes)

    def test_different_criteria_order(self):
        criteria = ["atom_economy", "e_factor", "toxicity"]
        steps    = [_step([BENZENE], ETHANOL)]
        route    = _make_route(steps)
        details  = _make_details(criteria)
        result   = build_route_report_pdf(0.6, details, route, criteria)
        assert isinstance(result, bytes)

    def test_conditions_with_no_temperature(self):
        cond  = {"solvent": "DCM", "reagents": ["KOH"]}
        steps = [_step([BENZENE], ETHANOL, cond=cond)]
        result = self._minimal_call(steps=steps)
        assert isinstance(result, bytes)

    def test_long_route_name_does_not_crash(self):
        steps  = [_step([BENZENE], ETHANOL)]
        route  = _make_route(steps, name="A" * 200)
        details = _make_details(CRITERIA)
        result  = build_route_report_pdf(0.7, details, route, CRITERIA)
        assert isinstance(result, bytes)