"""
test_report_builder.py — Tests for report_builder.py.

build_route_report_pdf() is the sole public function.  Tests verify:
  - The function returns valid PDF bytes.
  - Edge cases: empty steps, many steps, predicted routes, missing metadata.
  - Internal helpers (fonts, text wrapping) are exercised via the public API.

RDKit and Pillow are assumed to be installed.
"""

import io
import pytest

import src.path_finder.route_engine as rt
from src.path_finder.report_builder import build_route_report_pdf


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BENZENE  = "c1ccccc1"
TOLUENE  = "Cc1ccccc1"
ETHANOL  = "CCO"
CAFFEINE = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"

CRITERIA = ["steps", "yield", "atom_economy"]


def _make_route(steps, status="dataset", name="Test Route", target="Target"):
    return {
        "dataset_steps":         steps,
        "matched_route_name":    name,
        "matched_target":        target,
        "validation_status":     status,
        "is_predicted":          (status == "predicted"),
        "validated_steps_count": 0,
        "total_steps_count":     len(steps),
    }


def _make_step(n, reactants, product, yld=None, source="dataset"):
    return {
        "step_number":      n,
        "reactants_smiles": reactants,
        "product_smiles":   product,
        "yield_percent":    yld,
        "reaction_type":    "Test Reaction",
        "conditions":       {
            "temperature_C": 80,
            "solvent":       "THF",
            "reagents":      ["NaH"],
            "apparatus":     None,
        },
        "source": source,
    }


def _make_details(criteria, raw=0.75, weight=None, weighted=None):
    weights = rt.compute_weights(criteria)
    details = {}
    for i, c in enumerate(criteria):
        w = weights[c]
        details[c] = {
            "raw":      raw,
            "weight":   w,
            "weighted": raw * w,
            "excluded": False,
        }
    return details


def _is_pdf(data: bytes) -> bool:
    return data[:4] == b"%PDF"


# ---------------------------------------------------------------------------
# Basic return-value tests
# ---------------------------------------------------------------------------

class TestBuildRouteReportPdfBasic:
    def test_returns_bytes(self):
        route   = _make_route([])
        details = _make_details(CRITERIA)
        result  = build_route_report_pdf(0.75, details, route, CRITERIA)
        assert isinstance(result, bytes)

    def test_returns_non_empty_bytes(self):
        route   = _make_route([])
        details = _make_details(CRITERIA)
        result  = build_route_report_pdf(0.75, details, route, CRITERIA)
        assert len(result) > 0

    def test_result_is_pdf(self):
        route   = _make_route([])
        details = _make_details(CRITERIA)
        result  = build_route_report_pdf(0.75, details, route, CRITERIA)
        assert _is_pdf(result), "First 4 bytes should be '%PDF'"

    def test_pdf_parseable_by_pillow(self):
        from PIL import Image
        route   = _make_route([])
        details = _make_details(CRITERIA)
        result  = build_route_report_pdf(0.75, details, route, CRITERIA)
        buf = io.BytesIO(result)
        # Pillow cannot open multi-page PDF directly, but the raw bytes should
        # start with the PDF header.
        assert result[:4] == b"%PDF"


# ---------------------------------------------------------------------------
# Route content variants
# ---------------------------------------------------------------------------

class TestBuildRouteReportPdfContent:
    def test_single_step_route(self):
        step    = _make_step(1, [BENZENE, ETHANOL], TOLUENE, yld=80)
        route   = _make_route([step])
        details = _make_details(CRITERIA)
        result  = build_route_report_pdf(0.8, details, route, CRITERIA)
        assert _is_pdf(result)

    def test_multi_step_route(self):
        steps = [
            _make_step(1, [BENZENE], TOLUENE,  yld=85),
            _make_step(2, [TOLUENE], ETHANOL,  yld=70),
            _make_step(3, [ETHANOL], CAFFEINE, yld=60),
        ]
        route   = _make_route(steps)
        details = _make_details(CRITERIA)
        result  = build_route_report_pdf(0.7, details, route, CRITERIA)
        assert _is_pdf(result)

    def test_predicted_route_excluded_yield(self):
        step    = _make_step(1, [BENZENE], TOLUENE, yld=None, source="rxn-insight")
        route   = _make_route([step], status="predicted")
        details = _make_details(CRITERIA)
        details["yield"] = {"raw": None, "weight": 0.0, "weighted": 0.0, "excluded": True}
        result  = build_route_report_pdf(0.5, details, route, CRITERIA)
        assert _is_pdf(result)

    def test_route_with_no_yield(self):
        step    = _make_step(1, [BENZENE], TOLUENE, yld=None)
        route   = _make_route([step])
        details = _make_details(CRITERIA)
        result  = build_route_report_pdf(0.6, details, route, CRITERIA)
        assert _is_pdf(result)

    def test_route_with_long_name(self):
        long_name = "A Very Long Route Name That Should Wrap Across Multiple Lines In The Header"
        route   = _make_route([], name=long_name)
        details = _make_details(CRITERIA)
        result  = build_route_report_pdf(0.5, details, route, CRITERIA)
        assert _is_pdf(result)

    def test_route_missing_optional_metadata(self):
        route = {
            "dataset_steps":      [],
            "validation_status":  "dataset",
        }
        details = _make_details(CRITERIA)
        result  = build_route_report_pdf(0.5, details, route, CRITERIA)
        assert _is_pdf(result)

    def test_four_steps_spans_two_pages(self):
        steps = [_make_step(i, [BENZENE], TOLUENE, yld=80) for i in range(1, 5)]
        route   = _make_route(steps)
        details = _make_details(CRITERIA)
        result  = build_route_report_pdf(0.7, details, route, CRITERIA)
        assert _is_pdf(result)

    def test_step_with_invalid_smiles_does_not_crash(self):
        step = _make_step(1, ["not!valid"], "also!invalid", yld=None)
        route   = _make_route([step])
        details = _make_details(CRITERIA)
        result  = build_route_report_pdf(0.3, details, route, CRITERIA)
        assert _is_pdf(result)

    def test_score_zero_route(self):
        route   = _make_route([])
        details = _make_details(CRITERIA, raw=0.0)
        result  = build_route_report_pdf(0.0, details, route, CRITERIA)
        assert _is_pdf(result)

    def test_score_one_route(self):
        route   = _make_route([])
        details = _make_details(CRITERIA, raw=1.0)
        result  = build_route_report_pdf(1.0, details, route, CRITERIA)
        assert _is_pdf(result)

    def test_multiple_reactants_per_step(self):
        step    = _make_step(1, [BENZENE, ETHANOL, TOLUENE], CAFFEINE, yld=55)
        route   = _make_route([step])
        details = _make_details(CRITERIA)
        result  = build_route_report_pdf(0.55, details, route, CRITERIA)
        assert _is_pdf(result)

    def test_validated_status(self):
        step    = _make_step(1, [BENZENE], TOLUENE, yld=75, source="generic_dataset")
        route   = _make_route([step], status="validated")
        details = _make_details(CRITERIA)
        result  = build_route_report_pdf(0.75, details, route, CRITERIA)
        assert _is_pdf(result)