"""Tests for report_builder.py.

AiZynthFinder and rxn_insight are mocked so the suite runs without them.
PIL (Pillow) is required for PDF generation — tests are skipped if absent.
"""

import sys
import os
from unittest.mock import MagicMock, patch

# Mock heavy optional dependencies before any local import
for mod in [
    "aizynthfinder",
    "aizynthfinder.aizynthfinder",
    "rxn_insight",
    "rxn_insight.reaction",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

sys.modules["aizynthfinder.aizynthfinder"].AiZynthFinder = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "path_finder"))

import pytest

try:
    from PIL import Image  # noqa: F401
    PIL_OK = True
except ImportError:
    PIL_OK = False

PIL_SKIP = pytest.mark.skipif(not PIL_OK, reason="Pillow not available")


# Fixtures

BENZENE = "c1ccccc1"
TOLUENE = "Cc1ccccc1"
ASPIRIN  = "CC(=O)Oc1ccccc1C(=O)O"


def _make_step(n, reactants, product, yield_pct=None):
    return {
        "step_number":      n,
        "reactants_smiles": reactants,
        "product_smiles":   product,
        "yield_percent":    yield_pct,
        "reaction_type":    "Alkylation",
        "source":           "dataset",
        "conditions":       {"temperature_C": 60, "solvent": "THF", "reagents": ["NaH"]},
    }


def _make_route(steps, name="Test Route", target="Aspirin", status="dataset"):
    return {
        "dataset_steps":      steps,
        "matched_route_name": name,
        "matched_target":     target,
        "validation_status":  status,
    }


def _make_details(criteria):
    return {
        c: {"raw": 0.8, "weight": 0.33, "weighted": 0.264, "excluded": False}
        for c in criteria
    }


# Import

def test_report_builder_importable():
    import path_finder.report_builder as report_builder
    assert hasattr(report_builder, "build_route_report_pdf")


# build_route_report_pdf — return type and structure

@PIL_SKIP
def test_build_route_report_pdf_returns_bytes():
    from path_finder.report_builder import build_route_report_pdf
    steps = [_make_step(1, [BENZENE], TOLUENE, 75.0)]
    route = _make_route(steps)
    criteria = ["steps", "yield", "atom_economy"]
    details = _make_details(criteria)

    pdf_bytes = build_route_report_pdf(0.75, details, route, criteria)
    assert isinstance(pdf_bytes, bytes)


@PIL_SKIP
def test_build_route_report_pdf_non_empty():
    from path_finder.report_builder import build_route_report_pdf
    steps = [_make_step(1, [BENZENE], TOLUENE, 75.0)]
    route = _make_route(steps)
    criteria = ["steps", "yield", "atom_economy"]
    details = _make_details(criteria)

    pdf_bytes = build_route_report_pdf(0.75, details, route, criteria)
    assert len(pdf_bytes) > 0


@PIL_SKIP
def test_build_route_report_pdf_starts_with_pdf_header():
    from path_finder.report_builder import build_route_report_pdf
    steps = [_make_step(1, [BENZENE], TOLUENE, 75.0)]
    route = _make_route(steps)
    criteria = ["steps", "yield", "atom_economy"]
    details = _make_details(criteria)

    pdf_bytes = build_route_report_pdf(0.75, details, route, criteria)
    assert pdf_bytes[:4] == b"%PDF"


@PIL_SKIP
def test_build_route_report_pdf_empty_route():
    from path_finder.report_builder import build_route_report_pdf
    route = _make_route([], name="Empty Route")
    criteria = ["steps", "yield", "atom_economy"]
    details = _make_details(criteria)

    pdf_bytes = build_route_report_pdf(0.0, details, route, criteria)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0


@PIL_SKIP
def test_build_route_report_pdf_multi_step():
    from path_finder.report_builder import build_route_report_pdf
    steps = [
        _make_step(1, [BENZENE], TOLUENE, 80.0),
        _make_step(2, [TOLUENE], ASPIRIN, 65.0),
        _make_step(3, [ASPIRIN], BENZENE, 90.0),
        _make_step(4, [BENZENE], TOLUENE, 70.0),
    ]
    route = _make_route(steps)
    criteria = ["steps", "yield", "atom_economy"]
    details = _make_details(criteria)

    pdf_bytes = build_route_report_pdf(0.8, details, route, criteria)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0


@PIL_SKIP
def test_build_route_report_pdf_predicted_route():
    from path_finder.report_builder import build_route_report_pdf
    steps = [_make_step(1, [BENZENE], TOLUENE, None)]
    steps[0]["source"] = "rxn-insight"
    route = _make_route(steps, status="predicted")
    criteria = ["steps", "yield", "atom_economy"]
    details = {
        "steps":        {"raw": 0.8, "weight": 0.73, "weighted": 0.58, "excluded": False},
        "yield":        {"raw": None, "weight": 0.0, "weighted": 0.0,  "excluded": True},
        "atom_economy": {"raw": 0.6, "weight": 0.09, "weighted": 0.05, "excluded": False},
    }

    pdf_bytes = build_route_report_pdf(0.63, details, route, criteria)
    assert isinstance(pdf_bytes, bytes)


@PIL_SKIP
def test_build_route_report_pdf_no_yield_data():
    from path_finder.report_builder import build_route_report_pdf
    steps = [_make_step(1, [BENZENE], TOLUENE, None)]
    route = _make_route(steps)
    criteria = ["steps", "yield", "atom_economy"]
    details = _make_details(criteria)

    pdf_bytes = build_route_report_pdf(0.5, details, route, criteria)
    assert isinstance(pdf_bytes, bytes)


@PIL_SKIP
def test_build_route_report_pdf_all_criteria():
    from path_finder.report_builder import build_route_report_pdf
    steps = [_make_step(1, [BENZENE], TOLUENE, 80.0)]
    route = _make_route(steps)
    criteria = ["steps", "e_factor", "toxicity"]
    details = _make_details(criteria)

    pdf_bytes = build_route_report_pdf(0.7, details, route, criteria)
    assert isinstance(pdf_bytes, bytes)


@PIL_SKIP
def test_build_route_report_pdf_many_steps():
    from path_finder.report_builder import build_route_report_pdf
    steps = [_make_step(i + 1, [BENZENE], TOLUENE, 80.0) for i in range(9)]
    route = _make_route(steps)
    criteria = ["steps", "yield", "atom_economy"]
    details = _make_details(criteria)

    pdf_bytes = build_route_report_pdf(0.5, details, route, criteria)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0


@PIL_SKIP
def test_build_route_report_pdf_validated_status():
    from path_finder.report_builder import build_route_report_pdf
    steps = [_make_step(1, [BENZENE], TOLUENE, 85.0)]
    route = _make_route(steps, status="validated")
    criteria = ["steps", "yield", "atom_economy"]
    details = _make_details(criteria)

    pdf_bytes = build_route_report_pdf(0.85, details, route, criteria)
    assert isinstance(pdf_bytes, bytes)
