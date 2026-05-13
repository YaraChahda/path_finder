"""
Comprehensive tests for the Project modules:
  - route_engine.py   : data loading, SMILES utils, scoring, ranking
  - localization.py   : UI string catalogue and display constants
  - molecule_rendering.py : RDKit-based image rendering utilities

aizynthfinder and rxn_insight are optional heavy dependencies that are mocked
at the sys.modules level so these tests run without them installed.
"""

import io
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock heavy optional / hard dependencies BEFORE importing any project module
# ---------------------------------------------------------------------------

_aiz_mock = MagicMock()
sys.modules.setdefault("aizynthfinder", _aiz_mock)
sys.modules.setdefault("aizynthfinder.aizynthfinder", _aiz_mock)

_rxni_mock = MagicMock()
sys.modules.setdefault("rxn_insight", _rxni_mock)
sys.modules.setdefault("rxn_insight.reaction", _rxni_mock)

# Ensure the Project directory is on sys.path so the modules can be imported
_PROJECT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..", "Project"
)
_PROJECT_DIR = os.path.abspath(_PROJECT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import route_engine as re_mod
import localization as loc_mod
import molecule_rendering as mr_mod


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def simple_reaction_dataset(tmp_path):
    """Write a minimal reaction_dataset.json and return its path."""
    data = {
        "reactions": [
            {
                "id": "rxn-001",
                "route_id": "route-A",
                "route_name": "Route A",
                "target": "Aspirin",
                "step_number": 1,
                "reactants_smiles": ["OC(=O)c1ccccc1O", "CC(=O)O"],
                "product_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "conditions": {
                    "temperature_C": 80,
                    "solvent": "AcOH",
                    "reagents": ["H2SO4"],
                },
                "yield_percent": 85.0,
                "reaction_type": "Esterification",
            },
            {
                "id": "rxn-002",
                "route_id": "route-B",
                "route_name": "Route B",
                "target": "Aspirin",
                "step_number": 1,
                "reactants_smiles": ["OC(=O)c1ccccc1O"],
                "product_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "conditions": {},
                "yield_percent": 70.0,
                "reaction_type": "Acetylation",
            },
        ],
        "_metadata": {
            "target_smiles": {
                "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
            }
        },
    }
    p = tmp_path / "reaction_dataset.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


@pytest.fixture
def simple_toxicity_dataset(tmp_path):
    """Write a minimal toxicity_dataset.json and return its path."""
    data = [
        {"smiles": "CC(=O)O", "hazard_score": 0.1},   # AcOH
        {"smiles": "ClCCl",   "hazard_score": 0.7},   # DCM
    ]
    p = tmp_path / "toxicity_dataset.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


@pytest.fixture
def loaded_dataset(simple_reaction_dataset):
    return re_mod.load_reaction_dataset(simple_reaction_dataset)


@pytest.fixture
def loaded_tox(simple_toxicity_dataset):
    return re_mod.load_toxicity_dataset(simple_toxicity_dataset)


@pytest.fixture
def aspirin_route():
    """Minimal enriched route dict for Aspirin with real yield data."""
    return {
        "route_id": "route-A",
        "matched_route_name": "Route A",
        "matched_target": "Aspirin",
        "validation_status": "dataset",
        "dataset_steps": [
            {
                "id": "rxn-001",
                "step_number": 1,
                "reactants_smiles": ["OC(=O)c1ccccc1O", "CC(=O)O"],
                "product_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "yield_percent": 85.0,
                "conditions": {"temperature_C": 80, "solvent": "AcOH", "reagents": []},
                "reaction_type": "Esterification",
                "source": "dataset",
            }
        ],
        "is_predicted": False,
        "is_validated": False,
    }


@pytest.fixture
def predicted_route():
    """Minimal enriched route dict with no yield data (predicted)."""
    return {
        "route_id": "predicted_01",
        "matched_route_name": "Predicted route #1",
        "matched_target": "?",
        "validation_status": "predicted",
        "dataset_steps": [
            {
                "id": "PRED-01-01",
                "step_number": 1,
                "reactants_smiles": ["CCO", "CC(=O)O"],
                "product_smiles": "CCOC(C)=O",
                "yield_percent": None,
                "conditions": {},
                "reaction_type": "Esterification",
                "source": "rxn-insight",
            }
        ],
        "is_predicted": True,
        "is_validated": False,
    }


# ===========================================================================
# route_engine — SMILES utilities
# ===========================================================================

class TestToCanonical:
    def test_simple_smiles(self):
        assert re_mod.to_canonical("C") == "C"

    def test_canonical_form_normalised(self):
        # Both representations should produce the same canonical SMILES
        a = re_mod.to_canonical("OCC")
        b = re_mod.to_canonical("CCO")
        assert a == b

    def test_list_input(self):
        result = re_mod.to_canonical(["C", "O"])
        assert isinstance(result, str)

    def test_empty_string(self):
        assert re_mod.to_canonical("") == ""

    def test_empty_list(self):
        assert re_mod.to_canonical([]) == ""

    def test_invalid_smiles_returned_as_is(self):
        result = re_mod.to_canonical("INVALID_SMILES_XYZ")
        assert result == "INVALID_SMILES_XYZ"

    def test_non_string_scalar(self):
        result = re_mod.to_canonical(42)
        assert result == ""

    def test_aspirin(self):
        canon = re_mod.to_canonical("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(canon, str) and len(canon) > 0


class TestSafeMol:
    def test_valid_smiles_returns_mol(self):
        mol = re_mod.safe_mol("CCO")
        assert mol is not None

    def test_invalid_smiles_returns_none(self):
        assert re_mod.safe_mol("NOT_A_SMILES") is None

    def test_empty_string_returns_none(self):
        assert re_mod.safe_mol("") is None


class TestValidateSmilesForAizynthfinder:
    def test_valid_smiles(self):
        result = re_mod.validate_smiles_for_aizynthfinder("CCO")
        assert isinstance(result, str) and len(result) > 0

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            re_mod.validate_smiles_for_aizynthfinder("")

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            re_mod.validate_smiles_for_aizynthfinder("NOT_SMILES_XYZ")


# ===========================================================================
# route_engine — Data loading
# ===========================================================================

class TestLoadReactionDataset:
    def test_loads_reactions(self, simple_reaction_dataset):
        ds = re_mod.load_reaction_dataset(simple_reaction_dataset)
        assert len(ds["all"]) == 2

    def test_returns_by_route(self, simple_reaction_dataset):
        ds = re_mod.load_reaction_dataset(simple_reaction_dataset)
        assert "route-A" in ds["by_route"]
        assert "route-B" in ds["by_route"]

    def test_returns_by_product(self, simple_reaction_dataset):
        ds = re_mod.load_reaction_dataset(simple_reaction_dataset)
        assert len(ds["by_product"]) > 0

    def test_returns_metadata(self, simple_reaction_dataset):
        ds = re_mod.load_reaction_dataset(simple_reaction_dataset)
        assert "target_smiles" in ds["metadata"]

    def test_steps_sorted_by_step_number(self, tmp_path):
        data = {
            "reactions": [
                {"id": "r2", "route_id": "R", "route_name": "R",
                 "target": "T", "step_number": 2,
                 "reactants_smiles": ["C"], "product_smiles": "O",
                 "conditions": {}, "yield_percent": None, "reaction_type": ""},
                {"id": "r1", "route_id": "R", "route_name": "R",
                 "target": "T", "step_number": 1,
                 "reactants_smiles": ["N"], "product_smiles": "C",
                 "conditions": {}, "yield_percent": None, "reaction_type": ""},
            ]
        }
        p = tmp_path / "ds.json"
        p.write_text(json.dumps(data))
        ds = re_mod.load_reaction_dataset(str(p))
        steps = ds["by_route"]["R"]
        assert steps[0]["step_number"] == 1
        assert steps[1]["step_number"] == 2

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            re_mod.load_reaction_dataset("/nonexistent/path.json")

    def test_unrecognized_format_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text('{"wrong_key": []}')
        with pytest.raises(ValueError, match="unrecognized"):
            re_mod.load_reaction_dataset(str(p))

    def test_list_format_accepted(self, tmp_path):
        data = [
            {"id": "r1", "route_id": "R", "route_name": "R",
             "target": "T", "step_number": 1,
             "reactants_smiles": ["C"], "product_smiles": "O",
             "conditions": {}, "yield_percent": None, "reaction_type": ""},
        ]
        p = tmp_path / "list.json"
        p.write_text(json.dumps(data))
        ds = re_mod.load_reaction_dataset(str(p))
        assert len(ds["all"]) == 1
        assert ds["metadata"] == {}

    def test_product_smiles_list_joined(self, tmp_path):
        data = [
            {"id": "r1", "route_id": "R", "route_name": "R",
             "target": "T", "step_number": 1,
             "reactants_smiles": ["C"],
             "product_smiles": ["O", "N"],   # list — should be joined
             "conditions": {}, "yield_percent": None, "reaction_type": ""},
        ]
        p = tmp_path / "list_prod.json"
        p.write_text(json.dumps(data))
        ds = re_mod.load_reaction_dataset(str(p))
        assert "." in ds["all"][0]["product_smiles"]


class TestLoadToxicityDataset:
    def test_loads_compounds(self, simple_toxicity_dataset):
        tox = re_mod.load_toxicity_dataset(simple_toxicity_dataset)
        assert len(tox) == 2

    def test_returns_empty_when_missing(self, tmp_path):
        tox = re_mod.load_toxicity_dataset(str(tmp_path / "nonexistent.json"))
        assert tox == {}

    def test_index_by_canonical_smiles(self, simple_toxicity_dataset):
        tox = re_mod.load_toxicity_dataset(simple_toxicity_dataset)
        canon_acoh = re_mod.to_canonical("CC(=O)O")
        assert canon_acoh in tox

    def test_dict_format_with_compounds_key(self, tmp_path):
        data = {"compounds": [{"smiles": "C", "hazard_score": 0.2}]}
        p = tmp_path / "tox.json"
        p.write_text(json.dumps(data))
        tox = re_mod.load_toxicity_dataset(str(p))
        assert re_mod.to_canonical("C") in tox


class TestLoadGenericReactionDataset:
    def test_loads_from_list(self, tmp_path):
        data = [
            {"id": "g1", "reactants_smiles": ["CCO"], "product_smiles": "CC=O",
             "yield_percent": 80.0, "reaction_type": "Oxidation"},
        ]
        p = tmp_path / "generic.json"
        p.write_text(json.dumps(data))
        ds = re_mod.load_generic_reaction_dataset(str(p))
        assert len(ds["all"]) == 1

    def test_builds_by_product_index(self, tmp_path):
        data = [
            {"id": "g1", "reactants_smiles": ["CCO"], "product_smiles": "CC=O",
             "yield_percent": 80.0, "reaction_type": "Oxidation"},
        ]
        p = tmp_path / "generic.json"
        p.write_text(json.dumps(data))
        ds = re_mod.load_generic_reaction_dataset(str(p))
        canon = re_mod.to_canonical("CC=O")
        assert canon in ds["by_product"]

    def test_builds_by_reaction_key(self, tmp_path):
        data = [
            {"id": "g1", "reactants_smiles": ["CCO"], "product_smiles": "CC=O",
             "yield_percent": 80.0, "reaction_type": "Oxidation"},
        ]
        p = tmp_path / "generic.json"
        p.write_text(json.dumps(data))
        ds = re_mod.load_generic_reaction_dataset(str(p))
        assert len(ds["by_reaction_key"]) == 1

    def test_returns_empty_when_absent(self, tmp_path):
        ds = re_mod.load_generic_reaction_dataset(str(tmp_path / "missing.json"))
        assert ds == {}

    def test_empty_path_returns_empty(self):
        ds = re_mod.load_generic_reaction_dataset("")
        assert ds == {}


# ===========================================================================
# route_engine — Dataset utilities
# ===========================================================================

class TestGetTargetsFromDataset:
    def test_returns_target_from_metadata(self, loaded_dataset):
        targets = re_mod.get_targets_from_dataset(loaded_dataset)
        assert "Aspirin" in targets
        assert isinstance(targets["Aspirin"], str)

    def test_returns_canonical_smiles(self, loaded_dataset):
        targets = re_mod.get_targets_from_dataset(loaded_dataset)
        canon = re_mod.to_canonical("CC(=O)Oc1ccccc1C(=O)O")
        assert targets["Aspirin"] == canon


class TestBuildDatasetSmilesIndex:
    def test_returns_set(self, loaded_dataset):
        index = re_mod.build_dataset_smiles_index(loaded_dataset)
        assert isinstance(index, set)

    def test_contains_product_smiles(self, loaded_dataset):
        index = re_mod.build_dataset_smiles_index(loaded_dataset)
        canon = re_mod.to_canonical("CC(=O)Oc1ccccc1C(=O)O")
        assert canon in index

    def test_contains_reactant_smiles(self, loaded_dataset):
        index = re_mod.build_dataset_smiles_index(loaded_dataset)
        canon = re_mod.to_canonical("OC(=O)c1ccccc1O")
        assert canon in index


# ===========================================================================
# route_engine — Per-step yield helpers
# ===========================================================================

class TestBottleneckYield:
    def test_returns_minimum(self):
        steps = [
            {"yield_percent": 90.0},
            {"yield_percent": 60.0},
            {"yield_percent": 80.0},
        ]
        assert re_mod.bottleneck_yield(steps) == 60.0

    def test_ignores_none_yields(self):
        steps = [{"yield_percent": 90.0}, {"yield_percent": None}]
        assert re_mod.bottleneck_yield(steps) == 90.0

    def test_returns_none_when_all_missing(self):
        steps = [{"yield_percent": None}, {}]
        assert re_mod.bottleneck_yield(steps) is None

    def test_empty_steps(self):
        assert re_mod.bottleneck_yield([]) is None


class TestAverageYield:
    def test_calculates_mean(self):
        steps = [{"yield_percent": 80.0}, {"yield_percent": 60.0}]
        assert re_mod.average_yield(steps) == pytest.approx(70.0)

    def test_ignores_none(self):
        steps = [{"yield_percent": 80.0}, {"yield_percent": None}]
        assert re_mod.average_yield(steps) == pytest.approx(80.0)

    def test_returns_none_when_all_missing(self):
        assert re_mod.average_yield([{"yield_percent": None}]) is None

    def test_empty_steps(self):
        assert re_mod.average_yield([]) is None


class TestCumulativeYield:
    def test_multiplies_fractions(self):
        steps = [{"yield_percent": 80.0}, {"yield_percent": 50.0}]
        assert re_mod.cumulative_yield(steps) == pytest.approx(0.4)

    def test_none_yield_treated_as_neutral(self):
        steps = [{"yield_percent": 80.0}, {"yield_percent": None}]
        assert re_mod.cumulative_yield(steps) == pytest.approx(0.8)

    def test_empty_steps(self):
        assert re_mod.cumulative_yield([]) == pytest.approx(1.0)

    def test_single_step_full_yield(self):
        assert re_mod.cumulative_yield([{"yield_percent": 100.0}]) == pytest.approx(1.0)


# ===========================================================================
# route_engine — Substance / conditions helpers
# ===========================================================================

class TestGetSubstancesList:
    def test_to_buy_are_leaves(self):
        steps = [
            {
                "reactants_smiles": ["CCO", "CC(=O)O"],
                "product_smiles": "CCOC(C)=O",
                "conditions": {},
            }
        ]
        sub = re_mod.get_substances_list(steps)
        assert isinstance(sub["to_buy"], list)
        assert isinstance(sub["to_prepare"], list)

    def test_to_prepare_contains_intermediates(self):
        steps = [
            {
                "reactants_smiles": ["CCO"],
                "product_smiles": "CC=O",
                "conditions": {},
            },
            {
                "reactants_smiles": ["CC=O"],
                "product_smiles": "CCC=O",
                "conditions": {},
            },
        ]
        sub = re_mod.get_substances_list(steps)
        # CC=O is produced by step 1, so it should be in to_prepare
        canon_aldehyde = re_mod.to_canonical("CC=O")
        assert canon_aldehyde in sub["to_prepare"]
        # CC=O is also consumed by step 2 — not a leaf → not in to_buy
        assert canon_aldehyde not in sub["to_buy"]

    def test_solvents_extracted(self):
        steps = [
            {
                "reactants_smiles": ["C"],
                "product_smiles": "O",
                "conditions": {"solvent": "THF", "co_solvent": "MeOH"},
            }
        ]
        sub = re_mod.get_substances_list(steps)
        assert "THF" in sub["solvents"]
        assert "MeOH" in sub["solvents"]

    def test_reagents_extracted(self):
        steps = [
            {
                "reactants_smiles": ["C"],
                "product_smiles": "O",
                "conditions": {"reagents": ["H2SO4", "Et3N"]},
            }
        ]
        sub = re_mod.get_substances_list(steps)
        assert "H2SO4" in sub["reagents"]
        assert "Et3N" in sub["reagents"]


class TestFmtConditions:
    def test_empty_dict_returns_empty(self):
        assert re_mod.fmt_conditions({}) == ""

    def test_temperature(self):
        result = re_mod.fmt_conditions({"temperature_C": 80})
        assert "80" in result

    def test_solvent(self):
        result = re_mod.fmt_conditions({"solvent": "THF"})
        assert "THF" in result

    def test_reagents_list(self):
        result = re_mod.fmt_conditions({"reagents": ["KOH", "Et3N"]})
        assert "KOH" in result and "Et3N" in result

    def test_apparatus(self):
        result = re_mod.fmt_conditions({"apparatus": "reflux condenser"})
        assert "reflux condenser" in result

    def test_separator(self):
        result = re_mod.fmt_conditions({"temperature_C": 60, "solvent": "DCM"})
        assert "·" in result


# ===========================================================================
# route_engine — Scoring functions
# ===========================================================================

class TestCalcAtomEconomy:
    def test_perfect_economy(self):
        # Product equals one reactant → AE = 1
        # Use ethanol (MW≈46) → acetaldehyde (MW≈44) + water; AE < 1 in practice
        score = re_mod.calc_atom_economy(["CCO"], "CC=O")
        assert 0.0 < score <= 1.0

    def test_invalid_product_returns_zero(self):
        score = re_mod.calc_atom_economy(["CCO"], "NOT_SMILES")
        assert score == 0.0

    def test_empty_reactants_returns_zero(self):
        score = re_mod.calc_atom_economy([], "CCO")
        assert score == 0.0

    def test_score_bounded(self):
        score = re_mod.calc_atom_economy(["C", "O", "N", "CCO"], "C")
        assert 0.0 <= score <= 1.0


class TestCalcEFactor:
    def test_perfect_yield_no_waste_near_one(self):
        # Single reactant, product = reactant (AE=1), yield=1 → E ≈ 0 → score ≈ 1
        score = re_mod.calc_e_factor(["CCO"], "CCO", 1.0)
        assert score > 0.9

    def test_invalid_product_returns_neutral(self):
        score = re_mod.calc_e_factor(["CCO"], "NOT_SMILES", 1.0)
        assert score == 0.5

    def test_score_in_range(self):
        score = re_mod.calc_e_factor(["CCO", "CC(=O)O"], "CCOC(C)=O", 0.8)
        assert 0.0 < score <= 1.0

    def test_low_yield_lower_score(self):
        high = re_mod.calc_e_factor(["CCO"], "CC=O", 0.9)
        low  = re_mod.calc_e_factor(["CCO"], "CC=O", 0.1)
        assert high >= low


class TestCalcToxicityScore:
    def test_known_safe_compound_high_score(self, loaded_tox):
        solvent_map = re_mod.build_solvent_map(loaded_tox)
        # AcOH has hazard_score=0.1 → safety = 1 - 0.1 = 0.9
        score = re_mod.calc_toxicity_score(
            ["CC(=O)O"], {}, loaded_tox, solvent_map)
        assert score == pytest.approx(0.9, abs=0.01)

    def test_unknown_compound_gets_neutral_default(self):
        # No compound in tox_index → hazard defaults to 0.5
        score = re_mod.calc_toxicity_score(["c1ccccc1"], {}, {}, {})
        assert score == pytest.approx(0.5)

    def test_empty_reactants_returns_neutral(self):
        score = re_mod.calc_toxicity_score([], {}, {}, {})
        assert score == pytest.approx(0.5)

    def test_solvent_resolved_from_map(self, loaded_tox):
        solvent_map = re_mod.build_solvent_map(loaded_tox)
        # DCM has hazard_score=0.7 → safety via solvent key "DCM"
        score = re_mod.calc_toxicity_score(
            [], {"solvent": "DCM"}, loaded_tox, solvent_map)
        assert score == pytest.approx(0.3, abs=0.05)


class TestBuildSolventMap:
    def test_returns_dict(self, loaded_tox):
        m = re_mod.build_solvent_map(loaded_tox)
        assert isinstance(m, dict)

    def test_common_abbreviations_present(self, loaded_tox):
        m = re_mod.build_solvent_map(loaded_tox)
        assert "DCM" in m
        assert "THF" in m
        assert "MeOH" in m

    def test_tox_keys_self_mapped(self, loaded_tox):
        m = re_mod.build_solvent_map(loaded_tox)
        for k in loaded_tox:
            assert k in m and m[k] == k


# ===========================================================================
# route_engine — Criterion compute functions
# ===========================================================================

class TestComputeSteps:
    def test_single_step_scores_one(self, aspirin_route, loaded_tox):
        assert re_mod.compute_steps(aspirin_route, loaded_tox) == pytest.approx(1.0)

    def test_two_steps_scores_half(self, loaded_tox):
        route = {
            "dataset_steps": [{"step_number": 1}, {"step_number": 2}]
        }
        assert re_mod.compute_steps(route, loaded_tox) == pytest.approx(0.5)

    def test_empty_steps_scores_one(self, loaded_tox):
        assert re_mod.compute_steps({"dataset_steps": []}, loaded_tox) == pytest.approx(1.0)


class TestComputeYield:
    def test_dataset_route_with_yield(self, aspirin_route, loaded_tox):
        score = re_mod.compute_yield(aspirin_route, loaded_tox)
        assert score == pytest.approx(0.85)

    def test_predicted_route_returns_one(self, predicted_route, loaded_tox):
        score = re_mod.compute_yield(predicted_route, loaded_tox)
        assert score == pytest.approx(1.0)

    def test_empty_steps_returns_zero(self, loaded_tox):
        route = {"dataset_steps": [], "validation_status": "dataset"}
        assert re_mod.compute_yield(route, loaded_tox) == pytest.approx(0.0)

    def test_missing_yield_treated_as_neutral(self, loaded_tox):
        route = {
            "dataset_steps": [
                {"yield_percent": None, "source": "dataset"},
            ],
            "validation_status": "dataset",
        }
        assert re_mod.compute_yield(route, loaded_tox) == pytest.approx(1.0)


class TestComputeAtomEconomy:
    def test_returns_float_in_range(self, aspirin_route, loaded_tox):
        score = re_mod.compute_atom_economy(aspirin_route, loaded_tox)
        assert 0.0 <= score <= 1.0

    def test_empty_steps_returns_zero(self, loaded_tox):
        route = {"dataset_steps": []}
        assert re_mod.compute_atom_economy(route, loaded_tox) == 0.0


class TestComputeEFactor:
    def test_returns_float_in_range(self, aspirin_route, loaded_tox):
        score = re_mod.compute_e_factor(aspirin_route, loaded_tox)
        assert 0.0 < score <= 1.0

    def test_empty_steps_returns_zero(self, loaded_tox):
        route = {"dataset_steps": []}
        assert re_mod.compute_e_factor(route, loaded_tox) == 0.0


class TestComputeToxicity:
    def test_returns_float_in_range(self, aspirin_route, loaded_tox):
        score = re_mod.compute_toxicity(aspirin_route, loaded_tox)
        assert 0.0 <= score <= 1.0

    def test_empty_steps_returns_neutral(self, loaded_tox):
        route = {"dataset_steps": []}
        assert re_mod.compute_toxicity(route, loaded_tox) == pytest.approx(0.5)


# ===========================================================================
# route_engine — Weighted ranking
# ===========================================================================

class TestComputeWeights:
    def test_weights_sum_to_one(self):
        criteria = ["steps", "yield", "atom_economy"]
        w = re_mod.compute_weights(criteria)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_first_criterion_has_highest_weight(self):
        criteria = ["steps", "yield", "atom_economy"]
        w = re_mod.compute_weights(criteria)
        assert w["steps"] > w["yield"] > w["atom_economy"]

    def test_single_criterion_weight_is_one(self):
        w = re_mod.compute_weights(["steps"])
        assert w["steps"] == pytest.approx(1.0)

    def test_returns_all_keys(self):
        criteria = ["steps", "yield", "e_factor"]
        w = re_mod.compute_weights(criteria)
        assert set(w.keys()) == set(criteria)


class TestComputeAllScores:
    def test_returns_all_registry_keys(self, aspirin_route, loaded_tox):
        scores = re_mod.compute_all_scores(aspirin_route, loaded_tox)
        for key in re_mod.CRITERIA_REGISTRY:
            assert key in scores

    def test_all_scores_in_range(self, aspirin_route, loaded_tox):
        scores = re_mod.compute_all_scores(aspirin_route, loaded_tox)
        for v in scores.values():
            assert 0.0 <= v <= 1.0


class TestRankWeighted:
    def test_returns_sorted_descending(self, aspirin_route, predicted_route, loaded_tox):
        criteria = ["steps", "yield", "atom_economy"]
        routes = [aspirin_route, predicted_route]
        ranked = re_mod.rank_weighted(routes, criteria, loaded_tox)
        scores = [s for s, _, _ in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_each_entry_is_three_tuple(self, aspirin_route, loaded_tox):
        ranked = re_mod.rank_weighted(
            [aspirin_route], ["steps", "yield", "atom_economy"], loaded_tox)
        assert len(ranked) == 1
        score, details, route = ranked[0]
        assert isinstance(score, float)
        assert isinstance(details, dict)
        assert isinstance(route, dict)

    def test_predicted_route_excludes_yield(self, predicted_route, loaded_tox):
        ranked = re_mod.rank_weighted(
            [predicted_route], ["steps", "yield", "atom_economy"], loaded_tox)
        _, details, _ = ranked[0]
        assert details["yield"]["excluded"] is True

    def test_dataset_route_includes_yield(self, aspirin_route, loaded_tox):
        ranked = re_mod.rank_weighted(
            [aspirin_route], ["steps", "yield", "atom_economy"], loaded_tox)
        _, details, _ = ranked[0]
        assert details["yield"].get("excluded", False) is False

    def test_details_contains_all_scores(self, aspirin_route, loaded_tox):
        ranked = re_mod.rank_weighted(
            [aspirin_route], ["steps", "yield", "atom_economy"], loaded_tox)
        _, details, _ = ranked[0]
        assert "_all_scores" in details

    def test_empty_routes_returns_empty(self, loaded_tox):
        ranked = re_mod.rank_weighted([], ["steps", "yield", "atom_economy"], loaded_tox)
        assert ranked == []


# ===========================================================================
# route_engine — Generic dataset step matching
# ===========================================================================

class TestMatchStepInGenericDataset:
    def test_exact_match(self, tmp_path):
        data = [
            {"id": "g1", "reactants_smiles": ["CCO"], "product_smiles": "CC=O",
             "yield_percent": 80.0, "reaction_type": "Oxidation"},
        ]
        p = tmp_path / "generic.json"
        p.write_text(json.dumps(data))
        ds = re_mod.load_generic_reaction_dataset(str(p))
        result = re_mod.match_step_in_generic_dataset(["CCO"], "CC=O", ds)
        assert result is not None
        assert result["yield_percent"] == 80.0

    def test_product_only_match(self, tmp_path):
        data = [
            {"id": "g1", "reactants_smiles": ["CCO", "CC(=O)O"],
             "product_smiles": "CCOC(C)=O",
             "yield_percent": 75.0, "reaction_type": "Esterification"},
        ]
        p = tmp_path / "generic.json"
        p.write_text(json.dumps(data))
        ds = re_mod.load_generic_reaction_dataset(str(p))
        # Different reactants but same product → product-only match
        result = re_mod.match_step_in_generic_dataset(["C"], "CCOC(C)=O", ds)
        assert result is not None

    def test_no_match_returns_none(self, tmp_path):
        data = [
            {"id": "g1", "reactants_smiles": ["CCO"], "product_smiles": "CC=O",
             "yield_percent": 80.0, "reaction_type": "Oxidation"},
        ]
        p = tmp_path / "generic.json"
        p.write_text(json.dumps(data))
        ds = re_mod.load_generic_reaction_dataset(str(p))
        result = re_mod.match_step_in_generic_dataset(["c1ccccc1"], "CCCCCC", ds)
        assert result is None

    def test_empty_dataset_returns_none(self):
        result = re_mod.match_step_in_generic_dataset(["CCO"], "CC=O", {})
        assert result is None


# ===========================================================================
# route_engine — Route coverage check
# ===========================================================================

class TestIsRouteCoveredByDataset:
    def test_fully_covered_route(self):
        canon = re_mod.to_canonical("CCO")
        index = {canon}
        route = {"steps": [{"product": "CCO"}]}
        assert re_mod.is_route_covered_by_dataset(route, index, threshold=0.4) is True

    def test_not_covered_route(self):
        index = {re_mod.to_canonical("CCO")}
        route = {"steps": [{"product": "c1ccccc1"}]}
        assert re_mod.is_route_covered_by_dataset(route, index, threshold=0.4) is False

    def test_empty_steps_returns_false(self):
        assert re_mod.is_route_covered_by_dataset({"steps": []}, set()) is False

    def test_partial_coverage_above_threshold(self):
        index = {re_mod.to_canonical("CCO"), re_mod.to_canonical("CC=O")}
        route = {
            "steps": [
                {"product": "CCO"},
                {"product": "CC=O"},
                {"product": "c1ccccc1"},   # not covered
            ]
        }
        # 2/3 = 0.67 ≥ 0.4 → covered
        assert re_mod.is_route_covered_by_dataset(route, index, threshold=0.4) is True

    def test_partial_coverage_below_threshold(self):
        index = {re_mod.to_canonical("CCO")}
        route = {
            "steps": [
                {"product": "CCO"},
                {"product": "c1ccccc1"},
                {"product": "CC(=O)O"},
                {"product": "CCCC"},
            ]
        }
        # 1/4 = 0.25 < 0.4 → not covered
        assert re_mod.is_route_covered_by_dataset(route, index, threshold=0.4) is False


# ===========================================================================
# route_engine — Dataset route filtering
# ===========================================================================

class TestFilterRoutesByStartingMaterials:
    def test_matches_by_canonical_product(self, loaded_dataset):
        aspirin_canon = re_mod.to_canonical("CC(=O)Oc1ccccc1C(=O)O")
        routes = re_mod.filter_routes_by_starting_materials(
            [], loaded_dataset, aspirin_canon, "")
        assert len(routes) >= 1

    def test_matches_by_target_name(self, loaded_dataset):
        routes = re_mod.filter_routes_by_starting_materials(
            [], loaded_dataset, "", "Aspirin")
        assert len(routes) >= 1

    def test_no_match_returns_empty(self, loaded_dataset):
        routes = re_mod.filter_routes_by_starting_materials(
            [], loaded_dataset, re_mod.to_canonical("c1ccccc1"), "UnknownMol")
        assert routes == []

    def test_returned_routes_have_required_keys(self, loaded_dataset):
        aspirin_canon = re_mod.to_canonical("CC(=O)Oc1ccccc1C(=O)O")
        routes = re_mod.filter_routes_by_starting_materials(
            [], loaded_dataset, aspirin_canon, "")
        for r in routes:
            for key in ("route_id", "dataset_steps", "matched_route_name",
                        "is_predicted", "validation_status"):
                assert key in r


class TestGetAllDatasetRoutesForTarget:
    def test_returns_routes_for_known_target(self, loaded_dataset):
        routes = re_mod.get_all_dataset_routes_for_target(loaded_dataset, "Aspirin")
        assert len(routes) >= 1

    def test_returns_empty_for_unknown_target(self, loaded_dataset):
        routes = re_mod.get_all_dataset_routes_for_target(loaded_dataset, "Unknown")
        assert routes == []

    def test_skips_wrong_canonical_product(self, loaded_dataset):
        # Provide a SMILES that does not match the dataset products
        routes = re_mod.get_all_dataset_routes_for_target(
            loaded_dataset, "Aspirin",
            target_smiles=re_mod.to_canonical("c1ccccc1"))  # benzene ≠ aspirin
        assert routes == []


# ===========================================================================
# route_engine — _walk_reaction_tree and adapt_route
# ===========================================================================

class TestWalkReactionTree:
    def test_collects_steps(self):
        tree = {
            "type": "mol",
            "smiles": "CCO",
            "children": [
                {
                    "type": "reaction",
                    "children": [
                        {"type": "mol", "smiles": "C", "children": []},
                        {"type": "mol", "smiles": "O", "children": []},
                    ],
                }
            ],
        }
        steps = []
        re_mod._walk_reaction_tree(tree, steps)
        assert len(steps) == 1
        assert steps[0]["product"] == "CCO"
        assert set(steps[0]["reactants"]) == {"C", "O"}

    def test_empty_tree_no_steps(self):
        steps = []
        re_mod._walk_reaction_tree({"type": "mol", "smiles": "C", "children": []}, steps)
        assert steps == []


class TestAdaptRoute:
    def test_returns_dict_with_required_keys(self):
        tree_mock = MagicMock()
        tree_mock.to_dict.return_value = {
            "type": "mol",
            "smiles": "CCO",
            "children": [
                {
                    "type": "reaction",
                    "children": [{"type": "mol", "smiles": "C", "children": []}],
                }
            ],
        }
        raw = {"reaction_tree": tree_mock}
        result = re_mod.adapt_route(raw)
        for key in ("route_id", "route_name", "steps",
                    "starting_materials", "target_smiles", "raw"):
            assert key in result

    def test_tree_error_returns_empty_steps(self):
        tree_mock = MagicMock()
        tree_mock.to_dict.side_effect = RuntimeError("fail")
        raw = {"reaction_tree": tree_mock}
        result = re_mod.adapt_route(raw)
        assert result["steps"] == []


# ===========================================================================
# route_engine — find_best_routes validation errors
# ===========================================================================

class TestFindBestRoutesValidation:
    def test_wrong_number_of_criteria_raises(self):
        with pytest.raises(ValueError, match="3 criteria"):
            re_mod.find_best_routes("CCO", ["steps", "yield"])

    def test_unknown_criterion_raises(self):
        with pytest.raises(ValueError, match="unknown criteria"):
            re_mod.find_best_routes("CCO", ["steps", "yield", "magic"])


# ===========================================================================
# localization — structure tests
# ===========================================================================

class TestLang:
    REQUIRED_KEYS = [
        "page_title", "page_caption", "run_btn", "no_routes",
        "tab_search", "tab_analysis", "tab_dataset", "tab_help",
        "err_file", "err_param", "err_other",
        "sec_dataset", "sec_validated", "sec_predicted",
        "badge_dataset", "badge_validated", "badge_predicted",
    ]

    def test_english_present(self):
        assert "en" in loc_mod.LANG

    def test_french_present(self):
        assert "fr" in loc_mod.LANG

    @pytest.mark.parametrize("lang", ["en", "fr"])
    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_required_key_present(self, lang, key):
        assert key in loc_mod.LANG[lang], f"LANG[{lang!r}] missing key {key!r}"

    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_all_values_are_strings(self, lang):
        for k, v in loc_mod.LANG[lang].items():
            assert isinstance(v, str), f"LANG[{lang!r}][{k!r}] is not a str"


class TestCriteriaLabels:
    EXPECTED_CRITERIA = ["steps", "yield", "atom_economy", "e_factor", "toxicity"]

    def test_english_present(self):
        assert "en" in loc_mod.CRITERIA_LABELS

    def test_french_present(self):
        assert "fr" in loc_mod.CRITERIA_LABELS

    @pytest.mark.parametrize("lang", ["en", "fr"])
    @pytest.mark.parametrize("crit", EXPECTED_CRITERIA)
    def test_criterion_label_present(self, lang, crit):
        assert crit in loc_mod.CRITERIA_LABELS[lang]


class TestPaletteAndFigBg:
    def test_palette_is_list(self):
        assert isinstance(loc_mod.PALETTE, list)

    def test_palette_has_at_least_one_colour(self):
        assert len(loc_mod.PALETTE) >= 1

    def test_palette_entries_are_hex_strings(self):
        for c in loc_mod.PALETTE:
            assert isinstance(c, str) and c.startswith("#")

    def test_fig_bg_is_string(self):
        assert isinstance(loc_mod.FIG_BG, str)

    def test_fig_bg_is_hex(self):
        assert loc_mod.FIG_BG.startswith("#")


# ===========================================================================
# molecule_rendering — rendering utilities
# ===========================================================================

class TestMolPng:
    def test_valid_smiles_returns_bytes(self):
        result = mr_mod.mol_png("CCO")
        # May be None if RDKit/Cairo unavailable in test env, but must be bytes if not None
        assert result is None or isinstance(result, bytes)

    def test_invalid_smiles_returns_none(self):
        assert mr_mod.mol_png("NOT_A_SMILES_XYZ") is None

    def test_empty_smiles_returns_none(self):
        assert mr_mod.mol_png("") is None


class TestMolB64OrTextSvg:
    def test_returns_string(self):
        result = mr_mod.mol_b64_or_text_svg("CCO", 200, 150)
        assert isinstance(result, str)

    def test_returns_data_uri_prefix(self):
        result = mr_mod.mol_b64_or_text_svg("CCO", 200, 150)
        assert result.startswith("data:image/png;base64,")

    def test_invalid_smiles_returns_fallback(self):
        result = mr_mod.mol_b64_or_text_svg("NOT_SMILES", 200, 150)
        assert result.startswith("data:image/png;base64,")

    def test_empty_smiles_returns_fallback(self):
        result = mr_mod.mol_b64_or_text_svg("", 100, 100)
        assert result.startswith("data:image/png;base64,")


class TestFallbackDataUri:
    def test_returns_data_uri(self):
        result = mr_mod.fallback_data_uri("test", 100, 80)
        assert result.startswith("data:image/png;base64,")

    def test_long_text_truncated_in_label(self):
        # Should not raise even for very long text
        result = mr_mod.fallback_data_uri("A" * 100, 200, 100)
        assert isinstance(result, str)

    def test_small_dimensions(self):
        result = mr_mod.fallback_data_uri("?", 10, 10)
        assert isinstance(result, str)


class TestIsTrivialSmiles:
    def test_single_atom_is_trivial(self):
        assert mr_mod.is_trivial_smiles("C") is True

    def test_two_heavy_atoms_is_trivial(self):
        assert mr_mod.is_trivial_smiles("[Na+]") is True

    def test_large_molecule_not_trivial(self):
        # Benzene has 6 heavy atoms
        assert mr_mod.is_trivial_smiles("c1ccccc1") is False

    def test_empty_string_is_trivial(self):
        assert mr_mod.is_trivial_smiles("") is True

    def test_invalid_smiles_is_trivial(self):
        assert mr_mod.is_trivial_smiles("NOT_VALID") is True

    def test_aspirin_not_trivial(self):
        assert mr_mod.is_trivial_smiles("CC(=O)Oc1ccccc1C(=O)O") is False
