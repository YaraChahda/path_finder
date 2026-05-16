"""Tests for route_engine.py.

AiZynthFinder and rxn_insight are mocked at sys.modules level before import
so the test suite runs without those optional heavy dependencies installed.
"""

import sys
import os
import json
import tempfile
from unittest.mock import MagicMock, patch

# Mock heavy optional dependencies before importing route_engine
for mod in [
    "aizynthfinder",
    "aizynthfinder.aizynthfinder",
    "rxn_insight",
    "rxn_insight.reaction",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

_aiz_mock = MagicMock()
sys.modules["aizynthfinder.aizynthfinder"].AiZynthFinder = _aiz_mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "path_finder"))

import pytest
import src.path_finder.route_engine as rt


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_step(reactants, product, yield_pct=None, source="dataset", reaction_type=""):
    return {
        "reactants_smiles": reactants,
        "product_smiles":   product,
        "yield_percent":    yield_pct,
        "source":           source,
        "reaction_type":    reaction_type,
        "conditions":       {},
    }


def make_route(*steps):
    return {"dataset_steps": list(steps), "validation_status": "dataset"}


BENZENE = "c1ccccc1"
TOLUENE = "Cc1ccccc1"
ASPIRIN  = "CC(=O)Oc1ccccc1C(=O)O"


# ---------------------------------------------------------------------------
# to_canonical
# ---------------------------------------------------------------------------

def test_to_canonical_empty_string():
    assert rt.to_canonical("") == ""


def test_to_canonical_none_via_empty_list():
    assert rt.to_canonical([]) == ""


def test_to_canonical_valid_smiles():
    result = rt.to_canonical(BENZENE)
    assert isinstance(result, str)
    assert len(result) > 0


def test_to_canonical_list_input():
    result = rt.to_canonical([BENZENE, TOLUENE])
    assert isinstance(result, str)


def test_to_canonical_invalid_smiles_returns_original():
    bad = "not!!valid"
    result = rt.to_canonical(bad)
    assert result == bad


def test_to_canonical_idempotent():
    c1 = rt.to_canonical(BENZENE)
    c2 = rt.to_canonical(c1)
    assert c1 == c2


# ---------------------------------------------------------------------------
# safe_mol
# ---------------------------------------------------------------------------

def test_safe_mol_empty_returns_none():
    assert rt.safe_mol("") is None


def test_safe_mol_invalid_returns_none():
    assert rt.safe_mol("not_a_smiles!!!") is None


def test_safe_mol_valid_returns_mol():
    mol = rt.safe_mol(BENZENE)
    assert mol is not None


def test_safe_mol_type():
    mol = rt.safe_mol(BENZENE)
    assert hasattr(mol, "GetNumAtoms")


# ---------------------------------------------------------------------------
# validate_smiles_for_aizynthfinder
# ---------------------------------------------------------------------------

def test_validate_smiles_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        rt.validate_smiles_for_aizynthfinder("")


def test_validate_smiles_raises_on_invalid():
    with pytest.raises(ValueError, match="invalid"):
        rt.validate_smiles_for_aizynthfinder("not!valid")


def test_validate_smiles_returns_canonical():
    result = rt.validate_smiles_for_aizynthfinder(BENZENE)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# build_dataset_smiles_index
# ---------------------------------------------------------------------------

def test_build_dataset_smiles_index_empty():
    dataset = {"all": []}
    index = rt.build_dataset_smiles_index(dataset)
    assert isinstance(index, set)
    assert len(index) == 0


def test_build_dataset_smiles_index_collects_products():
    dataset = {"all": [{"product_smiles": BENZENE, "reactants_smiles": [TOLUENE]}]}
    index = rt.build_dataset_smiles_index(dataset)
    assert rt.to_canonical(BENZENE) in index


def test_build_dataset_smiles_index_collects_reactants():
    dataset = {"all": [{"product_smiles": BENZENE, "reactants_smiles": [TOLUENE]}]}
    index = rt.build_dataset_smiles_index(dataset)
    assert rt.to_canonical(TOLUENE) in index


# ---------------------------------------------------------------------------
# bottleneck_yield
# ---------------------------------------------------------------------------

def test_bottleneck_yield_empty_steps():
    assert rt.bottleneck_yield([]) is None


def test_bottleneck_yield_no_reported_yields():
    steps = [make_step([BENZENE], TOLUENE, None)]
    assert rt.bottleneck_yield(steps) is None


def test_bottleneck_yield_single_step():
    steps = [make_step([BENZENE], TOLUENE, 75.0)]
    assert rt.bottleneck_yield(steps) == 75.0


def test_bottleneck_yield_returns_minimum():
    steps = [
        make_step([BENZENE], TOLUENE, 80.0),
        make_step([TOLUENE], ASPIRIN, 60.0),
        make_step([ASPIRIN], BENZENE, 90.0),
    ]
    assert rt.bottleneck_yield(steps) == 60.0


def test_bottleneck_yield_ignores_none():
    steps = [
        make_step([BENZENE], TOLUENE, 70.0),
        make_step([TOLUENE], ASPIRIN, None),
    ]
    assert rt.bottleneck_yield(steps) == 70.0


# ---------------------------------------------------------------------------
# average_yield
# ---------------------------------------------------------------------------

def test_average_yield_empty():
    assert rt.average_yield([]) is None


def test_average_yield_no_yields():
    steps = [make_step([BENZENE], TOLUENE, None)]
    assert rt.average_yield(steps) is None


def test_average_yield_single():
    steps = [make_step([BENZENE], TOLUENE, 80.0)]
    assert rt.average_yield(steps) == pytest.approx(80.0)


def test_average_yield_multiple():
    steps = [
        make_step([BENZENE], TOLUENE, 60.0),
        make_step([TOLUENE], ASPIRIN, 80.0),
    ]
    assert rt.average_yield(steps) == pytest.approx(70.0)


def test_average_yield_ignores_none():
    steps = [
        make_step([BENZENE], TOLUENE, 60.0),
        make_step([TOLUENE], ASPIRIN, None),
        make_step([ASPIRIN], BENZENE, 80.0),
    ]
    assert rt.average_yield(steps) == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# cumulative_yield
# ---------------------------------------------------------------------------

def test_cumulative_yield_empty():
    assert rt.cumulative_yield([]) == pytest.approx(1.0)


def test_cumulative_yield_single_step():
    steps = [make_step([BENZENE], TOLUENE, 50.0)]
    assert rt.cumulative_yield(steps) == pytest.approx(0.5)


def test_cumulative_yield_multiple_steps():
    steps = [
        make_step([BENZENE], TOLUENE, 50.0),
        make_step([TOLUENE], ASPIRIN, 80.0),
    ]
    assert rt.cumulative_yield(steps) == pytest.approx(0.5 * 0.8)


def test_cumulative_yield_none_treated_as_neutral():
    steps = [
        make_step([BENZENE], TOLUENE, 50.0),
        make_step([TOLUENE], ASPIRIN, None),
    ]
    assert rt.cumulative_yield(steps) == pytest.approx(0.5)


def test_cumulative_yield_all_none():
    steps = [make_step([BENZENE], TOLUENE, None)]
    assert rt.cumulative_yield(steps) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# get_substances_list
# ---------------------------------------------------------------------------

def test_get_substances_list_empty():
    result = rt.get_substances_list([])
    assert result["to_buy"] == []
    assert result["to_prepare"] == []
    assert result["solvents"] == []
    assert result["reagents"] == []


def test_get_substances_list_keys():
    result = rt.get_substances_list([])
    assert "to_buy" in result
    assert "to_prepare" in result
    assert "solvents" in result
    assert "reagents" in result


def test_get_substances_list_starting_materials():
    steps = [
        {
            "reactants_smiles": [BENZENE],
            "product_smiles":   TOLUENE,
            "conditions":       {},
        }
    ]
    result = rt.get_substances_list(steps)
    assert rt.to_canonical(BENZENE) in result["to_buy"]


def test_get_substances_list_intermediates_not_in_to_buy():
    steps = [
        {"reactants_smiles": [BENZENE], "product_smiles": TOLUENE, "conditions": {}},
        {"reactants_smiles": [TOLUENE], "product_smiles": ASPIRIN, "conditions": {}},
    ]
    result = rt.get_substances_list(steps)
    # TOLUENE is produced by step 1 so not "to buy"
    assert rt.to_canonical(TOLUENE) not in result["to_buy"]


def test_get_substances_list_solvents():
    steps = [
        {
            "reactants_smiles": [BENZENE],
            "product_smiles":   TOLUENE,
            "conditions":       {"solvent": "THF"},
        }
    ]
    result = rt.get_substances_list(steps)
    assert "THF" in result["solvents"]


def test_get_substances_list_reagents():
    steps = [
        {
            "reactants_smiles": [BENZENE],
            "product_smiles":   TOLUENE,
            "conditions":       {"reagents": ["NaH", "BnBr"]},
        }
    ]
    result = rt.get_substances_list(steps)
    assert "NaH" in result["reagents"]
    assert "BnBr" in result["reagents"]


# ---------------------------------------------------------------------------
# fmt_conditions
# ---------------------------------------------------------------------------

def test_fmt_conditions_empty_dict():
    assert rt.fmt_conditions({}) == ""


def test_fmt_conditions_none():
    assert rt.fmt_conditions(None) == ""


def test_fmt_conditions_temperature():
    result = rt.fmt_conditions({"temperature_C": 80})
    assert "80" in result


def test_fmt_conditions_solvent():
    result = rt.fmt_conditions({"solvent": "THF"})
    assert "THF" in result


def test_fmt_conditions_reagents():
    result = rt.fmt_conditions({"reagents": ["NaH", "BnBr"]})
    assert "NaH" in result


def test_fmt_conditions_apparatus():
    result = rt.fmt_conditions({"apparatus": "microwave"})
    assert "microwave" in result


def test_fmt_conditions_multiple_fields():
    result = rt.fmt_conditions({"temperature_C": 60, "solvent": "DCM"})
    assert "60" in result
    assert "DCM" in result


# ---------------------------------------------------------------------------
# calc_atom_economy
# ---------------------------------------------------------------------------

def test_calc_atom_economy_invalid_product():
    result = rt.calc_atom_economy(["CC"], "not!valid")
    assert result == pytest.approx(0.0)


def test_calc_atom_economy_no_reactants():
    result = rt.calc_atom_economy([], BENZENE)
    assert result == pytest.approx(0.0)


def test_calc_atom_economy_returns_float_in_range():
    result = rt.calc_atom_economy([BENZENE, "CO"], TOLUENE)
    assert 0.0 <= result <= 1.0


def test_calc_atom_economy_capped_at_one():
    result = rt.calc_atom_economy([BENZENE], BENZENE)
    assert result <= 1.0


# ---------------------------------------------------------------------------
# calc_e_factor
# ---------------------------------------------------------------------------

def test_calc_e_factor_invalid_product():
    result = rt.calc_e_factor([BENZENE], "not!valid", 1.0)
    assert result == pytest.approx(0.5)


def test_calc_e_factor_returns_float():
    result = rt.calc_e_factor([BENZENE, "CO"], TOLUENE, 0.8)
    assert isinstance(result, float)
    assert 0.0 < result <= 1.0


def test_calc_e_factor_no_reactants():
    # total_mw = 0 → waste = 0 → score = 1/(1+0) = 1.0
    result = rt.calc_e_factor([], BENZENE, 1.0)
    assert result == pytest.approx(1.0)


def test_calc_e_factor_perfect_yield():
    result = rt.calc_e_factor([BENZENE], BENZENE, 1.0)
    assert 0.0 < result <= 1.0


# ---------------------------------------------------------------------------
# build_solvent_map
# ---------------------------------------------------------------------------

def test_build_solvent_map_returns_dict():
    result = rt.build_solvent_map({})
    assert isinstance(result, dict)


def test_build_solvent_map_contains_dcm():
    result = rt.build_solvent_map({})
    assert "DCM" in result
    assert result["DCM"] == "ClCCl"


def test_build_solvent_map_contains_thf():
    result = rt.build_solvent_map({})
    assert "THF" in result


def test_build_solvent_map_contains_toluene():
    result = rt.build_solvent_map({})
    assert "toluene" in result


def test_build_solvent_map_appends_tox_index_keys():
    tox_index = {rt.to_canonical(BENZENE): {"hazard_score": 0.3}}
    result = rt.build_solvent_map(tox_index)
    assert rt.to_canonical(BENZENE) in result


# ---------------------------------------------------------------------------
# compute_weights
# ---------------------------------------------------------------------------

def test_compute_weights_returns_dict():
    result = rt.compute_weights(["steps", "yield", "atom_economy"])
    assert isinstance(result, dict)


def test_compute_weights_keys_match_criteria():
    criteria = ["steps", "yield", "atom_economy"]
    result = rt.compute_weights(criteria)
    assert set(result.keys()) == set(criteria)


def test_compute_weights_sum_to_one():
    result = rt.compute_weights(["steps", "yield", "atom_economy"])
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)


def test_compute_weights_first_is_largest():
    result = rt.compute_weights(["steps", "yield", "atom_economy"])
    assert result["steps"] > result["yield"] > result["atom_economy"]


def test_compute_weights_first_approx_73_pct():
    result = rt.compute_weights(["steps", "yield", "atom_economy"])
    assert result["steps"] == pytest.approx(1 / (1 + 1/4 + 1/9), rel=1e-3)


def test_compute_weights_single_criterion():
    result = rt.compute_weights(["steps"])
    assert result["steps"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_all_scores
# ---------------------------------------------------------------------------

def test_compute_all_scores_returns_dict():
    route = make_route(make_step([BENZENE], TOLUENE, 80.0))
    result = rt.compute_all_scores(route, {})
    assert isinstance(result, dict)


def test_compute_all_scores_has_all_criteria():
    route = make_route(make_step([BENZENE], TOLUENE, 80.0))
    result = rt.compute_all_scores(route, {})
    for key in rt.CRITERIA_REGISTRY:
        assert key in result


def test_compute_all_scores_values_are_floats():
    route = make_route(make_step([BENZENE], TOLUENE, 80.0))
    result = rt.compute_all_scores(route, {})
    for v in result.values():
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# compute_steps
# ---------------------------------------------------------------------------

def test_compute_steps_one_step():
    route = make_route(make_step([BENZENE], TOLUENE))
    assert rt.compute_steps(route, {}) == pytest.approx(1.0)


def test_compute_steps_two_steps():
    route = make_route(
        make_step([BENZENE], TOLUENE),
        make_step([TOLUENE], ASPIRIN),
    )
    assert rt.compute_steps(route, {}) == pytest.approx(0.5)


def test_compute_steps_empty_route():
    route = {"dataset_steps": []}
    assert rt.compute_steps(route, {}) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_yield
# ---------------------------------------------------------------------------

def test_compute_yield_predicted_route_returns_neutral():
    route = {"dataset_steps": [make_step([BENZENE], TOLUENE, 50.0)],
             "validation_status": "predicted"}
    assert rt.compute_yield(route, {}) == pytest.approx(1.0)


def test_compute_yield_empty_steps():
    route = {"dataset_steps": [], "validation_status": "dataset"}
    assert rt.compute_yield(route, {}) == pytest.approx(0.0)


def test_compute_yield_cumulative():
    steps = [
        {**make_step([BENZENE], TOLUENE, 80.0), "source": "dataset"},
        {**make_step([TOLUENE], ASPIRIN, 50.0), "source": "dataset"},
    ]
    route = {"dataset_steps": steps, "validation_status": "dataset"}
    assert rt.compute_yield(route, {}) == pytest.approx(0.8 * 0.5)


# ---------------------------------------------------------------------------
# compute_atom_economy
# ---------------------------------------------------------------------------

def test_compute_atom_economy_empty():
    route = {"dataset_steps": []}
    assert rt.compute_atom_economy(route, {}) == pytest.approx(0.0)


def test_compute_atom_economy_returns_float_in_range():
    route = make_route(make_step([BENZENE, "CO"], TOLUENE))
    result = rt.compute_atom_economy(route, {})
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# rank_weighted
# ---------------------------------------------------------------------------

def test_rank_weighted_empty_routes():
    result = rt.rank_weighted([], ["steps", "yield", "atom_economy"], {})
    assert result == []


def test_rank_weighted_single_route():
    route = make_route(make_step([BENZENE], TOLUENE, 80.0))
    result = rt.rank_weighted([route], ["steps", "yield", "atom_economy"], {})
    assert len(result) == 1
    score, details, r = result[0]
    assert isinstance(score, float)
    assert "steps" in details
    assert r is route


def test_rank_weighted_descending_order():
    route_short = {"dataset_steps": [make_step([BENZENE], TOLUENE, 80.0)],
                   "validation_status": "dataset"}
    route_long = {"dataset_steps": [make_step([BENZENE], TOLUENE, 80.0)] * 5,
                  "validation_status": "dataset"}
    criteria = ["steps", "yield", "atom_economy"]
    result = rt.rank_weighted([route_long, route_short], criteria, {})
    assert result[0][0] >= result[1][0]


def test_rank_weighted_predicted_excludes_yield():
    route = {
        "dataset_steps": [make_step([BENZENE], TOLUENE, 80.0)],
        "validation_status": "predicted",
    }
    result = rt.rank_weighted([route], ["steps", "yield", "atom_economy"], {})
    score, details, _ = result[0]
    assert details["yield"]["excluded"] is True
    assert details["yield"]["raw"] is None


def test_rank_weighted_details_have_required_keys():
    route = make_route(make_step([BENZENE], TOLUENE, 80.0))
    _, details, _ = rt.rank_weighted([route], ["steps", "yield", "atom_economy"], {})[0]
    for c in ["steps", "yield", "atom_economy"]:
        assert "raw" in details[c]
        assert "weight" in details[c]
        assert "weighted" in details[c]
        assert "excluded" in details[c]


# ---------------------------------------------------------------------------
# load_reaction_dataset
# ---------------------------------------------------------------------------

def test_load_reaction_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        rt.load_reaction_dataset("/non/existent/path.json")


def test_load_reaction_dataset_list_format(tmp_path):
    data = [
        {
            "id": "1", "route_id": "r1", "route_name": "Test", "target": "aspirin",
            "step_number": 1, "reactants_smiles": [BENZENE],
            "product_smiles": TOLUENE, "conditions": {}, "yield_percent": 80,
            "reaction_type": "alkylation",
        }
    ]
    f = tmp_path / "ds.json"
    f.write_text(json.dumps(data))
    result = rt.load_reaction_dataset(str(f))
    assert "all" in result
    assert "by_route" in result
    assert "by_product" in result
    assert "by_reactant" in result
    assert len(result["all"]) == 1


def test_load_reaction_dataset_dict_format(tmp_path):
    data = {"reactions": [
        {
            "id": "1", "route_id": "r1", "route_name": "Test", "target": "aspirin",
            "step_number": 1, "reactants_smiles": [BENZENE],
            "product_smiles": TOLUENE, "conditions": {}, "yield_percent": 80,
            "reaction_type": "",
        }
    ]}
    f = tmp_path / "ds.json"
    f.write_text(json.dumps(data))
    result = rt.load_reaction_dataset(str(f))
    assert len(result["all"]) == 1


def test_load_reaction_dataset_invalid_format(tmp_path):
    f = tmp_path / "ds.json"
    f.write_text(json.dumps({"foo": "bar"}))
    with pytest.raises(ValueError):
        rt.load_reaction_dataset(str(f))


def test_load_reaction_dataset_routes_sorted_by_step(tmp_path):
    data = [
        {"id": "2", "route_id": "r1", "route_name": "T", "target": "t",
         "step_number": 2, "reactants_smiles": [TOLUENE], "product_smiles": ASPIRIN,
         "conditions": {}, "yield_percent": None, "reaction_type": ""},
        {"id": "1", "route_id": "r1", "route_name": "T", "target": "t",
         "step_number": 1, "reactants_smiles": [BENZENE], "product_smiles": TOLUENE,
         "conditions": {}, "yield_percent": None, "reaction_type": ""},
    ]
    f = tmp_path / "ds.json"
    f.write_text(json.dumps(data))
    result = rt.load_reaction_dataset(str(f))
    steps = result["by_route"]["r1"]
    assert steps[0]["step_number"] == 1
    assert steps[1]["step_number"] == 2


# ---------------------------------------------------------------------------
# load_toxicity_dataset
# ---------------------------------------------------------------------------

def test_load_toxicity_dataset_missing_file():
    result = rt.load_toxicity_dataset("/no/such/file.json")
    assert result == {}


def test_load_toxicity_dataset_list_format(tmp_path):
    data = [{"smiles": BENZENE, "hazard_score": 0.4}]
    f = tmp_path / "tox.json"
    f.write_text(json.dumps(data))
    result = rt.load_toxicity_dataset(str(f))
    assert isinstance(result, dict)
    assert len(result) == 1


def test_load_toxicity_dataset_dict_format(tmp_path):
    data = {"compounds": [{"smiles": BENZENE, "hazard_score": 0.4}]}
    f = tmp_path / "tox.json"
    f.write_text(json.dumps(data))
    result = rt.load_toxicity_dataset(str(f))
    assert len(result) == 1


def test_load_toxicity_dataset_keyed_by_canonical(tmp_path):
    data = [{"smiles": BENZENE, "hazard_score": 0.4}]
    f = tmp_path / "tox.json"
    f.write_text(json.dumps(data))
    result = rt.load_toxicity_dataset(str(f))
    canon = rt.to_canonical(BENZENE)
    assert canon in result


# ---------------------------------------------------------------------------
# is_route_covered_by_dataset
# ---------------------------------------------------------------------------

def test_is_route_covered_empty_steps():
    aiz_route = {"steps": []}
    index = set()
    assert rt.is_route_covered_by_dataset(aiz_route, index) is False


def test_is_route_covered_all_products_in_index():
    canon_b = rt.to_canonical(BENZENE)
    aiz_route = {"steps": [{"product": BENZENE}]}
    assert rt.is_route_covered_by_dataset(aiz_route, {canon_b}) is True


def test_is_route_covered_none_in_index():
    aiz_route = {"steps": [{"product": BENZENE}]}
    assert rt.is_route_covered_by_dataset(aiz_route, set()) is False


# ---------------------------------------------------------------------------
# filter_routes_by_starting_materials
# ---------------------------------------------------------------------------

def test_filter_routes_empty_dataset():
    dataset = {"by_route": {}}
    result = rt.filter_routes_by_starting_materials([], dataset, BENZENE)
    assert result == []


def test_filter_routes_match_by_smiles(tmp_path):
    canon = rt.to_canonical(TOLUENE)
    dataset = {
        "by_route": {
            "r1": [
                {"route_name": "Test", "target": "toluene",
                 "product_smiles": TOLUENE, "reactants_smiles": [BENZENE],
                 "conditions": {}, "yield_percent": None, "reaction_type": ""},
            ]
        }
    }
    result = rt.filter_routes_by_starting_materials([], dataset, TOLUENE)
    assert len(result) == 1
    assert result[0]["matched_route_id"] == "r1"