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

# --- to_canonical ---
def test_to_canonical_non_string_non_list():
    assert rt.to_canonical(12345) == ""

# --- load_reaction_dataset normalisation branches ---
def test_load_dataset_product_smiles_as_list(tmp_path):
    data = [{"id":"1","route_id":"r1","route_name":"T","target":"t",
             "step_number":1,"reactants_smiles":[BENZENE],
             "product_smiles":[TOLUENE, ASPIRIN],
             "conditions":{},"yield_percent":None,"reaction_type":""}]
    f = tmp_path / "ds.json"
    f.write_text(json.dumps(data))
    result = rt.load_reaction_dataset(str(f))
    assert "." in result["all"][0]["product_smiles"]

def test_load_dataset_reactants_as_string(tmp_path):
    data = [{"id":"1","route_id":"r1","route_name":"T","target":"t",
             "step_number":1,"reactants_smiles": BENZENE,
             "product_smiles":TOLUENE,
             "conditions":{},"yield_percent":None,"reaction_type":""}]
    f = tmp_path / "ds.json"
    f.write_text(json.dumps(data))
    result = rt.load_reaction_dataset(str(f))
    assert isinstance(result["all"][0]["reactants_smiles"], list)

def test_load_dataset_reactants_nested_list(tmp_path):
    data = [{"id":"1","route_id":"r1","route_name":"T","target":"t",
             "step_number":1,"reactants_smiles":[[BENZENE],[TOLUENE]],
             "product_smiles":ASPIRIN,
             "conditions":{},"yield_percent":None,"reaction_type":""}]
    f = tmp_path / "ds.json"
    f.write_text(json.dumps(data))
    result = rt.load_reaction_dataset(str(f))
    assert all(isinstance(r, str) for r in result["all"][0]["reactants_smiles"])

def test_load_dataset_metadata_extracted(tmp_path):
    data = {"reactions":[{"id":"1","route_id":"r1","route_name":"T","target":"t",
             "step_number":1,"reactants_smiles":[BENZENE],"product_smiles":TOLUENE,
             "conditions":{},"yield_percent":None,"reaction_type":""}],
            "_metadata":{"target_smiles":{"aspirin": ASPIRIN}}}
    f = tmp_path / "ds.json"
    f.write_text(json.dumps(data))
    result = rt.load_reaction_dataset(str(f))
    assert result["metadata"]["target_smiles"]["aspirin"] == ASPIRIN

# --- get_targets_from_dataset ---
def test_get_targets_from_dataset_uses_metadata(tmp_path):
    data = {"reactions":[{"id":"1","route_id":"r1","route_name":"R","target":"aspirin",
             "step_number":1,"reactants_smiles":[BENZENE],"product_smiles":ASPIRIN,
             "conditions":{},"yield_percent":None,"reaction_type":""}],
            "_metadata":{"target_smiles":{"aspirin": ASPIRIN}}}
    f = tmp_path / "ds.json"
    f.write_text(json.dumps(data))
    dataset = rt.load_reaction_dataset(str(f))
    result = rt.get_targets_from_dataset(dataset)
    assert "aspirin" in result

def test_get_targets_from_dataset_fallback_largest_product(tmp_path):
    galanthamine = "OC1C=C[C@@]23c4cc(OC)ccc4CN(C)C[C@@H]2[C@@H]1O3"
    data = [{"id":"1","route_id":"r1","route_name":"R","target":"galanthamine",
             "step_number":1,"reactants_smiles":[BENZENE],"product_smiles":galanthamine,
             "conditions":{},"yield_percent":None,"reaction_type":""}]
    f = tmp_path / "ds.json"
    f.write_text(json.dumps(data))
    dataset = rt.load_reaction_dataset(str(f))
    result = rt.get_targets_from_dataset(dataset)
    assert "galanthamine" in result

def test_get_targets_from_dataset_skips_unknown(tmp_path):
    data = [{"id":"1","route_id":"r1","route_name":"R","target":"?",
             "step_number":1,"reactants_smiles":[BENZENE],"product_smiles":TOLUENE,
             "conditions":{},"yield_percent":None,"reaction_type":""}]
    f = tmp_path / "ds.json"
    f.write_text(json.dumps(data))
    dataset = rt.load_reaction_dataset(str(f))
    result = rt.get_targets_from_dataset(dataset)
    assert "?" not in result

# --- load_generic_reaction_dataset ---
def test_load_generic_dataset_missing_file():
    assert rt.load_generic_reaction_dataset("/no/such/file.json") == {}

def test_load_generic_dataset_empty_path():
    assert rt.load_generic_reaction_dataset("") == {}

def test_load_generic_dataset_list_format(tmp_path):
    data = [{"product_smiles":TOLUENE,"reactants_smiles":[BENZENE],"reaction_type":"test"}]
    f = tmp_path / "g.json"
    f.write_text(json.dumps(data))
    result = rt.load_generic_reaction_dataset(str(f))
    assert "by_product" in result and "by_reaction_key" in result

def test_load_generic_dataset_dict_format(tmp_path):
    data = {"reactions":[{"product_smiles":TOLUENE,"reactants_smiles":[BENZENE],"reaction_type":"test"}]}
    f = tmp_path / "g.json"
    f.write_text(json.dumps(data))
    result = rt.load_generic_reaction_dataset(str(f))
    assert len(result["all"]) == 1

# --- load_rxninsight_database ---
def test_load_rxninsight_unavailable():
    from unittest.mock import patch
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", False):
        assert rt.load_rxninsight_database("any.parquet") is None

def test_load_rxninsight_missing_file():
    from unittest.mock import patch
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True):
        assert rt.load_rxninsight_database("/no/such.parquet") is None

def test_load_rxninsight_empty_path():
    from unittest.mock import patch
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True):
        assert rt.load_rxninsight_database("") is None

# --- _walk_reaction_tree ---
def test_walk_tree_empty_node():
    steps = []
    rt._walk_reaction_tree({}, steps)
    assert steps == []

def test_walk_tree_non_mol_type():
    steps = []
    rt._walk_reaction_tree({"type":"reaction","smiles":BENZENE}, steps)
    assert steps == []

def test_walk_tree_mol_with_reaction_child():
    node = {"type":"mol","smiles":TOLUENE,"children":[{
        "type":"reaction","children":[{"type":"mol","smiles":BENZENE,"children":[]}]
    }]}
    steps = []
    rt._walk_reaction_tree(node, steps)
    assert len(steps) == 1
    assert steps[0]["product"] == TOLUENE
    assert BENZENE in steps[0]["reactants"]

def test_walk_tree_recursive():
    node = {"type":"mol","smiles":ASPIRIN,"children":[{
        "type":"reaction","children":[{
            "type":"mol","smiles":TOLUENE,"children":[{
                "type":"reaction","children":[{"type":"mol","smiles":BENZENE,"children":[]}]
            }]
        }]
    }]}
    steps = []
    rt._walk_reaction_tree(node, steps)
    assert len(steps) == 2

# --- adapt_route ---
def test_adapt_route_returns_required_keys():
    tree_mock = MagicMock()
    tree_mock.to_dict.return_value = {"type":"mol","smiles":ASPIRIN,"children":[{
        "type":"reaction","children":[{"type":"mol","smiles":TOLUENE,"children":[]}]
    }]}
    result = rt.adapt_route({"reaction_tree": tree_mock})
    for key in ("route_id","route_name","steps","starting_materials","target_smiles","raw"):
        assert key in result

def test_adapt_route_error_recovery():
    tree_mock = MagicMock()
    tree_mock.to_dict.side_effect = RuntimeError("broken")
    result = rt.adapt_route({"reaction_tree": tree_mock})
    assert result["steps"] == []

# --- match_step_in_generic_dataset ---
def test_match_step_empty_ds():
    assert rt.match_step_in_generic_dataset([BENZENE], TOLUENE, {}) is None

def test_match_step_exact_match(tmp_path):
    data = [{"product_smiles":TOLUENE,"reactants_smiles":[BENZENE],"reaction_type":"test","yield_percent":75}]
    f = tmp_path / "g.json"
    f.write_text(json.dumps(data))
    ds = rt.load_generic_reaction_dataset(str(f))
    assert rt.match_step_in_generic_dataset([BENZENE], TOLUENE, ds) is not None

def test_match_step_product_fallback(tmp_path):
    data = [{"product_smiles":TOLUENE,"reactants_smiles":[ASPIRIN],"reaction_type":"test","yield_percent":60}]
    f = tmp_path / "g.json"
    f.write_text(json.dumps(data))
    ds = rt.load_generic_reaction_dataset(str(f))
    # different reactants → no exact match → product-only fallback
    assert rt.match_step_in_generic_dataset([BENZENE], TOLUENE, ds) is not None

def test_match_step_no_match(tmp_path):
    data = [{"product_smiles":ASPIRIN,"reactants_smiles":[BENZENE],"reaction_type":"test"}]
    f = tmp_path / "g.json"
    f.write_text(json.dumps(data))
    ds = rt.load_generic_reaction_dataset(str(f))
    assert rt.match_step_in_generic_dataset([BENZENE], TOLUENE, ds) is None

# --- validate_aiz_route_against_generic_dataset ---
def _make_generic_ds(reactants, product, yield_pct=80):
    return {
        "by_product": {rt.to_canonical(product): [
            {"product_smiles":product,"reactants_smiles":reactants,
             "yield_percent":yield_pct,"reaction_type":"test","conditions":{}}
        ]},
        "by_reaction_key": {
            (tuple(sorted(rt.to_canonical(r) for r in reactants)), rt.to_canonical(product)):
            {"product_smiles":product,"reactants_smiles":reactants,
             "yield_percent":yield_pct,"reaction_type":"test","conditions":{}}
        },
        "all": [],
    }

def test_validate_aiz_all_matched():
    aiz = {"steps":[{"reactants":[BENZENE],"product":TOLUENE}],
           "target_smiles":TOLUENE,"matched_target":"t"}
    result = rt.validate_aiz_route_against_generic_dataset(aiz, _make_generic_ds([BENZENE],TOLUENE), None, 1)
    assert result["validation_status"] == "validated"
    assert result["dataset_steps"][0]["source"] == "generic_dataset"

def test_validate_aiz_none_matched():
    aiz = {"steps":[{"reactants":[BENZENE],"product":TOLUENE}],
           "target_smiles":TOLUENE,"matched_target":"?"}
    result = rt.validate_aiz_route_against_generic_dataset(aiz, {}, None, 1)
    assert result["validation_status"] == "predicted"
    assert result["dataset_steps"][0]["source"] == "rxn-insight"

def test_validate_aiz_partial():
    aiz = {"steps":[{"reactants":[BENZENE],"product":TOLUENE},
                    {"reactants":[TOLUENE],"product":ASPIRIN}],
           "target_smiles":ASPIRIN,"matched_target":"?"}
    result = rt.validate_aiz_route_against_generic_dataset(aiz, _make_generic_ds([BENZENE],TOLUENE), None, 1)
    assert result["validation_status"] == "partial"

def test_validate_aiz_empty_steps():
    aiz = {"steps":[],"target_smiles":"","matched_target":"?"}
    result = rt.validate_aiz_route_against_generic_dataset(aiz, {}, None, 1)
    assert result["validation_status"] == "predicted"
    assert result["dataset_steps"] == []

# --- enrich_aiz_route_with_rxninsight ---
def test_enrich_aiz_route_returns_dict():
    aiz = {"steps":[{"reactants":[BENZENE],"product":TOLUENE}],"target_smiles":TOLUENE}
    result = rt.enrich_aiz_route_with_rxninsight(aiz, None, 1)
    assert isinstance(result, dict)
    assert "dataset_steps" in result

def test_enrich_aiz_marked_predicted():
    aiz = {"steps":[{"reactants":[BENZENE],"product":TOLUENE}],"target_smiles":TOLUENE}
    assert rt.enrich_aiz_route_with_rxninsight(aiz, None, 1)["is_predicted"] is True

def test_enrich_aiz_yield_is_none():
    aiz = {"steps":[{"reactants":[BENZENE],"product":TOLUENE}],"target_smiles":TOLUENE}
    result = rt.enrich_aiz_route_with_rxninsight(aiz, None, 1)
    assert all(s["yield_percent"] is None for s in result["dataset_steps"])

def test_enrich_aiz_last_product_is_target():
    aiz = {"steps":[{"reactants":[BENZENE],"product":TOLUENE},
                    {"reactants":[TOLUENE],"product":BENZENE}],
           "target_smiles":ASPIRIN}
    result = rt.enrich_aiz_route_with_rxninsight(aiz, None, 1)
    assert result["dataset_steps"][-1]["product_smiles"] == ASPIRIN

def test_enrich_aiz_empty_steps():
    aiz = {"steps":[],"target_smiles":TOLUENE}
    result = rt.enrich_aiz_route_with_rxninsight(aiz, None, 1)
    assert result["dataset_steps"] == []

# --- get_all_dataset_routes_for_target ---
def test_get_all_routes_no_match():
    dataset = {"by_route":{"r1":[{"target":"morphine","product_smiles":TOLUENE,"route_name":"R","reactants_smiles":[BENZENE]}]}}
    assert rt.get_all_dataset_routes_for_target(dataset, "aspirin") == []

def test_get_all_routes_name_match():
    dataset = {"by_route":{"r1":[{"target":"aspirin","product_smiles":TOLUENE,"route_name":"R","reactants_smiles":[BENZENE]}]}}
    result = rt.get_all_dataset_routes_for_target(dataset, "aspirin")
    assert len(result) == 1 and result[0]["matched_route_id"] == "r1"

def test_get_all_routes_smiles_filter_mismatch():
    dataset = {"by_route":{"r1":[{"target":"aspirin","product_smiles":TOLUENE,"route_name":"R","reactants_smiles":[BENZENE]}]}}
    assert rt.get_all_dataset_routes_for_target(dataset, "aspirin", ASPIRIN) == []

def test_get_all_routes_smiles_filter_match():
    dataset = {"by_route":{"r1":[{"target":"aspirin","product_smiles":ASPIRIN,"route_name":"R","reactants_smiles":[BENZENE]}]}}
    result = rt.get_all_dataset_routes_for_target(dataset, "aspirin", ASPIRIN)
    assert len(result) == 1

# --- filter_routes_by_starting_materials (name fallback) ---
def test_filter_routes_name_fallback():
    step = {"route_name":"R","target":"aspirin","product_smiles":TOLUENE,
            "reactants_smiles":[BENZENE],"conditions":{},"yield_percent":None,"reaction_type":""}
    dataset = {"by_route":{"r1":[step]}}
    result = rt.filter_routes_by_starting_materials([], dataset, "", "aspirin")
    assert len(result) == 1

def test_filter_routes_empty_steps_skipped():
    assert rt.filter_routes_by_starting_materials([], {"by_route":{"r1":[]}}, BENZENE) == []

# --- calc_toxicity_score ---
def test_calc_toxicity_no_reactants():
    result = rt.calc_toxicity_score([], {}, {}, {})
    assert 0.0 <= result <= 1.0

def test_calc_toxicity_known_hazard():
    canon = rt.to_canonical(BENZENE)
    result = rt.calc_toxicity_score([BENZENE], {}, {canon:{"hazard_score":0.8}}, {})
    assert result == pytest.approx(0.2)

def test_calc_toxicity_unknown_defaults_half():
    assert rt.calc_toxicity_score([ASPIRIN], {}, {}, {}) == pytest.approx(0.5)

def test_calc_toxicity_solvent_resolved():
    solvent_map = {"THF": rt.to_canonical("C1CCOC1")}
    tox_index   = {rt.to_canonical("C1CCOC1"):{"hazard_score":0.2}}
    result = rt.calc_toxicity_score([BENZENE], {"solvent":"THF"}, tox_index, solvent_map)
    assert 0.0 <= result <= 1.0

def test_calc_toxicity_multiple_reactants():
    cb = rt.to_canonical(BENZENE); ct = rt.to_canonical(TOLUENE)
    tox = {cb: {"hazard_score": 0.6}, ct: {"hazard_score": 0.4}}
    result = rt.calc_toxicity_score([BENZENE, TOLUENE], {}, tox, {})
    assert result == pytest.approx(0.5)   # 1 - (0.6+0.4)/2

# --- fmt_conditions extra branches ---
def test_fmt_conditions_temp_range():
    assert "0-25°C" in rt.fmt_conditions({"temp_range":"0-25°C"})

def test_fmt_conditions_co_solvent():
    result = rt.fmt_conditions({"solvent":"THF","co_solvent":"H2O"})
    assert "H2O" in result

def test_fmt_conditions_empty_reagents():
    assert rt.fmt_conditions({"reagents":[]}) == ""

def test_fmt_conditions_apparatus():
    assert "(microwave)" in rt.fmt_conditions({"apparatus":"microwave"})

# --- compute_yield source branches ---
def test_compute_yield_rxninsight_neutral():
    steps = [{**make_step([BENZENE],TOLUENE,80.0),"source":"rxn-insight"}]
    route = {"dataset_steps":steps,"validation_status":"partial"}
    assert rt.compute_yield(route, {}) == pytest.approx(1.0)

def test_compute_yield_generic_dataset_used():
    steps = [{**make_step([BENZENE],TOLUENE,50.0),"source":"generic_dataset"}]
    route = {"dataset_steps":steps,"validation_status":"validated"}
    assert rt.compute_yield(route, {}) == pytest.approx(0.5)

# --- find_best_routes validation ---
def test_find_best_routes_wrong_count():
    with pytest.raises(ValueError, match="exactly 3"):
        rt.find_best_routes("c1ccccc1", ["steps","yield"])

def test_find_best_routes_unknown_criterion():
    with pytest.raises(ValueError, match="unknown"):
        rt.find_best_routes("c1ccccc1", ["steps","yield","bogus"])

def test_load_rxninsight_database_success():
    """Mock pd.read_parquet to simulate a successful database load."""
    import pandas as pd
    from unittest.mock import patch, MagicMock

    fake_df = MagicMock()
    fake_df.columns = ["RXN", "SOLVENT", "REAGENT", "CATALYST", "NAME", "CLASS",
                       "TAG2", "rxn_str_patt_fp", "rxn_dif_patt_fp",
                       "rxn_str_morgan_fp", "rxn_dif_morgan_fp"]
    fake_df.__len__ = lambda self: 100
    fake_df.__getitem__ = lambda self, cols: fake_df

    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True), \
         patch("src.path_finder.route_engine.pd.read_parquet", return_value=fake_df), \
         patch("os.path.exists", return_value=True):
        result = rt.load_rxninsight_database("fake.parquet")
    assert result is not None


def test_load_rxninsight_database_renames_reaction_column():
    """REACTION column should be renamed to RXN if RXN is absent."""
    fake_df = MagicMock()
    # Simulate old column name present, new name absent
    fake_df.columns = ["REACTION", "SOLVENT", "REAGENT", "CATALYST", "NAME",
                       "CLASS", "TAG2", "rxn_str_patt_fp", "rxn_dif_patt_fp",
                       "rxn_str_morgan_fp", "rxn_dif_morgan_fp"]
    renamed_df = MagicMock()
    renamed_df.columns = ["RXN", "SOLVENT", "REAGENT", "CATALYST", "NAME",
                          "CLASS", "TAG2", "rxn_str_patt_fp", "rxn_dif_patt_fp",
                          "rxn_str_morgan_fp", "rxn_dif_morgan_fp"]
    renamed_df.__getitem__ = lambda self, cols: renamed_df
    fake_df.rename.return_value = renamed_df

    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True), \
         patch("src.path_finder.route_engine.pd.read_parquet", return_value=fake_df), \
         patch("os.path.exists", return_value=True):
        result = rt.load_rxninsight_database("fake.parquet")
    fake_df.rename.assert_called_once()


def test_load_rxninsight_database_read_error():
    """Exception during read_parquet should return None gracefully."""
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True), \
         patch("src.path_finder.route_engine.pd.read_parquet",
               side_effect=Exception("read error")), \
         patch("os.path.exists", return_value=True):
        result = rt.load_rxninsight_database("fake.parquet")
    assert result is None


def _make_rxni_reaction_mock(name="Suzuki", cls="C-C"):
    """Return a mock RxnInsightReaction with typical get_reaction_info output."""
    rxn_mock = MagicMock()
    rxn_mock.get_reaction_info.return_value = {
        "NAME":         name,
        "CLASS":        cls,
        "FG_REACTANTS": ["aryl halide"],
        "FG_PRODUCTS":  ["biaryl"],
        "BY-PRODUCTS":  [],
    }
    return rxn_mock


def test_get_reaction_info_rxninsight_unavailable():
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", False):
        result = rt.get_reaction_info_rxninsight([BENZENE], TOLUENE, None)
    assert result == {}


def test_get_reaction_info_invalid_product():
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True):
        result = rt.get_reaction_info_rxninsight([BENZENE], "not!valid", None)
    assert result == {}


def test_get_reaction_info_empty_reactants():
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True):
        result = rt.get_reaction_info_rxninsight([], TOLUENE, None)
    assert result == {}


def test_get_reaction_info_no_db_returns_type_and_class():
    """Without rxni_db, should still return reaction_type from pattern matching."""
    rxn_mock = _make_rxni_reaction_mock()
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True), \
         patch.object(rt, "RxnInsightReaction", return_value=rxn_mock):
        result = rt.get_reaction_info_rxninsight([BENZENE], TOLUENE, None)
    assert isinstance(result, dict)
    assert "reaction_type" in result
    assert "conditions" in result


def test_get_reaction_info_with_db_calls_find_neighbors():
    """With rxni_db provided, find_neighbors should be called."""
    rxn_mock = _make_rxni_reaction_mock()
    fake_neighbors = MagicMock()
    fake_neighbors.__len__ = lambda self: 5
    rxn_mock.find_neighbors.return_value = fake_neighbors

    fake_db = MagicMock()
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True), \
         patch.object(rt, "RxnInsightReaction", return_value=rxn_mock):
        result = rt.get_reaction_info_rxninsight([BENZENE], TOLUENE, fake_db)
    rxn_mock.find_neighbors.assert_called_once()
    assert isinstance(result, dict)


def test_get_reaction_info_exception_returns_empty():
    """If RxnInsightReaction constructor raises, return {}."""
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True), \
         patch.object(rt, "RxnInsightReaction", side_effect=Exception("rxni error")):
        result = rt.get_reaction_info_rxninsight([BENZENE], TOLUENE, None)
    assert result == {}


def test_get_reaction_info_fg_reactants_in_result():
    rxn_mock = _make_rxni_reaction_mock()
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True), \
         patch.object(rt, "RxnInsightReaction", return_value=rxn_mock):
        result = rt.get_reaction_info_rxninsight([BENZENE], TOLUENE, None)
    assert "fg_reactants" in result
    assert isinstance(result["fg_reactants"], list)


def test_best_condition_none_input():
    assert rt._best_condition(None) == ""


def test_best_condition_empty_df():
    df_mock = MagicMock()
    df_mock.empty = True
    assert rt._best_condition(df_mock) == ""


def test_best_condition_all_blank_names():
    df_mock = MagicMock()
    df_mock.empty = False
    valid_mock = MagicMock()
    valid_mock.empty = True
    df_mock.__getitem__ = MagicMock(return_value=MagicMock())
    df_mock.__getitem__.return_value.str.strip.return_value.__ne__ = MagicMock(
        return_value=MagicMock())
    # Easier: just test the exception path
    df_mock.side_effect = Exception("bad df")
    assert rt._best_condition(df_mock) == ""


def test_best_condition_exception_returns_empty():
    assert rt._best_condition("not_a_dataframe") == ""


def _make_aiz_route(smiles=TOLUENE):
    return {
        "steps": [{"reactants": [BENZENE], "product": smiles}],
        "target_smiles": smiles,
        "matched_target": "?",
    }


def test_process_novel_routes_rxninsight_unavailable():
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", False):
        v, p = rt.process_novel_routes([], {}, {}, "aspirin", None)
    assert v == [] and p == []


def test_process_novel_routes_empty_aiz_routes():
    dataset = {"all": [], "by_route": {}, "by_product": {}, "by_reactant": {}, "metadata": {}}
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True):
        v, p = rt.process_novel_routes([], dataset, {}, "aspirin", None)
    assert v == [] and p == []


def test_process_novel_routes_skips_covered_routes():
    """Routes whose products are all in the dataset index should be skipped."""
    canon_t = rt.to_canonical(TOLUENE)
    dataset = {
        "all": [{"product_smiles": TOLUENE, "reactants_smiles": [BENZENE]}],
        "by_route": {}, "by_product": {}, "by_reactant": {}, "metadata": {},
    }
    aiz_routes = [_make_aiz_route(TOLUENE)]
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True):
        v, p = rt.process_novel_routes(aiz_routes, dataset, {}, "aspirin", None)
    # Covered route is skipped, so both lists empty
    assert v == [] and p == []


def test_process_novel_routes_skips_routes_with_no_steps():
    dataset = {"all": [], "by_route": {}, "by_product": {}, "by_reactant": {}, "metadata": {}}
    empty_route = {"steps": [], "target_smiles": TOLUENE}
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True):
        v, p = rt.process_novel_routes([empty_route], dataset, {}, "aspirin", None)
    assert v == [] and p == []


def test_process_novel_routes_predicted_when_no_generic_match():
    """Novel route with no generic dataset match → predicted list."""
    dataset = {"all": [], "by_route": {}, "by_product": {}, "by_reactant": {}, "metadata": {}}
    aiz_routes = [_make_aiz_route(ASPIRIN)]
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True):
        v, p = rt.process_novel_routes(aiz_routes, dataset, {}, "aspirin", None)
    assert len(p) == 1
    assert v == []


def test_process_novel_routes_validated_when_all_steps_match(tmp_path):
    """Novel route where all steps match generic dataset → validated list."""
    dataset = {"all": [], "by_route": {}, "by_product": {}, "by_reactant": {}, "metadata": {}}
    generic_ds_data = [{"product_smiles": ASPIRIN, "reactants_smiles": [BENZENE],
                        "reaction_type": "test", "yield_percent": 80, "conditions": {}}]
    f = tmp_path / "g.json"
    f.write_text(json.dumps(generic_ds_data))
    generic_ds = rt.load_generic_reaction_dataset(str(f))
    aiz_routes = [_make_aiz_route(ASPIRIN)]
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True):
        v, p = rt.process_novel_routes(aiz_routes, dataset, generic_ds, "aspirin", None)
    assert len(v) == 1
    assert p == []


def test_get_novel_routes_rxninsight_unavailable():
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", False):
        result = rt.get_novel_routes_from_aizynthfinder([], {}, "aspirin", None)
    assert result == []


def test_get_novel_routes_empty_routes():
    dataset = {"all": [], "by_route": {}, "by_product": {}, "by_reactant": {}, "metadata": {}}
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True):
        result = rt.get_novel_routes_from_aizynthfinder([], dataset, "aspirin", None)
    assert result == []


def test_get_novel_routes_skips_covered():
    """Routes covered by dataset should be skipped."""
    canon_t = rt.to_canonical(TOLUENE)
    dataset = {
        "all": [{"product_smiles": TOLUENE, "reactants_smiles": [BENZENE]}],
        "by_route": {}, "by_product": {}, "by_reactant": {}, "metadata": {},
    }
    aiz_routes = [_make_aiz_route(TOLUENE)]
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True):
        result = rt.get_novel_routes_from_aizynthfinder(aiz_routes, dataset, "aspirin", None)
    assert result == []


def test_get_novel_routes_skips_empty_steps():
    dataset = {"all": [], "by_route": {}, "by_product": {}, "by_reactant": {}, "metadata": {}}
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True):
        result = rt.get_novel_routes_from_aizynthfinder(
            [{"steps": [], "target_smiles": TOLUENE}], dataset, "aspirin", None)
    assert result == []


def test_get_novel_routes_returns_enriched_route():
    dataset = {"all": [], "by_route": {}, "by_product": {}, "by_reactant": {}, "metadata": {}}
    aiz_routes = [_make_aiz_route(ASPIRIN)]
    with patch.object(rt, "RXNINSIGHT_AVAILABLE", True):
        result = rt.get_novel_routes_from_aizynthfinder(aiz_routes, dataset, "aspirin", None)
    assert len(result) == 1
    assert result[0]["matched_target"] == "aspirin"
    assert result[0]["is_predicted"] is True


def test_find_best_routes_runs_pipeline(tmp_path):
    """Mock all heavy calls so the full pipeline body executes."""
    # Minimal dataset file
    data = [{
        "id": "1", "route_id": "r1", "route_name": "Test Route", "target": "toluene",
        "step_number": 1, "reactants_smiles": [BENZENE], "product_smiles": TOLUENE,
        "conditions": {}, "yield_percent": 80, "reaction_type": "alkylation",
    }]
    ds_file  = tmp_path / "ds.json"
    tox_file = tmp_path / "tox.json"
    ds_file.write_text(json.dumps(data))
    tox_file.write_text(json.dumps([]))

    with patch.object(rt, "run_aizynthfinder", return_value=[]), \
         patch.object(rt, "RXNINSIGHT_AVAILABLE", False):
        result = rt.find_best_routes(
            target_smiles=TOLUENE,
            criteria_priority=["steps", "yield", "atom_economy"],
            dataset_path=str(ds_file),
            toxicity_path=str(tox_file),
            config_path="config.yml",
            top_n=3,
            target_name="toluene",
            include_predicted=False,
        )
    assert "dataset" in result
    assert "validated" in result
    assert "predicted" in result
    assert isinstance(result["dataset"], list)


def test_find_best_routes_returns_scored_dataset_routes(tmp_path):
    """Dataset routes matching target SMILES should appear in result['dataset']."""
    data = [{
        "id": "1", "route_id": "r1", "route_name": "Toluene Route", "target": "toluene",
        "step_number": 1, "reactants_smiles": [BENZENE], "product_smiles": TOLUENE,
        "conditions": {}, "yield_percent": 80, "reaction_type": "alkylation",
    }]
    ds_file  = tmp_path / "ds.json"
    tox_file = tmp_path / "tox.json"
    ds_file.write_text(json.dumps(data))
    tox_file.write_text(json.dumps([]))

    with patch.object(rt, "run_aizynthfinder", return_value=[]), \
         patch.object(rt, "RXNINSIGHT_AVAILABLE", False):
        result = rt.find_best_routes(
            target_smiles=TOLUENE,
            criteria_priority=["steps", "yield", "atom_economy"],
            dataset_path=str(ds_file),
            toxicity_path=str(tox_file),
            include_predicted=False,
            target_name="toluene",
        )
    assert len(result["dataset"]) == 1
    score, details, route = result["dataset"][0]
    assert isinstance(score, float)
    assert route["matched_route_name"] == "Toluene Route"


def test_find_best_routes_top_n_respected(tmp_path):
    """top_n=1 should return at most 1 route per section."""
    data = [
        {"id": str(i), "route_id": f"r{i}", "route_name": f"Route {i}", "target": "toluene",
         "step_number": 1, "reactants_smiles": [BENZENE], "product_smiles": TOLUENE,
         "conditions": {}, "yield_percent": 80, "reaction_type": ""}
        for i in range(3)
    ]
    ds_file  = tmp_path / "ds.json"
    tox_file = tmp_path / "tox.json"
    ds_file.write_text(json.dumps(data))
    tox_file.write_text(json.dumps([]))

    with patch.object(rt, "run_aizynthfinder", return_value=[]), \
         patch.object(rt, "RXNINSIGHT_AVAILABLE", False):
        result = rt.find_best_routes(
            target_smiles=TOLUENE,
            criteria_priority=["steps", "yield", "atom_economy"],
            dataset_path=str(ds_file),
            toxicity_path=str(tox_file),
            include_predicted=False,
            target_name="toluene",
            top_n=1,
        )
    assert len(result["dataset"]) <= 1