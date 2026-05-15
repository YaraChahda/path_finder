"""
test_route_engine.py — Tests for route_engine.py.

AiZynthFinder is mocked by conftest.py so no model files are required.
RDKit is assumed to be installed (it is a hard project dependency).

Tests are grouped by the logical section of the module they cover:
  - Data loading helpers (using temporary JSON files)
  - SMILES utilities
  - Yield and step metric helpers
  - Per-criterion scoring functions
  - Weighted ranking
  - Dataset route filtering and matching
"""

import json
import os
import tempfile
import pytest

import src.path_finder.route_engine as rt


# ---------------------------------------------------------------------------
# Fixtures — minimal synthetic data
# ---------------------------------------------------------------------------

BENZENE   = "c1ccccc1"
ETHANOL   = "CCO"
TOLUENE   = "Cc1ccccc1"
CAFFEINE  = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"

STEP_WITH_YIELD = {
    "step_number":      1,
    "reactants_smiles": [BENZENE, ETHANOL],
    "product_smiles":   TOLUENE,
    "yield_percent":    80.0,
    "reaction_type":    "Alkylation",
    "conditions":       {"temperature_C": 100, "solvent": "THF", "reagents": []},
    "source":           "dataset",
}

STEP_NO_YIELD = {
    "step_number":      2,
    "reactants_smiles": [TOLUENE],
    "product_smiles":   CAFFEINE,
    "yield_percent":    None,
    "reaction_type":    "Condensation",
    "conditions":       {},
    "source":           "dataset",
}

SIMPLE_ROUTE = {
    "dataset_steps":      [STEP_WITH_YIELD, STEP_NO_YIELD],
    "validation_status":  "dataset",
    "is_predicted":       False,
    "matched_route_name": "Test route",
    "matched_route_id":   "R001",
    "matched_target":     "Caffeine",
}

PREDICTED_ROUTE = {
    "dataset_steps":      [STEP_WITH_YIELD, STEP_NO_YIELD],
    "validation_status":  "predicted",
    "is_predicted":       True,
    "matched_route_name": "Predicted route #1",
    "matched_route_id":   "predicted_01",
    "matched_target":     "?",
}


@pytest.fixture
def minimal_dataset_json(tmp_path):
    """Write a minimal reaction_dataset.json to a temp file."""
    reactions = [
        {
            "id": "rxn-001",
            "route_id": "R001",
            "route_name": "Aspirin synthesis",
            "target": "Aspirin",
            "step_number": 1,
            "reactants_smiles": ["OC(=O)c1ccccc1O", "CC(=O)O"],
            "product_smiles":   "CC(=O)Oc1ccccc1C(=O)O",
            "conditions":       {"temperature_C": 85, "solvent": "AcOH"},
            "yield_percent":    75.0,
            "reaction_type":    "Esterification",
        },
        {
            "id": "rxn-002",
            "route_id": "R001",
            "route_name": "Aspirin synthesis",
            "target": "Aspirin",
            "step_number": 2,
            "reactants_smiles": ["CC(=O)Oc1ccccc1C(=O)Cl"],
            "product_smiles":   "CC(=O)Oc1ccccc1C(=O)O",
            "conditions":       {},
            "yield_percent":    None,
            "reaction_type":    "Hydrolysis",
        },
    ]
    p = tmp_path / "reaction_dataset.json"
    p.write_text(json.dumps(reactions), encoding="utf-8")
    return str(p)


@pytest.fixture
def minimal_toxicity_json(tmp_path):
    compounds = [
        {"smiles": BENZENE, "hazard_score": 0.7},
        {"smiles": ETHANOL, "hazard_score": 0.2},
    ]
    p = tmp_path / "toxicity_dataset.json"
    p.write_text(json.dumps(compounds), encoding="utf-8")
    return str(p)


@pytest.fixture
def minimal_generic_json(tmp_path):
    reactions = [
        {
            "id": "gen-001",
            "route_id": "gen",
            "route_name": "generic",
            "target": "any",
            "step_number": 1,
            "reactants_smiles": [BENZENE, ETHANOL],
            "product_smiles":   TOLUENE,
            "conditions":       {"temperature_C": 80},
            "yield_percent":    65.0,
            "reaction_type":    "Alkylation",
        }
    ]
    p = tmp_path / "generic_reactions.json"
    p.write_text(json.dumps(reactions), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class TestLoadReactionDataset:
    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            rt.load_reaction_dataset("/nonexistent/path/dataset.json")

    def test_raises_value_error_for_bad_format(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"not_reactions": True}))
        with pytest.raises(ValueError):
            rt.load_reaction_dataset(str(bad))

    def test_loads_list_format(self, minimal_dataset_json):
        ds = rt.load_reaction_dataset(minimal_dataset_json)
        assert "all" in ds
        assert "by_route" in ds
        assert "by_product" in ds
        assert "by_reactant" in ds
        assert "metadata" in ds

    def test_all_contains_all_reactions(self, minimal_dataset_json):
        ds = rt.load_reaction_dataset(minimal_dataset_json)
        assert len(ds["all"]) == 2

    def test_by_route_groups_correctly(self, minimal_dataset_json):
        ds = rt.load_reaction_dataset(minimal_dataset_json)
        assert "R001" in ds["by_route"]
        assert len(ds["by_route"]["R001"]) == 2

    def test_steps_sorted_by_step_number(self, minimal_dataset_json):
        ds = rt.load_reaction_dataset(minimal_dataset_json)
        steps = ds["by_route"]["R001"]
        nums = [s.get("step_number", 0) for s in steps]
        assert nums == sorted(nums)

    def test_loads_dict_format_with_reactions_key(self, tmp_path):
        data = {"reactions": [{"id": "x", "route_id": "r", "route_name": "n",
                                "target": "T", "step_number": 1,
                                "reactants_smiles": [BENZENE],
                                "product_smiles": TOLUENE,
                                "conditions": {}, "yield_percent": 50,
                                "reaction_type": ""}],
                "_metadata": {"target_smiles": {}}}
        p = tmp_path / "ds.json"
        p.write_text(json.dumps(data))
        ds = rt.load_reaction_dataset(str(p))
        assert len(ds["all"]) == 1

    def test_product_smiles_list_is_joined(self, tmp_path):
        reactions = [{"id": "x", "route_id": "r", "route_name": "n",
                      "target": "T", "step_number": 1,
                      "reactants_smiles": [BENZENE],
                      "product_smiles": [TOLUENE, ETHANOL],
                      "conditions": {}, "yield_percent": None, "reaction_type": ""}]
        p = tmp_path / "ds.json"
        p.write_text(json.dumps(reactions))
        ds = rt.load_reaction_dataset(str(p))
        assert "." in ds["all"][0]["product_smiles"]

    def test_reactants_string_is_wrapped_in_list(self, tmp_path):
        reactions = [{"id": "x", "route_id": "r", "route_name": "n",
                      "target": "T", "step_number": 1,
                      "reactants_smiles": BENZENE,  # string, not list
                      "product_smiles": TOLUENE,
                      "conditions": {}, "yield_percent": None, "reaction_type": ""}]
        p = tmp_path / "ds.json"
        p.write_text(json.dumps(reactions))
        ds = rt.load_reaction_dataset(str(p))
        assert isinstance(ds["all"][0]["reactants_smiles"], list)


class TestLoadToxicityDataset:
    def test_missing_file_returns_empty_dict(self):
        result = rt.load_toxicity_dataset("/nonexistent/path.json")
        assert result == {}

    def test_loads_list_format(self, minimal_toxicity_json):
        idx = rt.load_toxicity_dataset(minimal_toxicity_json)
        assert len(idx) == 2

    def test_index_keyed_by_canonical_smiles(self, minimal_toxicity_json):
        idx = rt.load_toxicity_dataset(minimal_toxicity_json)
        canon_benzene = rt.to_canonical(BENZENE)
        assert canon_benzene in idx

    def test_loads_dict_format_with_compounds_key(self, tmp_path):
        data = {"compounds": [{"smiles": ETHANOL, "hazard_score": 0.3}]}
        p = tmp_path / "tox.json"
        p.write_text(json.dumps(data))
        idx = rt.load_toxicity_dataset(str(p))
        assert len(idx) == 1


class TestLoadGenericReactionDataset:
    def test_missing_path_returns_empty_dict(self):
        result = rt.load_generic_reaction_dataset("/nonexistent/path.json")
        assert result == {}

    def test_empty_string_returns_empty_dict(self):
        result = rt.load_generic_reaction_dataset("")
        assert result == {}

    def test_loads_correctly(self, minimal_generic_json):
        ds = rt.load_generic_reaction_dataset(minimal_generic_json)
        assert "by_product" in ds
        assert "by_reaction_key" in ds
        assert "all" in ds

    def test_by_product_indexed(self, minimal_generic_json):
        ds = rt.load_generic_reaction_dataset(minimal_generic_json)
        canon_tol = rt.to_canonical(TOLUENE)
        assert canon_tol in ds["by_product"]


# ---------------------------------------------------------------------------
# SMILES utilities
# ---------------------------------------------------------------------------

class TestToCanonical:
    def test_empty_string_returns_empty(self):
        assert rt.to_canonical("") == ""

    def test_none_type_returns_empty(self):
        assert rt.to_canonical(None) == ""

    def test_valid_smiles_returns_canonical(self):
        result = rt.to_canonical("C1=CC=CC=C1")
        assert result == rt.to_canonical(BENZENE)

    def test_already_canonical_unchanged(self):
        canon = rt.to_canonical(BENZENE)
        assert rt.to_canonical(canon) == canon

    def test_list_input_joined_with_dot(self):
        result = rt.to_canonical([BENZENE, ETHANOL])
        assert "." in result

    def test_invalid_smiles_returned_as_is(self):
        bad = "not!a!smiles"
        result = rt.to_canonical(bad)
        assert result == bad

    def test_toluene_canonical(self):
        # Different representations of toluene should canonicalise identically
        r1 = rt.to_canonical("c1ccc(C)cc1")
        r2 = rt.to_canonical("Cc1ccccc1")
        assert r1 == r2


class TestSafeMol:
    def test_empty_returns_none(self):
        assert rt.safe_mol("") is None

    def test_invalid_returns_none(self):
        assert rt.safe_mol("not!valid") is None

    def test_valid_returns_mol(self):
        mol = rt.safe_mol(BENZENE)
        assert mol is not None

    def test_returned_mol_has_atoms(self):
        mol = rt.safe_mol(BENZENE)
        assert mol.GetNumAtoms() > 0


class TestValidateSmilesForAizynthfinder:
    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            rt.validate_smiles_for_aizynthfinder("")

    def test_invalid_smiles_raises_value_error(self):
        with pytest.raises(ValueError, match="invalid"):
            rt.validate_smiles_for_aizynthfinder("not!valid")

    def test_valid_returns_canonical_string(self):
        result = rt.validate_smiles_for_aizynthfinder(BENZENE)
        assert isinstance(result, str)
        assert result == rt.to_canonical(BENZENE)


class TestBuildDatasetSmilesIndex:
    def test_returns_set(self, minimal_dataset_json):
        ds = rt.load_reaction_dataset(minimal_dataset_json)
        index = rt.build_dataset_smiles_index(ds)
        assert isinstance(index, set)

    def test_contains_product_smiles(self, minimal_dataset_json):
        ds = rt.load_reaction_dataset(minimal_dataset_json)
        index = rt.build_dataset_smiles_index(ds)
        canon = rt.to_canonical("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        assert canon in index

    def test_contains_reactant_smiles(self, minimal_dataset_json):
        ds = rt.load_reaction_dataset(minimal_dataset_json)
        index = rt.build_dataset_smiles_index(ds)
        canon = rt.to_canonical("OC(=O)c1ccccc1O")
        assert canon in index


# ---------------------------------------------------------------------------
# Yield and step metric helpers
# ---------------------------------------------------------------------------

class TestBottleneckYield:
    def test_empty_steps_returns_none(self):
        assert rt.bottleneck_yield([]) is None

    def test_all_none_yields_returns_none(self):
        steps = [{"yield_percent": None}, {"yield_percent": None}]
        assert rt.bottleneck_yield(steps) is None

    def test_returns_minimum_yield(self):
        steps = [{"yield_percent": 80}, {"yield_percent": 60}, {"yield_percent": 95}]
        assert rt.bottleneck_yield(steps) == 60

    def test_ignores_none_yields(self):
        steps = [{"yield_percent": 80}, {"yield_percent": None}, {"yield_percent": 50}]
        assert rt.bottleneck_yield(steps) == 50

    def test_single_step(self):
        steps = [{"yield_percent": 73}]
        assert rt.bottleneck_yield(steps) == 73


class TestAverageYield:
    def test_empty_returns_none(self):
        assert rt.average_yield([]) is None

    def test_all_none_returns_none(self):
        assert rt.average_yield([{"yield_percent": None}]) is None

    def test_correct_average(self):
        steps = [{"yield_percent": 80}, {"yield_percent": 60}]
        assert rt.average_yield(steps) == pytest.approx(70.0)

    def test_ignores_none_values(self):
        steps = [{"yield_percent": 80}, {"yield_percent": None}, {"yield_percent": 60}]
        assert rt.average_yield(steps) == pytest.approx(70.0)


class TestCumulativeYield:
    def test_empty_returns_one(self):
        assert rt.cumulative_yield([]) == pytest.approx(1.0)

    def test_missing_yields_treated_as_100_percent(self):
        steps = [{"yield_percent": None}, {"yield_percent": None}]
        assert rt.cumulative_yield(steps) == pytest.approx(1.0)

    def test_product_of_yields(self):
        steps = [{"yield_percent": 80}, {"yield_percent": 50}]
        # 0.8 * 0.5 = 0.4
        assert rt.cumulative_yield(steps) == pytest.approx(0.4)

    def test_single_step(self):
        assert rt.cumulative_yield([{"yield_percent": 75}]) == pytest.approx(0.75)

    def test_mixed_none_and_values(self):
        steps = [{"yield_percent": 80}, {"yield_percent": None}, {"yield_percent": 50}]
        assert rt.cumulative_yield(steps) == pytest.approx(0.4)


class TestGetSubstancesList:
    def test_empty_steps(self):
        result = rt.get_substances_list([])
        assert result["to_buy"] == []
        assert result["to_prepare"] == []
        assert result["solvents"] == []
        assert result["reagents"] == []

    def test_to_buy_are_starting_materials(self):
        steps = [
            {"reactants_smiles": [BENZENE, ETHANOL],
             "product_smiles":   TOLUENE,
             "conditions":       {}},
        ]
        result = rt.get_substances_list(steps)
        canon_b = rt.to_canonical(BENZENE)
        canon_e = rt.to_canonical(ETHANOL)
        assert canon_b in result["to_buy"] or canon_e in result["to_buy"]

    def test_to_prepare_contains_product(self):
        steps = [{"reactants_smiles": [BENZENE], "product_smiles": TOLUENE,
                  "conditions": {}}]
        result = rt.get_substances_list(steps)
        canon_t = rt.to_canonical(TOLUENE)
        assert canon_t in result["to_prepare"]

    def test_solvents_collected(self):
        steps = [{"reactants_smiles": [], "product_smiles": TOLUENE,
                  "conditions": {"solvent": "THF"}}]
        result = rt.get_substances_list(steps)
        assert "THF" in result["solvents"]

    def test_reagents_collected(self):
        steps = [{"reactants_smiles": [], "product_smiles": TOLUENE,
                  "conditions": {"reagents": ["NaH", "KBr"]}}]
        result = rt.get_substances_list(steps)
        assert "NaH" in result["reagents"]
        assert "KBr" in result["reagents"]


class TestFmtConditions:
    def test_empty_dict_returns_empty_string(self):
        assert rt.fmt_conditions({}) == ""

    def test_none_returns_empty_string(self):
        assert rt.fmt_conditions(None) == ""

    def test_temperature_included(self):
        result = rt.fmt_conditions({"temperature_C": 100})
        assert "100" in result
        assert "°C" in result

    def test_solvent_included(self):
        result = rt.fmt_conditions({"solvent": "THF"})
        assert "THF" in result

    def test_reagents_list_included(self):
        result = rt.fmt_conditions({"reagents": ["NaH", "DMSO"]})
        assert "NaH" in result
        assert "DMSO" in result

    def test_apparatus_included(self):
        result = rt.fmt_conditions({"apparatus": "microwave"})
        assert "microwave" in result

    def test_temp_range_used_when_no_temperature_c(self):
        result = rt.fmt_conditions({"temp_range": "0–25 °C"})
        assert "0–25" in result

    def test_separator_between_fields(self):
        result = rt.fmt_conditions({"temperature_C": 80, "solvent": "DCM"})
        assert "·" in result


# ---------------------------------------------------------------------------
# Per-criterion scoring functions
# ---------------------------------------------------------------------------

class TestCalcAtomEconomy:
    def test_invalid_product_returns_zero(self):
        assert rt.calc_atom_economy([BENZENE], "not!valid") == pytest.approx(0.0)

    def test_same_reactant_and_product_returns_one(self):
        result = rt.calc_atom_economy([BENZENE], BENZENE)
        assert result == pytest.approx(1.0)

    def test_empty_reactants_returns_zero(self):
        result = rt.calc_atom_economy([], BENZENE)
        assert result == pytest.approx(0.0)

    def test_score_is_between_zero_and_one(self):
        result = rt.calc_atom_economy([BENZENE, ETHANOL], TOLUENE)
        assert 0.0 <= result <= 1.0

    def test_capped_at_one(self):
        # Product heavier than one reactant alone → still capped at 1
        result = rt.calc_atom_economy([ETHANOL], CAFFEINE)
        assert result <= 1.0


class TestCalcEFactor:
    def test_invalid_product_returns_half(self):
        result = rt.calc_e_factor([BENZENE], "not!valid", 1.0)
        assert result == pytest.approx(0.5)

    def test_perfect_yield_high_score(self):
        result = rt.calc_e_factor([BENZENE], BENZENE, 1.0)
        assert result == pytest.approx(1.0)

    def test_result_in_range(self):
        result = rt.calc_e_factor([BENZENE, ETHANOL], TOLUENE, 0.75)
        assert 0.0 < result <= 1.0

    def test_low_yield_lower_score_than_high_yield(self):
        high = rt.calc_e_factor([BENZENE, ETHANOL], TOLUENE, 0.9)
        low  = rt.calc_e_factor([BENZENE, ETHANOL], TOLUENE, 0.1)
        assert high > low


class TestBuildSolventMap:
    def test_returns_dict(self):
        result = rt.build_solvent_map({})
        assert isinstance(result, dict)

    def test_common_solvents_present(self):
        result = rt.build_solvent_map({})
        for key in ("THF", "DCM", "MeOH", "EtOH", "DMF", "toluene"):
            assert key in result, f"Expected {key!r} in solvent map"

    def test_tox_index_keys_added(self):
        tox_index = {rt.to_canonical(BENZENE): {"hazard_score": 0.7}}
        result = rt.build_solvent_map(tox_index)
        assert rt.to_canonical(BENZENE) in result


class TestCalcToxicityScore:
    def test_empty_reactants_no_conditions_returns_half(self):
        result = rt.calc_toxicity_score([], {}, {}, {})
        assert result == pytest.approx(0.5)

    def test_unknown_compounds_default_to_half_hazard(self):
        result = rt.calc_toxicity_score([BENZENE], {}, {}, {})
        # Hazard is 0.5 → safety score is 1 - 0.5 = 0.5
        assert result == pytest.approx(0.5)

    def test_safe_compound_gives_high_score(self):
        canon = rt.to_canonical(ETHANOL)
        tox_index = {canon: {"hazard_score": 0.1}}
        result = rt.calc_toxicity_score([ETHANOL], {}, tox_index, {})
        assert result > 0.8

    def test_hazardous_compound_gives_low_score(self):
        canon = rt.to_canonical(BENZENE)
        tox_index = {canon: {"hazard_score": 0.9}}
        result = rt.calc_toxicity_score([BENZENE], {}, tox_index, {})
        assert result < 0.2

    def test_score_between_zero_and_one(self):
        canon = rt.to_canonical(BENZENE)
        tox_index = {canon: {"hazard_score": 0.5}}
        result = rt.calc_toxicity_score([BENZENE, ETHANOL], {}, tox_index, {})
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Route-level scoring functions
# ---------------------------------------------------------------------------

class TestComputeSteps:
    def test_single_step_gives_one(self):
        route = {"dataset_steps": [STEP_WITH_YIELD]}
        assert rt.compute_steps(route, {}) == pytest.approx(1.0)

    def test_two_steps_give_half(self):
        route = {"dataset_steps": [STEP_WITH_YIELD, STEP_NO_YIELD]}
        assert rt.compute_steps(route, {}) == pytest.approx(0.5)

    def test_empty_steps_gives_one(self):
        route = {"dataset_steps": []}
        assert rt.compute_steps(route, {}) == pytest.approx(1.0)

    def test_score_decreases_with_more_steps(self):
        r3 = {"dataset_steps": [STEP_WITH_YIELD] * 3}
        r5 = {"dataset_steps": [STEP_WITH_YIELD] * 5}
        assert rt.compute_steps(r3, {}) > rt.compute_steps(r5, {})


class TestComputeYield:
    def test_predicted_route_returns_one(self):
        result = rt.compute_yield(PREDICTED_ROUTE, {})
        assert result == pytest.approx(1.0)

    def test_empty_steps_returns_zero(self):
        route = {"dataset_steps": [], "validation_status": "dataset"}
        assert rt.compute_yield(route, {}) == pytest.approx(0.0)

    def test_dataset_route_uses_reported_yields(self):
        route = {
            "dataset_steps":     [STEP_WITH_YIELD],
            "validation_status": "dataset",
        }
        result = rt.compute_yield(route, {})
        assert result == pytest.approx(0.8)

    def test_missing_yield_treated_as_neutral(self):
        route = {
            "dataset_steps":     [STEP_NO_YIELD],
            "validation_status": "dataset",
        }
        result = rt.compute_yield(route, {})
        assert result == pytest.approx(1.0)


class TestComputeAtomEconomy:
    def test_empty_steps_returns_zero(self):
        route = {"dataset_steps": []}
        assert rt.compute_atom_economy(route, {}) == pytest.approx(0.0)

    def test_result_in_range(self):
        result = rt.compute_atom_economy(SIMPLE_ROUTE, {})
        assert 0.0 <= result <= 1.0


class TestComputeEFactor:
    def test_empty_steps_returns_zero(self):
        route = {"dataset_steps": []}
        assert rt.compute_e_factor(route, {}) == pytest.approx(0.0)

    def test_result_in_range(self):
        result = rt.compute_e_factor(SIMPLE_ROUTE, {})
        assert 0.0 < result <= 1.0


class TestComputeToxicity:
    def test_empty_steps_returns_half(self):
        route = {"dataset_steps": []}
        assert rt.compute_toxicity(route, {}) == pytest.approx(0.5)

    def test_result_in_range(self):
        result = rt.compute_toxicity(SIMPLE_ROUTE, {})
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Weighted ranking
# ---------------------------------------------------------------------------

class TestComputeWeights:
    def test_weights_sum_to_one(self):
        criteria = ["steps", "yield", "atom_economy"]
        weights = rt.compute_weights(criteria)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_first_criterion_has_highest_weight(self):
        criteria = ["steps", "yield", "atom_economy"]
        weights = rt.compute_weights(criteria)
        assert weights["steps"] > weights["yield"] > weights["atom_economy"]

    def test_returns_dict_with_all_keys(self):
        criteria = ["steps", "yield", "atom_economy"]
        weights = rt.compute_weights(criteria)
        for c in criteria:
            assert c in weights

    def test_inverse_square_scheme(self):
        criteria = ["steps", "yield", "atom_economy"]
        weights = rt.compute_weights(criteria)
        # w1 / w2 should ≈ 4  (1/1² vs 1/2²)
        ratio = weights["steps"] / weights["yield"]
        assert ratio == pytest.approx(4.0, rel=1e-6)


class TestComputeAllScores:
    def test_returns_all_criteria(self):
        scores = rt.compute_all_scores(SIMPLE_ROUTE, {})
        for c in rt.CRITERIA_REGISTRY:
            assert c in scores

    def test_all_scores_in_range(self):
        scores = rt.compute_all_scores(SIMPLE_ROUTE, {})
        for c, v in scores.items():
            assert 0.0 <= v <= 1.0, f"Score for {c!r} = {v} out of range"

    def test_returns_floats(self):
        scores = rt.compute_all_scores(SIMPLE_ROUTE, {})
        for v in scores.values():
            assert isinstance(v, float)


class TestRankWeighted:
    def test_returns_list_of_tuples(self):
        routes   = [SIMPLE_ROUTE, PREDICTED_ROUTE]
        criteria = ["steps", "yield", "atom_economy"]
        result   = rt.rank_weighted(routes, criteria, {})
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 3  # (score, details, route)

    def test_sorted_descending(self):
        routes   = [SIMPLE_ROUTE, PREDICTED_ROUTE]
        criteria = ["steps", "yield", "atom_economy"]
        result   = rt.rank_weighted(routes, criteria, {})
        scores   = [r[0] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_details_contain_criterion_keys(self):
        routes   = [SIMPLE_ROUTE]
        criteria = ["steps", "yield", "atom_economy"]
        result   = rt.rank_weighted(routes, criteria, {})
        details  = result[0][1]
        for c in criteria:
            assert c in details

    def test_predicted_route_yield_excluded(self):
        routes   = [PREDICTED_ROUTE]
        criteria = ["steps", "yield", "atom_economy"]
        result   = rt.rank_weighted(routes, criteria, {})
        details  = result[0][1]
        assert details["yield"]["excluded"] is True
        assert details["yield"]["raw"] is None

    def test_dataset_route_yield_not_excluded(self):
        routes   = [SIMPLE_ROUTE]
        criteria = ["steps", "yield", "atom_economy"]
        result   = rt.rank_weighted(routes, criteria, {})
        details  = result[0][1]
        assert details["yield"]["excluded"] is False

    def test_empty_routes_returns_empty_list(self):
        result = rt.rank_weighted([], ["steps", "yield", "atom_economy"], {})
        assert result == []

    def test_total_score_between_zero_and_one(self):
        routes   = [SIMPLE_ROUTE, PREDICTED_ROUTE]
        criteria = ["steps", "yield", "atom_economy"]
        result   = rt.rank_weighted(routes, criteria, {})
        for score, _, _ in result:
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Dataset route filtering and matching
# ---------------------------------------------------------------------------

class TestIsRouteCoveredByDataset:
    def test_empty_steps_not_covered(self):
        route = {"steps": []}
        result = rt.is_route_covered_by_dataset(route, set())
        assert result is False

    def test_fully_covered_route(self):
        canon_t = rt.to_canonical(TOLUENE)
        index   = {canon_t}
        route   = {"steps": [{"product": TOLUENE}]}
        assert rt.is_route_covered_by_dataset(route, index, threshold=0.4) is True

    def test_not_covered_when_below_threshold(self):
        index = {rt.to_canonical(BENZENE)}
        route = {"steps": [{"product": TOLUENE}, {"product": ETHANOL}]}
        assert rt.is_route_covered_by_dataset(route, index, threshold=0.9) is False


class TestMatchStepInGenericDataset:
    def test_empty_generic_returns_none(self):
        result = rt.match_step_in_generic_dataset([BENZENE], TOLUENE, {})
        assert result is None

    def test_exact_match_found(self, minimal_generic_json):
        ds = rt.load_generic_reaction_dataset(minimal_generic_json)
        result = rt.match_step_in_generic_dataset([BENZENE, ETHANOL], TOLUENE, ds)
        assert result is not None

    def test_product_only_fallback(self, minimal_generic_json):
        ds = rt.load_generic_reaction_dataset(minimal_generic_json)
        # Use wrong reactants — should still find via product fallback
        result = rt.match_step_in_generic_dataset([CAFFEINE], TOLUENE, ds)
        assert result is not None  # product-only fallback

    def test_no_match_returns_none(self, minimal_generic_json):
        ds = rt.load_generic_reaction_dataset(minimal_generic_json)
        result = rt.match_step_in_generic_dataset([CAFFEINE], CAFFEINE, ds)
        assert result is None


class TestFilterRoutesByStartingMaterials:
    def test_returns_list(self, minimal_dataset_json):
        ds = rt.load_reaction_dataset(minimal_dataset_json)
        result = rt.filter_routes_by_starting_materials(
            [], ds, "CC(=O)Oc1ccccc1C(=O)O", "Aspirin")
        assert isinstance(result, list)

    def test_matching_target_included(self, minimal_dataset_json):
        ds = rt.load_reaction_dataset(minimal_dataset_json)
        aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
        result = rt.filter_routes_by_starting_materials(
            [], ds, aspirin_smiles, "Aspirin")
        assert len(result) >= 1

    def test_non_matching_target_excluded(self, minimal_dataset_json):
        ds = rt.load_reaction_dataset(minimal_dataset_json)
        result = rt.filter_routes_by_starting_materials(
            [], ds, CAFFEINE, "Caffeine")
        assert result == []

    def test_route_dict_has_required_fields(self, minimal_dataset_json):
        ds = rt.load_reaction_dataset(minimal_dataset_json)
        aspirin_smiles = rt.to_canonical("CC(=O)Oc1ccccc1C(=O)O")
        result = rt.filter_routes_by_starting_materials(
            [], ds, aspirin_smiles, "Aspirin")
        if result:
            r = result[0]
            for field in ("dataset_steps", "matched_route_name",
                          "validation_status", "is_predicted"):
                assert field in r


class TestGetAllDatasetRoutesForTarget:
    def test_matching_target_name_returned(self, minimal_dataset_json):
        ds = rt.load_reaction_dataset(minimal_dataset_json)
        result = rt.get_all_dataset_routes_for_target(ds, "Aspirin")
        assert len(result) >= 1

    def test_non_matching_target_returns_empty(self, minimal_dataset_json):
        ds = rt.load_reaction_dataset(minimal_dataset_json)
        result = rt.get_all_dataset_routes_for_target(ds, "Nonexistent")
        assert result == []

    def test_case_insensitive_match(self, minimal_dataset_json):
        ds = rt.load_reaction_dataset(minimal_dataset_json)
        r1 = rt.get_all_dataset_routes_for_target(ds, "Aspirin")
        r2 = rt.get_all_dataset_routes_for_target(ds, "aspirin")
        assert len(r1) == len(r2)


# ---------------------------------------------------------------------------
# find_best_routes — validation only (no AiZynthFinder execution)
# ---------------------------------------------------------------------------

class TestFindBestRoutesValidation:
    def test_wrong_number_of_criteria_raises(self):
        with pytest.raises(ValueError, match="3 criteria"):
            rt.find_best_routes("c1ccccc1", ["steps", "yield"],
                                dataset_path="/nonexistent/ds.json",
                                config_path="/nonexistent/cfg.yml")

    def test_unknown_criterion_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            rt.find_best_routes("c1ccccc1",
                                ["steps", "yield", "nonexistent_crit"],
                                dataset_path="/nonexistent/ds.json",
                                config_path="/nonexistent/cfg.yml")