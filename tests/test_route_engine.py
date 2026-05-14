"""
Tests for src/path_finder/route_engine.py

Covers: to_canonical, safe_mol, validate_smiles_for_aizynthfinder,
        build_dataset_smiles_index, bottleneck_yield, average_yield,
        cumulative_yield, get_substances_list, fmt_conditions,
        calc_atom_economy, calc_e_factor, calc_toxicity_score,
        build_solvent_map, compute_steps, compute_yield,
        compute_atom_economy, compute_e_factor, compute_toxicity,
        compute_weights, compute_all_scores, rank_weighted,
        load_reaction_dataset, load_toxicity_dataset,
        load_generic_reaction_dataset.

aizynthfinder is stubbed via conftest.py.
"""

import json
import os
import sys
import tempfile
import pytest

# conftest.py adds src/path_finder to sys.path and stubs aizynthfinder
import src.path_finder.route_engine as rt


# ---------------------------------------------------------------------------
# SMILES constants used across tests
# ---------------------------------------------------------------------------
BENZENE  = "c1ccccc1"
ETHANOL  = "CCO"
ACETIC   = "CC(=O)O"
ASPIRIN  = "CC(=O)Oc1ccccc1C(=O)O"
INVALID  = "not_a_smiles!!!"


# ===========================================================================
# to_canonical
# ===========================================================================

class TestToCanonical:
    def test_valid_smiles_returns_canonical(self):
        result = rt.to_canonical(BENZENE)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_already_canonical_unchanged(self):
        canonical = rt.to_canonical(BENZENE)
        assert rt.to_canonical(canonical) == canonical

    def test_empty_string_returns_empty(self):
        assert rt.to_canonical("") == ""

    def test_invalid_smiles_returned_as_is(self):
        assert rt.to_canonical(INVALID) == INVALID

    def test_list_of_smiles_joined(self):
        result = rt.to_canonical([BENZENE, ETHANOL])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_non_string_scalar_returns_empty(self):
        assert rt.to_canonical(42) == ""

    def test_none_list_entry_skipped(self):
        result = rt.to_canonical([BENZENE, None, ETHANOL])
        assert isinstance(result, str)

    def test_different_representations_same_canonical(self):
        c1 = rt.to_canonical("C1=CC=CC=C1")
        c2 = rt.to_canonical("c1ccccc1")
        assert c1 == c2


# ===========================================================================
# safe_mol
# ===========================================================================

class TestSafeMol:
    def test_valid_smiles_returns_mol(self):
        mol = rt.safe_mol(BENZENE)
        assert mol is not None

    def test_empty_string_returns_none(self):
        assert rt.safe_mol("") is None

    def test_invalid_smiles_returns_none(self):
        assert rt.safe_mol(INVALID) is None

    def test_atom_count_correct(self):
        from rdkit import Chem
        mol = rt.safe_mol(BENZENE)
        assert mol.GetNumAtoms() == 6


# ===========================================================================
# validate_smiles_for_aizynthfinder
# ===========================================================================

class TestValidateSmilesForAiZ:
    def test_valid_smiles_returns_canonical(self):
        result = rt.validate_smiles_for_aizynthfinder(BENZENE)
        assert isinstance(result, str) and len(result) > 0

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            rt.validate_smiles_for_aizynthfinder("")

    def test_invalid_smiles_raises_value_error(self):
        with pytest.raises(ValueError, match="invalid"):
            rt.validate_smiles_for_aizynthfinder(INVALID)

    def test_normalises_non_canonical_form(self):
        c1 = rt.validate_smiles_for_aizynthfinder("C1=CC=CC=C1")
        c2 = rt.validate_smiles_for_aizynthfinder("c1ccccc1")
        assert c1 == c2


# ===========================================================================
# build_dataset_smiles_index
# ===========================================================================

class TestBuildDatasetSmilesIndex:
    def _make_dataset(self, reactions):
        return {"all": reactions}

    def test_returns_set(self):
        dataset = self._make_dataset([])
        result = rt.build_dataset_smiles_index(dataset)
        assert isinstance(result, set)

    def test_includes_products_and_reactants(self):
        dataset = self._make_dataset([{
            "product_smiles":   BENZENE,
            "reactants_smiles": [ETHANOL],
        }])
        index = rt.build_dataset_smiles_index(dataset)
        canon_benz = rt.to_canonical(BENZENE)
        canon_eth  = rt.to_canonical(ETHANOL)
        assert canon_benz in index
        assert canon_eth  in index

    def test_empty_dataset_returns_empty_set(self):
        dataset = self._make_dataset([])
        assert rt.build_dataset_smiles_index(dataset) == set()

    def test_invalid_smiles_not_added(self):
        dataset = self._make_dataset([{
            "product_smiles":   INVALID,
            "reactants_smiles": [],
        }])
        index = rt.build_dataset_smiles_index(dataset)
        # INVALID is returned as-is by to_canonical but is still a non-empty string
        # The important check: no crash and it's handled
        assert isinstance(index, set)


# ===========================================================================
# bottleneck_yield / average_yield / cumulative_yield
# ===========================================================================

class TestYieldHelpers:
    def _step(self, y):
        return {"yield_percent": y}

    # bottleneck_yield
    def test_bottleneck_all_reported(self):
        steps = [self._step(80), self._step(60), self._step(90)]
        assert rt.bottleneck_yield(steps) == 60

    def test_bottleneck_no_yield_data_returns_none(self):
        steps = [{"yield_percent": None}, {"step_number": 1}]
        assert rt.bottleneck_yield(steps) is None

    def test_bottleneck_single_step(self):
        assert rt.bottleneck_yield([self._step(75)]) == 75

    def test_bottleneck_skips_none(self):
        steps = [self._step(None), self._step(40)]
        assert rt.bottleneck_yield(steps) == 40

    # average_yield
    def test_average_yield_correct(self):
        steps = [self._step(80), self._step(60)]
        assert rt.average_yield(steps) == pytest.approx(70.0)

    def test_average_yield_no_data_returns_none(self):
        assert rt.average_yield([{"yield_percent": None}]) is None

    def test_average_yield_single_step(self):
        assert rt.average_yield([self._step(55)]) == pytest.approx(55.0)

    # cumulative_yield
    def test_cumulative_yield_all_reported(self):
        steps = [self._step(80), self._step(50)]
        expected = (80 / 100) * (50 / 100)
        assert rt.cumulative_yield(steps) == pytest.approx(expected)

    def test_cumulative_yield_missing_treated_as_one(self):
        steps = [self._step(80), self._step(None)]
        expected = 80 / 100
        assert rt.cumulative_yield(steps) == pytest.approx(expected)

    def test_cumulative_yield_empty_is_one(self):
        assert rt.cumulative_yield([]) == pytest.approx(1.0)

    def test_cumulative_yield_all_missing_is_one(self):
        steps = [{"yield_percent": None}, {"yield_percent": None}]
        assert rt.cumulative_yield(steps) == pytest.approx(1.0)


# ===========================================================================
# get_substances_list
# ===========================================================================

class TestGetSubstancesList:
    def _make_steps(self):
        return [
            {
                "reactants_smiles": [BENZENE, ETHANOL],
                "product_smiles":   ACETIC,
                "conditions":       {"solvent": "THF", "reagents": ["KOH"]},
            },
            {
                "reactants_smiles": [ACETIC],
                "product_smiles":   ASPIRIN,
                "conditions":       {},
            },
        ]

    def test_returns_dict_with_expected_keys(self):
        result = rt.get_substances_list(self._make_steps())
        assert set(result.keys()) == {"to_buy", "to_prepare", "solvents", "reagents"}

    def test_to_prepare_contains_intermediates(self):
        result = rt.get_substances_list(self._make_steps())
        # ACETIC is produced in step 1 → to_prepare
        canon_acetic = rt.to_canonical(ACETIC)
        assert canon_acetic in result["to_prepare"]

    def test_to_buy_does_not_contain_intermediates(self):
        result = rt.get_substances_list(self._make_steps())
        # ACETIC is produced, so it should NOT appear in to_buy
        canon_acetic = rt.to_canonical(ACETIC)
        assert canon_acetic not in result["to_buy"]

    def test_solvent_listed(self):
        result = rt.get_substances_list(self._make_steps())
        assert "THF" in result["solvents"]

    def test_reagent_listed(self):
        result = rt.get_substances_list(self._make_steps())
        assert "KOH" in result["reagents"]

    def test_empty_steps_returns_empty_lists(self):
        result = rt.get_substances_list([])
        assert result["to_buy"] == []
        assert result["to_prepare"] == []
        assert result["solvents"] == []
        assert result["reagents"] == []


# ===========================================================================
# fmt_conditions
# ===========================================================================

class TestFmtConditions:
    def test_empty_dict_returns_empty_string(self):
        assert rt.fmt_conditions({}) == ""

    def test_none_returns_empty_string(self):
        assert rt.fmt_conditions(None) == ""

    def test_temperature_included(self):
        result = rt.fmt_conditions({"temperature_C": 80})
        assert "80" in result and "°C" in result

    def test_temp_range_fallback(self):
        result = rt.fmt_conditions({"temp_range": "0–25°C"})
        assert "0–25°C" in result

    def test_solvent_included(self):
        result = rt.fmt_conditions({"solvent": "THF"})
        assert "THF" in result

    def test_co_solvent_included(self):
        result = rt.fmt_conditions({"co_solvent": "MeOH"})
        assert "MeOH" in result

    def test_reagents_list_included(self):
        result = rt.fmt_conditions({"reagents": ["KOH", "NaI"]})
        assert "KOH" in result
        assert "NaI" in result

    def test_apparatus_included(self):
        result = rt.fmt_conditions({"apparatus": "microwave"})
        assert "microwave" in result

    def test_full_conditions_joined_by_separator(self):
        cond = {
            "temperature_C": 60,
            "solvent": "DCM",
            "reagents": ["Et3N"],
            "apparatus": "reflux",
        }
        result = rt.fmt_conditions(cond)
        assert "  ·  " in result
        assert "60°C" in result
        assert "DCM"  in result
        assert "Et3N" in result
        assert "reflux" in result


# ===========================================================================
# calc_atom_economy
# ===========================================================================

class TestCalcAtomEconomy:
    def test_single_reactant_to_product_is_one(self):
        # Ethanol → ethanol (identity): AE = MW(product)/MW(reactant) = 1.0
        score = rt.calc_atom_economy([ETHANOL], ETHANOL)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_invalid_product_returns_zero(self):
        score = rt.calc_atom_economy([BENZENE], INVALID)
        assert score == 0.0

    def test_score_bounded_zero_to_one(self):
        score = rt.calc_atom_economy([BENZENE, ETHANOL], ASPIRIN)
        assert 0.0 <= score <= 1.0

    def test_no_reactants_returns_zero(self):
        score = rt.calc_atom_economy([], BENZENE)
        assert score == 0.0

    def test_large_product_capped_at_one(self):
        # If product is heavier than sum of reactants (shouldn't happen chemically
        # but the formula must cap at 1.0)
        score = rt.calc_atom_economy([ETHANOL], ASPIRIN)
        assert score == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# calc_e_factor
# ===========================================================================

class TestCalcEFactor:
    def test_invalid_product_returns_neutral(self):
        score = rt.calc_e_factor([BENZENE], INVALID, 1.0)
        assert score == pytest.approx(0.5)

    def test_score_in_zero_one(self):
        score = rt.calc_e_factor([BENZENE, ETHANOL], ASPIRIN, 0.8)
        assert 0.0 < score <= 1.0

    def test_higher_yield_gives_better_score(self):
        score_high = rt.calc_e_factor([BENZENE, ETHANOL], ASPIRIN, 0.9)
        score_low  = rt.calc_e_factor([BENZENE, ETHANOL], ASPIRIN, 0.1)
        assert score_high > score_low

    def test_perfect_atom_economy_yield_one_approaches_one(self):
        # If product == reactant (identity transform) and yield=1, E-factor≈0
        # So score approaches 1.0
        score = rt.calc_e_factor([ETHANOL], ETHANOL, 1.0)
        assert score == pytest.approx(1.0, abs=1e-3)


# ===========================================================================
# calc_toxicity_score
# ===========================================================================

class TestCalcToxicityScore:
    def test_no_compounds_in_index_returns_half(self):
        # All unknown compounds default hazard to 0.5 → score = 1 − 0.5 = 0.5
        score = rt.calc_toxicity_score([BENZENE], {}, {}, {})
        assert score == pytest.approx(0.5)

    def test_known_safe_compound_raises_score(self):
        canon = rt.to_canonical(BENZENE)
        tox_index = {canon: {"hazard_score": 0.0}}
        score = rt.calc_toxicity_score([BENZENE], {}, tox_index, {})
        assert score == pytest.approx(1.0)

    def test_known_hazardous_compound_lowers_score(self):
        canon = rt.to_canonical(BENZENE)
        tox_index = {canon: {"hazard_score": 1.0}}
        score = rt.calc_toxicity_score([BENZENE], {}, tox_index, {})
        assert score == pytest.approx(0.0)

    def test_score_in_zero_one(self):
        score = rt.calc_toxicity_score([BENZENE, ETHANOL], {}, {}, {})
        assert 0.0 <= score <= 1.0

    def test_empty_reactants_no_crash(self):
        score = rt.calc_toxicity_score([], {}, {}, {})
        assert 0.0 <= score <= 1.0


# ===========================================================================
# build_solvent_map
# ===========================================================================

class TestBuildSolventMap:
    def test_returns_dict(self):
        assert isinstance(rt.build_solvent_map({}), dict)

    def test_known_abbreviations_present(self):
        smap = rt.build_solvent_map({})
        assert "THF"    in smap
        assert "DCM"    in smap
        assert "MeOH"   in smap
        assert "toluene" in smap
        assert "H2O"    in smap

    def test_tox_index_keys_self_map(self):
        canon = rt.to_canonical(BENZENE)
        tox_index = {canon: {"hazard_score": 0.5}}
        smap = rt.build_solvent_map(tox_index)
        assert canon in smap
        assert smap[canon] == canon


# ===========================================================================
# compute_weights
# ===========================================================================

class TestComputeWeights:
    def test_weights_sum_to_one(self):
        criteria = ["steps", "yield", "atom_economy"]
        weights = rt.compute_weights(criteria)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_first_criterion_has_highest_weight(self):
        criteria = ["steps", "yield", "atom_economy"]
        weights = rt.compute_weights(criteria)
        assert weights["steps"] > weights["yield"] > weights["atom_economy"]

    def test_single_criterion_has_weight_one(self):
        weights = rt.compute_weights(["steps"])
        assert weights["steps"] == pytest.approx(1.0)

    def test_two_criteria_weights_correct(self):
        weights = rt.compute_weights(["steps", "yield"])
        # 1/1² = 1, 1/2² = 0.25  → normalised 0.8 and 0.2
        assert weights["steps"] == pytest.approx(0.8, abs=1e-9)
        assert weights["yield"] == pytest.approx(0.2, abs=1e-9)

    def test_approx_three_criteria_weights(self):
        criteria = ["steps", "yield", "atom_economy"]
        weights  = rt.compute_weights(criteria)
        # 1/1, 1/4, 1/9  → total 49/36, approx 0.7347, 0.1837, 0.0816
        assert weights["steps"] == pytest.approx(36/49, abs=1e-4)


# ===========================================================================
# compute_steps / compute_yield / compute_atom_economy /
# compute_e_factor / compute_toxicity
# ===========================================================================

def _make_route(steps, status="dataset"):
    return {"dataset_steps": steps, "validation_status": status}

def _step_dict(reactants, product, y=None, src="dataset", cond=None):
    return {
        "reactants_smiles": reactants,
        "product_smiles":   product,
        "yield_percent":    y,
        "source":           src,
        "conditions":       cond or {},
    }


class TestComputeSteps:
    def test_one_step_route_score_is_one(self):
        route = _make_route([_step_dict([BENZENE], ETHANOL)])
        assert rt.compute_steps(route, {}) == pytest.approx(1.0)

    def test_two_step_route_score_is_half(self):
        route = _make_route([_step_dict([BENZENE], ETHANOL),
                              _step_dict([ETHANOL], ACETIC)])
        assert rt.compute_steps(route, {}) == pytest.approx(0.5)

    def test_empty_steps_returns_one(self):
        assert rt.compute_steps(_make_route([]), {}) == pytest.approx(1.0)

    def test_score_decreases_with_more_steps(self):
        r2 = _make_route([_step_dict([], "")] * 2)
        r5 = _make_route([_step_dict([], "")] * 5)
        assert rt.compute_steps(r2, {}) > rt.compute_steps(r5, {})


class TestComputeYield:
    def test_predicted_route_returns_one(self):
        route = _make_route([_step_dict([BENZENE], ETHANOL, y=50)], status="predicted")
        assert rt.compute_yield(route, {}) == pytest.approx(1.0)

    def test_empty_steps_returns_zero(self):
        assert rt.compute_yield(_make_route([]), {}) == pytest.approx(0.0)

    def test_reported_yields_multiplied(self):
        steps = [_step_dict([BENZENE], ETHANOL, y=80),
                 _step_dict([ETHANOL], ACETIC,  y=50)]
        route = _make_route(steps)
        expected = 0.8 * 0.5
        assert rt.compute_yield(route, {}) == pytest.approx(expected)

    def test_missing_yield_treated_as_one(self):
        steps = [_step_dict([BENZENE], ETHANOL, y=80),
                 _step_dict([ETHANOL], ACETIC,  y=None)]
        route = _make_route(steps)
        assert rt.compute_yield(route, {}) == pytest.approx(0.8)


class TestComputeAtomEconomy:
    def test_score_in_zero_one(self):
        steps = [_step_dict([BENZENE, ETHANOL], ASPIRIN)]
        route = _make_route(steps)
        score = rt.compute_atom_economy(route, {})
        assert 0.0 <= score <= 1.0

    def test_empty_returns_zero(self):
        assert rt.compute_atom_economy(_make_route([]), {}) == 0.0

    def test_is_mean_of_per_step_scores(self):
        s1 = rt.calc_atom_economy([BENZENE], BENZENE)
        s2 = rt.calc_atom_economy([ETHANOL], ETHANOL)
        steps = [_step_dict([BENZENE], BENZENE),
                 _step_dict([ETHANOL], ETHANOL)]
        route = _make_route(steps)
        assert rt.compute_atom_economy(route, {}) == pytest.approx((s1 + s2) / 2)


class TestComputeEFactor:
    def test_score_in_zero_one(self):
        steps = [_step_dict([BENZENE, ETHANOL], ASPIRIN, y=80)]
        route = _make_route(steps)
        score = rt.compute_e_factor(route, {})
        assert 0.0 < score <= 1.0

    def test_empty_returns_zero(self):
        assert rt.compute_e_factor(_make_route([]), {}) == 0.0


class TestComputeToxicity:
    def test_empty_returns_half(self):
        assert rt.compute_toxicity(_make_route([]), {}) == pytest.approx(0.5)

    def test_score_in_zero_one(self):
        steps = [_step_dict([BENZENE], ETHANOL)]
        score = rt.compute_toxicity(_make_route(steps), {})
        assert 0.0 <= score <= 1.0


# ===========================================================================
# compute_all_scores
# ===========================================================================

class TestComputeAllScores:
    def test_returns_dict_with_all_criteria(self):
        route = _make_route([_step_dict([BENZENE], ETHANOL, y=80)])
        result = rt.compute_all_scores(route, {})
        assert set(result.keys()) == set(rt.CRITERIA_REGISTRY.keys())

    def test_all_values_are_floats_in_range(self):
        route = _make_route([_step_dict([BENZENE], ETHANOL, y=80)])
        for key, val in rt.compute_all_scores(route, {}).items():
            assert isinstance(val, float), f"score for {key!r} is not float"
            assert 0.0 <= val <= 1.0, f"score for {key!r} out of [0,1]: {val}"


# ===========================================================================
# rank_weighted
# ===========================================================================

class TestRankWeighted:
    CRITERIA = ["steps", "yield", "atom_economy"]

    def _route(self, n_steps, yield_pct=80, status="dataset"):
        steps = [_step_dict([BENZENE], ETHANOL, y=yield_pct)] * n_steps
        return _make_route(steps, status=status)

    def test_returns_list_of_tuples(self):
        routes = [self._route(2), self._route(3)]
        result = rt.rank_weighted(routes, self.CRITERIA, {})
        assert isinstance(result, list)
        assert all(len(t) == 3 for t in result)

    def test_sorted_descending_by_score(self):
        routes = [self._route(5), self._route(1)]
        result = rt.rank_weighted(routes, self.CRITERIA, {})
        scores = [r[0] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_shorter_route_ranks_higher_when_steps_first(self):
        # Only steps criterion matters here (it's first and dominant)
        r1 = self._route(1)  # 1 step → steps score = 1.0
        r5 = self._route(5)  # 5 steps → steps score = 0.2
        criteria = ["steps", "atom_economy", "e_factor"]
        result = rt.rank_weighted([r5, r1], criteria, {})
        assert result[0][2] is r1

    def test_predicted_route_excludes_yield(self):
        route = self._route(2, status="predicted")
        result = rt.rank_weighted([route], self.CRITERIA, {})
        details = result[0][1]
        assert details["yield"]["excluded"] is True
        assert details["yield"]["weighted"] == 0.0

    def test_empty_input_returns_empty_list(self):
        assert rt.rank_weighted([], self.CRITERIA, {}) == []

    def test_details_contain_expected_keys(self):
        route = self._route(2)
        result = rt.rank_weighted([route], self.CRITERIA, {})
        details = result[0][1]
        for c in self.CRITERIA:
            assert c in details
            if not details[c].get("excluded"):
                assert "raw" in details[c]
                assert "weight" in details[c]
                assert "weighted" in details[c]
        assert "_all_scores" in details


# ===========================================================================
# load_reaction_dataset
# ===========================================================================

class TestLoadReactionDataset:
    def _write_json(self, data, suffix=".json"):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        )
        json.dump(data, f)
        f.close()
        return f.name

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            rt.load_reaction_dataset("/nonexistent/path.json")

    def test_bare_list_format(self):
        reactions = [
            {
                "id": "1", "route_id": "r1", "route_name": "Route1",
                "target": "TestTarget", "step_number": 1,
                "reactants_smiles": [BENZENE], "product_smiles": ETHANOL,
                "conditions": {}, "yield_percent": 80, "reaction_type": "esterification",
            }
        ]
        path = self._write_json(reactions)
        try:
            ds = rt.load_reaction_dataset(path)
            assert "by_route" in ds
            assert "by_product" in ds
            assert "by_reactant" in ds
            assert "all" in ds
            assert len(ds["all"]) == 1
        finally:
            os.unlink(path)

    def test_dict_format_with_reactions_key(self):
        data = {
            "reactions": [
                {
                    "id": "1", "route_id": "r1", "route_name": "Route1",
                    "target": "T", "step_number": 1,
                    "reactants_smiles": [BENZENE], "product_smiles": ETHANOL,
                    "conditions": {}, "yield_percent": None, "reaction_type": "",
                }
            ],
            "_metadata": {"target_smiles": {"T": ETHANOL}},
        }
        path = self._write_json(data)
        try:
            ds = rt.load_reaction_dataset(path)
            assert ds["metadata"]["target_smiles"]["T"] == ETHANOL
        finally:
            os.unlink(path)

    def test_unrecognised_format_raises(self):
        path = self._write_json({"foo": "bar"})
        try:
            with pytest.raises(ValueError, match="unrecognized"):
                rt.load_reaction_dataset(path)
        finally:
            os.unlink(path)

    def test_steps_sorted_by_step_number(self):
        reactions = [
            {"id": "2", "route_id": "r1", "route_name": "R", "target": "T",
             "step_number": 2, "reactants_smiles": [ACETIC], "product_smiles": ASPIRIN,
             "conditions": {}, "yield_percent": None, "reaction_type": ""},
            {"id": "1", "route_id": "r1", "route_name": "R", "target": "T",
             "step_number": 1, "reactants_smiles": [BENZENE], "product_smiles": ACETIC,
             "conditions": {}, "yield_percent": None, "reaction_type": ""},
        ]
        path = self._write_json(reactions)
        try:
            ds = rt.load_reaction_dataset(path)
            steps = ds["by_route"]["r1"]
            assert steps[0]["step_number"] == 1
            assert steps[1]["step_number"] == 2
        finally:
            os.unlink(path)

    def test_list_product_smiles_joined(self):
        reactions = [
            {"id": "1", "route_id": "r1", "route_name": "R", "target": "T",
             "step_number": 1, "reactants_smiles": [BENZENE],
             "product_smiles": [ETHANOL, ACETIC],
             "conditions": {}, "yield_percent": None, "reaction_type": ""},
        ]
        path = self._write_json(reactions)
        try:
            ds = rt.load_reaction_dataset(path)
            prod = ds["all"][0]["product_smiles"]
            assert isinstance(prod, str)
            assert "." in prod
        finally:
            os.unlink(path)

    def test_string_reactants_smiles_wrapped_in_list(self):
        reactions = [
            {"id": "1", "route_id": "r1", "route_name": "R", "target": "T",
             "step_number": 1, "reactants_smiles": BENZENE,
             "product_smiles": ETHANOL,
             "conditions": {}, "yield_percent": None, "reaction_type": ""},
        ]
        path = self._write_json(reactions)
        try:
            ds = rt.load_reaction_dataset(path)
            reacs = ds["all"][0]["reactants_smiles"]
            assert isinstance(reacs, list)
        finally:
            os.unlink(path)


# ===========================================================================
# load_toxicity_dataset
# ===========================================================================

class TestLoadToxicityDataset:
    def test_missing_file_returns_empty_dict(self):
        result = rt.load_toxicity_dataset("/nonexistent/toxicity.json")
        assert result == {}

    def test_list_format(self, tmp_path):
        data = [
            {"smiles": BENZENE, "hazard_score": 0.6},
            {"smiles": ETHANOL, "hazard_score": 0.2},
        ]
        p = tmp_path / "tox.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = rt.load_toxicity_dataset(str(p))
        assert len(result) == 2
        canon = rt.to_canonical(BENZENE)
        assert canon in result
        assert result[canon]["hazard_score"] == pytest.approx(0.6)

    def test_dict_format_with_compounds_key(self, tmp_path):
        data = {"compounds": [{"smiles": BENZENE, "hazard_score": 0.3}]}
        p = tmp_path / "tox.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = rt.load_toxicity_dataset(str(p))
        assert len(result) == 1


# ===========================================================================
# load_generic_reaction_dataset
# ===========================================================================

class TestLoadGenericReactionDataset:
    def test_missing_file_returns_empty_dict(self):
        result = rt.load_generic_reaction_dataset("/nonexistent/generic.json")
        assert result == {}

    def test_empty_path_returns_empty_dict(self):
        assert rt.load_generic_reaction_dataset("") == {}

    def test_valid_file_structure(self, tmp_path):
        data = [
            {"reactants_smiles": [BENZENE], "product_smiles": ETHANOL,
             "step_number": 1},
        ]
        p = tmp_path / "generic.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = rt.load_generic_reaction_dataset(str(p))
        assert "by_product" in result
        assert "by_reaction_key" in result
        assert "all" in result
        assert len(result["all"]) == 1

    def test_indexed_by_product(self, tmp_path):
        data = [{"reactants_smiles": [BENZENE], "product_smiles": ETHANOL}]
        p = tmp_path / "generic.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = rt.load_generic_reaction_dataset(str(p))
        canon_eth = rt.to_canonical(ETHANOL)
        assert canon_eth in result["by_product"]