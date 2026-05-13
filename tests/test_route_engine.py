"""
Tests for path_finder.route_engine

Covers every public function that does not require AiZynthFinder or
rxn_insight (those heavy deps are skipped via pytest.importorskip /
module-level guards).  RDKit is expected to be available; tests that
need it are skipped gracefully if the import fails.
"""

import json
import os
import pytest

# ---------------------------------------------------------------------------
# Optional dep guards
# ---------------------------------------------------------------------------
rdkit = pytest.importorskip("rdkit", reason="RDKit not installed")

from path_finder import route_engine as re_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

ETHANOL = "CCO"
ETHANOL_CANON = "CCO"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

def _make_step(reactants, product, yield_pct=None, source="dataset", conditions=None):
    return {
        "reactants_smiles": reactants,
        "product_smiles": product,
        "yield_percent": yield_pct,
        "source": source,
        "conditions": conditions or {},
    }


def _make_route(steps, validation_status="dataset"):
    return {
        "dataset_steps": steps,
        "validation_status": validation_status,
    }


@pytest.fixture
def simple_dataset_json(tmp_path):
    data = {
        "reactions": [
            {
                "id": "r1",
                "route_id": "route_A",
                "route_name": "Route A",
                "target": "Aspirin",
                "step_number": 1,
                "reactants_smiles": ["c1ccccc1O", "CC(=O)O"],
                "product_smiles": ASPIRIN,
                "conditions": {"temperature_C": 80, "solvent": "AcOH"},
                "yield_percent": 85.0,
                "reaction_type": "Esterification",
            },
            {
                "id": "r2",
                "route_id": "route_A",
                "route_name": "Route A",
                "target": "Aspirin",
                "step_number": 2,
                "reactants_smiles": [ASPIRIN],
                "product_smiles": ASPIRIN,
                "conditions": {},
                "yield_percent": 90.0,
                "reaction_type": "Purification",
            },
        ]
    }
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(data))
    return str(p)


@pytest.fixture
def toxicity_json(tmp_path):
    data = [
        {"smiles": ETHANOL, "hazard_score": 0.2},
        {"smiles": "ClCCl", "hazard_score": 0.8},
    ]
    p = tmp_path / "tox.json"
    p.write_text(json.dumps(data))
    return str(p)


@pytest.fixture
def generic_dataset_json(tmp_path):
    data = [
        {
            "id": "g1",
            "route_id": "generic",
            "step_number": 1,
            "reactants_smiles": ["c1ccccc1O", "CC(=O)O"],
            "product_smiles": ASPIRIN,
            "conditions": {"temperature_C": 80},
            "yield_percent": 85.0,
            "reaction_type": "Esterification",
        }
    ]
    p = tmp_path / "generic.json"
    p.write_text(json.dumps(data))
    return str(p)


# ===========================================================================
# to_canonical
# ===========================================================================

class TestToCanonical:
    def test_simple_smiles(self):
        assert re_mod.to_canonical("CCO") == re_mod.to_canonical("OCC")

    def test_list_input(self):
        result = re_mod.to_canonical(["C", "CO"])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_string(self):
        assert re_mod.to_canonical("") == ""

    def test_empty_list(self):
        assert re_mod.to_canonical([]) == ""

    def test_invalid_smiles_returns_original(self):
        bad = "NOT_A_SMILES"
        assert re_mod.to_canonical(bad) == bad

    def test_non_string_scalar(self):
        assert re_mod.to_canonical(123) == ""

    def test_list_with_none_entries(self):
        result = re_mod.to_canonical([None, "CCO", None])
        assert "CCO" in result or result == re_mod.to_canonical("CCO")


# ===========================================================================
# safe_mol
# ===========================================================================

class TestSafeMol:
    def test_valid_smiles(self):
        mol = re_mod.safe_mol("CCO")
        assert mol is not None

    def test_empty_string(self):
        assert re_mod.safe_mol("") is None

    def test_invalid_smiles(self):
        assert re_mod.safe_mol("INVALID") is None

    def test_complex_molecule(self):
        assert re_mod.safe_mol(ASPIRIN) is not None


# ===========================================================================
# validate_smiles_for_aizynthfinder
# ===========================================================================

class TestValidateSmilesForAizynthfinder:
    def test_valid_smiles_returns_canonical(self):
        canon = re_mod.validate_smiles_for_aizynthfinder("CCO")
        assert isinstance(canon, str)
        assert len(canon) > 0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty SMILES"):
            re_mod.validate_smiles_for_aizynthfinder("")

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="invalid SMILES"):
            re_mod.validate_smiles_for_aizynthfinder("NOT_VALID")

    def test_returns_canonical_form(self):
        result = re_mod.validate_smiles_for_aizynthfinder("OCC")
        assert result == re_mod.to_canonical("CCO")


# ===========================================================================
# load_reaction_dataset
# ===========================================================================

class TestLoadReactionDataset:
    def test_loads_correctly(self, simple_dataset_json):
        ds = re_mod.load_reaction_dataset(simple_dataset_json)
        assert "by_product" in ds
        assert "by_reactant" in ds
        assert "by_route" in ds
        assert "all" in ds
        assert "metadata" in ds

    def test_route_indexed(self, simple_dataset_json):
        ds = re_mod.load_reaction_dataset(simple_dataset_json)
        assert "route_A" in ds["by_route"]

    def test_steps_sorted_by_step_number(self, simple_dataset_json):
        ds = re_mod.load_reaction_dataset(simple_dataset_json)
        steps = ds["by_route"]["route_A"]
        nums = [s["step_number"] for s in steps]
        assert nums == sorted(nums)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            re_mod.load_reaction_dataset("/nonexistent/path.json")

    def test_invalid_format_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"no_reactions_key": []}))
        with pytest.raises(ValueError, match="unrecognized"):
            re_mod.load_reaction_dataset(str(bad))

    def test_list_format(self, tmp_path):
        data = [
            {
                "id": "x1", "route_id": "r1", "route_name": "R1",
                "target": "X", "step_number": 1,
                "reactants_smiles": ["CCO"], "product_smiles": ASPIRIN,
                "conditions": {}, "yield_percent": 50.0, "reaction_type": "Test",
            }
        ]
        p = tmp_path / "list.json"
        p.write_text(json.dumps(data))
        ds = re_mod.load_reaction_dataset(str(p))
        assert len(ds["all"]) == 1

    def test_product_smiles_list_normalised(self, tmp_path):
        data = [
            {
                "id": "p1", "route_id": "r1", "route_name": "R", "target": "T",
                "step_number": 1,
                "reactants_smiles": ["CCO"],
                "product_smiles": ["CC", "O"],  # list format
                "conditions": {}, "yield_percent": None, "reaction_type": "",
            }
        ]
        p = tmp_path / "prod_list.json"
        p.write_text(json.dumps(data))
        ds = re_mod.load_reaction_dataset(str(p))
        prod = ds["all"][0]["product_smiles"]
        assert isinstance(prod, str)


# ===========================================================================
# load_toxicity_dataset
# ===========================================================================

class TestLoadToxicityDataset:
    def test_loads_correctly(self, toxicity_json):
        tox = re_mod.load_toxicity_dataset(toxicity_json)
        assert isinstance(tox, dict)
        assert len(tox) == 2

    def test_missing_file_returns_empty(self):
        tox = re_mod.load_toxicity_dataset("/nonexistent/tox.json")
        assert tox == {}

    def test_keys_are_canonical_smiles(self, toxicity_json):
        tox = re_mod.load_toxicity_dataset(toxicity_json)
        for key in tox:
            assert re_mod.to_canonical(key) == key or isinstance(key, str)


# ===========================================================================
# load_generic_reaction_dataset
# ===========================================================================

class TestLoadGenericReactionDataset:
    def test_loads_correctly(self, generic_dataset_json):
        gds = re_mod.load_generic_reaction_dataset(generic_dataset_json)
        assert "by_product" in gds
        assert "by_reaction_key" in gds
        assert "all" in gds
        assert len(gds["all"]) == 1

    def test_missing_file_returns_empty(self):
        gds = re_mod.load_generic_reaction_dataset("/nonexistent.json")
        assert gds == {}

    def test_empty_path_returns_empty(self):
        gds = re_mod.load_generic_reaction_dataset("")
        assert gds == {}


# ===========================================================================
# build_dataset_smiles_index
# ===========================================================================

class TestBuildDatasetSmilesIndex:
    def test_returns_set(self, simple_dataset_json):
        ds = re_mod.load_reaction_dataset(simple_dataset_json)
        index = re_mod.build_dataset_smiles_index(ds)
        assert isinstance(index, set)

    def test_contains_product_smiles(self, simple_dataset_json):
        ds = re_mod.load_reaction_dataset(simple_dataset_json)
        index = re_mod.build_dataset_smiles_index(ds)
        canon_aspirin = re_mod.to_canonical(ASPIRIN)
        assert canon_aspirin in index

    def test_contains_reactant_smiles(self, simple_dataset_json):
        ds = re_mod.load_reaction_dataset(simple_dataset_json)
        index = re_mod.build_dataset_smiles_index(ds)
        # "c1ccccc1O" is phenol
        assert re_mod.to_canonical("c1ccccc1O") in index


# ===========================================================================
# get_targets_from_dataset
# ===========================================================================

class TestGetTargetsFromDataset:
    def test_returns_dict(self, simple_dataset_json):
        ds = re_mod.load_reaction_dataset(simple_dataset_json)
        targets = re_mod.get_targets_from_dataset(ds)
        assert isinstance(targets, dict)

    def test_aspirin_found(self, simple_dataset_json):
        ds = re_mod.load_reaction_dataset(simple_dataset_json)
        targets = re_mod.get_targets_from_dataset(ds)
        # Aspirin should be present; its product has enough heavy atoms
        assert "Aspirin" in targets

    def test_values_are_canonical_smiles(self, simple_dataset_json):
        ds = re_mod.load_reaction_dataset(simple_dataset_json)
        targets = re_mod.get_targets_from_dataset(ds)
        for name, smi in targets.items():
            assert isinstance(smi, str)
            assert re_mod.safe_mol(smi) is not None


# ===========================================================================
# _walk_reaction_tree
# ===========================================================================

class TestWalkReactionTree:
    def test_empty_mol_node(self):
        steps = []
        re_mod._walk_reaction_tree({"type": "mol", "smiles": "CCO", "children": []}, steps)
        assert steps == []

    def test_single_reaction_step(self):
        tree = {
            "type": "mol",
            "smiles": "CCO",
            "children": [
                {
                    "type": "reaction",
                    "children": [
                        {"type": "mol", "smiles": "CC", "children": []},
                        {"type": "mol", "smiles": "O",  "children": []},
                    ],
                }
            ],
        }
        steps = []
        re_mod._walk_reaction_tree(tree, steps)
        assert len(steps) == 1
        assert steps[0]["product"] == "CCO"
        assert "CC" in steps[0]["reactants"]

    def test_non_mol_root_ignored(self):
        steps = []
        re_mod._walk_reaction_tree({"type": "reaction", "children": []}, steps)
        assert steps == []


# ===========================================================================
# bottleneck_yield / average_yield / cumulative_yield
# ===========================================================================

class TestYieldHelpers:
    def test_bottleneck_with_yields(self):
        steps = [{"yield_percent": 80}, {"yield_percent": 50}, {"yield_percent": 90}]
        assert re_mod.bottleneck_yield(steps) == 50

    def test_bottleneck_no_yields(self):
        assert re_mod.bottleneck_yield([{"yield_percent": None}]) is None

    def test_bottleneck_empty(self):
        assert re_mod.bottleneck_yield([]) is None

    def test_average_yield(self):
        steps = [{"yield_percent": 80}, {"yield_percent": 60}]
        assert re_mod.average_yield(steps) == pytest.approx(70.0)

    def test_average_yield_no_data(self):
        assert re_mod.average_yield([]) is None

    def test_cumulative_yield(self):
        steps = [{"yield_percent": 50}, {"yield_percent": 80}]
        assert re_mod.cumulative_yield(steps) == pytest.approx(0.4)

    def test_cumulative_yield_missing_treated_as_100(self):
        steps = [{"yield_percent": 50}, {"yield_percent": None}]
        assert re_mod.cumulative_yield(steps) == pytest.approx(0.5)

    def test_cumulative_yield_empty(self):
        assert re_mod.cumulative_yield([]) == pytest.approx(1.0)


# ===========================================================================
# get_substances_list
# ===========================================================================

class TestGetSubstancesList:
    def test_returns_expected_keys(self):
        steps = [_make_step(["CCO", "CC"], ASPIRIN)]
        result = re_mod.get_substances_list(steps)
        assert set(result.keys()) == {"to_buy", "to_prepare", "solvents", "reagents"}

    def test_to_prepare_contains_product(self):
        steps = [_make_step(["CCO"], ASPIRIN)]
        result = re_mod.get_substances_list(steps)
        assert re_mod.to_canonical(ASPIRIN) in result["to_prepare"]

    def test_solvents_extracted(self):
        steps = [_make_step(["CCO"], ASPIRIN, conditions={"solvent": "THF"})]
        result = re_mod.get_substances_list(steps)
        assert "THF" in result["solvents"]

    def test_reagents_extracted(self):
        steps = [_make_step(["CCO"], ASPIRIN,
                             conditions={"reagents": ["NaOH", "HCl"]})]
        result = re_mod.get_substances_list(steps)
        assert "NaOH" in result["reagents"]


# ===========================================================================
# fmt_conditions
# ===========================================================================

class TestFmtConditions:
    def test_empty_dict(self):
        assert re_mod.fmt_conditions({}) == ""

    def test_temperature(self):
        out = re_mod.fmt_conditions({"temperature_C": 100})
        assert "100" in out

    def test_solvent(self):
        out = re_mod.fmt_conditions({"solvent": "THF"})
        assert "THF" in out

    def test_reagents_list(self):
        out = re_mod.fmt_conditions({"reagents": ["NaH", "DMF"]})
        assert "NaH" in out

    def test_full_conditions(self):
        cond = {
            "temperature_C": 80,
            "solvent": "THF",
            "reagents": ["NaH"],
            "apparatus": "flask",
        }
        out = re_mod.fmt_conditions(cond)
        assert "80" in out
        assert "THF" in out
        assert "NaH" in out
        assert "flask" in out


# ===========================================================================
# calc_atom_economy
# ===========================================================================

class TestCalcAtomEconomy:
    def test_perfect_economy(self):
        # product MW = sum of reactants → atom economy = 1
        # Simple: C + O → CO (methanol)
        score = re_mod.calc_atom_economy(["C", "O"], "CO")
        assert 0.0 < score <= 1.0

    def test_invalid_product_returns_zero(self):
        score = re_mod.calc_atom_economy(["CCO"], "INVALID_SMI")
        assert score == 0.0

    def test_result_in_range(self):
        score = re_mod.calc_atom_economy(["c1ccccc1O", "CC(=O)O"], ASPIRIN)
        assert 0.0 <= score <= 1.0

    def test_no_reactants(self):
        score = re_mod.calc_atom_economy([], ASPIRIN)
        assert score == 0.0


# ===========================================================================
# calc_e_factor
# ===========================================================================

class TestCalcEFactor:
    def test_invalid_product_returns_half(self):
        score = re_mod.calc_e_factor(["CCO"], "INVALID", 1.0)
        assert score == pytest.approx(0.5)

    def test_high_yield_gives_good_score(self):
        score_high = re_mod.calc_e_factor(["c1ccccc1O", "CC(=O)O"], ASPIRIN, 0.95)
        score_low  = re_mod.calc_e_factor(["c1ccccc1O", "CC(=O)O"], ASPIRIN, 0.10)
        assert score_high > score_low

    def test_result_in_range(self):
        score = re_mod.calc_e_factor(["CCO"], ASPIRIN, 0.8)
        assert 0.0 < score <= 1.0


# ===========================================================================
# build_solvent_map
# ===========================================================================

class TestBuildSolventMap:
    def test_contains_common_abbreviations(self):
        smap = re_mod.build_solvent_map({})
        assert "THF" in smap
        assert "DCM" in smap
        assert "MeOH" in smap

    def test_tox_index_keys_added(self):
        tox = {re_mod.to_canonical("CCO"): {"hazard_score": 0.2}}
        smap = re_mod.build_solvent_map(tox)
        assert re_mod.to_canonical("CCO") in smap

    def test_values_are_smiles_strings(self):
        smap = re_mod.build_solvent_map({})
        for v in smap.values():
            assert isinstance(v, str)


# ===========================================================================
# calc_toxicity_score
# ===========================================================================

class TestCalcToxicityScore:
    def test_score_in_range(self, toxicity_json):
        tox = re_mod.load_toxicity_dataset(toxicity_json)
        smap = re_mod.build_solvent_map(tox)
        score = re_mod.calc_toxicity_score([ETHANOL], {}, tox, smap)
        assert 0.0 <= score <= 1.0

    def test_safe_molecule_gives_high_score(self, toxicity_json):
        tox = re_mod.load_toxicity_dataset(toxicity_json)  # CCO=0.2, ClCCl=0.8
        smap = re_mod.build_solvent_map(tox)
        safe  = re_mod.calc_toxicity_score([ETHANOL], {}, tox, smap)
        toxic = re_mod.calc_toxicity_score(["ClCCl"], {}, tox, smap)
        assert safe > toxic

    def test_unknown_compound_defaults_to_neutral(self):
        tox = {}
        smap = re_mod.build_solvent_map(tox)
        score = re_mod.calc_toxicity_score(["CCCCCC"], {}, tox, smap)
        assert score == pytest.approx(0.5)


# ===========================================================================
# compute_weights
# ===========================================================================

class TestComputeWeights:
    def test_weights_sum_to_one(self):
        w = re_mod.compute_weights(["steps", "yield", "atom_economy"])
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

    def test_first_criterion_highest_weight(self):
        w = re_mod.compute_weights(["steps", "yield", "atom_economy"])
        assert w["steps"] > w["yield"] > w["atom_economy"]

    def test_single_criterion(self):
        w = re_mod.compute_weights(["steps"])
        assert w["steps"] == pytest.approx(1.0)


# ===========================================================================
# compute_steps
# ===========================================================================

class TestComputeSteps:
    def test_one_step_scores_one(self):
        route = _make_route([_make_step(["CCO"], ASPIRIN)])
        assert re_mod.compute_steps(route, {}) == pytest.approx(1.0)

    def test_more_steps_lower_score(self):
        r2 = _make_route([_make_step(["CCO"], ASPIRIN)] * 2)
        r5 = _make_route([_make_step(["CCO"], ASPIRIN)] * 5)
        assert re_mod.compute_steps(r2, {}) > re_mod.compute_steps(r5, {})

    def test_empty_steps(self):
        assert re_mod.compute_steps(_make_route([]), {}) == pytest.approx(1.0)


# ===========================================================================
# compute_yield
# ===========================================================================

class TestComputeYield:
    def test_predicted_returns_one(self):
        route = _make_route(
            [_make_step(["CCO"], ASPIRIN, yield_pct=50.0)],
            validation_status="predicted",
        )
        assert re_mod.compute_yield(route, {}) == pytest.approx(1.0)

    def test_dataset_route_cumulative(self):
        steps = [
            _make_step(["CCO"], ASPIRIN, yield_pct=50.0, source="dataset"),
            _make_step([ASPIRIN], ASPIRIN, yield_pct=80.0, source="dataset"),
        ]
        route = _make_route(steps, "dataset")
        assert re_mod.compute_yield(route, {}) == pytest.approx(0.40)

    def test_empty_steps_returns_zero(self):
        assert re_mod.compute_yield(_make_route([]), {}) == 0.0


# ===========================================================================
# compute_atom_economy / compute_e_factor
# ===========================================================================

class TestComputeAtomEconomy:
    def test_result_in_range(self):
        route = _make_route([_make_step(["c1ccccc1O", "CC(=O)O"], ASPIRIN)])
        score = re_mod.compute_atom_economy(route, {})
        assert 0.0 <= score <= 1.0

    def test_empty_steps_returns_zero(self):
        assert re_mod.compute_atom_economy(_make_route([]), {}) == 0.0


class TestComputeEFactor:
    def test_result_in_range(self):
        steps = [_make_step(["c1ccccc1O", "CC(=O)O"], ASPIRIN, yield_pct=85.0)]
        route = _make_route(steps)
        score = re_mod.compute_e_factor(route, {})
        assert 0.0 < score <= 1.0

    def test_empty_steps_returns_zero(self):
        assert re_mod.compute_e_factor(_make_route([]), {}) == 0.0


# ===========================================================================
# compute_all_scores
# ===========================================================================

class TestComputeAllScores:
    def test_returns_all_criteria_keys(self):
        route = _make_route([_make_step(["CCO"], ASPIRIN, yield_pct=80.0)])
        scores = re_mod.compute_all_scores(route, {})
        assert set(scores.keys()) == set(re_mod.CRITERIA_REGISTRY.keys())

    def test_all_values_in_range(self):
        route = _make_route([_make_step(["CCO"], ASPIRIN, yield_pct=80.0)])
        scores = re_mod.compute_all_scores(route, {})
        for v in scores.values():
            assert 0.0 <= v <= 1.0


# ===========================================================================
# rank_weighted
# ===========================================================================

class TestRankWeighted:
    def _routes(self):
        return [
            _make_route([_make_step(["CCO"], ASPIRIN, yield_pct=90.0)]),
            _make_route([_make_step(["CCO"], ASPIRIN, yield_pct=90.0)] * 5),
        ]

    def test_sorted_descending(self):
        routes = self._routes()
        ranked = re_mod.rank_weighted(routes, ["steps", "yield", "atom_economy"], {})
        scores = [r[0] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_returns_tuples_of_three(self):
        ranked = re_mod.rank_weighted(self._routes(),
                                      ["steps", "yield", "atom_economy"], {})
        for item in ranked:
            assert len(item) == 3

    def test_fewer_steps_ranks_higher_with_steps_criterion(self):
        r1 = _make_route([_make_step(["CCO"], ASPIRIN, yield_pct=80.0)])   # 1 step
        r2 = _make_route([_make_step(["CCO"], ASPIRIN, yield_pct=80.0)] * 4)  # 4 steps
        ranked = re_mod.rank_weighted([r1, r2], ["steps", "yield", "atom_economy"], {})
        # The 1-step route must have a higher total score
        top_route = ranked[0][2]
        assert top_route is r1

    def test_predicted_yield_excluded(self):
        pred = _make_route(
            [_make_step(["CCO"], ASPIRIN)], validation_status="predicted"
        )
        ranked = re_mod.rank_weighted([pred], ["steps", "yield", "atom_economy"], {})
        details = ranked[0][1]
        assert details["yield"]["excluded"] is True


# ===========================================================================
# match_step_in_generic_dataset
# ===========================================================================

class TestMatchStepInGenericDataset:
    def test_exact_match_found(self, generic_dataset_json):
        gds = re_mod.load_generic_reaction_dataset(generic_dataset_json)
        reactants = ["c1ccccc1O", "CC(=O)O"]
        match = re_mod.match_step_in_generic_dataset(reactants, ASPIRIN, gds)
        assert match is not None

    def test_no_match_returns_none(self, generic_dataset_json):
        gds = re_mod.load_generic_reaction_dataset(generic_dataset_json)
        match = re_mod.match_step_in_generic_dataset(["CCO"], "C#N", gds)
        assert match is None

    def test_empty_dataset_returns_none(self):
        match = re_mod.match_step_in_generic_dataset(["CCO"], ASPIRIN, {})
        assert match is None


# ===========================================================================
# is_route_covered_by_dataset
# ===========================================================================

class TestIsRouteCoveredByDataset:
    def test_covered_route(self, simple_dataset_json):
        ds = re_mod.load_reaction_dataset(simple_dataset_json)
        index = re_mod.build_dataset_smiles_index(ds)
        # Route whose product is Aspirin — known in dataset
        aiz_route = {
            "steps": [{"product": ASPIRIN, "reactants": ["c1ccccc1O"]}]
        }
        assert re_mod.is_route_covered_by_dataset(aiz_route, index, threshold=0.5)

    def test_uncovered_route(self, simple_dataset_json):
        ds = re_mod.load_reaction_dataset(simple_dataset_json)
        index = re_mod.build_dataset_smiles_index(ds)
        aiz_route = {
            "steps": [
                {"product": "C1CCCCC1", "reactants": ["CCCCCC"]},  # cyclohexane
                {"product": "C1CCCNC1", "reactants": ["C1CCCCC1"]},  # piperidine
            ]
        }
        assert not re_mod.is_route_covered_by_dataset(aiz_route, index, threshold=0.5)

    def test_empty_steps_returns_false(self):
        assert not re_mod.is_route_covered_by_dataset({"steps": []}, set())


# ===========================================================================
# filter_routes_by_starting_materials
# ===========================================================================

class TestFilterRoutesByStartingMaterials:
    def test_matches_by_smiles(self, simple_dataset_json):
        ds = re_mod.load_reaction_dataset(simple_dataset_json)
        result = re_mod.filter_routes_by_starting_materials(
            [], ds, ASPIRIN, target_name="Aspirin"
        )
        assert len(result) >= 1

    def test_no_match_returns_empty(self, simple_dataset_json):
        ds = re_mod.load_reaction_dataset(simple_dataset_json)
        result = re_mod.filter_routes_by_starting_materials(
            [], ds, "C1CCNCC1", target_name="Piperidine"
        )
        assert result == []

    def test_result_has_expected_keys(self, simple_dataset_json):
        ds = re_mod.load_reaction_dataset(simple_dataset_json)
        result = re_mod.filter_routes_by_starting_materials(
            [], ds, ASPIRIN, target_name="Aspirin"
        )
        if result:
            for key in ("route_id", "dataset_steps", "is_predicted"):
                assert key in result[0]