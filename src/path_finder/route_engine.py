# route_engine.py
# Backend for the Path Finder retrosynthesis interface.
# Handles: dataset loading, SMILES canonicalisation, AiZynthFinder search,
# Rxn-INSIGHT condition prediction, step validation against the generic dataset,
# per-criterion scoring and weighted route ranking.
# Main entry point: find_best_routes()

import json
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from aizynthfinder.aizynthfinder import AiZynthFinder

# optional Rxn-INSIGHT import
# rxn_insight is not required; if absent, predicted-route enrichment is disabled
# and RXNINSIGHT_AVAILABLE is set to False throughout the module.
try:
    from rxn_insight.reaction import Reaction as RxnInsightReaction
    RXNINSIGHT_AVAILABLE = True
except ImportError:
    RXNINSIGHT_AVAILABLE = False

# Data loading
def load_reaction_dataset(path: str) -> dict:
    """
    Load and index the main curated reaction dataset from a JSON file.

    Accepted formats: a bare list of reaction dicts, or a dict with a
    ``"reactions"`` key and an optional ``"_metadata"`` section storing
    canonical target SMILES keyed by target name.

    Each reaction dict is expected to contain: ``id``, ``route_id``,
    ``route_name``, ``target``, ``step_number``, ``reactants_smiles``,
    ``product_smiles``, ``conditions``, ``yield_percent``, ``reaction_type``.

    List-valued ``product_smiles`` fields are joined with ``'.'``.
    String ``reactants_smiles`` fields are wrapped in a list; nested lists
    are flattened. Steps within each route are sorted by ``step_number``.

    Parameters
    ----------
    path : str
        Filesystem path to the JSON dataset file.

    Returns
    -------
    dict
        A dict with keys ``by_product``, ``by_reactant``, ``by_route``,
        ``all``, and ``metadata``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the JSON structure is not a recognised format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Detect format and extract flat reaction list
    if isinstance(raw, list):
        reactions = raw
        metadata  = {}
    elif isinstance(raw, dict) and "reactions" in raw:
        reactions = raw["reactions"]
        metadata  = raw.get("_metadata", {})
    else:
        raise ValueError("unrecognized dataset format")
    print(f"[dataset] {len(reactions)} reactions loaded")

    # Build secondary indices
    by_product = {}; by_reactant = {}; by_route = {}
    for rxn in reactions:
        # Normalise product_smiles: convert list → single SMILES string
        prod = rxn.get("product_smiles", "")
        if isinstance(prod, list):
            rxn["product_smiles"] = '.'.join(str(s) for s in prod if s)

        # Normalise reactants_smiles: ensure it is always a flat list of strings
        reac = rxn.get("reactants_smiles", [])
        if isinstance(reac, str):
            rxn["reactants_smiles"] = [reac]
        elif isinstance(reac, list):
            flat = []
            for r in reac:
                if isinstance(r, list):
                    flat.extend(str(s) for s in r if s)
                elif r is not None:
                    flat.append(str(r))
            rxn["reactants_smiles"] = flat

        # Index by canonical product SMILES
        pk = to_canonical(rxn.get("product_smiles", ""))
        if pk:
            by_product.setdefault(pk, []).append(rxn)

        # Index by each canonical reactant SMILES
        for r in rxn.get("reactants_smiles", []):
            rk = to_canonical(r)
            if rk:
                by_reactant.setdefault(rk, []).append(rxn)

        # Group steps by route_id
        by_route.setdefault(rxn.get("route_id", "unknown"), []).append(rxn)

    # Sort each route's steps by step_number for correct synthetic order
    for rid in by_route:
        by_route[rid].sort(key=lambda x: x.get("step_number", 0))

    print(f"[dataset] {len(by_route)} distinct routes indexed")
    return {
        "by_product":  by_product,
        "by_reactant": by_reactant,
        "by_route":    by_route,
        "all":         reactions,
        "metadata":    metadata,
    }


def get_targets_from_dataset(dataset: dict) -> dict:
    """
    Extract a target-name to canonical-SMILES mapping from a loaded dataset.

    Resolution strategy applied in priority order:

    1. Use the canonical SMILES stored in
    ``dataset["metadata"]["target_smiles"]`` if it parses correctly.
    2. Walk the route steps in reverse to find the largest product molecule
    (≥ 15 heavy atoms) as a proxy for the final target structure.

    Targets for which no valid SMILES can be determined are skipped with a
    warning.

    Parameters
    ----------
    dataset : dict
        As returned by ``load_reaction_dataset()``.

    Returns
    -------
    dict
        ``{target_name (str): canonical_SMILES (str)}``.
    """
    by_route    = dataset["by_route"]
    meta_smiles = dataset["metadata"].get("target_smiles", {})
    by_target   = {}
    for rid, steps in by_route.items():
        t = steps[0].get("target", "?")
        if t != "?":
            by_target.setdefault(t, []).append((rid, steps))

    result = {}
    for target in sorted(by_target.keys()):
        # Priority 1: explicit entry in metadata
        candidate = meta_smiles.get(target, "")
        if candidate and Chem.MolFromSmiles(candidate):
            result[target] = to_canonical(candidate)
            continue
        # Priority 2: largest product found by scanning steps in reverse
        found = None
        for rid, steps in by_target[target]:
            for step in reversed(steps):
                prod = step.get("product_smiles", "")
                mol  = Chem.MolFromSmiles(prod) if prod else None
                if mol and mol.GetNumAtoms() >= 15:
                    found = to_canonical(prod)
                    break
            if found:
                break
        if found:
            result[target] = found
        else:
            print(f"[dataset] WARNING {target} — no valid SMILES")
    return result


def load_toxicity_dataset(path: str) -> dict:
    """
    Load safety/hazard scores from JSON. Returns {canonical_SMILES: compound_dict}.
    Missing file returns {} and scores default to 0.5 (neutral).

    Parameters
    ----------
    path : str
        Path to the JSON toxicity file.

    Returns
    -------
    dict
        {canonical_SMILES: compound_dict}
    """
    if not os.path.exists(path):
        print("[toxicity] missing — Safety scores will default to 0.5 (required for correct scoring)")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    compounds = raw if isinstance(raw, list) else raw.get("compounds", [])
    index = {}
    for c in compounds:
        key = to_canonical(c.get("smiles", ""))
        if key:
            index[key] = c
    print(f"[toxicity] {len(index)} compounds indexed")
    return index


def load_rxninsight_database(path: str):
    """
    Load the Rxn-INSIGHT USPTO parquet file for condition suggestion.
    Returns None if the file is absent or rxn_insight is not installed.

    Parameters
    ----------
    path : str
        Path to the .gzip parquet file.

    Returns
    -------
    pandas.DataFrame or None
    """
    if not RXNINSIGHT_AVAILABLE:
        return None
    if not path or not os.path.exists(path):
        print("[rxn-insight] USPTO database missing — condition suggestion disabled")
        return None
    try:
        df = pd.read_parquet(path)
        # Rename legacy column name if present
        if "REACTION" in df.columns and "RXN" not in df.columns:
            df = df.rename(columns={"REACTION": "RXN"})
        # Keep only the columns needed by find_neighbors()
        cols_needed = [
            "RXN", "SOLVENT", "REAGENT", "CATALYST", "NAME", "CLASS",
            "TAG2", "rxn_str_patt_fp", "rxn_dif_patt_fp",
            "rxn_str_morgan_fp", "rxn_dif_morgan_fp",
        ]
        cols_present = [c for c in cols_needed if c in df.columns]
        df = df[cols_present]
        print(f"[rxn-insight] database loaded — {len(df)} USPTO reactions")
        return df
    except Exception as e:
        print(f"[rxn-insight] error loading database: {e}")
        return None


def load_generic_reaction_dataset(path: str) -> dict:
    """
    Load individual USPTO reactions used to validate AiZynthFinder steps.
    Returns dict with keys by_product, by_reaction_key, all.
    Returns {} if the file is missing.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    dict
    """
    if not path or not os.path.exists(path):
        print("[generic dataset] absent — step validation disabled")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    reactions = raw if isinstance(raw, list) else raw.get("reactions", [])
    print(f"[generic dataset] {len(reactions)} individual reactions loaded")

    by_product      = {}
    by_reaction_key = {}  # (tuple_of_sorted_canonical_reactants, canonical_product) → rxn

    for rxn in reactions:
        prod  = to_canonical(rxn.get("product_smiles", ""))
        reacs = tuple(sorted([to_canonical(r) for r in rxn.get("reactants_smiles", []) if r]))
        if prod:
            by_product.setdefault(prod, []).append(rxn)
        key = (reacs, prod)
        if key not in by_reaction_key:
            by_reaction_key[key] = rxn

    return {
        "by_product":      by_product,
        "by_reaction_key": by_reaction_key,
        "all":             reactions,
    }


# SMILES utilities
def to_canonical(smiles) -> str:
    """
    Convert a SMILES string to its canonical form using RDKit. Returns an empty
    """
    if isinstance(smiles, list):
        smiles = '.'.join(str(s) for s in smiles if s)
    if not smiles:
        return ""
    if not isinstance(smiles, str):
        return ""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else smiles


def safe_mol(smiles: str):
    """
    Convert a SMILES string to an RDKit Mol object. Returns None if parsing fails.
    """
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def validate_smiles_for_aizynthfinder(smiles: str) -> str:
    """
    Validate and canonicalise a SMILES string before passing it to
    AiZynthFinder.

    Parameters
    ----------
    smiles : str
        User-supplied or dataset-derived SMILES string.

    Returns
    -------
    str
        Canonical SMILES guaranteed parseable by RDKit.

    Raises
    ------
    ValueError
        If ``smiles`` is empty or cannot be parsed by RDKit.
    """
    if not smiles:
        raise ValueError("empty SMILES")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol)


def build_dataset_smiles_index(dataset: dict) -> set:
    """
    Build a set of all canonical SMILES appearing in the dataset.

    Covers both products and reactants. Used by
    ``is_route_covered_by_dataset()`` to check whether an AiZynthFinder
    route overlaps with molecules already in the curated dataset.

    Parameters
    ----------
    dataset : dict
        As returned by ``load_reaction_dataset()``.

    Returns
    -------
    set of str
        Canonical SMILES strings of all molecules in the dataset.
    """
    index = set()
    for rxn in dataset["all"]:
        p = to_canonical(rxn.get("product_smiles", ""))
        if p:
            index.add(p)
        for r in rxn.get("reactants_smiles", []):
            c = to_canonical(r)
            if c:
                index.add(c)
    return index


def _walk_reaction_tree(node: dict, steps: list) -> None:
    """
    Recursively traverse an AiZynthFinder reaction tree and collect steps.

    Traverses in retrosynthetic order (target → starting materials). The
    caller reverses the result to obtain the forward synthetic order. Each
    collected step is a dict with keys ``reactants`` (list of SMILES) and
    ``product`` (SMILES string).

    Parameters
    ----------
    node : dict
        Node from AiZynthFinder's ``tree.to_dict()`` output.
    steps : list
        Accumulator list modified in place.

    Returns
    -------
    None
    """
    if node.get("type") != "mol":
        return
    mol_smiles = node.get("smiles", "")
    for child in node.get("children", []):
        if child.get("type") != "reaction":
            continue
        reactant_nodes = [c for c in child.get("children", []) if c.get("type") == "mol"]
        steps.append({
            "reactants": [c.get("smiles", "") for c in reactant_nodes],
            "product":   mol_smiles,
        })
        for rnode in reactant_nodes:
            _walk_reaction_tree(rnode, steps)


def adapt_route(route: dict) -> dict:
    """
    Convert a raw AiZynthFinder route object into the internal route dict
    format.

    Reverses the retrosynthetic step order to produce a forward synthetic
    sequence. Also derives the list of starting materials (leaf nodes) and
    the canonical target SMILES.

    Parameters
    ----------
    route : dict
        One entry from ``finder.routes`` after ``build_routes()``.

    Returns
    -------
    dict
        Route dict with keys: ``route_id``, ``route_name``, ``steps``,
        ``starting_materials``, ``target_smiles``, ``raw``.
    """
    tree = route["reaction_tree"]
    try:
        tree_dict    = tree.to_dict()
        steps_retro  = []
        _walk_reaction_tree(tree_dict, steps_retro)
        # Reverse retrosynthetic steps to get forward synthetic order
        steps_forward = list(reversed(steps_retro))
        all_products  = {to_canonical(s["product"]) for s in steps_forward if s["product"]}
        all_reactants = {to_canonical(r) for s in steps_forward for r in s["reactants"]}
        # Leaves = reactants that are never produced in the route → starting materials
        leaves        = [r for r in all_reactants if r not in all_products]
        target_smiles = to_canonical(tree_dict.get("smiles", ""))
    except Exception as e:
        print(f"[adapt_route] error: {e}")
        steps_forward = []; leaves = []; target_smiles = ""

    return {
        "route_id":          "aiz",
        "route_name":        "AiZynthFinder Route",
        "steps":             steps_forward,
        "starting_materials": leaves,
        "target_smiles":     target_smiles,
        "raw":               route,
    }


def run_aizynthfinder(target_smiles: str, config_path: str = "config.yml",
                      n_routes: int = 25) -> list:
    """
    Run AiZynthFinder MCTS on a target molecule and return adapted routes.

    Uses the ``"zinc"`` stock and ``"uspto"`` expansion and filter policies.
    Requests more routes than ``top_n`` to maximise the chance of matching
    steps in the generic dataset during validation.

    Parameters
    ----------
    target_smiles : str
        Canonical SMILES of the synthesis target.
    config_path : str, optional
        Path to the AiZynthFinder YAML config file (default ``"config.yml"``).
    n_routes : int, optional
        Maximum number of routes to retrieve from the tree (default 25).

    Returns
    -------
    list of dict
        Each element is an adapted route dict from ``adapt_route()``.
    """
    canon = validate_smiles_for_aizynthfinder(target_smiles)
    print(f"[AiZynthFinder] target: {canon} (requesting up to {n_routes} routes)")
    finder = AiZynthFinder(configfile=config_path)
    finder.stock.select("zinc")
    finder.expansion_policy.select("uspto")
    finder.filter_policy.select("uspto")
    finder.target_smiles = canon
    finder.tree_search()
    finder.build_routes()
    routes_raw = list(finder.routes)[:n_routes]
    print(f"[AiZynthFinder] {len(routes_raw)} routes found")
    return [adapt_route(r) for r in routes_raw]


def _best_condition(df_cond) -> str:
    """
    Extract the most frequently suggested condition from a Rxn-INSIGHT
    suggestion DataFrame.

    Each DataFrame has columns ``NAME`` (condition string) and ``COUNT``
    (frequency). Empty ``NAME`` strings are filtered out before selecting
    the top row.

    Parameters
    ----------
    df_cond : pandas.DataFrame or None
        Suggestion DataFrame from Rxn-INSIGHT (e.g. ``suggested_solvent``).

    Returns
    -------
    str
        Most common non-empty condition name, or ``""`` on failure.
    """
    try:
        if df_cond is None or not hasattr(df_cond, "empty") or df_cond.empty:
            return ""
        valid = df_cond[df_cond["NAME"].str.strip() != ""]
        if valid.empty:
            return ""
        return str(valid.sort_values("COUNT", ascending=False).iloc[0]["NAME"]).strip()
    except Exception:
        return ""


def get_reaction_info_rxninsight(reactants: list, product: str, rxni_db) -> dict:
    """
    Classify a reaction and suggest conditions using Rxn-INSIGHT.

    Without ``rxni_db``: reaction type and class are returned via pattern
    matching. With ``rxni_db``: additionally calls ``find_neighbors()`` and
    ``suggest_conditions()`` for data-driven solvent/catalyst/reagent
    prediction. Returns an empty dict if Rxn-INSIGHT is unavailable or
    inputs are invalid.

    Parameters
    ----------
    reactants : list of str
        SMILES strings of the reactants.
    product : str
        SMILES string of the product.
    rxni_db : pandas.DataFrame or None
        USPTO database from ``load_rxninsight_database()``.

    Returns
    -------
    dict
        Keys: ``reaction_type``, ``reaction_class``, ``fg_reactants``,
        ``fg_products``, ``by_products``, ``conditions``.
    """
    if not RXNINSIGHT_AVAILABLE:
        return {}
    valid_r = [r for r in reactants if r and Chem.MolFromSmiles(r)]
    if not valid_r or not product or not Chem.MolFromSmiles(product):
        return {}
    try:
        rxn_smi = ".".join(valid_r) + ">>" + product
        rxn     = RxnInsightReaction(rxn_smi)
        info    = rxn.get_reaction_info()

        result = {
            "reaction_type":  info.get("NAME") or info.get("CLASS") or "unknown",
            "reaction_class": info.get("CLASS", "?"),
            "fg_reactants":   list(info.get("FG_REACTANTS", [])),
            "fg_products":    list(info.get("FG_PRODUCTS", [])),
            "by_products":    list(info.get("BY-PRODUCTS", [])),
            "conditions": {
                "temperature_C": None,
                "temp_range":    None,
                "solvent":       None,
                "co_solvent":    None,
                "reagents":      [],
                "apparatus":     None,
            },
        }

        # Condition suggestion requires the USPTO database
        if rxni_db is not None:
            try:
                df_nb = rxn.find_neighbors(
                    rxni_db, fp="MACCS", concatenate=True,
                    threshold=0.3, broaden=True, full_search=False,
                )
                if df_nb is not None and len(df_nb) > 0:
                    rxn.suggest_conditions(df_nb)
                    solv = _best_condition(rxn.suggested_solvent)
                    cata = _best_condition(rxn.suggested_catalyst)
                    reag = _best_condition(rxn.suggested_reagent)
                    result["conditions"]["solvent"]  = solv or None
                    result["conditions"]["reagents"] = [r for r in [cata, reag] if r]
            except Exception as e:
                print(f"    [rxn-insight] condition error: {e}")
        return result
    except Exception as e:
        print(f"  [rxn-insight] error: {e}")
        return {}


def enrich_aiz_route_with_rxninsight(aiz_route: dict, rxni_db, route_index: int) -> dict:
    """
    Create synthetic dataset steps for a fully predicted AiZynthFinder route.

    Each step is annotated by Rxn-INSIGHT for reaction type and suggested
    conditions. Yield is set to ``None`` and excluded from scoring.

    Parameters
    ----------
    aiz_route : dict
        Adapted AiZ route from ``adapt_route()``.
    rxni_db : pandas.DataFrame or None
        USPTO database for condition prediction.
    route_index : int
        1-based index used to build unique step IDs.

    Returns
    -------
    dict
        Enriched route dict with added keys: ``dataset_steps``,
        ``matched_route_id``, ``matched_route_name``, ``matched_target``,
        ``coverage``, ``is_predicted``.
    """
    steps         = aiz_route.get("steps", [])
    target_smiles = aiz_route.get("target_smiles", "")
    dataset_steps = []

    for i, step in enumerate(steps, 1):
        reactants = step.get("reactants", [])
        product   = step.get("product", "")
        rxni_info = get_reaction_info_rxninsight(reactants, product, rxni_db)
        dataset_steps.append({
            "id":               f"PRED-{route_index:02d}-{i:02d}",
            "route_id":         f"predicted_{route_index:02d}",
            "step_number":      i,
            "reactants_smiles": reactants,
            "product_smiles":   product,
            "yield_percent":    None,   # yield excluded from scoring for predicted routes
            "source":           "rxn-insight",
            **rxni_info,
        })

    # Ensure the last step's product is the canonical target SMILES
    if dataset_steps and target_smiles:
        dataset_steps[-1]["product_smiles"] = target_smiles

    enriched = dict(aiz_route)
    enriched.update({
        "dataset_steps":      dataset_steps,
        "matched_route_id":   f"predicted_{route_index:02d}",
        "matched_route_name": f"Predicted route #{route_index}",
        "matched_target":     "?",
        "coverage":           0,
        "is_predicted":       True,
    })
    return enriched


def is_route_covered_by_dataset(aiz_route: dict, dataset_smiles_index: set,
                                 threshold: float = 0.4) -> bool:
    """
    Determine whether an AiZ route is substantially covered by the main
    dataset.

    A route is considered covered if at least ``threshold`` fraction of its
    product SMILES appear in the main dataset's SMILES index.

    Parameters
    ----------
    aiz_route : dict
        Adapted AiZ route from ``adapt_route()``.
    dataset_smiles_index : set
        Built by ``build_dataset_smiles_index()``.
    threshold : float, optional
        Minimum fraction of products that must appear in the dataset
        (default 0.4).

    Returns
    -------
    bool
        ``True`` if the route should be skipped as already covered.
    """
    steps    = aiz_route.get("steps", [])
    products = [to_canonical(s.get("product", "")) for s in steps if s.get("product")]
    if not products:
        return False
    return sum(1 for p in products if p in dataset_smiles_index) / len(products) >= threshold


# Generic dataset step matching
def match_step_in_generic_dataset(reactants: list, product: str, generic_ds: dict):
    """
    Search for a reaction step in the generic reactions dataset.

    Uses a two-level strategy: exact match on (sorted canonical reactants,
    canonical product), then product-only match as a fallback.

    Parameters
    ----------
    reactants : list of str
        SMILES strings of the AiZ-proposed precursors.
    product : str
        SMILES string of the step product.
    generic_ds : dict
        As returned by ``load_generic_reaction_dataset()``.

    Returns
    -------
    dict or None
        Matching reaction entry, or ``None`` if no match is found.
    """
    if not generic_ds:
        return None
    prod_canon  = to_canonical(product)
    reac_canons = tuple(sorted([to_canonical(r) for r in reactants if r]))
    # Level 1: exact key match
    exact = generic_ds.get("by_reaction_key", {}).get((reac_canons, prod_canon))
    if exact:
        return exact
    # Level 2: product-only match
    by_prod = generic_ds.get("by_product", {}).get(prod_canon, [])
    return by_prod[0] if by_prod else None


def validate_aiz_route_against_generic_dataset(aiz_route: dict, generic_ds: dict,
                                                rxni_db, route_index: int) -> dict:
    """
    Validate each step of an AiZynthFinder route against the generic dataset.

    Matched steps use real conditions and yield; unmatched steps receive
    Rxn-INSIGHT predicted conditions. Assigns one of three validation
    statuses: ``"validated"`` (all steps matched), ``"partial"`` (some),
    or ``"predicted"`` (none).

    Parameters
    ----------
    aiz_route : dict
        Adapted AiZ route from ``adapt_route()``.
    generic_ds : dict
        As returned by ``load_generic_reaction_dataset()``.
    rxni_db : pandas.DataFrame or None
        USPTO database for condition prediction fallback.
    route_index : int
        1-based index for unique ID generation.

    Returns
    -------
    dict
        Enriched route dict with added keys: ``dataset_steps``,
        ``matched_route_id``, ``matched_route_name``, ``matched_target``,
        ``coverage``, ``validation_status``, ``validated_steps_count``,
        ``total_steps_count``, ``is_predicted``, ``is_validated``.
    """
    steps         = aiz_route.get("steps", [])
    target_smiles = aiz_route.get("target_smiles", "")
    dataset_steps   = []
    validated_count = 0

    for i, step in enumerate(steps, 1):
        reactants = step.get("reactants", [])
        product   = step.get("product", "")
        match     = match_step_in_generic_dataset(reactants, product, generic_ds)

        if match:
            validated_count += 1
            dataset_steps.append({
                "id":               f"VAL-{route_index:02d}-{i:02d}",
                "route_id":         f"validated_{route_index:02d}",
                "step_number":      i,
                "reactants_smiles": reactants,
                "product_smiles":   product,
                "yield_percent":    match.get("yield_percent"),
                "reaction_type":    match.get("reaction_type", ""),
                "reaction_class":   match.get("reaction_class", ""),
                "fg_reactants":     match.get("fg_reactants", []),
                "by_products":      match.get("by_products", []),
                "conditions":       match.get("conditions", {
                    "temperature_C": None, "temp_range": None,
                    "solvent":       None, "co_solvent": None,
                    "reagents":      [], "apparatus": None,
                }),
                "source": "generic_dataset",  # real experimental conditions
            })
        else:
            # Fall back to Rxn-INSIGHT for condition prediction
            rxni_info = get_reaction_info_rxninsight(reactants, product, rxni_db)
            dataset_steps.append({
                "id":               f"VAL-{route_index:02d}-{i:02d}",
                "route_id":         f"validated_{route_index:02d}",
                "step_number":      i,
                "reactants_smiles": reactants,
                "product_smiles":   product,
                "yield_percent":    None,
                "source":           "rxn-insight",
                **rxni_info,
            })

    # Ensure the last step product equals the canonical target
    if dataset_steps and target_smiles:
        dataset_steps[-1]["product_smiles"] = target_smiles

    n = len(steps)
    if   n == 0:                 status = "predicted"
    elif validated_count == n:   status = "validated"
    elif validated_count > 0:    status = "partial"
    else:                        status = "predicted"

    enriched = dict(aiz_route)
    enriched.update({
        "dataset_steps":         dataset_steps,
        "matched_route_id":      f"validated_{route_index:02d}",
        "matched_route_name":    f"AiZ route #{route_index} ({validated_count}/{n} validated)",
        "matched_target":        aiz_route.get("matched_target", "?"),
        "coverage":              validated_count,
        "validation_status":     status,
        "validated_steps_count": validated_count,
        "total_steps_count":     n,
        "is_predicted":          status == "predicted",
        "is_validated":          status in ("validated", "partial"),
    })
    return enriched


def process_novel_routes(aiz_routes: list, dataset: dict, generic_ds: dict,
                          target_name: str, rxni_db) -> tuple:
    """
    Process AiZynthFinder routes not covered by the main dataset.

    Validates each route step-by-step against the generic dataset and
    classifies routes as validated, partial, or predicted. Routes already
    covered by the main dataset are silently skipped.

    Parameters
    ----------
    aiz_routes : list of dict
        Adapted AiZ routes from ``run_aizynthfinder()``.
    dataset : dict
        As returned by ``load_reaction_dataset()``.
    generic_ds : dict
        As returned by ``load_generic_reaction_dataset()``.
    target_name : str
        Human-readable target name used to label routes.
    rxni_db : pandas.DataFrame or None
        USPTO database for condition prediction.

    Returns
    -------
    tuple of (list, list)
        ``(validated_routes, predicted_routes)`` — both lists of enriched
        route dicts.
    """
    if not RXNINSIGHT_AVAILABLE:
        print("[rxn-insight] not installed")
        return [], []

    dataset_smiles_index = build_dataset_smiles_index(dataset)
    validated_routes = []
    predicted_routes = []
    counter = 1

    for aiz_route in aiz_routes:
        if not aiz_route.get("steps"):
            continue
        if is_route_covered_by_dataset(aiz_route, dataset_smiles_index):
            print(f"  [novel] covered by main dataset — skipped")
            continue

        print(f"  [novel] route #{counter} — validating against generic dataset...")
        enriched = validate_aiz_route_against_generic_dataset(
            aiz_route, generic_ds, rxni_db, counter)
        enriched["matched_target"] = target_name
        status = enriched.get("validation_status", "predicted")
        v, t   = enriched.get("validated_steps_count", 0), enriched.get("total_steps_count", 0)

        if status == "validated":
            # 100% steps matched — goes to validated section
            validated_routes.append(enriched)
            print(f"    → validated ({v}/{t} steps in generic dataset)")
        elif status == "partial":
            # Some steps matched — goes to predicted with validation info preserved
            enriched["is_predicted"] = True
            predicted_routes.append(enriched)
            print(f"    → partial ({v}/{t} steps validated) — shown in predicted with badge")
        else:
            pure = enrich_aiz_route_with_rxninsight(aiz_route, rxni_db, counter)
            pure["matched_target"] = target_name
            pure.update({
                "validation_status":     "predicted",
                "is_validated":          False,
                "validated_steps_count": 0,
                "total_steps_count":     t,
            })
            predicted_routes.append(pure)
            print(f"    → predicted (no steps in generic dataset)")
        counter += 1

    print(f"  {len(validated_routes)} validated/partial, {len(predicted_routes)} predicted")
    return validated_routes, predicted_routes


def get_novel_routes_from_aizynthfinder(aiz_routes, dataset, target_name, rxni_db) -> list:
    """
    Legacy helper that enriches novel AiZ routes with Rxn-INSIGHT only,
    without generic dataset validation.

    Kept for backward compatibility. Prefer ``process_novel_routes()``
    which also cross-validates steps against the generic dataset.

    Parameters
    ----------
    aiz_routes : list of dict
        Adapted AiZ routes from ``run_aizynthfinder()``.
    dataset : dict
        As returned by ``load_reaction_dataset()``.
    target_name : str
        Human-readable target name.
    rxni_db : pandas.DataFrame or None
        USPTO database for condition prediction.

    Returns
    -------
    list of dict
        Enriched route dicts, all classified as ``"predicted"``.
    """
    if not RXNINSIGHT_AVAILABLE:
        print("[rxn-insight] not installed")
        return []
    dataset_smiles_index = build_dataset_smiles_index(dataset)
    novel   = []
    counter = 1
    for route in aiz_routes:
        if not route.get("steps"):
            continue
        if is_route_covered_by_dataset(route, dataset_smiles_index):
            print(f"  [novel] covered by dataset — skipped")
            continue
        print(f"  [novel] route #{counter} — Rxn-INSIGHT enrichment...")
        enriched = enrich_aiz_route_with_rxninsight(route, rxni_db, counter)
        enriched["matched_target"] = target_name
        novel.append(enriched)
        counter += 1
    print(f"  {len(novel)} novel routes predicted")
    return novel


def get_all_dataset_routes_for_target(dataset: dict, target_name: str,
                                       target_smiles: str = "") -> list:
    """
    Return all routes whose target name and final product match the query.

    Both the target name (case-insensitive) and the canonical SMILES of the
    last step's product must match when ``target_smiles`` is provided. Used
    as a fallback in ``filter_routes_by_starting_materials()`` to ensure
    known dataset routes are always included.

    Parameters
    ----------
    dataset : dict
        As returned by ``load_reaction_dataset()``.
    target_name : str
        Human-readable target name (case-insensitive).
    target_smiles : str, optional
        Canonical SMILES of the search target. When provided, routes whose
        last product does not match are excluded.

    Returns
    -------
    list of dict
        Route dicts with ``dataset_steps`` and ``matched_*`` fields set.
    """
    canon_target = to_canonical(target_smiles) if target_smiles else ""
    result = []
    for rid, steps in dataset["by_route"].items():
        # Name filter (always applied)
        if steps[0].get("target", "").lower() != target_name.lower():
            continue
        # Canonical product filter (applied when target_smiles is provided)
        if canon_target:
            last_product = to_canonical(steps[-1].get("product_smiles", ""))
            if last_product != canon_target:
                print(f"  [dataset] fallback skipped {rid} — "
                      f"product {last_product!r} ≠ target {canon_target!r}")
                continue
        result.append({
            "route_id":           rid,
            "route_name":         steps[0].get("route_name", rid),
            "steps":              [],
            "starting_materials": [],
            "dataset_steps":      steps,
            "matched_route_id":   rid,
            "matched_route_name": steps[0].get("route_name", rid),
            "matched_target":     steps[0].get("target", "?"),
            "coverage":           0,
            "is_predicted":       False,
            "is_validated":       False,
            "validation_status":  "dataset",
        })
        print(f"  [dataset] OK {rid} — {len(steps)} steps")
    return result

def filter_routes_by_starting_materials(aiz_routes, dataset, target_smiles,
                                          target_name="") -> list:
    """
    Return all dataset routes whose final product matches the target SMILES.

    Falls back to target-name matching when canonical SMILES comparison
    fails. All matching routes are returned; ``top_n`` truncation happens
    later in ``find_best_routes()`` after scoring.

    Parameters
    ----------
    aiz_routes : list
        Kept for API compatibility — no longer used internally.
    dataset : dict
        As returned by ``load_reaction_dataset()``.
    target_smiles : str
        Canonical SMILES of the search target.
    target_name : str, optional
        Human-readable target name used as a secondary filter.

    Returns
    -------
    list of dict
        Enriched route dicts ready for ``rank_weighted()``.
    """
    canon_target = to_canonical(target_smiles) if target_smiles else ""
    by_route     = dataset["by_route"]
    result       = []

    for rid, steps in by_route.items():
        if not steps:
            continue

        matched = False

        # Primary: canonical SMILES of the last step's product == target
        if canon_target:
            last_product = to_canonical(steps[-1].get("product_smiles", ""))
            if last_product == canon_target:
                matched = True

        # Secondary fallback: match by target name when SMILES comparison fails
        if not matched and target_name:
            route_target = steps[0].get("target", "")
            if route_target.lower() == target_name.lower():
                # Only accept if no canon_target was provided, OR if
                # the last product actually parses (avoids broken entries)
                if not canon_target or to_canonical(steps[-1].get("product_smiles", "")):
                    matched = True

        if not matched:
            continue

        result.append({
            "route_id":           rid,
            "route_name":         steps[0].get("route_name", rid),
            "steps":              [],
            "starting_materials": [],
            "dataset_steps":      steps,
            "matched_route_id":   rid,
            "matched_route_name": steps[0].get("route_name", rid),
            "matched_target":     steps[0].get("target", "?"),
            "coverage":           len(steps),
            "is_predicted":       False,
            "is_validated":       False,
            "validation_status":  "dataset",
        })
        print(f"  [dataset] matched {rid} — {len(steps)} steps")

    print(f"  {len(result)} dataset routes retained")
    return result

def bottleneck_yield(steps: list) -> float | None:
    """
    Return the bottleneck step yield (lowest reported yield) in percent.
    """
    ys = [s.get("yield_percent") for s in steps if s.get("yield_percent") is not None]
    return min(ys) if ys else None


def average_yield(steps: list) -> float | None:
    """
    Return the average step yield in percent, ignoring steps with missing
    """
    ys = [s.get("yield_percent") for s in steps if s.get("yield_percent") is not None]
    return sum(ys) / len(ys) if ys else None


def cumulative_yield(steps: list) -> float:
    """
    Return the overall cumulative yield as a fraction in [0, 1].

    Computed as the product of all reported step yields. Missing yields are
    treated as 1.0 (neutral) to avoid penalising routes with incomplete data.

    Parameters
    ----------
    steps : list of dict
        Step dicts from ``route["dataset_steps"]``.

    Returns
    -------
    float
        Cumulative yield in [0, 1].
    """
    r = 1.0
    for s in steps:
        y = s.get("yield_percent")
        r *= (y / 100.0) if y is not None else 1.0
    return r


def get_substances_list(steps_data: list) -> dict:
    """
    Extract categorised substance lists from a route's dataset steps.

    Parameters
    ----------
    steps_data : list of dict
        Step dicts from ``route["dataset_steps"]``.

    Returns
    -------
    dict
        Keys ``to_buy`` (starting materials), ``to_prepare``
        (intermediates), ``solvents``, and ``reagents`` — all sorted lists.
    """
    all_prod = {to_canonical(s.get("product_smiles", "")) for s in steps_data}
    all_reac = {to_canonical(r) for s in steps_data for r in s.get("reactants_smiles", [])}
    solvents = set(); reagents = set()
    for s in steps_data:
        cond = s.get("conditions", {})
        if cond.get("solvent"):    solvents.add(cond["solvent"])
        if cond.get("co_solvent"): solvents.add(cond["co_solvent"])
        for r in (cond.get("reagents") or []):
            if r: reagents.add(r)
    return {
        "to_buy":     sorted(all_reac - all_prod - {""}),
        "to_prepare": sorted(all_prod - {""}),
        "solvents":   sorted(solvents - {""}),
        "reagents":   sorted(reagents - {""}),
    }


def fmt_conditions(cond: dict) -> str:
    """
    Format a conditions dict into a human-readable string.

    Fields included in order: temperature, solvent, co-solvent, reagents,
    apparatus. Empty and ``None`` fields are silently skipped.

    Parameters
    ----------
    cond : dict
        Conditions dict with optional keys ``temperature_C``,
        ``temp_range``, ``solvent``, ``co_solvent``, ``reagents``,
        ``apparatus``.

    Returns
    -------
    str
        ``'  ·  '``-separated conditions string, or ``""`` if ``cond``
        is empty.
    """
    if not cond:
        return ""
    parts = []
    if cond.get("temperature_C"):   parts.append(f"{cond['temperature_C']}°C")
    elif cond.get("temp_range"):    parts.append(cond["temp_range"])
    if cond.get("solvent"):
        solv = cond["solvent"]
        mol  = Chem.MolFromSmiles(solv) if True else None
        parts.append(solv if mol is None else solv)
    if cond.get("co_solvent"):      parts.append(f"/ {cond['co_solvent']}")
    reag = cond.get("reagents", [])
    if isinstance(reag, list) and reag:
        parts.append(", ".join(reag))
    if cond.get("apparatus"):       parts.append(f"({cond['apparatus']})")
    return "  ·  ".join(parts)


def calc_atom_economy(reactants_smiles, product_smiles) -> float:
    """
    Calculate atom economy for a single reaction step.

    Atom economy = MW(product) / sum(MW(reactants)), capped at 1.0.

    Parameters
    ----------
    reactants_smiles : list of str
        SMILES strings of the reactants.
    product_smiles : str
        SMILES string of the product.

    Returns
    -------
    float
        Atom economy score in [0, 1]. Returns 0.0 if the product is invalid.
    """
    prod_mol = safe_mol(product_smiles)
    if not prod_mol:
        return 0.0
    prod_mw  = Descriptors.MolWt(prod_mol)
    total_mw = sum(Descriptors.MolWt(m) for s in reactants_smiles if (m := safe_mol(s)))
    return min(prod_mw / total_mw, 1.0) if total_mw else 0.0


def calc_e_factor(reactants_smiles, product_smiles, yield_fraction) -> float:
    """
    Calculate the E-factor score for a single reaction step.

    Score = 1 / (1 + E-factor), where E-factor = kg waste / kg product.
    Missing yield is treated as 1.0 to avoid penalising unreported steps.

    Parameters
    ----------
    reactants_smiles : list of str
        SMILES strings of the reactants.
    product_smiles : str
        SMILES string of the product.
    yield_fraction : float
        Step yield as a fraction in (0, 1].

    Returns
    -------
    float
        E-factor score in (0, 1].
    """
    prod_mol = safe_mol(product_smiles)
    if not prod_mol:
        return 0.5
    prod_mw  = Descriptors.MolWt(prod_mol)
    total_mw = sum(Descriptors.MolWt(m) for s in reactants_smiles if (m := safe_mol(s)))
    obtained = prod_mw * max(yield_fraction, 0.01)
    waste    = max(total_mw - obtained, 0)
    return 1.0 / (1.0 + waste / obtained)


def calc_toxicity_score(reactants_smiles, conditions, tox_index, solvent_map) -> float:
    """
    Calculate the safety score for a single reaction step.

    Score = 1 − mean(hazard_score) across all reactants and solvents.
    Compounds absent from ``tox_index`` are assigned a hazard of 0.5.

    Parameters
    ----------
    reactants_smiles : list of str
        SMILES strings of the reactants.
    conditions : dict
        Conditions dict (may contain ``"solvent"`` and ``"co_solvent"``).
    tox_index : dict
        As returned by ``load_toxicity_dataset()``.
    solvent_map : dict
        As returned by ``build_solvent_map()``.

    Returns
    -------
    float
        Safety score in [0, 1].
    """
    to_check = list(reactants_smiles)
    for key in ("solvent", "co_solvent"):
        s = conditions.get(key)
        if s and (smi := solvent_map.get(s)):
            to_check.append(smi)
    scores = [
        tox_index[to_canonical(s)]["hazard_score"]
        if to_canonical(s) in tox_index else 0.5
        for s in to_check
    ]
    return 1.0 - (sum(scores) / len(scores) if scores else 0.5)


def build_solvent_map(tox_index: dict) -> dict:
    """
    Build a mapping from common solvent abbreviations and names to SMILES.

    Used by ``calc_toxicity_score()`` to resolve solvent names in conditions
    dicts to canonical SMILES for toxicity index lookup. The hard-coded
    abbreviation table is augmented with all keys already in ``tox_index``.

    Parameters
    ----------
    tox_index : dict
        As returned by ``load_toxicity_dataset()``.

    Returns
    -------
    dict
        ``{abbreviation_or_name (str): SMILES (str)}``.
    """
    abbrev = {
        "PhMe": "Cc1ccccc1", 
        "toluene": "Cc1ccccc1", # same as PhMe
        "DCM": "ClCCl", 
        "CH2Cl2": "ClCCl",
        "THF": "C1CCOC1",
        "MeOH": "CO", 
        "EtOH": "CCO",
        "DMF": "CN(C)C=O", 
        "MeCN": "CC#N",
        "AcOH": "CC(=O)O", 
        "Et2O": "CCOCC",
        "CHCl3": "ClC(Cl)Cl", 
        "CCl4": "ClC(Cl)(Cl)Cl",
        "PhH": "c1ccccc1", 
        "benzene": "c1ccccc1",
        "TFE": "OCC(F)(F)F", 
        "t-BuOH": "CC(C)(C)O",
        "MeNO2": "C[N+](=O)[O-]",
        "H2O": "O",
        "dioxane": "C1COCCO1", 
        "1,4-dioxane": "C1COCCO1",
        "hexane": "CCCCCC", 
        "acetone": "CC(C)=O",
        "pyridine": "c1ccncc1",
        "i-PrOH": "CC(C)O", 
        "DMSO": "CS(C)=O",
    }
    result = dict(abbrev)
    # Add all tox_index keys so known SMILES are always self-mapping
    for canon in tox_index:
        result[canon] = canon
    return result

def compute_steps(route_data: dict, tox_index: dict) -> float:
    """
    Compute the steps criterion score for a route.

    Score = 1 / number_of_steps. Fewer steps gives a score closer to 1.

    Parameters
    ----------
    route_data : dict
        Enriched route dict with a ``"dataset_steps"`` key.
    tox_index : dict
        Unused — kept for uniform function signature across all criteria.

    Returns
    -------
    float
        Steps score in (0, 1].
    """
    return 1.0 / max(len(route_data.get("dataset_steps", [])), 1)


def compute_yield(route_data: dict, tox_index: dict) -> float:
    """
    Compute the yield criterion score for a route.

    For predicted routes returns 1.0 (neutral placeholder; yield is always
    excluded from scoring by ``rank_weighted()``). For all other routes,
    returns the product of all reported step yields (missing yields treated
    as 1.0).

    Parameters
    ----------
    route_data : dict
        Enriched route dict.
    tox_index : dict
        Unused — kept for uniform function signature.

    Returns
    -------
    float
        Cumulative yield score as a fraction in [0, 1].
    """
    steps  = route_data.get("dataset_steps", [])
    status = route_data.get("validation_status", "dataset")
    if not steps:
        return 0.0
    if status == "predicted":
        return 1.0  # neutral value; never used in final score for predicted routes
    result = 1.0
    for s in steps:
        y   = s.get("yield_percent")
        src = s.get("source", "dataset")
        if y is not None and src in ("generic_dataset", "dataset"):
            result *= y / 100.0
        else:
            result *= 1.0   # missing yield → neutral (does not penalise route)
    return result


def compute_atom_economy(route_data: dict, tox_index: dict) -> float:
    """
    Compute the atom economy criterion score averaged across all steps.

    Parameters
    ----------
    route_data : dict
        Enriched route dict.
    tox_index : dict
        Unused — kept for uniform function signature.

    Returns
    -------
    float
        Mean atom economy score in [0, 1]. Returns 0.0 for empty routes.
    """
    steps  = route_data.get("dataset_steps", [])
    scores = [
        calc_atom_economy(s.get("reactants_smiles", []), s.get("product_smiles", ""))
        for s in steps
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_e_factor(route_data: dict, tox_index: dict) -> float:
    """
    Compute the E-factor criterion score averaged across all steps.

    Missing yields are treated as 1.0 (100 %) to avoid penalising steps
    with no reported data.

    Parameters
    ----------
    route_data : dict
        Enriched route dict.
    tox_index : dict
        Unused — kept for uniform function signature.

    Returns
    -------
    float
        Mean E-factor score in (0, 1]. Returns 0.0 for empty routes.
    """
    steps  = route_data.get("dataset_steps", [])
    scores = [
        calc_e_factor(
            s.get("reactants_smiles", []),
            s.get("product_smiles", ""),
            (s.get("yield_percent") or 100) / 100.0,
        )
        for s in steps
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_toxicity(route_data: dict, tox_index: dict) -> float:
    """
    Compute the safety criterion score averaged across all steps.

    Parameters
    ----------
    route_data : dict
        Enriched route dict.
    tox_index : dict
        As returned by ``load_toxicity_dataset()``.

    Returns
    -------
    float
        Mean safety score in [0, 1]. Returns 0.5 for empty routes.
    """
    solvent_map = build_solvent_map(tox_index)
    steps  = route_data.get("dataset_steps", [])
    scores = [
        calc_toxicity_score(
            s.get("reactants_smiles", []),
            s.get("conditions", {}),
            tox_index,
            solvent_map,
        )
        for s in steps
    ]
    return sum(scores) / len(scores) if scores else 0.5


# scoring functions, one per criterion
CRITERIA_REGISTRY = {
    "steps":        {"fn": compute_steps,        "description": "number of steps"},
    "yield":        {"fn": compute_yield,        "description": "cumulative yield"},
    "atom_economy": {"fn": compute_atom_economy, "description": "atom economy"},
    "e_factor":     {"fn": compute_e_factor,     "description": "e-factor"},
    "toxicity":     {"fn": compute_toxicity,     "description": "safety score"},
}


def compute_weights(criteria: list) -> dict:
    """Inverse-square weights (1/i²), normalised to sum 1."""
    raw   = {c: 1.0 / (i + 1) ** 2 for i, c in enumerate(criteria)}
    total = sum(raw.values())
    return {c: w / total for c, w in raw.items()}


def compute_all_scores(route: dict, tox_index: dict) -> dict:
    """
    Compute raw scores for all criteria in CRITERIA_REGISTRY on one route.

    Used by the Analysis tab to show scores for criteria not selected as
    primary ranking criteria.

    Parameters
    ----------
    route : dict
        Enriched route dict.
    tox_index : dict
        As returned by ``load_toxicity_dataset()``.

    Returns
    -------
    dict
        ``{criterion_key (str): raw_score (float)}``, rounded to 4 d.p.
    """
    return {
        c: round(CRITERIA_REGISTRY[c]["fn"](route, tox_index), 4)
        for c in CRITERIA_REGISTRY
    }


def rank_weighted(routes: list, criteria: list, tox_index: dict) -> list:
    """
    Score and rank a list of routes using weighted criterion scores.

    For purely predicted routes the yield criterion is excluded and the
    remaining criteria are re-weighted to sum to 1.0. All other routes
    use the full 1/i² weight scheme.

    Parameters
    ----------
    routes : list of dict
        Enriched route dicts to score.
    criteria : list of str
        Three criterion keys in priority order.
    tox_index : dict
        As returned by ``load_toxicity_dataset()``.

    Returns
    -------
    list of tuple
        ``(total_score, details_dict, route_dict)`` tuples sorted in
        descending order of ``total_score``. Each ``details_dict`` contains
        per-criterion keys ``raw``, ``weight``, ``weighted``, ``excluded``,
        and ``_all_scores``.
    """
    weights = compute_weights(criteria)
    scored  = []

    for route in routes:
        status = route.get("validation_status", "dataset")
        # Yield is only excluded for purely predicted routes
        exclude_yield = (status == "predicted")

        details = {}
        total   = 0.0

        # Re-weight if yield is excluded so remaining weights sum to 1
        active_criteria = [c for c in criteria if not (exclude_yield and c == "yield")]
        active_weights  = (
            compute_weights(active_criteria) if len(active_criteria) < len(criteria)
            else weights
        )

        for c in criteria:
            raw = CRITERIA_REGISTRY[c]["fn"](route, tox_index)
            if exclude_yield and c == "yield":
                details[c] = {"raw": None, "weight": 0.0, "weighted": 0.0, "excluded": True}
            else:
                w = active_weights.get(c, weights[c])
                details[c] = {
                    "raw":      round(raw, 4),
                    "weight":   round(w, 4),
                    "weighted": round(raw * w, 4),
                    "excluded": False,
                }
                total += raw * w

        # Store all-criteria scores for the Analysis tab comparison table
        details["_all_scores"] = compute_all_scores(route, tox_index)
        scored.append((round(total, 4), details, route))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored


def find_best_routes(
    target_smiles:        str,
    criteria_priority:    list,
    dataset_path:         str  = "reaction_dataset.json",
    toxicity_path:        str  = "toxicity_dataset.json",
    config_path:          str  = "config.yml",
    top_n:                int  = 3,
    target_name:          str  = "",
    include_predicted:    bool = True,
    rxninsight_db_path:   str  = "",
    generic_dataset_path: str  = "",
    n_aiz_routes:         int  = 25,
) -> dict:
    """
    Find and rank synthesis routes for a target molecule.

    Runs a four-stage pipeline: load datasets, run AiZynthFinder MCTS,
    classify routes into dataset / validated / predicted sections, then
    score and rank each section independently.

    Parameters
    ----------
    target_smiles : str
        Canonical or parseable SMILES of the synthesis target.
    criteria_priority : list of str
        Exactly 3 criterion keys in descending priority order.
    dataset_path : str, optional
        Path to ``reaction_dataset.json`` (default ``"reaction_dataset.json"``).
    toxicity_path : str, optional
        Path to ``toxicity_dataset.json`` (default ``"toxicity_dataset.json"``).
    config_path : str, optional
        Path to AiZynthFinder ``config.yml`` (default ``"config.yml"``).
    top_n : int, optional
        Maximum routes shown per section (default 3).
    target_name : str, optional
        Human-readable target name for display and filtering.
    include_predicted : bool, optional
        Whether to process predicted routes (default ``True``).
    rxninsight_db_path : str, optional
        Path to the Rxn-INSIGHT USPTO Parquet file.
    generic_dataset_path : str, optional
        Path to ``generic_reactions.json``.
    n_aiz_routes : int, optional
        Number of AiZynthFinder MCTS routes to retrieve (default 25).

    Returns
    -------
    dict
        Keys ``"dataset"``, ``"validated"``, ``"predicted"`` — each a list
        of ``(score, details_dict, route_dict)`` tuples.

    Raises
    ------
    ValueError
        If ``criteria_priority`` does not contain exactly 3 known criteria.
    FileNotFoundError
        If ``dataset_path`` or ``config_path`` does not exist.
    """
    if len(criteria_priority) != 3:
        raise ValueError(f"exactly 3 criteria needed, got {len(criteria_priority)}")
    unknown = [c for c in criteria_priority if c not in CRITERIA_REGISTRY]
    if unknown:
        raise ValueError(f"unknown criteria: {unknown}")

    print(f"\n[path_finder] target: {target_smiles}")
    if target_name:
        print(f"[pathfinder] name: {target_name}")
    print(f"[pathfinder] criteria: {criteria_priority}")

    # Stage 1 — Load all data sources
    print("\n[1/4] loading datasets...")
    dataset    = load_reaction_dataset(dataset_path)
    tox_index  = load_toxicity_dataset(toxicity_path)
    generic_ds = load_generic_reaction_dataset(generic_dataset_path)

    rxni_db = None
    if include_predicted and RXNINSIGHT_AVAILABLE:
        rxni_db = load_rxninsight_database(rxninsight_db_path)

    # Stage 2 — AiZynthFinder MCTS search
    print(f"\n[2/4] AiZynthFinder search (up to {n_aiz_routes} routes)...")
    all_aiz = run_aizynthfinder(target_smiles, config_path, n_routes=n_aiz_routes)
    print(f"  {len(all_aiz)} AiZ routes retrieved")

    # Stage 3 — Dataset route matching
    print("\n[3/4] dataset routes...")
    dataset_routes = filter_routes_by_starting_materials(
        all_aiz, dataset, target_smiles, target_name)

    validated_routes = []
    predicted_routes = []
    if include_predicted and RXNINSIGHT_AVAILABLE and all_aiz:
        print("\n[3b] processing novel routes (generic dataset + Rxn-INSIGHT)...")
        validated_routes, predicted_routes = process_novel_routes(
            all_aiz, dataset, generic_ds, target_name, rxni_db)

    # Stage 4 — Score and rank
    print("\n[4/4] scoring...")
    scored_dataset   = rank_weighted(dataset_routes,   criteria_priority, tox_index)[:top_n]
    scored_validated = rank_weighted(validated_routes, criteria_priority, tox_index)[:top_n]
    scored_predicted = rank_weighted(predicted_routes, criteria_priority, tox_index)[:top_n]

    # Console summary for debugging
    print(f"\n[pathfinder] {len(scored_dataset)} dataset / "
          f"{len(scored_validated)} validated / {len(scored_predicted)} predicted")
    for i, (sc, _, r) in enumerate(scored_dataset, 1):
        print(f"  [dataset]   {i}. {r.get('matched_route_name','?')} — score {sc:.4f}")
    for i, (sc, _, r) in enumerate(scored_validated, 1):
        v = r.get("validated_steps_count", 0); t = r.get("total_steps_count", 0)
        print(f"  [validated] {i}. {r.get('matched_route_name','?')} ({v}/{t}) — score {sc:.4f}")
    for i, (sc, _, r) in enumerate(scored_predicted, 1):
        print(f"  [predicted] {i}. {r.get('matched_route_name','?')} — score {sc:.4f}")

    return {
        "dataset":   scored_dataset,
        "validated": scored_validated,
        "predicted": scored_predicted,
    }