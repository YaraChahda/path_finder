# function.py
# pathfinder : dataset + AiZynthFinder + Rxn-INSIGHT

import json
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from aizynthfinder.aizynthfinder import AiZynthFinder

try:
    from rxn_insight.reaction import Reaction as RxnInsightReaction
    RXNINSIGHT_AVAILABLE = True
except ImportError:
    RXNINSIGHT_AVAILABLE = False


# =============================================================================
# data loading
# =============================================================================

def load_reaction_dataset(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        reactions = raw
        metadata  = {}
    elif isinstance(raw, dict) and "reactions" in raw:
        reactions = raw["reactions"]
        metadata  = raw.get("_metadata", {})
    else:
        raise ValueError("unrecognized dataset format")
    print(f"[dataset] {len(reactions)} reactions loaded")

    by_product = {}; by_reactant = {}; by_route = {}
    for rxn in reactions:
        prod = rxn.get("product_smiles", "")
        if isinstance(prod, list):
            rxn["product_smiles"] = '.'.join(str(s) for s in prod if s)

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
        pk = to_canonical(rxn.get("product_smiles", ""))
        if pk:
            by_product.setdefault(pk, []).append(rxn)
        for r in rxn.get("reactants_smiles", []):
            rk = to_canonical(r)
            if rk:
                by_reactant.setdefault(rk, []).append(rxn)
        by_route.setdefault(rxn.get("route_id", "unknown"), []).append(rxn)
    for rid in by_route:
        by_route[rid].sort(key=lambda x: x.get("step_number", 0))
    print(f"[dataset] {len(by_route)} distinct routes indexed")
    return {"by_product": by_product, "by_reactant": by_reactant,
            "by_route": by_route, "all": reactions, "metadata": metadata}


def get_targets_from_dataset(dataset: dict) -> dict:
    by_route    = dataset["by_route"]
    meta_smiles = dataset["metadata"].get("target_smiles", {})
    by_target   = {}
    for rid, steps in by_route.items():
        t = steps[0].get("target", "?")
        if t != "?":
            by_target.setdefault(t, []).append((rid, steps))
    result = {}
    for target in sorted(by_target.keys()):
        candidate = meta_smiles.get(target, "")
        if candidate and Chem.MolFromSmiles(candidate):
            result[target] = to_canonical(candidate); continue
        found = None
        for rid, steps in by_target[target]:
            for step in reversed(steps):
                prod = step.get("product_smiles", "")
                mol  = Chem.MolFromSmiles(prod) if prod else None
                if mol and mol.GetNumAtoms() >= 15:
                    found = to_canonical(prod); break
            if found: break
        if found:
            result[target] = found
        else:
            print(f"[dataset] WARNING {target} — no valid SMILES")
    return result


def load_toxicity_dataset(path: str) -> dict:
    if not os.path.exists(path):
        print("[toxicity] missing — Safety scores will default to 0.5 (required for correct scoring)"); return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    compounds = raw if isinstance(raw, list) else raw.get("compounds", [])
    index = {}
    for c in compounds:
        key = to_canonical(c.get("smiles", ""))
        if key: index[key] = c
    print(f"[toxicity] {len(index)} compounds indexed")
    return index


def load_rxninsight_database(path: str):
    """
    charge la base USPTO pour Rxn-INSIGHT (format Parquet malgré extension .gzip)
    TAG2 et les fingerprints sont obligatoires pour find_neighbors()
    """
    if not RXNINSIGHT_AVAILABLE: return None
    if not path or not os.path.exists(path):
        print("[rxn-insight] USPTO database missing — condition suggestion disabled")
        return None
    try:
        df = pd.read_parquet(path)
        if "REACTION" in df.columns and "RXN" not in df.columns:
            df = df.rename(columns={"REACTION": "RXN"})
        # TAG2 et fingerprints sont obligatoires pour find_neighbors()
        cols_needed = ["RXN", "SOLVENT", "REAGENT", "CATALYST", "NAME", "CLASS",
                       "TAG2", "rxn_str_patt_fp", "rxn_dif_patt_fp",
                       "rxn_str_morgan_fp", "rxn_dif_morgan_fp"]
        cols_present = [c for c in cols_needed if c in df.columns]
        df = df[cols_present]
        print(f"[rxn-insight] database loaded — {len(df)} USPTO reactions")
        return df
    except Exception as e:
        print(f"[rxn-insight] error loading database: {e}")
        return None


# =============================================================================
# SMILES utilities
# =============================================================================

def to_canonical(smiles) -> str:
    # Handle list-valued SMILES (e.g. ["CCO"] instead of "CCO")
    if isinstance(smiles, list):
        smiles = '.'.join(str(s) for s in smiles if s)
    if not smiles: return ""
    if not isinstance(smiles, str):
        return ""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else smiles

def safe_mol(smiles: str):
    if not smiles: return None
    return Chem.MolFromSmiles(smiles)


def validate_smiles_for_aizynthfinder(smiles: str) -> str:
    if not smiles: raise ValueError("empty SMILES")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol)


def build_dataset_smiles_index(dataset: dict) -> set:
    index = set()
    for rxn in dataset["all"]:
        p = to_canonical(rxn.get("product_smiles", ""))
        if p: index.add(p)
        for r in rxn.get("reactants_smiles", []):
            c = to_canonical(r)
            if c: index.add(c)
    return index


# =============================================================================
# AiZynthFinder
# =============================================================================

def _walk_reaction_tree(node: dict, steps: list) -> None:
    """
    parcours récursif de to_dict() — ordre rétrosynthétique (cible → SM)
    l'appelant inverse pour avoir l'ordre synthétique
    """
    if node.get("type") != "mol": return
    mol_smiles = node.get("smiles", "")
    for child in node.get("children", []):
        if child.get("type") != "reaction": continue
        reactant_nodes = [c for c in child.get("children", []) if c.get("type") == "mol"]
        steps.append({"reactants": [c.get("smiles", "") for c in reactant_nodes],
                      "product":   mol_smiles})
        for rnode in reactant_nodes:
            _walk_reaction_tree(rnode, steps)


def adapt_route(route: dict) -> dict:
    """
    extrait étapes dans l'ordre synthétique correct (SM → cible)
    utilise to_dict() pour avoir la hiérarchie — plus fiable que tree.reactions()
    """
    tree = route["reaction_tree"]
    try:
        tree_dict   = tree.to_dict()
        steps_retro = []
        _walk_reaction_tree(tree_dict, steps_retro)
        steps_forward = list(reversed(steps_retro))
        all_products  = {to_canonical(s["product"]) for s in steps_forward if s["product"]}
        all_reactants = {to_canonical(r) for s in steps_forward for r in s["reactants"]}
        leaves        = [r for r in all_reactants if r not in all_products]
        target_smiles = to_canonical(tree_dict.get("smiles", ""))
    except Exception as e:
        print(f"[adapt_route] error: {e}")
        steps_forward = []; leaves = []; target_smiles = ""
    return {"route_id": "aiz", "route_name": "AiZynthFinder Route",
            "steps": steps_forward, "starting_materials": leaves,
            "target_smiles": target_smiles, "raw": route}


def run_aizynthfinder(target_smiles: str, config_path: str = "config.yml",
                      n_routes: int = 25) -> list:
    # n_routes : on en prend plus (25 par défaut) pour maximiser les chances
    # de matching avec le dataset générique
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


# =============================================================================
# Rxn-INSIGHT
# =============================================================================

def _best_condition(df_cond) -> str:
    """
    extrait la condition la plus fréquente depuis un DataFrame Rxn-INSIGHT
    suggested_solvent/catalyst/reagent retournent un DataFrame NAME/COUNT
    """
    try:
        if df_cond is None or not hasattr(df_cond, "empty") or df_cond.empty:
            return ""
        valid = df_cond[df_cond["NAME"].str.strip() != ""]
        if valid.empty: return ""
        return str(valid.sort_values("COUNT", ascending=False).iloc[0]["NAME"]).strip()
    except Exception:
        return ""


def get_reaction_info_rxninsight(reactants: list, product: str, rxni_db) -> dict:
    """
    analyse une étape avec Rxn-INSIGHT :
    - type/classe de réaction (disponible SANS la base USPTO)
    - conditions suggérées par similarité (NÉCESSITE la base USPTO)

    sans base USPTO : type/classe OK mais pas de conditions
    avec base USPTO : tout disponible
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
            "conditions":     {"temperature_C": None, "temp_range": None,
                               "solvent": None, "co_solvent": None,
                               "reagents": [], "apparatus": None},
        }

        if rxni_db is not None:
            try:
                df_nb = rxn.find_neighbors(rxni_db, fp="MACCS", concatenate=True,
                                           threshold=0.3, broaden=True, full_search=False)
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
    crée des dataset_steps synthétiques pour une route AiZ absente du dataset
    le yield est None — pas affiché, pas utilisé dans le scoring
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
            "yield_percent":    None,   # pas de yield pour les routes prédites
            "source":           "rxn-insight",
            **rxni_info,
        })
    if dataset_steps and target_smiles:
        dataset_steps[-1]["product_smiles"] = target_smiles
    enriched = dict(aiz_route)
    enriched.update({"dataset_steps": dataset_steps,
                     "matched_route_id":   f"predicted_{route_index:02d}",
                     "matched_route_name": f"Predicted route #{route_index}",
                     "matched_target":     "?", "coverage": 0, "is_predicted": True})
    return enriched


def is_route_covered_by_dataset(aiz_route: dict, dataset_smiles_index: set,
                                 threshold: float = 0.4) -> bool:
    steps    = aiz_route.get("steps", [])
    products = [to_canonical(s.get("product", "")) for s in steps if s.get("product")]
    if not products: return False
    return sum(1 for p in products if p in dataset_smiles_index) / len(products) >= threshold




def load_generic_reaction_dataset(path: str) -> dict:
    """
    charge un dataset de réactions individuelles (pas liées à une synthèse totale)
    même format JSON que le dataset principal MAIS :
    - chaque entrée = une réaction indépendante
    - target absent ou vide
    - step_number = 1 partout
    permet de valider les étapes des routes AiZ avec des conditions expérimentales réelles
    """
    if not path or not os.path.exists(path):
        print("[generic dataset] absent — step validation disabled")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    reactions = raw if isinstance(raw, list) else raw.get("reactions", [])
    print(f"[generic dataset] {len(reactions)} individual reactions loaded")

    by_product      = {}
    by_reaction_key = {}   # (tuple réactifs triés, produit) → réaction

    for rxn in reactions:
        prod  = to_canonical(rxn.get("product_smiles", ""))
        reacs = tuple(sorted([to_canonical(r) for r in rxn.get("reactants_smiles", []) if r]))
        if prod:
            by_product.setdefault(prod, []).append(rxn)
        key = (reacs, prod)
        if key not in by_reaction_key:
            by_reaction_key[key] = rxn

    return {"by_product": by_product, "by_reaction_key": by_reaction_key, "all": reactions}


def match_step_in_generic_dataset(reactants: list, product: str, generic_ds: dict):
    """
    cherche si une étape AiZ est dans le dataset générique
    matching en deux niveaux :
    1. clé exacte (réactifs canoniques triés + produit) — matching strict
    2. produit seul — matching moins strict, retourne le premier match
    retourne la réaction du dataset générique si trouvée, None sinon
    """
    if not generic_ds:
        return None
    prod_canon  = to_canonical(product)
    reac_canons = tuple(sorted([to_canonical(r) for r in reactants if r]))
    exact = generic_ds.get("by_reaction_key", {}).get((reac_canons, prod_canon))
    if exact:
        return exact
    by_prod = generic_ds.get("by_product", {}).get(prod_canon, [])
    return by_prod[0] if by_prod else None


def validate_aiz_route_against_generic_dataset(aiz_route: dict, generic_ds: dict,
                                                rxni_db, route_index: int) -> dict:
    """
    pour chaque étape d'une route AiZ, cherche dans le dataset générique :
    - étape trouvée → conditions réelles du dataset (source="generic_dataset")
    - étape absente → conditions Rxn-INSIGHT (source="rxn-insight")

    classification :
    - "validated" : toutes les étapes dans le dataset générique → passe en section validée
    - "partial"   : certaines étapes validées → passe aussi en section validée
    - "predicted" : aucune étape → reste en section prédite
    """
    steps         = aiz_route.get("steps", [])
    target_smiles = aiz_route.get("target_smiles", "")
    dataset_steps = []
    validated_count = 0

    for i, step in enumerate(steps, 1):
        reactants = step.get("reactants", [])
        product   = step.get("product", "")
        match     = match_step_in_generic_dataset(reactants, product, generic_ds)

        if match:
            validated_count += 1
            dataset_steps.append({
                "id": f"VAL-{route_index:02d}-{i:02d}",
                "route_id": f"validated_{route_index:02d}",
                "step_number": i,
                "reactants_smiles": reactants,
                "product_smiles":   product,
                "yield_percent":    match.get("yield_percent"),
                "reaction_type":    match.get("reaction_type", ""),
                "reaction_class":   match.get("reaction_class", ""),
                "fg_reactants":     match.get("fg_reactants", []),
                "by_products":      match.get("by_products", []),
                "conditions":       match.get("conditions", {
                    "temperature_C": None, "temp_range": None,
                    "solvent": None, "co_solvent": None,
                    "reagents": [], "apparatus": None,
                }),
                "source": "generic_dataset",   # conditions réelles !
            })
        else:
            rxni_info = get_reaction_info_rxninsight(reactants, product, rxni_db)
            dataset_steps.append({
                "id": f"VAL-{route_index:02d}-{i:02d}",
                "route_id": f"validated_{route_index:02d}",
                "step_number": i,
                "reactants_smiles": reactants,
                "product_smiles":   product,
                "yield_percent":    None,
                "source":           "rxn-insight",
                **rxni_info,
            })

    if dataset_steps and target_smiles:
        dataset_steps[-1]["product_smiles"] = target_smiles

    n = len(steps)
    if   n == 0:                          status = "predicted"
    elif validated_count == n:            status = "validated"
    elif validated_count > 0:             status = "partial"
    else:                                 status = "predicted"

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
    traite les routes AiZ non couvertes par le dataset principal :
    1. vérifie chaque étape contre le dataset générique
    2. "validated"/"partial" → validated_routes (conditions réelles disponibles)
       "predicted"           → predicted_routes (Rxn-INSIGHT uniquement)
    retourne (validated_routes, predicted_routes)
    """
    if not RXNINSIGHT_AVAILABLE:
        print("[rxn-insight] not installed"); return [], []

    dataset_smiles_index = build_dataset_smiles_index(dataset)
    validated_routes = []
    predicted_routes = []
    counter = 1

    for aiz_route in aiz_routes:
        if not aiz_route.get("steps"): continue
        if is_route_covered_by_dataset(aiz_route, dataset_smiles_index):
            print(f"  [novel] covered by main dataset — skipped"); continue

        print(f"  [novel] route #{counter} — validating against generic dataset...")
        enriched = validate_aiz_route_against_generic_dataset(
            aiz_route, generic_ds, rxni_db, counter)
        enriched["matched_target"] = target_name
        status = enriched.get("validation_status", "predicted")
        v, t   = enriched.get("validated_steps_count", 0), enriched.get("total_steps_count", 0)

        if status in ("validated", "partial"):
            validated_routes.append(enriched)
            print(f"    → {status} ({v}/{t} steps in generic dataset)")
        else:
            # aucune étape dans le dataset générique → Rxn-INSIGHT pur
            pure = enrich_aiz_route_with_rxninsight(aiz_route, rxni_db, counter)
            pure["matched_target"] = target_name
            pure.update({"validation_status": "predicted", "is_validated": False,
                         "validated_steps_count": 0, "total_steps_count": t})
            predicted_routes.append(pure)
            print(f"    → predicted (no steps in generic dataset)")
        counter += 1

    print(f"  {len(validated_routes)} validated/partial, {len(predicted_routes)} predicted")
    return validated_routes, predicted_routes

def get_novel_routes_from_aizynthfinder(aiz_routes, dataset, target_name, rxni_db) -> list:
    if not RXNINSIGHT_AVAILABLE:
        print("[rxn-insight] not installed"); return []
    dataset_smiles_index = build_dataset_smiles_index(dataset)
    novel = []; counter = 1
    for route in aiz_routes:
        if not route.get("steps"): continue
        if is_route_covered_by_dataset(route, dataset_smiles_index):
            print(f"  [novel] covered by dataset — skipped"); continue
        print(f"  [novel] route #{counter} — Rxn-INSIGHT enrichment...")
        enriched = enrich_aiz_route_with_rxninsight(route, rxni_db, counter)
        enriched["matched_target"] = target_name
        novel.append(enriched); counter += 1
    print(f"  {len(novel)} novel routes predicted")
    return novel


# =============================================================================
# filtering + dataset routes
# =============================================================================

def get_all_dataset_routes_for_target(dataset: dict, target_name: str) -> list:
    result = []
    for rid, steps in dataset["by_route"].items():
        if steps[0].get("target", "").lower() != target_name.lower(): continue
        result.append({"route_id": rid, "route_name": steps[0].get("route_name", rid),
                       "steps": [], "starting_materials": [], "dataset_steps": steps,
                       "matched_route_id": rid, "matched_route_name": steps[0].get("route_name", rid),
                       "matched_target": steps[0].get("target", "?"),
                       "coverage": 0, "is_predicted": False, "is_validated": False, "validation_status": "dataset"})
        print(f"  [dataset] OK {rid} — {len(steps)} steps")
    return result


def filter_routes_by_starting_materials(aiz_routes, dataset, target_smiles, target_name="") -> list:
    by_reactant = dataset["by_reactant"]; by_route = dataset["by_route"]
    best = {}
    for route in aiz_routes:
        sm_list = route.get("starting_materials", [])
        if not sm_list: continue
        cands = set()
        for sm in sm_list:
            for rxn in by_reactant.get(to_canonical(sm), []):
                rid = rxn.get("route_id")
                if rid: cands.add(rid)
        if not cands: continue
        if target_name:
            cands = {rid for rid in cands
                     if by_route.get(rid,[{}])[0].get("target","").lower() == target_name.lower()}
        if not cands: continue
        best_id, best_cov = None, 0
        for rid in cands:
            dr = {to_canonical(r) for rxn in by_route.get(rid,[]) for r in rxn.get("reactants_smiles",[])}
            cov = sum(1 for sm in sm_list if to_canonical(sm) in dr)
            if cov > best_cov: best_cov, best_id = cov, rid
        if not best_id: continue
        ex = best.get(best_id)
        if ex and ex["coverage"] >= best_cov: continue
        sd = by_route[best_id]
        enriched = dict(route)
        enriched.update({"dataset_steps": sd, "matched_route_id": best_id,
                         "matched_route_name": sd[0].get("route_name", best_id),
                         "matched_target": sd[0].get("target","?"),
                         "coverage": best_cov, "is_predicted": False})
        best[best_id] = enriched
    for route in get_all_dataset_routes_for_target(dataset, target_name):
        if route["route_id"] not in best:
            best[route["route_id"]] = route
    validated = list(best.values())
    print(f"  {len(validated)} dataset routes retained")
    return validated


# =============================================================================
# metrics
# =============================================================================

def calc_atom_economy(reactants_smiles, product_smiles) -> float:
    prod_mol = safe_mol(product_smiles)
    if not prod_mol: return 0.0
    prod_mw  = Descriptors.MolWt(prod_mol)
    total_mw = sum(Descriptors.MolWt(m) for s in reactants_smiles if (m := safe_mol(s)))
    return min(prod_mw / total_mw, 1.0) if total_mw else 0.0


def calc_e_factor(reactants_smiles, product_smiles, yield_fraction) -> float:
    prod_mol = safe_mol(product_smiles)
    if not prod_mol: return 0.5
    prod_mw  = Descriptors.MolWt(prod_mol)
    total_mw = sum(Descriptors.MolWt(m) for s in reactants_smiles if (m := safe_mol(s)))
    obtained = prod_mw * max(yield_fraction, 0.01)
    waste    = max(total_mw - obtained, 0)
    return 1.0 / (1.0 + waste / obtained)


def calc_toxicity_score(reactants_smiles, conditions, tox_index, solvent_map) -> float:
    to_check = list(reactants_smiles)
    for key in ("solvent", "co_solvent"):
        s = conditions.get(key)
        if s and (smi := solvent_map.get(s)): to_check.append(smi)
    scores = [tox_index[to_canonical(s)]["hazard_score"]
              if to_canonical(s) in tox_index else 0.5 for s in to_check]
    return 1.0 - (sum(scores) / len(scores) if scores else 0.5)


def build_solvent_map(tox_index: dict) -> dict:
    abbrev = {"PhMe":"Cc1ccccc1","toluene":"Cc1ccccc1","DCM":"ClCCl","CH2Cl2":"ClCCl",
              "THF":"C1CCOC1","MeOH":"CO","EtOH":"CCO","DMF":"CN(C)C=O","MeCN":"CC#N",
              "AcOH":"CC(=O)O","Et2O":"CCOCC","CHCl3":"ClC(Cl)Cl","CCl4":"ClC(Cl)(Cl)Cl",
              "PhH":"c1ccccc1","benzene":"c1ccccc1","TFE":"OCC(F)(F)F","t-BuOH":"CC(C)(C)O",
              "MeNO2":"C[N+](=O)[O-]","H2O":"O","dioxane":"C1COCCO1","1,4-dioxane":"C1COCCO1",
              "hexane":"CCCCCC","acetone":"CC(C)=O","pyridine":"c1ccncc1",
              "i-PrOH":"CC(C)O","DMSO":"CS(C)=O"}
    result = dict(abbrev)
    for canon in tox_index: result[canon] = canon
    return result


# =============================================================================
# scoring functions
# =============================================================================

def compute_steps(route_data: dict, tox_index: dict) -> float:
    return 1.0 / max(len(route_data.get("dataset_steps", [])), 1)


def compute_yield(route_data: dict, tox_index: dict) -> float:
    """
    predicted : 0.5 fixe (pas affiché, critère exclu du scoring)
    validated/partial : yield réel pour les étapes generic_dataset,
                        0.5 pour les étapes rxn-insight
    dataset : calcul réel complet
    """
    steps  = route_data.get("dataset_steps", [])
    status = route_data.get("validation_status", "dataset")
    if not steps: return 0.0
    if status == "predicted":
        return 1.0   # valeur neutre, jamais affichée (yield exclu du scoring pour predicted)
    result = 1.0
    for s in steps:
        y   = s.get("yield_percent")
        src = s.get("source", "dataset")
        if y is not None and src in ("generic_dataset", "dataset"):
            result *= y / 100.0
        else:
            result *= 1.0   # yield manquant → 1.0 (optimiste, ne pénalise pas)
    return result


def compute_atom_economy(route_data: dict, tox_index: dict) -> float:
    steps  = route_data.get("dataset_steps", [])
    scores = [calc_atom_economy(s.get("reactants_smiles",[]), s.get("product_smiles",""))
              for s in steps]
    return sum(scores) / len(scores) if scores else 0.0


def compute_e_factor(route_data: dict, tox_index: dict) -> float:
    steps  = route_data.get("dataset_steps", [])
    scores = [calc_e_factor(s.get("reactants_smiles",[]), s.get("product_smiles",""),
                            (s.get("yield_percent") or 100) / 100.0)   # yield manquant → 1.0
              for s in steps]
    return sum(scores) / len(scores) if scores else 0.0


def compute_toxicity(route_data: dict, tox_index: dict) -> float:
    solvent_map = build_solvent_map(tox_index)
    steps  = route_data.get("dataset_steps", [])
    scores = [calc_toxicity_score(s.get("reactants_smiles",[]), s.get("conditions",{}),
                                  tox_index, solvent_map) for s in steps]
    return sum(scores) / len(scores) if scores else 0.5


CRITERIA_REGISTRY = {
    "steps":        {"fn": compute_steps,        "description": "number of steps"},
    "yield":        {"fn": compute_yield,        "description": "cumulative yield"},
    "atom_economy": {"fn": compute_atom_economy, "description": "atom economy"},
    "e_factor":     {"fn": compute_e_factor,     "description": "e-factor"},
    "toxicity":     {"fn": compute_toxicity,     "description": "safety score"},
}


# =============================================================================
# weighted ranking
# =============================================================================

def compute_weights(criteria: list) -> dict:
    """poids en 1/i² : #1→73%, #2→18%, #3→9%"""
    raw   = {c: 1.0 / (i + 1) ** 2 for i, c in enumerate(criteria)}
    total = sum(raw.values())
    return {c: w / total for c, w in raw.items()}


def compute_all_scores(route: dict, tox_index: dict) -> dict:
    """calcule tous les critères disponibles sur une route — pour le comparateur"""
    return {c: round(CRITERIA_REGISTRY[c]["fn"](route, tox_index), 4)
            for c in CRITERIA_REGISTRY}


def rank_weighted(routes: list, criteria: list, tox_index: dict) -> list:
    """
    classe par somme pondérée
    pour les routes prédites, le critère 'yield' est exclu du score
    (remplacé par la moyenne des autres critères pondérée)
    """
    weights = compute_weights(criteria)
    scored  = []
    for route in routes:
        status  = route.get("validation_status", "dataset")
        # yield exclu seulement pour les routes purement prédites
        # les routes validated/partial ont des données réelles → yield inclus
        exclude_yield = (status == "predicted")
        details = {}
        total   = 0.0

        active_criteria = [c for c in criteria if not (exclude_yield and c == "yield")]
        active_weights  = compute_weights(active_criteria) if len(active_criteria) < len(criteria)                           else weights

        for c in criteria:
            raw = CRITERIA_REGISTRY[c]["fn"](route, tox_index)
            if exclude_yield and c == "yield":
                details[c] = {"raw": None, "weight": 0.0, "weighted": 0.0, "excluded": True}
            else:
                w = active_weights.get(c, weights[c])
                details[c] = {"raw": round(raw, 4), "weight": round(w, 4),
                               "weighted": round(raw * w, 4)}
                total += raw * w

        # calcule aussi tous les autres critères pour le comparateur
        details["_all_scores"] = compute_all_scores(route, tox_index)
        scored.append((round(total, 4), details, route))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored


# =============================================================================
# main entry point
# =============================================================================

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
    generic_dataset_path: str  = "",   # NOUVEAU — dataset de réactions individuelles
    n_aiz_routes:         int  = 25,   # NOUVEAU — nombre de routes AiZ à récupérer
) -> dict:
    """
    retourne {"dataset": [...], "validated": [...], "predicted": [...]}

    dataset   : routes du dataset principal Chemistry by Design (toutes retournées)
    validated : routes AiZ dont au moins une étape est dans le dataset générique
                conditions réelles pour les étapes référencées, Rxn-INSIGHT pour les autres
                include routes "validated" (100%) et "partial" (partiellement validées)
    predicted : routes AiZ purement hypothétiques, aucune étape dans le dataset générique
                conditions Rxn-INSIGHT uniquement, yield exclu du scoring
    """
    if len(criteria_priority) != 3:
        raise ValueError(f"exactly 3 criteria needed, got {len(criteria_priority)}")
    unknown = [c for c in criteria_priority if c not in CRITERIA_REGISTRY]
    if unknown:
        raise ValueError(f"unknown criteria: {unknown}")

    print(f"\n[path_finder] target: {target_smiles}")
    if target_name: print(f"[pathfinder] name: {target_name}")
    print(f"[pathfinder] criteria: {criteria_priority}")

    print("\n[1/4] loading datasets...")
    dataset    = load_reaction_dataset(dataset_path)
    tox_index  = load_toxicity_dataset(toxicity_path)
    generic_ds = load_generic_reaction_dataset(generic_dataset_path)

    rxni_db = None
    if include_predicted and RXNINSIGHT_AVAILABLE:
        rxni_db = load_rxninsight_database(rxninsight_db_path)

    print(f"\n[2/4] AiZynthFinder search (up to {n_aiz_routes} routes)...")
    all_aiz = run_aizynthfinder(target_smiles, config_path, n_routes=n_aiz_routes)
    print(f"  {len(all_aiz)} AiZ routes retrieved")

    print("\n[3/4] dataset routes...")
    dataset_routes = filter_routes_by_starting_materials(all_aiz, dataset, target_smiles, target_name)

    validated_routes = []
    predicted_routes = []
    if include_predicted and RXNINSIGHT_AVAILABLE and all_aiz:
        print("\n[3b] processing novel routes (generic dataset + Rxn-INSIGHT)...")
        validated_routes, predicted_routes = process_novel_routes(
            all_aiz, dataset, generic_ds, target_name, rxni_db)

    print("\n[4/4] scoring...")
    scored_dataset   = rank_weighted(dataset_routes,   criteria_priority, tox_index)
    scored_validated = rank_weighted(validated_routes, criteria_priority, tox_index)[:top_n]
    scored_predicted = rank_weighted(predicted_routes, criteria_priority, tox_index)[:top_n]

    print(f"\n[pathfinder] {len(scored_dataset)} dataset / "
          f"{len(scored_validated)} validated / {len(scored_predicted)} predicted")
    for i, (sc, _, r) in enumerate(scored_dataset, 1):
        print(f"  [dataset]   {i}. {r.get('matched_route_name','?')} — score {sc:.4f}")
    for i, (sc, _, r) in enumerate(scored_validated, 1):
        v = r.get("validated_steps_count",0); t = r.get("total_steps_count",0)
        print(f"  [validated] {i}. {r.get('matched_route_name','?')} ({v}/{t}) — score {sc:.4f}")
    for i, (sc, _, r) in enumerate(scored_predicted, 1):
        print(f"  [predicted] {i}. {r.get('matched_route_name','?')} — score {sc:.4f}")

    return {"dataset": scored_dataset, "validated": scored_validated, "predicted": scored_predicted}