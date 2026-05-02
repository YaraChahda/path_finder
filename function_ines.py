# function_ines.py — pipeline de rétrosynthèse
# -----------------------------------------------
# Le but : prendre une molécule cible, lancer AiZynthFinder dessus,
# et classer les routes trouvées selon 3 critères choisis par l'utilisateur.
#
# Tout le scoring se base sur les données réelles du dataset Chemistry by Design
# (rendements, conditions, solvants) + le fichier toxicity_dataset.json.
# Il n'y a aucune donnée de toxicité codée en dur dans ce fichier.
#
# Pour l'utiliser depuis Streamlit ou un notebook :
#   results = find_best_routes(
#       target_smiles="OC1C=C...",
#       criteria_priority=["steps", "toxicity", "atom_economy"],
#       dataset_path="reaction_dataset.json",
#       toxicity_path="toxicity_dataset.json",
#       config_path="config.yml",
#       target_name="galanthamine",
#   )

import json
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from aizynthfinder.aizynthfinder import AiZynthFinder


# =============================================================================
# 1. chargement des données
# =============================================================================

def load_reaction_dataset(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Mets reaction_dataset.json dans le même dossier que ce script."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        reactions = raw
        metadata  = {}
    elif isinstance(raw, dict) and "reactions" in raw:
        reactions = raw["reactions"]
        metadata  = raw.get("_metadata", {})
    else:
        raise ValueError(
            "Format du dataset non reconnu.\n"
            "Attendu : une liste [...] ou un dict avec clé 'reactions'."
        )

    print(f"[dataset] {len(reactions)} réactions chargées")

    by_product  = {}
    by_reactant = {}
    by_route    = {}

    for rxn in reactions:
        prod_key = to_canonical(rxn.get("product_smiles", ""))
        if prod_key:
            by_product.setdefault(prod_key, []).append(rxn)

        for rsmi in rxn.get("reactants_smiles", []):
            rkey = to_canonical(rsmi)
            if rkey:
                by_reactant.setdefault(rkey, []).append(rxn)

        rid = rxn.get("route_id", "inconnu")
        by_route.setdefault(rid, []).append(rxn)

    for rid in by_route:
        by_route[rid].sort(key=lambda x: x.get("step_number", 0))

    print(f"[dataset] {len(by_route)} routes distinctes indexées")
    return {
        "by_product":  by_product,
        "by_reactant": by_reactant,
        "by_route":    by_route,
        "all":         reactions,
        "metadata":    metadata,
    }


def get_targets_from_dataset(dataset: dict) -> dict:
    """
    Lit les molécules cibles et leurs SMILES directement depuis le dataset.
    C'est cette fonction qui alimente la liste dans Streamlit —
    si tu ajoutes une molécule dans le dataset elle apparaît automatiquement,
    si tu la retires elle disparaît.

    On essaie d'abord _metadata["target_smiles"], puis les produits du dataset.
    Chaque SMILES est validé par RDKit — les SMILES invalides sont corrigés
    ou ignorés avec un message.
    """
    by_route = dataset["by_route"]
    metadata = dataset["metadata"]
    meta_smiles = metadata.get("target_smiles", {})

    # on groupe toutes les routes par cible
    by_target = {}
    for rid, steps in by_route.items():
        target = steps[0].get("target", "?")
        if target != "?":
            by_target.setdefault(target, []).append((rid, steps))

    result = {}
    for target in sorted(by_target.keys()):
        # essai 1 : SMILES dans _metadata
        candidate = meta_smiles.get(target, "")
        if candidate and Chem.MolFromSmiles(candidate) is not None:
            result[target] = to_canonical(candidate)
            continue

        # essai 2 : produit final valide dans le dataset
        # (utile quand le SMILES de _metadata est invalide comme aspidospermidine)
        found = None
        for rid, steps in by_target[target]:
            for step in reversed(steps):
                prod = step.get("product_smiles", "")
                mol  = Chem.MolFromSmiles(prod) if prod else None
                # on vérifie que c'est une vraie molécule complexe (>= 15 atomes)
                # pour éviter de prendre un intermédiaire simple
                if mol and mol.GetNumAtoms() >= 15:
                    found = to_canonical(prod)
                    break
            if found:
                break

        if found:
            result[target] = found
            print(f"[dataset] {target} — SMILES metadata invalide, fallback sur produit du dataset")
        else:
            print(f"[dataset] ⚠ {target} — aucun SMILES valide trouvé, cible ignorée")

    return result


def load_toxicity_dataset(path: str) -> dict:
    if not os.path.exists(path):
        print("[toxicité] fichier absent — scores à 0.5 par défaut")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    compounds = raw if isinstance(raw, list) else raw.get("compounds", [])

    index = {}
    for c in compounds:
        key = to_canonical(c.get("smiles", ""))
        if key:
            index[key] = c

    print(f"[toxicité] {len(index)} composés indexés")
    return index


# =============================================================================
# 2. utilitaires SMILES
# =============================================================================

def to_canonical(smiles: str) -> str:
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else smiles


def safe_mol(smiles: str):
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def validate_smiles_for_aizynthfinder(smiles: str) -> str:
    """
    Valide le SMILES avant de le passer à AiZynthFinder.
    Sans ça, AiZynthFinder plante avec une AttributeError obscure :
    'NoneType object has no attribute GetNumAtoms'
    On préfère lever une ValueError claire avec un message utile.
    Retourne le SMILES canonique RDKit si valide.
    """
    if not smiles:
        raise ValueError("SMILES vide — impossible de lancer AiZynthFinder.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(
            f"SMILES invalide : {smiles}\n"
            "Vérifie la syntaxe. Les NH entre parenthèses s'écrivent [NH] ou N, pas (NH)."
        )
    return Chem.MolToSmiles(mol)


# =============================================================================
# 3. AiZynthFinder
# =============================================================================

def run_aizynthfinder(target_smiles: str, config_path: str = "config.yml") -> list:
    # validation obligatoire — évite le crash obscur de AiZynthFinder
    canon_smiles = validate_smiles_for_aizynthfinder(target_smiles)
    print(f"[AiZynthFinder] démarrage — cible : {canon_smiles}")

    finder = AiZynthFinder(configfile=config_path)
    finder.stock.select("zinc")
    finder.expansion_policy.select("uspto")
    finder.filter_policy.select("uspto")

    finder.target_smiles = canon_smiles
    finder.tree_search()
    finder.build_routes()

    routes_raw = list(finder.routes)
    print(f"[AiZynthFinder] {len(routes_raw)} routes trouvées")

    return [adapt_route(r) for r in routes_raw]


def adapt_route(route: dict) -> dict:
    # tree.graph est un DiGraph NetworkX :
    #   mol → rxn  = mol est réactif de rxn
    #   rxn → mol  = mol est le produit de rxn
    tree   = route["reaction_tree"]
    steps  = []
    leaves = []

    try:
        for mol_node in tree.leafs():
            canon = to_canonical(mol_node.smiles)
            if canon:
                leaves.append(canon)

        graph = tree.graph
        for rxn_node in tree.reactions():
            reactants = [
                pred.smiles for pred in graph.predecessors(rxn_node)
                if hasattr(pred, "smiles")
            ]
            products = [
                succ.smiles for succ in graph.successors(rxn_node)
                if hasattr(succ, "smiles")
            ]
            product = products[0] if products else ""
            if reactants or product:
                steps.append({"reactants": reactants, "product": product})

    except Exception as e:
        print(f"[adapt_route] erreur : {e}")

    return {
        "route_id":           "inconnu",
        "route_name":         "Route AiZynthFinder",
        "steps":              steps,
        "starting_materials": leaves,
        "raw":                route,
    }


# =============================================================================
# 4. filtrage par starting materials
# =============================================================================
#
# Pour chaque route AiZ, on cherche dans le dataset la route qui :
#   1. cible LA MÊME molécule que ce qu'on cherche  ← point critique
#   2. a le plus de réactifs de départ en commun avec la route AiZ
#
# Sans la condition 1, une recherche de galanthamine pourrait retourner
# une route de morphine si elle partage des réactifs communs.

def get_all_dataset_routes_for_target(dataset: dict, target_name: str) -> list:
    """
    Retourne toutes les routes du dataset pour une cible donnée,
    formatées comme des routes enrichies prêtes pour le scoring.

    C'est la fonction centrale du pipeline : on ne filtre pas par matching
    avec AiZynthFinder (les SM d'AiZ sont trop simples pour matcher le dataset),
    on retourne simplement toutes les routes connues pour cette molécule.
    """
    by_route = dataset["by_route"]
    result   = []

    for rid, steps in by_route.items():
        route_target = steps[0].get("target", "").lower()
        if route_target != target_name.lower():
            continue

        route = {
            "route_id":           rid,
            "route_name":         steps[0].get("route_name", rid),
            "steps":              [],
            "starting_materials": [],
            "dataset_steps":      steps,
            "matched_route_id":   rid,
            "matched_route_name": steps[0].get("route_name", rid),
            "matched_target":     steps[0].get("target", "?"),
            "coverage":           0,
        }
        result.append(route)
        print(f"  [dataset] ✓ '{rid}' — {len(steps)} étapes")

    return result


def filter_routes_by_starting_materials(
    aiz_routes: list,
    dataset: dict,
    target_smiles: str,
    target_name:   str = "",
) -> list:
    """
    Stratégie en deux temps :

    1. On essaie de matcher les routes AiZynthFinder avec le dataset
       via les starting materials (fonctionne quand AiZ et dataset
       ont des réactifs en commun).

    2. Si le matching donne moins de routes que ce qu'il y a dans le dataset
       pour cette cible, on complète avec toutes les routes du dataset
       non encore couvertes.

    En pratique avec ton dataset Chemistry by Design, les SM d'AiZ (molécules
    commerciales simples, 2-6 atomes) ne matchent presque jamais les réactifs
    du dataset (molécules complexes, 15-25 atomes). On tombe donc toujours
    dans le cas 2 et on retourne toutes les routes du dataset pour la cible.
    """
    by_reactant = dataset["by_reactant"]
    by_route    = dataset["by_route"]

    # tentative de matching via SM — peut fonctionner si le dataset est étendu
    best_per_dataset_route = {}

    for route in aiz_routes:
        sm_list = route.get("starting_materials", [])
        if not sm_list:
            continue

        candidate_ids = set()
        for sm in sm_list:
            for rxn in by_reactant.get(to_canonical(sm), []):
                rid = rxn.get("route_id")
                if rid:
                    candidate_ids.add(rid)

        if not candidate_ids:
            continue

        # on ne garde que les routes pour la bonne cible
        if target_name:
            candidate_ids = {
                rid for rid in candidate_ids
                if by_route.get(rid, [{}])[0].get("target", "").lower()
                == target_name.lower()
            }

        if not candidate_ids:
            continue

        best_id, best_cov = None, 0
        for rid in candidate_ids:
            dataset_reactants = {
                to_canonical(rsmi)
                for rxn in by_route.get(rid, [])
                for rsmi in rxn.get("reactants_smiles", [])
            }
            cov = sum(1 for sm in sm_list if to_canonical(sm) in dataset_reactants)
            if cov > best_cov:
                best_cov, best_id = cov, rid

        if not best_id:
            continue

        existing = best_per_dataset_route.get(best_id)
        if existing and existing["coverage"] >= best_cov:
            continue

        steps_data = by_route[best_id]
        enriched   = dict(route)
        enriched["dataset_steps"]      = steps_data
        enriched["matched_route_id"]   = best_id
        enriched["matched_route_name"] = steps_data[0].get("route_name", best_id)
        enriched["matched_target"]     = steps_data[0].get("target", "?")
        enriched["coverage"]           = best_cov
        best_per_dataset_route[best_id] = enriched

    matched_ids = set(best_per_dataset_route.keys())

    # on récupère toutes les routes du dataset pour cette cible
    all_dataset_routes = get_all_dataset_routes_for_target(dataset, target_name)
    all_dataset_ids    = {r["route_id"] for r in all_dataset_routes}

    # on ajoute les routes du dataset non couvertes par le matching AiZ
    for route in all_dataset_routes:
        rid = route["route_id"]
        if rid not in matched_ids:
            best_per_dataset_route[rid] = route

    validated = list(best_per_dataset_route.values())
    n_matched = len(matched_ids & all_dataset_ids)
    n_direct  = len(validated) - n_matched

    if n_matched > 0:
        print(f"  {n_matched} route(s) matchée(s) via AiZynthFinder")
    if n_direct > 0:
        print(f"  {n_direct} route(s) ajoutée(s) directement depuis le dataset")
    print(f"\n  {len(validated)} routes uniques retenues au total")
    return validated


# =============================================================================
# 5. calcul des métriques
# =============================================================================

def calc_atom_economy(reactants_smiles: list, product_smiles: str) -> float:
    prod_mol = safe_mol(product_smiles)
    if prod_mol is None:
        return 0.0
    prod_mw  = Descriptors.MolWt(prod_mol)
    total_mw = sum(
        Descriptors.MolWt(m)
        for smi in reactants_smiles
        if (m := safe_mol(smi)) is not None
    )
    if total_mw == 0:
        return 0.0
    return min(prod_mw / total_mw, 1.0)


def calc_e_factor(reactants_smiles: list, product_smiles: str, yield_fraction: float) -> float:
    prod_mol = safe_mol(product_smiles)
    if prod_mol is None:
        return 0.5
    prod_mw  = Descriptors.MolWt(prod_mol)
    total_mw = sum(
        Descriptors.MolWt(m)
        for smi in reactants_smiles
        if (m := safe_mol(smi)) is not None
    )
    obtained = prod_mw * max(yield_fraction, 0.01)
    waste    = max(total_mw - obtained, 0)
    return 1.0 / (1.0 + waste / obtained)


def calc_toxicity_score(
    reactants_smiles: list,
    conditions: dict,
    tox_index: dict,
    solvent_smiles_map: dict,
) -> float:
    to_check = list(reactants_smiles)

    solvent = conditions.get("solvent")
    if solvent and (smi := solvent_smiles_map.get(solvent)):
        to_check.append(smi)

    co_solvent = conditions.get("co_solvent")
    if co_solvent and (smi := solvent_smiles_map.get(co_solvent)):
        to_check.append(smi)

    scores = [
        tox_index[to_canonical(smi)]["hazard_score"]
        if to_canonical(smi) in tox_index else 0.5
        for smi in to_check
    ]
    avg_danger = sum(scores) / len(scores) if scores else 0.5
    return 1.0 - avg_danger


def build_solvent_map(tox_index: dict) -> dict:
    # abréviations labo → SMILES (alias de nommage seulement, pas des données tox)
    abbrev = {
        "PhMe": "Cc1ccccc1", "toluene": "Cc1ccccc1",
        "DCM": "ClCCl", "CH2Cl2": "ClCCl",
        "THF": "C1CCOC1",
        "MeOH": "CO",
        "EtOH": "CCO",
        "DMF": "CN(C)C=O",
        "MeCN": "CC#N",
        "AcOH": "CC(=O)O", "HOAc": "CC(=O)O",
        "Et2O": "CCOCC",
        "CHCl3": "ClC(Cl)Cl",
        "CCl4": "ClC(Cl)(Cl)Cl",
        "PhH": "c1ccccc1", "benzene": "c1ccccc1",
        "TFE": "OCC(F)(F)F",
        "t-BuOH": "CC(C)(C)O",
        "MeNO2": "C[N+](=O)[O-]",
        "H2O": "O",
        "dioxane": "C1COCCO1", "1,4-dioxane": "C1COCCO1",
        "hexane": "CCCCCC",
        "acetone": "CC(C)=O",
        "pyridine": "c1ccncc1",
        "i-PrOH": "CC(C)O",
        "DMSO": "CS(C)=O",
    }
    result = dict(abbrev)
    # les SMILES bruts du tox_index s'auto-mappent
    for canon_smi in tox_index:
        result[canon_smi] = canon_smi
    return result


# =============================================================================
# 6. fonctions de scoring
# =============================================================================

def compute_steps(route_data: dict, tox_index: dict) -> float:
    n = len(route_data.get("dataset_steps", []))
    return 1.0 / max(n, 1)


def compute_yield(route_data: dict, tox_index: dict) -> float:
    steps = route_data.get("dataset_steps", [])
    if not steps:
        return 0.0
    result = 1.0
    for s in steps:
        yld = s.get("yield_percent")
        result *= (yld / 100.0) if yld is not None else 0.5
    return result


def compute_atom_economy(route_data: dict, tox_index: dict) -> float:
    steps  = route_data.get("dataset_steps", [])
    scores = [
        calc_atom_economy(s.get("reactants_smiles", []), s.get("product_smiles", ""))
        for s in steps
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_e_factor(route_data: dict, tox_index: dict) -> float:
    steps  = route_data.get("dataset_steps", [])
    scores = [
        calc_e_factor(
            s.get("reactants_smiles", []),
            s.get("product_smiles", ""),
            (s.get("yield_percent") or 50) / 100.0,
        )
        for s in steps
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_toxicity(route_data: dict, tox_index: dict) -> float:
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


CRITERIA_REGISTRY = {
    "steps":        {"fn": compute_steps,        "description": "nombre d'étapes (moins = mieux)"},
    "yield":        {"fn": compute_yield,        "description": "rendement global cumulé"},
    "atom_economy": {"fn": compute_atom_economy, "description": "économie atomique moyenne"},
    "e_factor":     {"fn": compute_e_factor,     "description": "e-factor (déchets générés)"},
    "toxicity":     {"fn": compute_toxicity,     "description": "sécurité réactifs + solvants"},
}


# =============================================================================
# 7. classement par somme pondérée (poids en 1/i²)
# =============================================================================

def compute_weights(criteria: list) -> dict:
    raw   = {c: 1.0 / (i + 1) ** 2 for i, c in enumerate(criteria)}
    total = sum(raw.values())
    return {c: w / total for c, w in raw.items()}


def rank_weighted(routes: list, criteria: list, tox_index: dict) -> list:
    weights = compute_weights(criteria)
    scored  = []
    for route in routes:
        details = {}
        total   = 0.0
        for c in criteria:
            raw = CRITERIA_REGISTRY[c]["fn"](route, tox_index)
            w   = weights[c]
            details[c] = {
                "raw":      round(raw, 4),
                "weight":   round(w, 4),
                "weighted": round(raw * w, 4),
            }
            total += raw * w
        scored.append((round(total, 4), details, route))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored


# =============================================================================
# 8. point d'entrée principal
# =============================================================================

def find_best_routes(
    target_smiles:     str,
    criteria_priority: list,
    dataset_path:      str = "reaction_dataset.json",
    toxicity_path:     str = "toxicity_dataset.json",
    config_path:       str = "config.yml",
    top_n:             int = 3,
    target_name:       str = "",
) -> list:
    """
    Lance le pipeline complet.

    target_name : nom de la cible (ex: "galanthamine") — fortement recommandé.
                  Sans ça, des routes vers d'autres molécules peuvent être retournées.

    Retourne : [(score_total, details_par_critère, route_dict), ...]
               liste vide si aucune route trouvée pour cette cible dans le dataset.
    """
    if len(criteria_priority) != 3:
        raise ValueError(
            f"il faut exactement 3 critères, tu en as donné {len(criteria_priority)}.\n"
            f"disponibles : {list(CRITERIA_REGISTRY.keys())}"
        )
    unknown = [c for c in criteria_priority if c not in CRITERIA_REGISTRY]
    if unknown:
        raise ValueError(
            f"critères inconnus : {unknown}\n"
            f"disponibles : {list(CRITERIA_REGISTRY.keys())}"
        )

    print(f"\n[pipeline] cible : {target_smiles}")
    if target_name:
        print(f"[pipeline] nom cible : {target_name}")
    print(f"[pipeline] critères : {criteria_priority}")

    print("\n[1/4] chargement des datasets...")
    dataset   = load_reaction_dataset(dataset_path)
    tox_index = load_toxicity_dataset(toxicity_path)

    print("\n[2/4] recherche AiZynthFinder...")
    all_routes = run_aizynthfinder(target_smiles, config_path)
    print(f"  {len(all_routes)} routes récupérées")

    print("\n[3/4] filtrage par le dataset...")
    valid_routes = filter_routes_by_starting_materials(
        all_routes, dataset, target_smiles, target_name
    )

    if not valid_routes:
        print("\n[pipeline] aucune route trouvée dans le dataset pour cette cible.")
        return []

    print("\n[4/4] scoring...")
    top_routes = rank_weighted(valid_routes, criteria_priority, tox_index)[:top_n]

    print(f"\n[pipeline] {len(top_routes)} routes retournées :")
    for i, (score, _, route) in enumerate(top_routes, 1):
        n_steps = len(route.get("dataset_steps", []))
        print(f"  {i}. {route.get('matched_route_name', '?')} "
              f"({route.get('matched_target', '?')}) "
              f"— {n_steps} étapes — score {score:.4f}")

    return top_routes