# Le but : prendre une molécule cible, lancer AiZynthFinder dessus, et classer les routes trouvées selon 3 critères choisis par l'utilisateur.
# Tout le scoring se base sur les données réelles du dataset Chemistry by Design
# (rendements, conditions, solvants) + le fichier toxicity_dataset.json.
# Pour l'utiliser depuis Streamlit on utilise:
#   results = find_best_routes(
#       target_smiles="OC1C=C...",
#       criteria_priority=["steps", "toxicity", "atom_economy"],
#       dataset_path="reaction_dataset.json",
#       toxicity_path="toxicity_dataset.json",
#       config_path="config.yml",
#   )

import json
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from aizynthfinder.aizynthfinder import AiZynthFinder


# 1. chargement des données

def load_reaction_dataset(path: str) -> dict: 
    if not os.path.exists(path): # on lit le fichier JSON et on vérifie qu'il existe
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Mets reaction_dataset.json dans le même dossier que ce script."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list): # le dataset peut être une liste directe ou un dict avec clé "reactions"
        reactions = raw
    elif isinstance(raw, dict) and "reactions" in raw:
        reactions = raw["reactions"]
    else:
        raise ValueError(
            "Format du dataset non reconnu.\n"
            "Attendu : une liste [...] ou un dict avec clé 'reactions'."
        )

    print(f"[dataset] {len(reactions)} réactions chargées")

    # on construit 3 index pour éviter de parcourir toute la liste à chaque fois
    by_product  = {}   # SMILES produit → liste de réactions
    by_reactant = {}   # SMILES réactif → liste de réactions (sert au matching AiZ)
    by_route    = {}   # route_id → liste d'étapes dans l'ordre

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

    for rid in by_route:  # on trie les étapes de chaque route par step_number
        by_route[rid].sort(key=lambda x: x.get("step_number", 0))

    print(f"[dataset] {len(by_route)} routes distinctes indexées")
    return {
        "by_product":  by_product,
        "by_reactant": by_reactant,
        "by_route":    by_route,
        "all":         reactions,
    }


def load_toxicity_dataset(path: str) -> dict: # si le fichier est absent on continue sans planter, les scores de toxicité seront 0.5 (neutre) par défaut
    if not os.path.exists(path):
        print("[toxicité] fichier absent — scores à 0.5 par défaut")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    compounds = raw if isinstance(raw, list) else raw.get("compounds", [])

    index = {}  # on indexe par SMILES canonique pour pouvoir retrouver un composé même si l'écriture SMILES est légèrement différente
    for c in compounds:
        key = to_canonical(c.get("smiles", ""))
        if key:
            index[key] = c

    print(f"[toxicité] {len(index)} composés indexés")
    return index


# 2. utilitaires SMILES

def to_canonical(smiles: str) -> str: # RDKit normalise le SMILES pour que deux écritures du même composé donnent toujours la même chaîne — indispensable pour les comparaisons
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else smiles  # si invalide on garde brut


def safe_mol(smiles: str): # retourne None si le SMILES est invalide plutôt que de planter
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


# 3. AiZynthFinder

def run_aizynthfinder(target_smiles: str, config_path: str = "config.yml") -> list: #fonctionnement basique de Aizynthfinder
    print(f"[AiZynthFinder] démarrage — cible : {target_smiles}")

    finder = AiZynthFinder(configfile=config_path)
    finder.stock.select("zinc")              # molécules commerciales disponibles
    finder.expansion_policy.select("uspto")  # modèle rétrosynthétique entraîné sur USPTO
    finder.filter_policy.select("uspto")     # filtre les réactions peu plausibles

    finder.target_smiles = target_smiles
    finder.tree_search()    # MCTS — explore l'espace rétrosynthétique
    finder.build_routes()   # construit les routes complètes depuis l'arbre

    routes_raw = list(finder.routes)
    print(f"[AiZynthFinder] {len(routes_raw)} routes trouvées")

    return [adapt_route(r) for r in routes_raw]


def adapt_route(route: dict) -> dict: # AiZynthFinder retourne un dict avec une clé "reaction_tree" qui contient un objet ReactionTree (pas un dict !) — on doit utiliser son API pour extraire les étapes et les starting materials
    tree   = route["reaction_tree"]
    steps  = []
    leaves = []

    try:
        # leafs() = molécules jamais produites dans la route = starting materials
        for mol_node in tree.leafs():
            canon = to_canonical(mol_node.smiles)
            if canon:
                leaves.append(canon)

        # tree.graph est un DiGraph NetworkX :
        #   mol → rxn  signifie que mol est un réactif de rxn
        #   rxn → mol  signifie que mol est le produit de rxn
        graph = tree.graph

        for rxn_node in tree.reactions():# les prédécesseurs d'un nœud réaction sont ses réactifs
            reactants = [
                pred.smiles
                for pred in graph.predecessors(rxn_node)
                if hasattr(pred, "smiles")
            ]
            # les successeurs d'un nœud réaction sont ses produits (toujours 1)
            products = [
                succ.smiles
                for succ in graph.successors(rxn_node)
                if hasattr(succ, "smiles")
            ]
            product = products[0] if products else ""

            if reactants or product:
                steps.append({"reactants": reactants, "product": product})

    except Exception as e:
        print(f"[adapt_route] erreur lors de l'extraction : {e}")

    return {
        "route_id":           "inconnu",
        "route_name":         "Route AiZynthFinder",
        "steps":              steps,
        "starting_materials": leaves,
        "raw":                route,
    }


# 4. filtrage par starting materials
# Pourquoi on filtre comme ça et pas étape par étape ?
# Si on comparait chaque étape AiZ avec le dataset, une seule étape manquante ferait rejeter toute la route — ça serait trop strict et ça ne garderait que les synthèses déjà exactement dans le dataset.
# À la place, on regarde les starting materials qu'AiZ propose, et on cherche quelle route du dataset utilise les mêmes réactifs de départ.
# La route avec le plus de réactifs en commun est retenue.

def filter_routes_by_starting_materials(routes: list, dataset: dict) -> list:
    by_reactant = dataset["by_reactant"]
    by_route    = dataset["by_route"]
    validated   = []

    for route in routes:
        sm_list = route.get("starting_materials", [])

        if not sm_list: # si AiZ n'a pas trouvé de starting materials, la route est inutilisable
            print(f"  [filtrage] '{route.get('route_name')}' — pas de starting materials")
            continue

        # on cherche toutes les routes du dataset qui ont au moins un réactif commun
        candidate_ids = set()
        for sm in sm_list:
            for rxn in by_reactant.get(to_canonical(sm), []):
                rid = rxn.get("route_id")
                if rid:
                    candidate_ids.add(rid)

        if not candidate_ids:
            print(f"  [filtrage] '{route.get('route_name')}' — aucun réactif reconnu dans le dataset")
            continue

        # parmi les candidates, on garde celle avec le plus de réactifs en commun
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

        if best_id:
            steps_data = by_route[best_id]
            enriched   = dict(route)
            # on enrichit la route AiZ avec toutes les données réelles du dataset
            enriched["dataset_steps"]      = steps_data
            enriched["matched_route_id"]   = best_id
            enriched["matched_route_name"] = steps_data[0].get("route_name", best_id)
            enriched["matched_target"]     = steps_data[0].get("target", "?")
            enriched["coverage"]           = best_cov
            validated.append(enriched)
            print(f"  [filtrage] ✓ → '{best_id}' ({best_cov} réactifs communs)")
        else:
            print(f"  [filtrage] ✗ '{route.get('route_name')}' — rejetée")

    print(f"\n  {len(validated)}/{len(routes)} routes retenues")
    return validated


# 5. calcul des métriques

def calc_atom_economy(reactants_smiles: list, product_smiles: str) -> float:
    # économie atomique = MW produit / somme MW réactifs
    # proche de 1 = peu d'atomes perdus en sous-produits
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
    # e-factor = déchets produits / produit obtenu
    # on retourne 1/(1+e_factor) pour avoir un score entre 0 et 1
    # où 1 = peu de déchets et 0 = beaucoup de déchets
    prod_mol = safe_mol(product_smiles)
    if prod_mol is None:
        return 0.5  # valeur neutre si on ne peut pas calculer

    prod_mw  = Descriptors.MolWt(prod_mol)
    total_mw = sum(
        Descriptors.MolWt(m)
        for smi in reactants_smiles
        if (m := safe_mol(smi)) is not None
    )
    obtained = prod_mw * max(yield_fraction, 0.01)  # évite la division par zéro
    waste    = max(total_mw - obtained, 0)
    return 1.0 / (1.0 + waste / obtained)


def calc_toxicity_score(
    reactants_smiles: list,
    conditions: dict,
    tox_index: dict,
    solvent_smiles_map: dict,   # mapping nom solvant → SMILES, vient du dataset de toxicité
) -> float:
    # on construit la liste de tout ce qu'on va évaluer :
    # réactifs + solvant + co-solvant de l'étape
    to_check = list(reactants_smiles)

    # on convertit les noms de solvants en SMILES via le mapping
    # pour pouvoir les chercher dans tox_index
    solvent = conditions.get("solvent")
    if solvent and (smi := solvent_smiles_map.get(solvent)):
        to_check.append(smi)

    co_solvent = conditions.get("co_solvent")
    if co_solvent and (smi := solvent_smiles_map.get(co_solvent)):
        to_check.append(smi)

    # pour chaque composé : on cherche son hazard_score dans le dataset de toxicité
    # si on ne le trouve pas → 0.5 (prudence, on suppose un risque modéré)
    scores = []
    for smi in to_check:
        canon = to_canonical(smi)
        if canon in tox_index:
            scores.append(tox_index[canon]["hazard_score"])
        else:
            scores.append(0.5)

    # hazard_score proche de 1 = dangereux → on retourne 1 - danger
    # pour que le score de sécurité soit "plus haut = mieux"
    avg_danger = sum(scores) / len(scores) if scores else 0.5
    return 1.0 - avg_danger


def build_solvent_map(tox_index: dict) -> dict:
    # on construit le mapping nom courant → SMILES canonique
    # à partir des noms dans le dataset de toxicité
    # comme ça on n'a aucune donnée codée en dur dans le code
    #
    # en plus de ça, on ajoute les abréviations courantes des labos
    # qui ne correspondent pas forcément aux noms dans tox_index
    abbrev = {
        # abréviations labo → SMILES direct (les seules données codées en dur,
        # mais ce sont juste des alias de nommage, pas des données de toxicité)
        "PhMe":   "Cc1ccccc1",
        "DCM":    "ClCCl",
        "THF":    "C1CCOC1",
        "MeOH":   "CO",
        "EtOH":   "CCO",
        "DMF":    "CN(C)C=O",
        "MeCN":   "CC#N",
        "AcOH":   "CC(=O)O",
        "HOAc":   "CC(=O)O",
        "Et2O":   "CCOCC",
        "CHCl3":  "ClC(Cl)Cl",
        "CCl4":   "ClC(Cl)(Cl)Cl",
        "PhH":    "c1ccccc1",
        "PhMe":   "Cc1ccccc1",
        "TFE":    "OCC(F)(F)F",
        "t-BuOH": "CC(C)(C)O",
        "MeNO2":  "C[N+](=O)[O-]",
        "H2O":    "O",
        "o-DCB":  "Clc1ccccc1Cl",
        "dioxane":    "C1COCCO1",
        "1,4-dioxane":"C1COCCO1",
        "hexane":     "CCCCCC",
        "toluene":    "Cc1ccccc1",
        "benzene":    "c1ccccc1",
        "acetone":    "CC(C)=O",
        "pyridine":   "c1ccncc1",
        "i-PrOH":     "CC(C)O",
    }

    # on commence avec les abréviations
    result = dict(abbrev)

    # on ajoute aussi les SMILES bruts qui apparaissent directement
    # dans le champ "solvent" de ton dataset (ex: "C1CCOC1" au lieu de "THF")
    for canon_smi in tox_index:
        result[canon_smi] = canon_smi  # SMILES → lui-même

    return result


# 6. fonctions de scoring par critère
# chaque fonction prend (route_data, tox_index) et retourne un float entre 0 et 1
# route_data contient "dataset_steps" avec les vraies données du dataset

def compute_steps(route_data: dict, tox_index: dict) -> float:
    # score = 1/n : moins d'étapes = mieux
    # 5 étapes → 0.2, 10 étapes → 0.1, 20 étapes → 0.05
    n = len(route_data.get("dataset_steps", []))
    return 1.0 / max(n, 1)


def compute_yield(route_data: dict, tox_index: dict) -> float:
    # rendement global = produit de tous les rendements d'étape
    # étape sans yield_percent dans le dataset → on suppose 50%
    steps = route_data.get("dataset_steps", [])
    if not steps:
        return 0.0
    result = 1.0
    for s in steps:
        yld = s.get("yield_percent")
        result *= (yld / 100.0) if yld is not None else 0.5
    return result


def compute_atom_economy(route_data: dict, tox_index: dict) -> float:
    # moyenne de l'économie atomique sur toutes les étapes
    steps  = route_data.get("dataset_steps", [])
    scores = [
        calc_atom_economy(s.get("reactants_smiles", []), s.get("product_smiles", ""))
        for s in steps
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_e_factor(route_data: dict, tox_index: dict) -> float:
    # e-factor moyen sur toutes les étapes, calculé avec les vraies données du dataset
    steps  = route_data.get("dataset_steps", [])
    scores = [
        calc_e_factor(
            s.get("reactants_smiles", []),
            s.get("product_smiles", ""),
            (s.get("yield_percent") or 50) / 100.0
        )
        for s in steps
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_toxicity(route_data: dict, tox_index: dict) -> float:
    # score de sécurité moyen — on construit le mapping solvant à la volée
    # depuis le dataset de toxicité, pas depuis des données codées en dur
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


# registre des critères disponibles
# pour en ajouter un : écrire compute_X(route_data, tox_index) → float,
# puis l'ajouter ici avec sa description
CRITERIA_REGISTRY = {
    "steps":        {"fn": compute_steps,        "description": "nombre d'étapes (moins = mieux)"},
    "yield":        {"fn": compute_yield,        "description": "rendement global cumulé"},
    "atom_economy": {"fn": compute_atom_economy, "description": "économie atomique moyenne"},
    "e_factor":     {"fn": compute_e_factor,     "description": "e-factor (déchets générés)"},
    "toxicity":     {"fn": compute_toxicity,     "description": "sécurité réactifs + solvants"},
}


# 7. classement par somme pondérée
# poids en 1/i² (i = position du critère, commence à 1) :
#   critère #1 → 1/1  = 1.000 → ~73% du score
#   critère #2 → 1/4  = 0.250 → ~18%
#   critère #3 → 1/9  = 0.111 → ~9%

def compute_weights(criteria: list) -> dict:
    # calcule les poids normalisés pour que leur somme fasse exactement 1
    raw   = {c: 1.0 / (i + 1) ** 2 for i, c in enumerate(criteria)}
    total = sum(raw.values())
    return {c: w / total for c, w in raw.items()}


def rank_weighted(routes: list, criteria: list, tox_index: dict) -> list:
    # calcule le score total de chaque route et trie par ordre décroissant
    # retourne des tuples (score_total, details_par_critère, route_dict)
    # les details permettent d'afficher la contribution de chaque critère dans Streamlit
    weights = compute_weights(criteria)
    scored  = []

    for route in routes:
        details = {}
        total   = 0.0
        for c in criteria:
            raw = CRITERIA_REGISTRY[c]["fn"](route, tox_index)
            w   = weights[c]
            details[c] = {
                "raw":      round(raw, 4),   # score brut du critère (0 à 1)
                "weight":   round(w, 4),     # poids normalisé
                "weighted": round(raw * w, 4) # contribution au score total
            }
            total += raw * w
        scored.append((round(total, 4), details, route))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored


# 8. point d'entrée principal

def find_best_routes(
    target_smiles:     str,
    criteria_priority: list,
    dataset_path:      str = "reaction_dataset.json",
    toxicity_path:     str = "toxicity_dataset.json",
    config_path:       str = "config.yml",
    top_n:             int = 3,
) -> list:
    """
    Lance le pipeline complet et retourne les meilleures routes.

    criteria_priority : liste de EXACTEMENT 3 critères dans l'ordre de priorité.
                        valeurs possibles : "steps", "yield", "atom_economy",
                                           "e_factor", "toxicity"

    retourne : [(score_total, details_par_critère, route_dict), ...]
               trié par score décroissant, les top_n premiers
    """
    # vérification des paramètres avant de lancer quoi que ce soit
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
    print(f"[pipeline] critères : {criteria_priority}")

    # étape 1 — on charge les deux datasets
    print("\n[1/4] chargement des datasets...")
    dataset   = load_reaction_dataset(dataset_path)
    tox_index = load_toxicity_dataset(toxicity_path)

    # étape 2 — AiZynthFinder explore la molécule cible
    print("\n[2/4] recherche AiZynthFinder...")
    all_routes = run_aizynthfinder(target_smiles, config_path)
    print(f"  {len(all_routes)} routes récupérées")

    # étape 3 — on filtre pour ne garder que les routes
    # dont les starting materials sont dans notre dataset
    print("\n[3/4] filtrage par le dataset...")
    valid_routes = filter_routes_by_starting_materials(all_routes, dataset)

    if not valid_routes:
        print("\n[pipeline] aucune route validée.")
        print("  → les starting materials d'AiZynthFinder ne correspondent à aucun réactif du dataset.")
        return []

    # étape 4 — on score et on classe
    print("\n[4/4] scoring...")
    top_routes = rank_weighted(valid_routes, criteria_priority, tox_index)[:top_n]

    print(f"\n[pipeline] {len(top_routes)} routes retournées :")
    for i, (score, _, route) in enumerate(top_routes, 1):
        n_steps = len(route.get("dataset_steps", []))
        print(f"  {i}. {route.get('matched_route_name', '?')} "
              f"({route.get('matched_target', '?')}) "
              f"— {n_steps} étapes — score {score:.4f}")

    return top_routes