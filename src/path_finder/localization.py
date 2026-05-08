# localization.py
# =============================================================================
# All user-visible strings and display constants for the Retrosynthesis Interface.
#
# Contents:
#   LANG             : nested dict of UI strings keyed by language code
#                      ("en" | "fr") then by string key.
#   CRITERIA_LABELS  : per-language display labels (with emoji) for the five
#                      scoring criteria.
#   PALETTE          : ordered list of hex colour strings used for multi-route
#                      matplotlib charts (navy → light blue, cycles if > 5 routes).
#   FIG_BG           : background hex colour shared by all matplotlib figures
#                      and molecule image placeholders.
#
# Usage (in app_path_finder.py):
#   from localization import LANG, CRITERIA_LABELS, PALETTE, FIG_BG
#   T  = LANG[lang]          # e.g. lang = "en"
#   CL = CRITERIA_LABELS[lang]
#
# Adding a new language:
#   1. Add a new key block to LANG with all required string keys.
#   2. Add a matching key block to CRITERIA_LABELS.
#   3. Update the language selector radio widget in app_path_finder.py.
#
# No Streamlit, RDKit, or other runtime dependency — this module is pure Python
# and can be imported or tested independently.
# =============================================================================

# =============================================================================
# UI string catalogue
# =============================================================================
# Every string that appears in the Streamlit UI is defined here so that
# adding or changing a language never requires touching app logic.

LANG = {
    "en": {
        "page_title":       "⚗️ Synthesis Route Finder",
        "page_caption":     "AiZynthFinder (MCTS)  ·  Chemistry by Design  ·  Rxn-INSIGHT",
        "sidebar_title":    "⚗️ Settings",
        "files_section":    "⚙️ Files & options",
        "ds_label":         "Dataset",
        "tox_label":        "Toxicity file",
        "cfg_label":        "AiZ config",
        "rxni_label":       "Rxn-INSIGHT database (optional)",
        "generic_label":    "Generic reactions dataset (optional)",
        "topn_label":       "Routes to display",
        "naiz_label":       "Max AiZ routes to explore",
        "weights_exp":      "📐 View criterion weights",
        "tab_search":       "🔍 Route Search",
        "tab_analysis":     "📊 Analysis",
        "tab_dataset":      "📂 Dataset Explorer",
        "tab_help":         "❓ Help",
        "target_section":   "🎯 Target molecule",
        "mode_pre":         "Predefined",
        "mode_custom":      "Custom SMILES",
        "mol_label":        "Molecule",
        "smiles_label":     "SMILES",
        "smiles_ph":        "e.g. c1ccccc1",
        "smiles_invalid":   "Invalid SMILES",
        "smiles_valid":     "Valid SMILES ✓",
        "criteria_section": "📊 Scoring criteria (priority order)",
        "criteria_caption": "Criterion #1 ≈ 73% of score  ·  #2 ≈ 18%  ·  #3 ≈ 9%",
        "c1_label":         "Criterion #1 — highest weight",
        "c2_label":         "Criterion #2 — medium weight",
        "c3_label":         "Criterion #3 — lowest weight",
        "run_btn":          "🔍 Run search",
        "welcome":          "Choose a target molecule, set your criteria, and click **Run search**.",
        "avail_mols":       "### Available molecules in the dataset",
        "loading_ds":       "📂 Loading dataset…",
        "loading_aiz":      "🔬 Running AiZynthFinder (1–2 min)…",
        "loading_rxni":     "🔮 Rxn-INSIGHT — analysing routes…",
        "search_ok":        "✅ Search complete",
        "err_file":         "❌ File not found",
        "err_param":        "❌ Invalid parameter",
        "err_other":        "❌ Unexpected error",
        "no_routes":        "**No routes found.** Try Galanthamine or Morphine.",
        "n_found":          "{n} route(s) found and ranked.",
        "chart_title":      "Route ranking — {target}",
        "score_axis":       "Total score",
        "metric_score":     "Total score",
        "metric_score_help":"Sum of (raw score × weight) across all criteria.",
        "metric_steps":     "Steps",
        "metric_bottleneck":"Bottleneck yield",
        "metric_bn_help":   "Yield of the worst single step.",
        "metric_avg":       "Avg step yield",
        "contrib_title":    "Score breakdown",
        "score_th_crit":    "Criterion",
        "score_th_raw":     "Raw score (0–1)",
        "score_th_raw_tip": "How well this route performs on this criterion (0=worst, 1=best).",
        "score_th_weight":  "Weight",
        "score_th_contrib": "Contribution",
        "why_best_title":   "🧠 Why is this route ranked #{r}?",
        "flow_title":       "🧬 Animated reaction flow",
        "steps_title":      "### 🧪 Detailed synthesis pathway — {n} steps",
        "yield_ok":         "**Yield:** {y}%",
        "yield_na":         "**Yield:** *not reported (50% default applied)*",
        "yield_predicted":  "**Yield:** *not shown for predicted routes*",
        "cond_lbl":         "**Conditions:** {c}",
        "smi_exp":          "SMILES — step {n}",
        "react_lbl":        "Reactants:",
        "prod_lbl":         "→ Product:",
        "compare_title":    "🆚 Side-by-side route comparison",
        "sel_routes":       "Select routes to compare",
        "radar_title":      "Criterion profile per route",
        "no_analysis":      "Run a search first to unlock the Analysis tab.",
        "ds_not_found":     "Dataset `{path}` not found.",
        "filter_lbl":       "Filter by target",
        "filter_all":       "All",
        "total_rxn":        "Total reactions",
        "dist_routes":      "Distinct routes",
        "tgt_mols":         "Target molecules",
        "col_route":        "Route",
        "col_target":       "Target",
        "col_steps":        "Steps",
        "col_cumyield":     "Cumulative yield",
        "col_missing":      "Missing yields",
        "col_avg":          "Avg yield",
        "detail_sel":       "View route details",
        "help_title":       "How it works",
        "footer":           "Chemistry by Design  ·  AiZynthFinder MCTS  ·  Rxn-INSIGHT",
        "mod_err":          "❌ Cannot load route_engine:",
        "searching":        "🔍 Searching…",
        "sec_dataset":      "## 📚 Routes from the dataset",
        "sec_validated":    "## ✅ Validated routes (AiZ + generic dataset)",
        "sec_predicted":    "## 🔮 Purely predicted routes (Rxn-INSIGHT only)",
        "cap_dataset":      "Experimentally validated — real conditions and yields",
        "cap_validated":    "AiZynthFinder routes with steps referenced in the generic dataset",
        "cap_predicted":    "AiZynthFinder routes with no steps in any dataset — Rxn-INSIGHT conditions only",
        "badge_dataset":    "📚 dataset",
        "badge_validated":  "✅ validated",
        "badge_partial":    "⚡ partial",
        "badge_predicted":  "🔮 predicted",
        "why_steps":        "it has only <strong>{n} steps</strong> — one of the shortest pathways found",
        "why_yield":        "it achieves a bottleneck yield of <strong>{y}%</strong> on the limiting step",
        "why_avg_yield":    "its average step yield of <strong>{a}%</strong> is above the field average",
        "why_top_score":    "it dominates on <strong>{crit}</strong> (raw score {s:.2f}), the highest-weighted criterion ({w:.0%})",
        "why_balanced":     "it presents a well-balanced profile across all three selected criteria",
        "why_prefix":       "This route is ranked <strong>#{r}</strong> because:",
        "why_suffix":       "No single criterion alone determines ranking — the weighted combination reflects your stated priorities.",
    },
    "fr": {
        "page_title":       "⚗️ Recherche de routes de synthèse",
        "page_caption":     "AiZynthFinder (MCTS)  ·  Chemistry by Design  ·  Rxn-INSIGHT",
        "sidebar_title":    "⚗️ Paramètres",
        "files_section":    "⚙️ Fichiers & options",
        "ds_label":         "Dataset",
        "tox_label":        "Fichier toxicité",
        "cfg_label":        "Config AiZ",
        "rxni_label":       "Base Rxn-INSIGHT (optionnel)",
        "generic_label":    "Dataset réactions génériques (optionnel)",
        "topn_label":       "Routes à afficher",
        "naiz_label":       "Max routes AiZ à explorer",
        "weights_exp":      "📐 Voir les poids des critères",
        "tab_search":       "🔍 Recherche de routes",
        "tab_analysis":     "📊 Analyse",
        "tab_dataset":      "📂 Explorer le dataset",
        "tab_help":         "❓ Aide",
        "target_section":   "🎯 Molécule cible",
        "mode_pre":         "Prédéfinie",
        "mode_custom":      "SMILES personnalisé",
        "mol_label":        "Molécule",
        "smiles_label":     "SMILES",
        "smiles_ph":        "ex : c1ccccc1",
        "smiles_invalid":   "SMILES invalide",
        "smiles_valid":     "SMILES valide ✓",
        "criteria_section": "📊 Critères de scoring (ordre de priorité)",
        "criteria_caption": "Critère #1 ≈ 73% du score  ·  #2 ≈ 18%  ·  #3 ≈ 9%",
        "c1_label":         "Critère #1 — poids le plus élevé",
        "c2_label":         "Critère #2 — poids moyen",
        "c3_label":         "Critère #3 — poids le plus bas",
        "run_btn":          "🔍 Lancer la recherche",
        "welcome":          "Choisissez une molécule cible, définissez vos critères et cliquez sur **Lancer la recherche**.",
        "avail_mols":       "### Molécules disponibles dans le dataset",
        "loading_ds":       "📂 Chargement du dataset…",
        "loading_aiz":      "🔬 Lancement d'AiZynthFinder (1–2 min)…",
        "loading_rxni":     "🔮 Rxn-INSIGHT — analyse des routes…",
        "search_ok":        "✅ Recherche terminée",
        "err_file":         "❌ Fichier introuvable",
        "err_param":        "❌ Paramètre invalide",
        "err_other":        "❌ Erreur inattendue",
        "no_routes":        "**Aucune route trouvée.** Essayez la galanthamine ou la morphine.",
        "n_found":          "{n} route(s) trouvée(s) et classée(s).",
        "chart_title":      "Classement des routes — {target}",
        "score_axis":       "Score total",
        "metric_score":     "Score total",
        "metric_score_help":"Somme de (score brut × poids) sur tous les critères.",
        "metric_steps":     "Étapes",
        "metric_bottleneck":"Rendement limitant",
        "metric_bn_help":   "Rendement de l'étape la plus faible.",
        "metric_avg":       "Rendement moyen / étape",
        "contrib_title":    "Décomposition du score",
        "score_th_crit":    "Critère",
        "score_th_raw":     "Score brut (0–1)",
        "score_th_raw_tip": "Performance de la route sur ce critère (0=pire, 1=meilleur).",
        "score_th_weight":  "Poids",
        "score_th_contrib": "Contribution",
        "why_best_title":   "🧠 Pourquoi cette route est-elle classée #{r} ?",
        "flow_title":       "🧬 Flux de réaction animé",
        "steps_title":      "### 🧪 Chemin de synthèse détaillé — {n} étapes",
        "yield_ok":         "**Rendement :** {y}%",
        "yield_na":         "**Rendement :** *non renseigné (50% par défaut)*",
        "yield_predicted":  "**Rendement :** *non affiché pour les routes prédites*",
        "cond_lbl":         "**Conditions :** {c}",
        "smi_exp":          "SMILES — étape {n}",
        "react_lbl":        "Réactifs :",
        "prod_lbl":         "→ Produit :",
        "compare_title":    "🆚 Comparaison côte à côte des routes",
        "sel_routes":       "Sélectionner les routes à comparer",
        "radar_title":      "Profil par critère",
        "no_analysis":      "Lancez une recherche d'abord pour accéder à l'onglet Analyse.",
        "ds_not_found":     "Dataset `{path}` introuvable.",
        "filter_lbl":       "Filtrer par cible",
        "filter_all":       "Toutes",
        "total_rxn":        "Réactions totales",
        "dist_routes":      "Routes distinctes",
        "tgt_mols":         "Molécules cibles",
        "col_route":        "Route",
        "col_target":       "Cible",
        "col_steps":        "Étapes",
        "col_cumyield":     "Rendement cumulé",
        "col_missing":      "Rendements manquants",
        "col_avg":          "Rendement moyen",
        "detail_sel":       "Voir le détail d'une route",
        "help_title":       "Comment ça marche",
        "footer":           "Chemistry by Design  ·  AiZynthFinder MCTS  ·  Rxn-INSIGHT",
        "mod_err":          "❌ Impossible de charger route_engine :",
        "searching":        "🔍 Recherche en cours…",
        "sec_dataset":      "## 📚 Routes du dataset",
        "sec_validated":    "## ✅ Routes validées (AiZ + dataset générique)",
        "sec_predicted":    "## 🔮 Routes purement prédites (Rxn-INSIGHT uniquement)",
        "cap_dataset":      "Routes validées expérimentalement — conditions et rendements réels",
        "cap_validated":    "Routes AiZynthFinder avec étapes référencées dans le dataset générique",
        "cap_predicted":    "Routes AiZynthFinder sans aucune étape dans le dataset — conditions Rxn-INSIGHT uniquement",
        "badge_dataset":    "📚 dataset",
        "badge_validated":  "✅ validée",
        "badge_partial":    "⚡ partielle",
        "badge_predicted":  "🔮 prédite",
        "why_steps":        "elle ne compte que <strong>{n} étapes</strong> — l'un des chemins les plus courts",
        "why_yield":        "elle atteint un rendement limitant de <strong>{y}%</strong> sur l'étape critique",
        "why_avg_yield":    "son rendement moyen de <strong>{a}%</strong> est au-dessus de la moyenne",
        "why_top_score":    "elle domine sur <strong>{crit}</strong> (score brut {s:.2f}), le critère le plus pondéré ({w:.0%})",
        "why_balanced":     "elle présente un profil équilibré sur les trois critères sélectionnés",
        "why_prefix":       "Cette route est classée <strong>#{r}</strong> car :",
        "why_suffix":       "Aucun critère seul ne détermine le classement — la combinaison pondérée reflète vos priorités.",
    },
}

# =============================================================================
# Per-criterion display labels
# =============================================================================
# Each criterion key maps to its emoji-prefixed display label.
# These labels appear in: sidebar weight bars, score tables, chart axes,
# the Analysis tab comparison dataframe, and the criteria selectboxes.

CRITERIA_LABELS = {
    "en": {
        "steps":        "🔢 Steps",
        "yield":        "📈 Yield",
        "atom_economy": "⚗️ Atom economy",
        "e_factor":     "♻️ E-factor",
        "toxicity":     "☣️ Safety",
    },
    "fr": {
        "steps":        "🔢 Étapes",
        "yield":        "📈 Rendement",
        "atom_economy": "⚗️ Écon. atomique",
        "e_factor":     "♻️ E-factor",
        "toxicity":     "☣️ Sécurité",
    },
}

# =============================================================================
# Chart and figure constants
# =============================================================================

# Colour palette for multi-route bar charts — navy → light blue.
# Cycles automatically when more than 5 routes are plotted.
PALETTE = ["#1a2e44", "#2d5986", "#4a86c8", "#6aabd2", "#9fc8e0"]

# Background colour applied to all matplotlib figures and PIL fallback images.
# Matches the Streamlit page background to avoid jarring white boxes in charts.
FIG_BG = "#f9fbfd"
