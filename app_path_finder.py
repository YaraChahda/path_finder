# app_with_pur_aizyn.py — Retrosynthesis Interface
# ============================================================
# Base: app_test.py — design, bilingual, Pareto, score table, animated flow
# Extensions: Rxn-INSIGHT, 3 categories (dataset/validated/predicted),
#             generic dataset, reaction scheme with SMILES co-reactant visualization
#
# Run: streamlit run app_with_pur_aizyn.py

import re
import io
import os
import json
import base64
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── chemistry imports ─────────────────────────────────────────────────────────
try:
    import function_path_finder as fi
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    MODULE_OK  = True
    MODULE_ERR = ""
except Exception as e:
    MODULE_OK  = False
    MODULE_ERR = str(e)

try:
    from rxn_insight.reaction import Reaction as RxnInsightReaction
    RXNINSIGHT_OK = True
except ImportError:
    RXNINSIGHT_OK = False

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retrosynthesis — Chemistry by Design",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _hires_fig(*args, dpi: int = 180, **kwargs):
    """Creates a high-resolution matplotlib figure. Drop-in for plt.subplots()."""
    fig, ax = plt.subplots(*args, dpi=dpi, **kwargs)
    fig.patch.set_facecolor(FIG_BG)
    if hasattr(ax, '__iter__'):
        for a in ax: a.set_facecolor(FIG_BG)
    else:
        ax.set_facecolor(FIG_BG)
    return fig, ax

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; color: #1a2e44; }

[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif;
    font-size: 1.45rem !important;
    color: #1a2e44;
}
[data-testid="stMetricLabel"] {
    font-size: 0.76rem !important;
    color: #6b7a8d;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="stExpander"] {
    border: 1px solid #dce3ec;
    border-radius: 12px;
    background: #f9fbfd;
}
.stButton > button {
    background: linear-gradient(135deg, #1a2e44 0%, #2d5986 100%);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0.04em;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ── CSS for isolated HTML components ─────────────────────────────────────────
COMPONENT_STYLE = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

  .score-table {
    width:100%; border-collapse:collapse;
    font-family:'DM Sans',sans-serif; font-size:0.87rem; margin:4px 0 14px 0;
  }
  .score-table th {
    background:#1a2e44; color:white; padding:8px 14px;
    text-align:left; font-weight:600; font-size:0.75rem; letter-spacing:0.05em;
  }
  .score-table th:nth-child(2),
  .score-table th:nth-child(3),
  .score-table th:nth-child(4) { text-align:right; }
  .score-table td {
    padding:7px 14px; border-bottom:1px solid #edf1f7;
    color:#1a2e44; vertical-align:middle;
  }
  .score-table td:nth-child(2),
  .score-table td:nth-child(3),
  .score-table td:nth-child(4) {
    text-align:right; font-weight:600; font-variant-numeric:tabular-nums;
  }
  .score-table tr:nth-child(even) { background:#f4f7fb; }
  .th-help { cursor:help; border-bottom:1px dashed rgba(255,255,255,0.6); display:inline-block; }
  .th-info {
    display:inline-block; background:rgba(255,255,255,0.25);
    border-radius:50%; width:14px; height:14px;
    font-size:0.65rem; line-height:14px; text-align:center;
    font-weight:700; margin-left:4px; cursor:help; vertical-align:middle;
  }

  .why-box {
    background:linear-gradient(135deg,#edf4fb,#f4f7fb);
    border-left:4px solid #2d5986;
    border-radius:0 10px 10px 0;
    padding:14px 18px; margin:10px 0 18px 0;
    font-family:'DM Sans',sans-serif; font-size:0.89rem;
    color:#1a2e44; line-height:1.65;
  }
  .why-box strong { color:#1a2e44; font-weight:700; }
  .why-box ul { margin:6px 0 6px 20px; padding:0; }
  .why-box li { margin-bottom:3px; }
  .why-box em { font-size:0.82rem; color:#6b7a8d; }

  /* source badge for steps */
  .src-badge {
    display:inline-block; border-radius:10px;
    padding:1px 8px; font-size:0.7rem; font-weight:600;
    font-family:'DM Sans',sans-serif; margin-left:6px;
  }
  .src-dataset    { background:#d4edda; color:#155724; }
  .src-generic    { background:#cce5ff; color:#004085; }
  .src-rxninsight { background:#fff3cd; color:#856404; }

  @keyframes fadeSlideIn {
    from { opacity:0; transform:translateX(-14px); }
    to   { opacity:1; transform:translateX(0); }
  }
  @keyframes arrowPulse {
    0%,100% { opacity:0.55; }
    50%      { opacity:1; }
  }

  .rxn-flow-wrap {
    display:flex; align-items:center; gap:0;
    overflow-x:auto; padding:18px 12px;
    background:#f9fbfd; border-radius:14px;
    border:1px solid #dce3ec; margin-bottom:14px;
  }
  .rxn-step-card {
    min-width:165px; max-width:205px;
    background:white; border-radius:10px;
    border:1.5px solid #dce3ec;
    padding:10px 8px 8px 8px;
    display:flex; flex-direction:column; align-items:center;
    animation:fadeSlideIn 0.45s ease both;
    box-shadow:0 2px 8px rgba(26,46,68,0.07); flex-shrink:0;
  }
  .rxn-step-badge {
    background:linear-gradient(135deg,#1a2e44,#2d5986);
    color:white; font-size:0.65rem; font-weight:700;
    letter-spacing:0.08em; text-transform:uppercase;
    border-radius:20px; padding:2px 9px; margin-bottom:6px;
    font-family:'DM Sans',sans-serif;
  }
  .rxn-step-type {
    font-size:0.69rem; color:#6b7a8d;
    text-align:center; margin-bottom:5px;
    font-family:'DM Sans',sans-serif;
  }
  .rxn-step-mol {
    width:132px; height:92px;
    object-fit:contain; border-radius:6px; background:#f0f4f8;
  }
  .rxn-step-yield {
    margin-top:6px; font-size:0.76rem;
    font-weight:600; color:#1a2e44;
    font-family:'DM Sans',sans-serif;
  }
  .rxn-arrow-zone {
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    padding:0 4px; min-width:90px; flex-shrink:0;
  }
  .rxn-arrow-cond {
    font-size:0.59rem; color:#37506e;
    max-width:86px; text-align:center;
    margin-bottom:5px; line-height:1.35;
    font-family:'DM Sans',sans-serif;
    word-break:break-word; white-space:normal;
  }
  .rxn-arrow-shaft {
    width:62px; height:3px;
    background:linear-gradient(90deg,#2d5986,#1a2e44);
    border-radius:2px 0 0 2px;
    animation:arrowPulse 2s ease-in-out infinite;
  }
  .rxn-arrow-head {
    width:0; height:0;
    border-top:7px solid transparent;
    border-bottom:7px solid transparent;
    border-left:11px solid #1a2e44;
    animation:arrowPulse 2s ease-in-out infinite; flex-shrink:0;
  }
  .rxn-arrow-row { display:flex; align-items:center; justify-content:center; }
  .rxn-start-label {
    font-size:0.65rem; font-weight:700; color:#4a86c8;
    text-transform:uppercase; letter-spacing:0.07em;
    margin-bottom:3px; font-family:'DM Sans',sans-serif;
  }
  .rxn-end-label {
    font-size:0.65rem; font-weight:700; color:#1a2e44;
    text-transform:uppercase; letter-spacing:0.07em;
    margin-bottom:3px; font-family:'DM Sans',sans-serif;
  }
</style>
"""

# ── strip emoji for matplotlib ────────────────────────────────────────────────
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U00002600-\U000027BF"
    "\U0001F004-\U0001F0CF"
    "\U0001F1E0-\U0001F1FF"
    "\u2702-\u27B0"
    "\u24C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
def strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text).strip()

# ── language strings ──────────────────────────────────────────────────────────
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
        "pareto_title":     "Pareto Front",
        "pareto_x":         "X axis",
        "pareto_y":         "Y axis (must differ from X)",
        "pareto_dominated": "Dominated routes",
        "pareto_front":     "Pareto-optimal routes",
        "pareto_note":      "**Pareto-optimal routes** (navy dots) cannot be improved on one criterion without worsening another.",
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
        "mod_err":          "❌ Cannot load function_path_finder:",
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
        "quiz_title":       "🧩 Retrosynthesis Quiz",
        "quiz_caption":     "Test your knowledge while AiZynthFinder is running!",
        "quiz_q":           "Which molecule is the most likely precursor of this structure?",
        "quiz_correct":     "✅ Correct!",
        "quiz_wrong":       "❌ Wrong. The correct answer was: {a}",
        "quiz_next":        "Next question →",
        "quiz_score":       "Score: {c}/{t}",
        "quiz_hint":        "Think about which bond disconnection makes the most chemical sense.",
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
        "pareto_title":     "Front de Pareto",
        "pareto_x":         "Axe X",
        "pareto_y":         "Axe Y (doit différer de X)",
        "pareto_dominated": "Routes dominées",
        "pareto_front":     "Routes Pareto-optimales",
        "pareto_note":      "**Routes Pareto-optimales** (points bleu marine) : impossible de les améliorer sur un critère sans en dégrader un autre.",
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
        "mod_err":          "❌ Impossible de charger function_path_finder :",
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
        "quiz_title":       "🧩 Quiz de rétrosynthèse",
        "quiz_caption":     "Testez vos connaissances pendant qu'AiZynthFinder travaille !",
        "quiz_q":           "Quel composé est le précurseur le plus probable de cette structure ?",
        "quiz_correct":     "✅ Correct !",
        "quiz_wrong":       "❌ Faux. La bonne réponse était : {a}",
        "quiz_next":        "Question suivante →",
        "quiz_score":       "Score : {c}/{t}",
        "quiz_hint":        "Pensez à quelle déconnexion de liaison est la plus logique chimiquement.",
    },
}

CRITERIA_LABELS = {
    "en": {"steps":"🔢 Steps","yield":"📈 Yield","atom_economy":"⚗️ Atom economy","e_factor":"♻️ E-factor","toxicity":"☣️ Safety"},
    "fr": {"steps":"🔢 Étapes","yield":"📈 Rendement","atom_economy":"⚗️ Écon. atomique","e_factor":"♻️ E-factor","toxicity":"☣️ Sécurité"},
}

PALETTE = ["#1a2e44","#2d5986","#4a86c8","#6aabd2","#9fc8e0"]
FIG_BG  = "#f9fbfd"

# ── language selector ─────────────────────────────────────────────────────────
with st.sidebar:
    lang_choice = st.radio("🌐 Language / Langue", ["🇬🇧 English", "🇫🇷 Français"],
                           horizontal=True, index=0)
    lang = "en" if "English" in lang_choice else "fr"

T  = LANG[lang]
CL = CRITERIA_LABELS[lang]


# =============================================================================
# image utilities
# =============================================================================

def mol_png(smiles: str, w: int = 800, h: int = 540) -> bytes | None:
    """
    Cairo 2x rendering — used by st.image() throughout the app for high-DPI
    molecule display (target preview, Dataset Explorer step-by-step, substances panel).
    """
    if not MODULE_OK or not smiles: return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(w * 2, h * 2)
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = True
        opts.bondLineWidth = 2.5
        opts.padding = 0.14
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        img = Draw.MolToImage(mol, size=(w*2, h*2))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def _mol_b64_or_text_svg(smiles: str, w: int, h: int) -> str:
    """
    Returns a PNG data-URI (Cairo) for a molecule — used exclusively inside
    build_clickable_scheme_html() to embed molecule images in the HTML reaction
    scheme (main molecules and co-reactants above arrows).
    Handles ions and single atoms ([Pd], [Na+], etc.) without Compute2DCoords.
    Falls back to a labelled grey rectangle if SMILES is invalid.
    """
    if not smiles or not MODULE_OK:
        return _fallback_data_uri(smiles or "?", w, h)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _fallback_data_uri(smiles, w, h)
    try:
        from PIL import Image as _PI
        import io as _io
        if mol.GetNumAtoms() > 1:
            rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(w * 4, h * 4)
        drawer.drawOptions().bondLineWidth     = 1.0
        drawer.drawOptions().padding           = 0.15
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        data = drawer.GetDrawingText()
        img  = _PI.open(_io.BytesIO(data))
        img  = img.resize((w * 2, h * 2), _PI.LANCZOS)
        buf  = _io.BytesIO()
        img.save(buf, format="PNG")
        b64  = base64.b64encode(buf.getvalue()).decode()
        return "data:image/png;base64," + b64
    except Exception:
        return _fallback_data_uri(smiles, w, h)


def _fallback_data_uri(text: str, w: int, h: int) -> str:
    """
    PNG fallback with a text label — returned by _mol_b64_or_text_svg() when
    RDKit cannot parse the SMILES (ions, malformed strings, single atoms).
    """
    try:
        from PIL import Image as _PI, ImageDraw as _PID
        img  = _PI.new("RGB", (w, h), (248, 248, 248))
        draw = _PID.Draw(img)
        display = text[:18] + "…" if len(text) > 18 else text
        draw.rectangle([2, 2, w-3, h-3], outline=(200, 200, 200))
        try:
            bbox = draw.textbbox((0,0), display)
            tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        except Exception:
            tw = len(display) * 6; th = 12
        draw.text(((w-tw)//2, (h-th)//2), display, fill=(80, 80, 80))
        import io as _io
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return "data:image/png;base64," + b64
    except Exception:
        return ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
                "AAAADULEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def _is_trivial_smiles(smiles: str) -> bool:
    """
    Returns True for single atoms, ions, or salts where all fragments have ≤ 2
    atoms. Used in build_clickable_scheme_html() to decide whether a co-reactant
    should be rendered as a 2D structure image or shown as plain text below the
    arrow (avoids cluttering the scheme with uninformative single-atom images).
    """
    if not smiles or not MODULE_OK: return True
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return True
    if mol.GetNumAtoms() <= 2: return True
    frags = smiles.split('.')
    if all(
        Chem.MolFromSmiles(f) is not None and Chem.MolFromSmiles(f).GetNumAtoms() <= 2
        for f in frags if f
    ): return True
    return False


# =============================================================================
# Reaction scheme HTML builder
# =============================================================================

def _is_purification_step(step: dict) -> bool:
    """
    Returns True when a step should be treated as a purification step and
    displayed unconditionally in the scheme, even if reactant SMILES equals
    product SMILES.

    A step is considered a purification step when ANY of the following holds:
      - the reaction_type field contains 'purif' (case-insensitive), OR
      - the reaction_type field contains 'recryst' (recrystallisation), OR
      - the reaction_type field contains 'chroma' (chromatography), OR
      - there are no co-reactants other than the substrate itself
        AND no reagents listed in conditions AND the product SMILES equals
        one of the reactant SMILES (identity transform = isolation step).

    NOTE: the last condition alone (identity transform) is the key fix requested.
    The reaction_type checks are additional guards for explicitly labelled steps.
    """
    rtype = (step.get("reaction_type") or "").lower()
    if any(kw in rtype for kw in ("purif", "recryst", "chroma", "isolation", "workup")):
        return True
    prod   = step.get("product_smiles", "")
    reacs  = step.get("reactants_smiles", [])
    # Identity transform: the product is identical to one of the reactants
    if prod and prod in reacs:
        return True
    # Canonical comparison via RDKit (handles tautomers / different SMILES notations)
    if MODULE_OK and prod:
        canon_prod = Chem.MolToSmiles(Chem.MolFromSmiles(prod)) if Chem.MolFromSmiles(prod) else prod
        for r in reacs:
            mol_r = Chem.MolFromSmiles(r) if r else None
            if mol_r and Chem.MolToSmiles(mol_r) == canon_prod:
                return True
    return False


def build_clickable_scheme_html(steps_data: list, route_id: str,
                                 is_predicted: bool = False) -> str:
    """
    Builds the interactive HTML reaction scheme shown in the Route Search and
    Analysis tabs. Each molecule is displayed as a Cairo PNG; clicking an arrow
    opens a detail panel with conditions, yield, and per-step SMILES.

    Key behaviour for purification steps
    ─────────────────────────────────────
    When _is_purification_step() returns True for a step, the product is shown
    in the molecule sequence even if its SMILES is identical to the preceding
    molecule (standard reactions skip duplicates to avoid visual repetition).
    The arrow is drawn with a dashed style and labelled "Purification" when no
    explicit reaction_type is set, so the user can immediately see it is an
    isolation / workup step rather than a bond-forming transformation.

    ── LAYOUT PARAMETERS (adjust these to resize the scheme) ──────────────────
    MOL_W, MOL_H    : main molecule image size in pixels
    CO_W,  CO_H     : co-reactant image size (solvents, reagents above the arrow)
    SCHEME_H        : height of the grey band containing the molecules
    PAD_TOP         : grey space ABOVE the molecule band (where co-reactants float)
    PAD_BOTTOM      : grey space BELOW the molecule band
    ARROW_SHAFT_W   : width of the horizontal arrow shaft in pixels
    ARROW_CELL_W    : total min-width of the arrow cell including labels
    ────────────────────────────────────────────────────────────────────────────
    """
    if not steps_data:
        return "<p style='color:#888'>No steps.</p>"

    arrow_color = "#E65100" if is_predicted else "#1a2e44"
    hover_bg    = "#FFF3E0" if is_predicted else "#E8F0FC"
    panel_bg    = "#FFF8F0" if is_predicted else "#F0F7FF"
    rid_js      = "".join(c for c in route_id if c.isalnum())

    # ── LAYOUT PARAMETERS ─────────────────────────────────────────────────────
    MOL_W, MOL_H = 158, 112
    CO_W,  CO_H  = 96,  70
    CELL_W = MOL_W + 20
    SCHEME_H = 200

    max_co_rows = 1
    for step in steps_data:
        reactants = step.get("reactants_smiles", [])
        product   = step.get("product_smiles", "")
        cond      = step.get("conditions", {})
        co_count = 0
        for r in reactants:
            if r != product and not _is_trivial_smiles(r):
                co_count += 1
        solv = cond.get("solvent", "")
        if solv and not _is_trivial_smiles(solv):
            co_count += 1
        for r in (cond.get("reagents", []) or []):
            if r and not _is_trivial_smiles(r):
                co_count += 1
        rows = max(1, (co_count + 2) // 3)
        max_co_rows = max(max_co_rows, rows)

    display_rows = min(max_co_rows, 2)
    PAD_TOP    = display_rows * (CO_H + 2) + 30 + 20
    PAD_BOTTOM = display_rows * (CO_H + 2) // 3 + 10

    mol_sequence = []
    arrow_data   = []

    for i, step in enumerate(steps_data):
        reactants   = step.get("reactants_smiles", [])
        product     = step.get("product_smiles", "")
        cond        = step.get("conditions", {})
        rtype       = step.get("reaction_type", "") or ""
        yld         = step.get("yield_percent")
        snum        = step.get("step_number", i + 1)
        src_step    = step.get("source", "dataset")
        is_purif    = _is_purification_step(step)

        # ── Build molecule sequence ──────────────────────────────────────────
        # For purification steps, always add the product even when it equals
        # the preceding molecule in the sequence (identity transform).
        if i == 0 and reactants:
            if reactants[0] not in mol_sequence:
                mol_sequence.append(reactants[0])

        if product:
            # Standard behaviour: skip if already the last molecule
            if mol_sequence and mol_sequence[-1] == product and not is_purif:
                pass  # duplicate, do not add
            else:
                mol_sequence.append(product)

        prev = mol_sequence[-2] if len(mol_sequence) >= 2 else ""
        co_draw = []
        co_text = []
        for r in reactants:
            # For purification steps the reactant == product, but we still want
            # to show any additional reagents/solvents above the arrow.
            if r == prev or r == product:
                continue
            if _is_trivial_smiles(r): co_text.append(r)
            else: co_draw.append(r)

        all_cond_parts = []
        temp = cond.get("temperature_C")
        if temp: all_cond_parts.append(str(temp) + "°C")
        elif cond.get("temp_range"): all_cond_parts.append(cond["temp_range"])
        solv = cond.get("solvent", "")
        if solv:
            if _is_trivial_smiles(solv): all_cond_parts.append(solv)
            else:
                if solv not in co_draw: co_draw.append(solv)
                all_cond_parts.append(solv)
        for r in (cond.get("reagents", []) or []):
            if not r: continue
            if _is_trivial_smiles(r):
                if r not in all_cond_parts: all_cond_parts.append(r)
            else:
                if r not in co_draw: co_draw.append(r)
                if r not in all_cond_parts: all_cond_parts.append(r)
        if cond.get("apparatus"): all_cond_parts.append("(" + cond["apparatus"] + ")")
        cond_display = "  ·  ".join(all_cond_parts)

        # Label purification steps explicitly when no reaction_type is set
        display_rtype = rtype
        if is_purif and not rtype:
            display_rtype = "Purification / Isolation"

        below_parts = []
        if yld is not None and src_step != "rxn-insight":
            below_parts.append(str(yld) + "%")
        for t in co_text[:3]: below_parts.append(t)
        co_draw_set = set(co_draw)
        for t in all_cond_parts[:4]:
            if t not in below_parts and t not in co_draw_set:
                below_parts.append(t)

        yld_str = (str(yld) + "%" if yld is not None and src_step != "rxn-insight" else "")

        arrow_data.append({
            "step": snum, "rtype": display_rtype, "yld": yld_str,
            "cond": cond_display, "co_draw": co_draw, "co_text": co_text,
            "below": below_parts, "reactants": reactants, "product": product,
            "source": src_step, "fg": step.get("fg_reactants", []),
            "is_purif": is_purif,
        })

    mol_imgs = {smi: _mol_b64_or_text_svg(smi, MOL_W, MOL_H)
                for smi in mol_sequence if smi}
    co_imgs  = {}
    for a in arrow_data:
        for smi in a["co_draw"]:
            if smi and smi not in co_imgs:
                co_imgs[smi] = _mol_b64_or_text_svg(smi, CO_W, CO_H)

    step_imgs = {}
    for a in arrow_data:
        r_b64s = [_mol_b64_or_text_svg(rsmi, 110, 78) for rsmi in a["reactants"][:4]]
        p_b64  = _mol_b64_or_text_svg(a["product"], 110, 78)
        step_imgs[str(a["step"])] = {
            "reactants": r_b64s, "product": p_b64,
            "reactant_smiles": a["reactants"], "product_smiles": a["product"],
        }
    step_imgs_json = json.dumps(step_imgs)

    n_mols = len(mol_sequence)
    items  = []

    for idx, smi in enumerate(mol_sequence):
        uri  = mol_imgs.get(smi, _fallback_data_uri(smi, MOL_W, MOL_H))
        role = ""
        if idx == 0:
            role = '<div class="mol-role sm-role">Starting material</div>'
        elif idx == n_mols - 1:
            role = '<div class="mol-role tgt-role">Target</div>'

        items.append(
            '<div class="mol-cell">' + role +
            '<img src="' + uri + '" width="' + str(MOL_W) + '" height="' + str(MOL_H) +
            '" style="display:block;border-radius:4px;"/>'
            '</div>'
        )

        if idx < len(arrow_data):
            a = arrow_data[idx]
            co_draw_slice = a["co_draw"][:4]
            n_co = len(co_draw_slice)

            def _co_img_tag(s):
                uco = co_imgs.get(s, _fallback_data_uri(s, CO_W, CO_H))
                return ('<img src="' + uco + '" width="' + str(CO_W) +
                        '" height="' + str(CO_H) +
                        '" style="border:1px solid #dce3ec;border-radius:3px;background:#fff;" '
                        'title="' + s.replace('"', '&quot;') + '"/>')

            if n_co == 0:
                co_html = ""
            elif n_co == 1:
                co_html = ('<div class="co-row">'
                           + _co_img_tag(co_draw_slice[0])
                           + '</div>')
            else:
                top_row = '<div class="co-row">'
                for s in co_draw_slice[:-1]:
                    top_row += _co_img_tag(s)
                top_row += '</div>'
                bot_row = ('<div class="co-row">'
                           + _co_img_tag(co_draw_slice[-1])
                           + '</div>')
                co_html = top_row + bot_row

            step_json_esc = json.dumps({
                k: v for k, v in a.items() if k not in ("co_draw","co_text","below","is_purif")
            }).replace('"', "&quot;")

            # Purification steps use a dashed arrow style to signal visually
            # that no bond-forming chemistry is happening at this stage.
            purif_dash = "stroke-dasharray:6,3;" if a["is_purif"] else ""
            purif_color = "#8B5E3C" if a["is_purif"] else arrow_color

            below_html = ""
            if a["yld"]: below_html += '<div class="b-yld">' + a["yld"] + '</div>'
            if a["rtype"]: below_html += '<div class="b-rtype">' + a["rtype"] + '</div>'
            for t in a["below"]:
                if t != a["yld"]:
                    below_html += '<div class="b-cond">' + str(t)[:32] + '</div>'

            items.append(
                '<div class="arrow-cell" data-step="' + step_json_esc + '"'
                ' onclick="showStepFn' + rid_js + '(' + str(idx) + ',this)">'
                '<div class="co-above">' + co_html + '</div>'
                '<div class="arrow-line">'
                '<div class="a-shaft" style="background:' + purif_color + ';' + purif_dash + '"></div>'
                '<div class="a-head" style="border-left-color:' + purif_color + ';"></div>'
                '</div>'
                '<div class="step-lbl" style="color:' + purif_color + ';">Step ' + str(a["step"]) + '</div>'
                + below_html +
                '</div>'
            )

    ARROW_SHAFT_W = 90
    ARROW_CELL_W  = 145

    css = (
        "html,body{margin:0;padding:0;background:#fff;"
        "font-family:'DM Sans',Arial,sans-serif;}"
        "*{box-sizing:border-box}"

        "#scroll-wrap{"
        "overflow-x:auto;"
        "overflow-y:visible;"
        "background:#FAFAFA;"
        "border-radius:10px;border:1px solid #dce3ec;"
        "}"

        ".band{"
        "display:flex;"
        "align-items:center;"
        "justify-content:flex-start;"
        "height:" + str(SCHEME_H) + "px;"
        "gap:2px;"
        "padding:0 8px;"
        "overflow:visible;"
        "padding-top:" + str(PAD_TOP) + "px;"
        "padding-bottom:" + str(PAD_BOTTOM) + "px;"
        "box-sizing:content-box;"
        "}"

        ".mol-cell{"
        "display:flex;flex-direction:column;align-items:center;justify-content:center;"
        "min-width:" + str(CELL_W) + "px;max-width:" + str(CELL_W) + "px;"
        "height:" + str(SCHEME_H) + "px;"
        "background:#fff;border:1px solid #dce3ec;border-radius:8px;padding:4px;"
        "box-shadow:0 1px 4px rgba(26,46,68,0.06);flex-shrink:0;"
        "}"
        ".mol-role{font-size:0.55rem;font-weight:700;letter-spacing:0.06em;"
        "text-transform:uppercase;border-radius:10px;padding:1px 6px;margin-bottom:3px;"
        "white-space:nowrap;}"
        ".sm-role{background:#cce5ff;color:#004085}"
        ".tgt-role{background:#d4edda;color:#155724}"

        ".arrow-cell{"
        "position:relative;"
        "display:flex;flex-direction:column;align-items:center;justify-content:center;"
        "cursor:pointer;"
        "min-width:" + str(ARROW_CELL_W) + "px;"
        "height:" + str(SCHEME_H) + "px;"
        "padding:2px 4px;"
        "border-radius:6px;"
        "transition:background 0.12s;"
        "user-select:none;flex-shrink:0;"
        "overflow:visible;"
        "}"
        ".arrow-cell:hover{background:" + hover_bg + "}"

        ".co-above{"
        "position:absolute;"
        "bottom:100%;"
        "margin-bottom:-35px;"
        "left:50%;"
        "transform:translateX(-50%);"
        "display:flex;"
        "flex-direction:column;"
        "align-items:center;"
        "gap:2px;"
        "pointer-events:none;"
        "}"
        ".co-row{display:flex;flex-direction:row;gap:2px;align-items:center;justify-content:center;}"

        ".arrow-line{display:flex;align-items:center;width:" + str(ARROW_SHAFT_W + 10) + "px;flex-shrink:0;}"
        ".a-shaft{width:" + str(ARROW_SHAFT_W) + "px;"
        "height:2.5px;background:" + arrow_color + ";border-radius:2px 0 0 2px;flex-shrink:0;}"
        ".a-head{width:0;height:0;"
        "border-top:6px solid transparent;border-bottom:6px solid transparent;"
        "border-left:10px solid " + arrow_color + ";flex-shrink:0;}"

        ".step-lbl{font-size:10px;font-weight:700;color:" + arrow_color + ";"
        "text-align:center;margin-top:2px;}"
        ".b-yld{font-size:12px;font-weight:700;color:" + arrow_color + ";text-align:center;}"
        ".b-rtype{font-size:8.5px;color:#6b7a8d;text-align:center;"
        "word-break:break-word;white-space:normal;line-height:1.2;"
        "max-width:136px;font-style:italic;}"
        ".b-cond{font-size:8px;color:#37506e;text-align:center;"
        "word-break:break-word;max-width:136px;line-height:1.15;}"

        ".detail-panel{margin-top:6px;padding:12px 16px;"
        "background:" + panel_bg + ";"
        "border-left:4px solid " + arrow_color + ";"
        "border-radius:0 8px 8px 0;display:none;}"
        ".detail-panel.active{display:block}"
        ".dp-title{font-weight:700;color:" + arrow_color + ";margin-bottom:5px;font-size:13px}"
        ".dp-grid{display:grid;grid-template-columns:140px 1fr;gap:3px 10px;margin-bottom:6px}"
        ".dk{color:#6b7a8d;font-size:11px}.dv{color:#1a2e44;font-size:11px;font-weight:500}"
        ".step-imgs{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:5px 0}"
        ".step-imgs img{border:1px solid #dce3ec;border-radius:4px}"
        ".step-arrow-txt{font-size:15px;color:" + arrow_color + ";font-weight:700}"
        ".smi-section{margin-top:5px}"
        ".smi-row{display:flex;align-items:center;gap:5px;margin:2px 0}"
        ".smi-code{font-family:monospace;font-size:9px;background:#f5f5f5;"
        "padding:2px 5px;border-radius:3px;word-break:break-all;flex:1}"
        ".copy-btn{background:" + arrow_color + ";color:white;border:none;"
        "border-radius:3px;padding:2px 6px;font-size:9px;cursor:pointer;"
        "white-space:nowrap;flex-shrink:0}.copy-btn:hover{opacity:0.8}"
    )

    js = (
        "var __si=JSON.parse(document.getElementById('__si_" + route_id + "').textContent);"
        "function notifyResize(){"
        "var h=document.getElementById('wrap-" + route_id + "').scrollHeight;"
        "try{window.parent.postMessage({isStreamlitMessage:true,"
        "type:'streamlit:setFrameHeight',height:h+8},'*');}catch(e){}}"
        "function copyTxt(t){"
        "try{navigator.clipboard.writeText(t)}"
        "catch(e){var x=document.createElement('textarea');x.value=t;"
        "document.body.appendChild(x);x.select();document.execCommand('copy');"
        "document.body.removeChild(x);}}"
        "function showStepFn" + rid_js + "(idx,el){"
        "var panel=document.getElementById('dp-" + route_id + "');"
        "if(panel.getAttribute('data-idx')==idx&&panel.classList.contains('active')){"
        "panel.classList.remove('active');panel.removeAttribute('data-idx');"
        "notifyResize();return;}"
        "panel.setAttribute('data-idx',idx);panel.classList.add('active');"
        "var d=JSON.parse(el.getAttribute('data-step').replace(/&quot;/g,'\"'));"
        "document.getElementById('dpt-" + route_id + "').textContent='Step '+d.step+' — '+(d.rtype||'');"
        "var grid=document.getElementById('dpg-" + route_id + "');"
        "var rows=[['Reaction type',d.rtype||'—'],"
        "['Yield',d.yld||(d.source==='rxn-insight'?'not shown':'not reported')],"
        "['Conditions',d.cond||'—'],['Source',d.source]];"
        "if(d.source==='rxn-insight'&&d.fg&&d.fg.length>0)"
        "{rows.push(['Functional groups',d.fg.join(', ')]);}"
        "grid.innerHTML=rows.map(function(r){"
        "return '<span class=\"dk\">'+r[0]+'</span><span class=\"dv\">'+r[1]+'</span>';}).join('');"
        "var si=__si[String(d.step)]||{};"
        "var imgDiv=document.getElementById('dpi-" + route_id + "');"
        "var ih='';"
        "(si.reactants||[]).forEach(function(b64){if(b64)ih+='<img src=\"'+b64+'\" width=\"105\" height=\"74\"/>';});"
        "if((si.reactants||[]).some(Boolean))ih+='<span class=\"step-arrow-txt\">&#8594;</span>';"
        "if(si.product)ih+='<img src=\"'+si.product+'\" width=\"105\" height=\"74\"/>';"
        "imgDiv.innerHTML=ih;"
        "var sDiv=document.getElementById('dps-" + route_id + "');"
        "var sh='<b style=\"font-size:10px;color:#1a2e44\">SMILES</b>';"
        "(si.reactant_smiles||[]).forEach(function(s){"
        "sh+='<div class=\"smi-row\"><span class=\"smi-code\">'+s+'</span>'"
        "+'<button class=\"copy-btn\" onclick=\"copyTxt(this.dataset.s)\" data-s=\"'+s+'\">Copy</button></div>';});"
        "if(si.product_smiles){"
        "sh+='<div class=\"smi-row\"><span class=\"smi-code\">&#8594; '+si.product_smiles+'</span>'"
        "+'<button class=\"copy-btn\" onclick=\"copyTxt(this.dataset.s)\" data-s=\"'+si.product_smiles+'\">Copy</button></div>';}"
        "sDiv.innerHTML=sh;notifyResize();}"
        "window.addEventListener('load',notifyResize);"
        "new ResizeObserver(notifyResize).observe(document.getElementById('wrap-" + route_id + "'));"
    )

    return (
        "<!DOCTYPE html><html><head>"
        "<style>" + css + "</style></head><body>"
        '<div id="wrap-' + route_id + '">'
        '<div id="scroll-wrap">'
        '<div class="band">' + "".join(items) + '</div>'
        '</div>'
        '<div class="detail-panel" id="dp-' + route_id + '">'
        '<div class="dp-title" id="dpt-' + route_id + '">Step details</div>'
        '<div class="dp-grid" id="dpg-' + route_id + '"></div>'
        '<div class="step-imgs" id="dpi-' + route_id + '"></div>'
        '<div class="smi-section" id="dps-' + route_id + '"></div>'
        '</div>'
        '</div>'
        '<script type="application/json" id="__si_' + route_id + '">'
        + step_imgs_json + '</script>'
        '<script>' + js + '</script>'
        '</body></html>'
    )


# Per-criterion explanations shown in the score table "?" tooltip
CRITERIA_SCORE_DESC = {
    "en": {
        "steps":        "Score = 1 / number_of_steps. Fewer steps → score closer to 1.",
        "yield":        "Score = cumulative yield (product of all step yields). "
                        "Missing yields are treated as 100% (neutral). Score closer to 1 = higher overall yield.",
        "atom_economy": "Score = fraction of reactant atoms incorporated into the product. "
                        "Score closer to 1 = less atomic waste.",
        "e_factor":     "Score = 1 / (1 + E-factor). E-factor = kg waste per kg product. "
                        "Score closer to 1 = less waste generated.",
        "toxicity":     "Score = 1 − (average hazard of reagents/solvents). "
                        "Score closer to 1 = safer, less toxic conditions.",
    },
    "fr": {
        "steps":        "Score = 1 / nombre_d'étapes. Moins d'étapes → score proche de 1.",
        "yield":        "Score = rendement cumulé (produit de tous les rendements). "
                        "Rendements manquants traités à 100% (neutres). Score proche de 1 = meilleur rendement.",
        "atom_economy": "Score = fraction des atomes des réactifs retrouvés dans le produit. "
                        "Score proche de 1 = moins de déchets atomiques.",
        "e_factor":     "Score = 1 / (1 + E-factor). E-factor = kg déchets / kg produit. "
                        "Score proche de 1 = moins de déchets.",
        "toxicity":     "Score = 1 − (danger moyen des réactifs/solvants). "
                        "Score proche de 1 = conditions plus sûres.",
    },
}


def build_score_table_html(details: dict, criteria: list, weights: dict) -> str:
    """
    Renders the per-criterion score breakdown table shown inside each route
    expander. Returns an HTML string with inline CSS (COMPONENT_STYLE).
    Each row shows the raw score (0–1) with a tooltip explaining how it is
    computed, the criterion weight, and the weighted contribution to the total.
    """
    rows = []
    csd  = CRITERIA_SCORE_DESC.get(lang, CRITERIA_SCORE_DESC["en"])
    for c in criteria:
        raw  = details[c].get("raw")
        tip_c = csd.get(c, T["score_th_raw_tip"]).replace('"', "&quot;")
        if raw is None:
            rows.append(
                f"<tr><td>{CL[c]}</td>"
                f"<td colspan='3'><em>excluded (predicted route)</em></td></tr>"
            )
            continue
        w    = weights.get(c, 0)
        cont = details[c]["weighted"]
        rows.append(f"""
  <tr>
    <td>{CL[c]}</td>
    <td><span class="th-help" title="{tip_c}" style="cursor:help;border-bottom:1px dashed #888">{raw:.3f}<span class="th-info" style="margin-left:3px">?</span></span></td>
    <td>{w*100:.0f}%</td>
    <td>{cont:.3f}</td>
  </tr>""")
    return f"""{COMPONENT_STYLE}
<table class="score-table">
<thead><tr>
  <th>{T['score_th_crit']}</th>
  <th>{T['score_th_raw']}</th>
  <th>{T['score_th_weight']}</th>
  <th>{T['score_th_contrib']}</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>"""


def fmt_conditions(cond: dict) -> str:
    """
    Formats a conditions dict into a human-readable string for the Dataset
    Explorer step detail cards. Returns a '·'-separated string of temperature,
    solvent, reagents, and apparatus fields.
    """
    if not cond: return ""
    parts = []
    if cond.get("temperature_C"):   parts.append(f"{cond['temperature_C']}°C")
    elif cond.get("temp_range"):    parts.append(cond["temp_range"])
    if cond.get("solvent"):
        solv = cond["solvent"]
        mol  = Chem.MolFromSmiles(solv) if MODULE_OK else None
        parts.append(solv if mol is None else solv)
    if cond.get("co_solvent"):      parts.append(f"/ {cond['co_solvent']}")
    reag = cond.get("reagents",[])
    if isinstance(reag,list) and reag:
        parts.append(", ".join(reag))
    if cond.get("apparatus"):       parts.append(f"({cond['apparatus']})")
    return "  ·  ".join(parts)


def bottleneck_yield(steps: list) -> float | None:
    """
    Returns the minimum step yield in the route (the rate-limiting step).
    Used in route metric cards and the Analysis tab comparison table.
    Returns None if no yield data is available.
    """
    ys = [s.get("yield_percent") for s in steps if s.get("yield_percent") is not None]
    return min(ys) if ys else None


def average_yield(steps: list) -> float | None:
    """
    Returns the mean step yield across all steps that report a yield.
    Used in route metric cards and the Analysis tab comparison table.
    Returns None if no yield data is available.
    """
    ys = [s.get("yield_percent") for s in steps if s.get("yield_percent") is not None]
    return sum(ys)/len(ys) if ys else None


def cumulative_yield(steps: list) -> float:
    """
    Returns the overall cumulative yield (product of all step yields, 0–1).
    Missing yields are treated as 1.0 (neutral — not penalised).
    Used for the yield criterion raw score and the Dataset Explorer summary table.
    """
    r = 1.0
    for s in steps:
        y = s.get("yield_percent")
        r *= (y / 100.0) if y is not None else 1.0
    return r


@st.cache_data(show_spinner=False)
def load_dataset_cached(path: str) -> dict:
    """
    Cached wrapper around fi.load_reaction_dataset(). Prevents reloading the
    JSON dataset on every Streamlit re-run during the same session.
    """
    return fi.load_reaction_dataset(path)

@st.cache_data(show_spinner=False)
def get_targets_cached(path: str) -> dict:
    """
    Cached wrapper that extracts the {target_name: SMILES} dict from the
    dataset. Used to populate the predefined molecule selector in the sidebar.
    """
    return fi.get_targets_from_dataset(load_dataset_cached(path))


def get_substances_list(steps_data: list) -> dict:
    """
    Extracts categorised substance lists from a route's steps for the
    'Substances needed' expander in each route card.

    Returns a dict with keys:
      - to_buy      : starting materials (reactants never produced in the route)
      - to_prepare  : intermediates (produced in at least one step)
      - solvents    : unique solvent SMILES / names
      - reagents    : unique reagent SMILES / names
    """
    all_prod = {fi.to_canonical(s.get("product_smiles","")) for s in steps_data}
    all_reac = {fi.to_canonical(r) for s in steps_data for r in s.get("reactants_smiles",[])}
    solvents=set(); reagents=set()
    for s in steps_data:
        cond=s.get("conditions",{})
        if cond.get("solvent"):    solvents.add(cond["solvent"])
        if cond.get("co_solvent"): solvents.add(cond["co_solvent"])
        for r in (cond.get("reagents") or []):
            if r: reagents.add(r)
    return {"to_buy":sorted(all_reac-all_prod-{""}),
            "to_prepare":sorted(all_prod-{""}),
            "solvents":sorted(solvents-{""}),
            "reagents":sorted(reagents-{""})}


# =============================================================================
# matplotlib charts
# =============================================================================

def make_ranking_chart(results: list, target_name: str):
    """
    Horizontal bar chart ranking routes by total score.
    Displayed above the route cards when more than one route is found in a
    given category (Dataset / Validated / Predicted).
    """
    labels = [r[2].get("matched_route_name", f"Route {i+1}")[:32]
              for i, r in enumerate(results)]
    scores = [r[0] for r in results]
    n = len(scores)

    fig, ax = _hires_fig(figsize=(7, max(2.0, n * 0.65)), dpi=180)
    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]
    bars   = ax.barh(np.arange(n), scores, color=colors, height=0.52, edgecolor="none")

    for bar, s in zip(bars, scores):
        ax.text(
            s + max(scores) * 0.013,
            bar.get_y() + bar.get_height() / 2,
            f"{s:.4f}",
            va="center", fontsize=9.5, color="#1a2e44", fontweight="600",
        )

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=10, color="#1a2e44")
    ax.set_xlabel(T["score_axis"], fontsize=10, color="#6b7a8d")
    ax.set_xlim(0, max(scores) * 1.30)
    ax.set_title(
        T["chart_title"].format(target=target_name),
        fontsize=13, fontweight="bold", color="#1a2e44", pad=14,
    )
    ax.tick_params(colors="#6b7a8d", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.grid(True, color="#dce3ec", linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.6)
    return fig


def make_yield_chart(steps_route: list):
    """
    Bar chart of reported yield per step — shown in the Dataset Explorer route
    detail view. Steps with no reported yield are simply absent from the chart.
    Yield labels use integer format (no decimal places). To add decimals, change
    f"{v:.0f}%" to f"{v:.1f}%" or f"{v:.2f}%" in the label text call below.
    """
    n = len(steps_route)
    step_nums      = list(range(1, n + 1))
    reported_vals  = []
    reported_steps = []
    for i, s in enumerate(steps_route):
        y = s.get("yield_percent")
        if y is not None:
            reported_vals.append(y)
            reported_steps.append(i + 1)

    fig, ax = _hires_fig(figsize=(max(4.5, n * 0.7), 2.6), dpi=180)

    if reported_vals:
        bars = ax.bar(
            reported_steps, reported_vals,
            color="#1a2e44", width=0.52, edgecolor="none",
        )
        for bar, v in zip(bars, reported_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{v:.0f}%",
                ha="center", va="bottom",
                fontsize=7.5, color="#1a2e44", fontweight="600",
            )

    ax.set_ylim(0, 125)
    ax.set_xticks(step_nums)
    ax.set_xlabel(strip_emoji(T["col_steps"]), fontsize=10, color="#6b7a8d")
    ax.set_ylabel("Yield (%)", fontsize=10, color="#6b7a8d")
    ax.legend(
        handles=[mpatches.Patch(color="#1a2e44", label="Reported yield")],
        fontsize=8, framealpha=0.8, facecolor=FIG_BG, edgecolor="#dce3ec",
    )
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color="#dce3ec", linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#6b7a8d", length=0)
    fig.tight_layout(pad=1.2)
    return fig


def make_comparison_chart(sel_results: list, criteria: list):
    """
    Grouped horizontal bar chart comparing raw scores per criterion across
    the routes selected in the Analysis tab multi-select.
    Each route is a different colour; each row is a criterion.
    """
    route_names = [
        r[2].get("matched_route_name", f"R{i+1}")[:20]
        for i, r in enumerate(sel_results)
    ]
    n_routes = len(sel_results)
    n_crit   = len(criteria)
    x        = np.arange(n_crit)
    bar_h    = 0.72 / n_routes
    offsets  = np.linspace(-(n_routes - 1) / 2, (n_routes - 1) / 2, n_routes) * bar_h

    fig, ax = _hires_fig(figsize=(6.5, max(2.8, n_crit * 1.0)), dpi=180)

    for i, (score, details, route) in enumerate(sel_results):
        vals   = [details[c].get("raw", 0) or 0 for c in criteria]
        color  = PALETTE[i % len(PALETTE)]
        name   = route_names[i]
        bars   = ax.barh(
            x + offsets[i], vals, bar_h * 0.88,
            color=color, label=name, edgecolor="none", alpha=0.92,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                v + 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}",
                va="center", fontsize=8, color=color, fontweight="600",
            )

    ax.set_yticks(x)
    ax.set_yticklabels(
        [strip_emoji(CL[c]) for c in criteria], fontsize=10, color="#1a2e44",
    )
    ax.set_xlim(0, 1.28)
    ax.set_xlabel("Raw score (0–1)", fontsize=10, color="#6b7a8d")
    ax.set_title(
        strip_emoji(T["radar_title"]),
        fontsize=12, fontweight="bold", color="#1a2e44", pad=12,
    )
    ax.legend(
        loc="lower right", fontsize=8,
        framealpha=0.8, facecolor=FIG_BG, edgecolor="#dce3ec",
    )
    ax.tick_params(colors="#6b7a8d", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.grid(True, color="#dce3ec", linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.4)
    return fig


# =============================================================================
# route display helpers
# =============================================================================

def build_route_report_pdf(score_total: float, details: dict, route: dict,
                            criteria: list) -> bytes:
    """
    Generates a multi-page PDF report for a single synthesis route using PIL
    only (no reportlab dependency).

    Page layout:
      - Page 1 : summary header (route name, target, status, score), metric
                 cards (steps, cumulative yield, bottleneck, avg yield), score
                 breakdown table, and starting material structures.
      - Pages 2+: up to 3 steps per page, each rendered with Cairo molecule
                 images at 200 DPI.
    """
    from PIL import Image as _PI, ImageDraw as _PID, ImageFont as _PIF
    import io as _io

    DPI   = 200
    PW    = int(8.27 * DPI)
    PH    = int(11.69 * DPI)
    MG    = int(0.55 * DPI)
    CW    = PW - 2 * MG

    NAVY  = (26,  46,  68)
    WHITE = (255, 255, 255)
    LIGHT = (235, 242, 250)
    LGREY = (245, 248, 252)
    GREY  = (107, 122, 141)
    ORANGE= (230,  81,   0)
    GREEN = ( 21,  87,  36)
    SEP   = (220, 227, 236)
    BG    = (249, 251, 253)

    def _fnt(size, bold=False):
        candidates = (
            ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]
            if bold else
            ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/System/Library/Fonts/Supplemental/Arial.ttf"]
        )
        for p in candidates:
            try: return _PIF.truetype(p, size)
            except Exception: pass
        return _PIF.load_default()

    def _mol_pil(smi, w, h):
        if not smi or not MODULE_OK: return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        try:
            if mol.GetNumAtoms() > 1:
                rdDepictor.Compute2DCoords(mol)
            drw = rdMolDraw2D.MolDraw2DCairo(w * 2, h * 2)
            drw.drawOptions().bondLineWidth      = 1.0
            drw.drawOptions().padding            = 0.1
            drw.drawOptions().addStereoAnnotation = True
            drw.DrawMolecule(mol)
            drw.FinishDrawing()
            img = _PI.open(_io.BytesIO(drw.GetDrawingText())).convert("RGBA")
            bg  = _PI.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            return bg.convert("RGB").resize((w, h), _PI.LANCZOS)
        except Exception:
            return None

    def _trunc(txt, n):
        return txt[:n] + "…" if len(txt) > n else txt

    def _text_w(draw, txt, font):
        try: return int(draw.textlength(txt, font=font))
        except Exception: return len(txt) * max(6, font.size // 2)

    def _wrap(draw, txt, max_px, font):
        words = txt.split(); lines = []; line = ""
        for w in words:
            test = (line + " " + w).strip()
            if _text_w(draw, test, font) <= max_px:
                line = test
            else:
                if line: lines.append(line)
                line = w
        if line: lines.append(line)
        return lines or [""]

    def _draw_band(draw, y, h, color=NAVY):
        draw.rectangle([0, y, PW, y + h], fill=color)

    def _draw_rounded(draw, x, y, w, h, fill, outline=None, r=5):
        draw.rounded_rectangle([x, y, x+w, y+h], radius=r,
                                fill=fill, outline=outline or fill)

    steps_data = route.get("dataset_steps", [])
    rname      = route.get("matched_route_name", "Unknown route")
    target     = route.get("matched_target", "?")
    status     = route.get("validation_status", "dataset")
    status_lbl = {"dataset":"Dataset","validated":"Validated",
                  "partial":"Partial","predicted":"Predicted"}.get(status, status)
    sub        = get_substances_list(steps_data)
    bn_        = bottleneck_yield(steps_data)
    av_        = average_yield(steps_data)
    cumyl      = cumulative_yield(steps_data)

    pages = []

    p1 = _PI.new("RGB", (PW, PH), BG)
    d1 = _PID.Draw(p1)

    BH = int(0.72 * DPI)
    _draw_band(d1, 0, BH)
    d1.text((MG, 20),  _trunc(rname, 60), fill=WHITE, font=_fnt(38, True))
    d1.text((MG, 64),
            f"Target: {target}   ·   {status_lbl}   ·   Score: {score_total:.4f}",
            fill=(180, 200, 220), font=_fnt(22))

    y1 = BH + 22

    mets = [("Steps", str(len(steps_data))),
            ("Cumul. yield", f"{cumyl*100:.4f}%"),
            ("Bottleneck",   f"{bn_:.4f}%" if bn_ else "—"),
            ("Avg yield",    f"{av_:.4f}%"  if av_ else "—")]
    mc = CW // len(mets); mx = MG
    for lbl, val in mets:
        _draw_rounded(d1, mx, y1, mc - 8, 64, LIGHT, SEP)
        d1.text((mx + 10, y1 + 5),  lbl, fill=GREY, font=_fnt(18))
        d1.text((mx + 10, y1 + 28), val, fill=NAVY, font=_fnt(26, True))
        mx += mc
    y1 += 78

    d1.line([(MG, y1), (PW - MG, y1)], fill=SEP, width=2); y1 += 18

    d1.text((MG, y1), "Score breakdown", fill=NAVY, font=_fnt(24, True)); y1 += 32
    col_ws = [int(CW * p) for p in [0.34, 0.22, 0.18, 0.26]]
    hdrs   = ["Criterion", "Raw (0–1)", "Weight", "Contribution"]
    hx = MG
    for txt, cw in zip(hdrs, col_ws):
        _draw_rounded(d1, hx, y1, cw - 4, 30, NAVY, NAVY, 3)
        d1.text((hx + 7, y1 + 6), txt, fill=WHITE, font=_fnt(16, True))
        hx += cw
    y1 += 32
    for ci, c in enumerate(criteria):
        dtl  = details.get(c, {})
        raw  = dtl.get("raw"); excl = dtl.get("excluded", False)
        row  = [c,
                "excluded" if excl else (f"{raw:.4f}" if raw is not None else "—"),
                "0%" if excl else f"{(dtl.get('weight') or 0)*100:.0f}%",
                "—" if excl else f"{dtl.get('weighted') or 0:.4f}"]
        bg_c = LIGHT if ci % 2 == 0 else LGREY
        hx = MG
        for txt, cw in zip(row, col_ws):
            _draw_rounded(d1, hx, y1, cw - 4, 28, bg_c, SEP, 3)
            d1.text((hx + 7, y1 + 5), txt, fill=NAVY, font=_fnt(16))
            hx += cw
        y1 += 28
    y1 += 16

    d1.line([(MG, y1), (PW - MG, y1)], fill=SEP, width=2); y1 += 16

    if sub["to_buy"]:
        d1.text((MG, y1), "Starting materials", fill=NAVY, font=_fnt(22, True)); y1 += 28
        n_sm  = min(len(sub["to_buy"]), 5)
        sm_w  = min(int(CW / n_sm) - 8, int(1.6 * DPI))
        sm_h  = int(sm_w * 0.68)
        sx    = MG
        for smi in sub["to_buy"][:n_sm]:
            img = _mol_pil(smi, sm_w, sm_h)
            if img:
                p1.paste(img, (sx, y1))
                lbl = _trunc(smi, 20)
                tw  = _text_w(d1, lbl, _fnt(13))
                d1.text((sx + (sm_w - tw) // 2, y1 + sm_h + 2),
                        lbl, fill=GREY, font=_fnt(13))
            sx += sm_w + 8
        y1 += sm_h + 20

    pages.append(p1)

    def _render_step_at(draw, page, step, x0, y0, avail_w, avail_h):
        L = x0 + 4
        W = avail_w - 8

        snum  = step.get("step_number", "?")
        rtype = step.get("reaction_type", "—") or "—"
        yld   = step.get("yield_percent")
        src_s = step.get("source", "dataset")
        cond  = step.get("conditions", {})
        reac  = step.get("reactants_smiles", [])
        prod  = step.get("product_smiles", "")

        hdr_c = ORANGE if src_s == "rxn-insight" else NAVY
        yld_s = (f"{yld}%" if yld is not None else
                 "not shown" if src_s == "rxn-insight" else "—")
        cond_parts = []
        t_ = cond.get("temperature_C")
        if t_: cond_parts.append(f"{t_}°C")
        s_ = cond.get("solvent","")
        if s_: cond_parts.append(s_)
        rr = cond.get("reagents",[]) or []
        if rr: cond_parts.append(", ".join(str(r) for r in rr[:2]))
        cond_s = "  ·  ".join(cond_parts) or "—"

        y = y0
        HDR_H = int(avail_h * 0.22)
        COND_H= int(avail_h * 0.16)

        _draw_band(draw, y, HDR_H, hdr_c)
        lbl = f"Step {snum} — {_trunc(rtype, 30)}"
        draw.text((L, y + 5), lbl, fill=WHITE, font=_fnt(16, True))
        draw.text((L, y + 24), f"Yield: {yld_s}", fill=(200,215,230), font=_fnt(13))
        y += HDR_H

        _draw_rounded(draw, L, y, W, COND_H - 4, LIGHT, SEP, 3)
        clines = _wrap(draw, "Cond: " + cond_s, W - 8, _fnt(13))[:2]
        cy = y + 4
        for cl in clines:
            draw.text((L + 5, cy), cl, fill=NAVY, font=_fnt(13))
            cy += 17
        y += COND_H

        mol_zone = avail_h - HDR_H - COND_H - 18
        mol_h = max(50, min(mol_zone, int(avail_h * 0.45)))
        n_reac = max(len(reac), 1)
        ARW = int(W * 0.22)
        PLS = int(W * 0.08)
        mol_w = max(40, (W - ARW - (n_reac-1)*PLS) // (n_reac+1))
        mol_w = min(mol_w, int(mol_h * 1.4))
        mol_h = min(mol_h, int(mol_w * 0.68))

        y += 6
        mx = L
        for i, rsmi in enumerate(reac[:3]):
            img = _mol_pil(rsmi, mol_w, mol_h)
            if img:
                page.paste(img, (mx, y))
                lbl2 = _trunc(rsmi, 14)
                tw   = _text_w(draw, lbl2, _fnt(11))
                draw.text((mx + (mol_w-tw)//2, y+mol_h+2), lbl2, fill=GREY, font=_fnt(11))
            mx += mol_w
            if i < len(reac)-1:
                draw.text((mx+4, y+mol_h//2-10), "+", fill=NAVY, font=_fnt(22, True))
                mx += PLS
        ay = y + mol_h//2
        ax0=mx+4; ax1=mx+ARW-10
        draw.line([(ax0,ay),(ax1,ay)], fill=NAVY, width=3)
        draw.polygon([(ax1,ay-6),(ax1+9,ay),(ax1,ay+6)], fill=NAVY)
        mx += ARW
        p_img = _mol_pil(prod, mol_w, mol_h)
        if p_img:
            page.paste(p_img, (mx, y))
            lbl3 = _trunc(prod, 14)
            tw   = _text_w(draw, lbl3, _fnt(11))
            draw.text((mx+(mol_w-tw)//2, y+mol_h+2), lbl3, fill=GREEN, font=_fnt(11))

        draw.line([(x0, y0+avail_h-2),(x0+avail_w, y0+avail_h-2)], fill=SEP, width=1)

    STEPS_PER_PAGE = 3
    STEP_H = (PH - 2 * MG - int(0.15 * DPI) * (STEPS_PER_PAGE - 1)) // STEPS_PER_PAGE
    STEP_W = CW
    it = iter(steps_data)
    while True:
        chunk = [next(it, None) for _ in range(STEPS_PER_PAGE)]
        if not any(chunk): break
        pg  = _PI.new("RGB", (PW, PH), BG)
        drw = _PID.Draw(pg)
        for slot, step in enumerate(chunk):
            if step is None: continue
            y0 = MG + slot * (STEP_H + int(0.15 * DPI))
            _render_step_at(drw, pg, step, MG, y0, STEP_W, STEP_H)
        pages.append(pg)

    buf = _io.BytesIO()
    if pages:
        pages[0].save(buf, format="PDF", save_all=True,
                      append_images=pages[1:], resolution=DPI)
    return buf.getvalue()


def display_route_card(score_total, details, route, criteria, weights,
                       all_results, rank=1, badge="📚 dataset", lang="en") -> None:
    """
    Renders a full route card inside a Streamlit expander.

    Displays (in order):
      1. Validation status banner (info/success/warning)
      2. Four metric columns (score, steps, bottleneck yield, avg yield)
      3. Score breakdown table (build_score_table_html)
      4. PDF download button (build_route_report_pdf)
      5. Interactive reaction scheme (build_clickable_scheme_html)
      6. Substances needed expander (get_substances_list)
    """
    route_name = route.get("matched_route_name","?")
    tgt        = route.get("matched_target","?")
    steps_data = route.get("dataset_steps",[])
    n_steps    = len(steps_data)
    bn         = bottleneck_yield(steps_data)
    av         = average_yield(steps_data)
    medals     = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
    medal      = medals[rank-1] if rank<=10 else f"#{rank}"
    route_key  = "".join(c for c in route.get("matched_route_id","r") if c.isalnum())
    is_pred    = route.get("is_predicted",False)
    status     = route.get("validation_status","dataset")

    with st.expander(
        f"{medal}  [{badge}]  {route_name}  ·  {tgt}  ·  Score: **{score_total:.3f}**",
        expanded=(rank==1),
    ):
        if status=="predicted":
            st.warning("⚠️ Predicted route — not experimentally validated. Yield excluded from scoring.")
        elif status=="partial":
            v=route.get("validated_steps_count",0); t=route.get("total_steps_count",0)
            st.info(f"⚡ Partial validation — {v}/{t} steps found in generic dataset (real conditions).")
        elif status=="validated":
            st.success("✅ Fully validated — all steps found in generic dataset (real conditions).")

        m1,m2,m3,m4 = st.columns(4)
        m1.metric(T["metric_score"], f"{score_total:.3f}", help=T["metric_score_help"])
        m2.metric(T["metric_steps"], n_steps)
        m3.metric(T["metric_bottleneck"],
                  f"{bn:.1f}%" if bn is not None else "—", help=T["metric_bn_help"])
        m4.metric(T["metric_avg"], f"{av:.1f}%" if av is not None else "—")

        st.markdown(f"**{T['contrib_title']}**")
        st.html(build_score_table_html(details, criteria, weights))

        dl_name = "".join(c for c in route_name if c.isalnum() or c in " _-")[:40].strip()
        dl_col, _ = st.columns([1, 3])
        with dl_col:
            try:
                pdf_data = build_route_report_pdf(score_total, details, route, criteria)
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_data,
                    file_name=f"{dl_name}.pdf",
                    mime="application/pdf",
                    key=f"dl_pdf_{route_key}_{rank}",
                    help="Download full synthesis route report as high-resolution PDF",
                )
            except Exception as e:
                st.caption(f"PDF unavailable: {e}")

        st.markdown("---")
        st.markdown("**Reaction scheme** *(click an arrow to see step details below)*")
        components.html(
            build_clickable_scheme_html(steps_data, route_key, is_pred),
            height=480, scrolling=True,
        )

        st.markdown("---")
        with st.expander("🧪 Substances needed", expanded=False):
            sub=get_substances_list(steps_data)
            all_mols = sub["to_buy"] + sub["to_prepare"][:6]
            if all_mols:
                n_cols = min(4, len(all_mols))
                cols_sub = st.columns(n_cols)
                for i, smi in enumerate(all_mols):
                    with cols_sub[i % n_cols]:
                        png = mol_png(smi, 160, 110)
                        label = "🛒" if smi in sub["to_buy"] else "⚗️"
                        if png: st.image(png, caption=label + " " + smi[:20], width="content")
            if sub["solvents"]: st.markdown("**🧴 Solvents**"); st.write("  ·  ".join(sub["solvents"]))
            if sub["reagents"]: st.markdown("**🔬 Reagents**"); st.write("  ·  ".join(sub["reagents"][:10]))


# =============================================================================
# sidebar settings
# =============================================================================

with st.sidebar:
    st.title(T["sidebar_title"])
    if not MODULE_OK:
        st.error(f"{T['mod_err']}\n\n`{MODULE_ERR}`"); st.stop()

    st.divider()
    st.subheader(T["files_section"])

    dataset_path = st.text_input(
        T["ds_label"],
        value="reaction_dataset.json",
        help=(
            "Path to your main curated reaction dataset (JSON). "
            "Format: list of reaction dicts with fields: id, route_id, route_name, "
            "target, step_number, reactants_smiles, product_smiles, conditions, "
            "yield_percent, reaction_type."
        ),
    )
    toxicity_path = st.text_input(
        T["tox_label"] + " *",
        value="toxicity_dataset.json",
        help=(
            "Path to your toxicity/safety scores file (JSON). "
            "Required for the Safety criterion. "
            "If not found, safety scores default to 0.5 (neutral)."
        ),
    )
    if not os.path.exists(toxicity_path):
        st.warning(f"⚠️ Toxicity file not found: `{toxicity_path}` — Safety scores will default to 0.5")

    config_path = st.text_input(
        T["cfg_label"],
        value="config.yml",
        help=(
            "Path to the AiZynthFinder YAML config file. "
            "Specifies the policy network, filter, and stock files. "
            "Required — AiZynthFinder will not run without it."
        ),
    )
    rxninsight_db_path = st.text_input(
        T["rxni_label"],
        value="data/uspto_rxn_insight.gzip",
        help=(
            "Path to the Rxn-INSIGHT USPTO reaction database (gzip). "
            "Optional — only needed if 'Include predicted routes' is enabled. "
            "Enables reaction classification and condition prediction for novel routes."
        ),
    )
    generic_dataset_path = st.text_input(
        T["generic_label"],
        value="generic_reactions.json",
        help=(
            "Path to a flat JSON list of individual reactions (same format as the main dataset). "
            "Used to cross-validate AiZynthFinder-proposed steps with real experimental data. "
            "Without this file, all AiZ routes are classified as 'predicted'."
        ),
    )

    st.divider()

    top_n = st.slider(
        T["topn_label"],
        1, 5, 3,
        help=(
            "Maximum number of routes shown per category (Dataset / Validated / Predicted). "
            "Does not affect the AiZynthFinder search — only how many results are displayed."
        ),
    )
    n_aiz = st.slider(
        T["naiz_label"],
        5, 50, 25,
        help=(
            "Number of routes AiZynthFinder explores internally via MCTS. "
            "Higher = more routes explored, better chance of matching your generic dataset, "
            "but longer search time (roughly +2–4 s per extra route). "
            "Recommended: 20–30 for speed, 40–50 for thoroughness."
        ),
    )

    include_predicted = False
    if RXNINSIGHT_OK:
        include_predicted = st.toggle(
            "Include predicted routes (Rxn-INSIGHT)",
            value=True,
            help=(
                "When enabled, AiZynthFinder routes that have no matching steps in the "
                "generic dataset are classified as 'predicted' and annotated by Rxn-INSIGHT "
                "(reaction type, conditions). Their yield is excluded from scoring. "
                "Disable to show only experimentally validated routes."
            ),
        )
    else:
        st.caption("🔮 Predicted routes disabled (`pip install rxn-insight`)")

    st.divider()
    if "criteria" in st.session_state:
        with st.expander(T["weights_exp"]):
            crit_w = fi.compute_weights(st.session_state["criteria"])
            for c in st.session_state["criteria"]:
                w=crit_w[c]; st.progress(float(w),text=f"{CL[c]} — {w*100:.1f}%")


# =============================================================================
# main layout
# =============================================================================

st.title(T["page_title"])
st.caption(T["page_caption"])

tab_search, tab_analysis, tab_dataset, tab_help = st.tabs([
    T["tab_search"], T["tab_analysis"], T["tab_dataset"], T["tab_help"]
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ROUTE SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:
    top_left, top_right = st.columns([3,2])

    with top_left:
        st.subheader(T["target_section"])
        mode = st.radio("Mode",[T["mode_pre"],T["mode_custom"]],horizontal=True,label_visibility="collapsed")

        if mode == T["mode_pre"]:
            if os.path.exists(dataset_path):
                try:   targets_ds = get_targets_cached(dataset_path)
                except: targets_ds = {}
            else: targets_ds = {}

            if targets_ds:
                target_name   = st.selectbox(T["mol_label"], list(targets_ds.keys()),
                                              format_func=lambda x: x.capitalize())
                target_smiles = targets_ds[target_name]
            else:
                target_name   = "Galanthamine"
                target_smiles = "OC1C=C[C@@]23c4cc(OC)ccc4CN(C)C[C@@H]2[C@@H]1O3"
        else:
            target_smiles = st.text_input(T["smiles_label"], placeholder=T["smiles_ph"])
            target_name   = "Custom"
            if target_smiles:
                if Chem.MolFromSmiles(target_smiles) is None:
                    st.error(T["smiles_invalid"]); target_smiles = ""
                else: st.success(T["smiles_valid"])

        st.subheader(T["criteria_section"])
        st.caption(T["criteria_caption"])
        all_crit = list(CL.keys())
        c1   = st.selectbox(T["c1_label"], all_crit, format_func=lambda x:CL[x])
        rem2 = [c for c in all_crit if c!=c1]
        c2   = st.selectbox(T["c2_label"], rem2, format_func=lambda x:CL[x])
        rem3 = [c for c in rem2 if c!=c2]
        c3   = st.selectbox(T["c3_label"], rem3, format_func=lambda x:CL[x])
        criteria = [c1,c2,c3]
        st.session_state["criteria"] = criteria

        run_search = st.button(T["run_btn"], type="primary", disabled=not target_smiles)

    with top_right:
        if target_smiles:
            png=mol_png(target_smiles,320,220)
            if png: st.image(png,caption=target_name,width="stretch")
        else:
            st.markdown(T["welcome"])
            st.markdown(T["avail_mols"])
            try:
                prev=get_targets_cached(dataset_path)
                cols=st.columns(3)
                for idx,(name,smi) in enumerate(prev.items()):
                    with cols[idx%3]:
                        png=mol_png(smi,200,135)
                        if png: st.image(png,caption=name.capitalize(),width="stretch")
            except Exception: pass

    st.divider()

    if run_search and target_smiles:
        errs=[]
        if not os.path.exists(dataset_path): errs.append(f"`{dataset_path}`")
        if not os.path.exists(config_path):  errs.append(f"`{config_path}`")
        for e in errs: st.error(f"{T['err_file']}: {e}")
        if not errs:
            with st.status(T["searching"], expanded=True) as status:
                try:
                    st.write(T["loading_ds"])
                    st.write(T["loading_aiz"])
                    if include_predicted and RXNINSIGHT_OK:
                        st.write(T["loading_rxni"])

                    results_raw = fi.find_best_routes(
                        target_smiles        = target_smiles,
                        criteria_priority    = criteria,
                        dataset_path         = dataset_path,
                        toxicity_path        = toxicity_path,
                        config_path          = config_path,
                        top_n                = top_n,
                        target_name          = target_name if mode==T["mode_pre"] else "",
                        include_predicted    = include_predicted,
                        rxninsight_db_path   = rxninsight_db_path,
                        generic_dataset_path = generic_dataset_path,
                        n_aiz_routes         = n_aiz,
                    )
                    tox_index = fi.load_toxicity_dataset(toxicity_path)
                    weights   = fi.compute_weights(criteria)
                    st.session_state.update({
                        "results":        results_raw,
                        "weights":        weights,
                        "tox_index":      tox_index,
                        "target_name":    target_name,
                        "target_smiles":  target_smiles,
                        "criteria":       criteria,
                    })
                    status.update(label=T["search_ok"],state="complete",expanded=False)
                except FileNotFoundError as e:
                    status.update(label=T["err_file"],state="error"); st.error(str(e))
                except ValueError as e:
                    status.update(label=T["err_param"],state="error"); st.error(str(e))
                except Exception as e:
                    status.update(label=T["err_other"],state="error"); st.exception(e)

    results_raw = st.session_state.get("results",   None)
    weights     = st.session_state.get("weights",   {})
    criteria    = st.session_state.get("criteria",  criteria)
    tox_index   = st.session_state.get("tox_index", {})

    if results_raw is None:
        pass
    elif not isinstance(results_raw, dict):
        st.warning(T["no_routes"])
    else:
        scored_dataset   = results_raw.get("dataset",   [])
        scored_validated = results_raw.get("validated", [])
        scored_predicted = results_raw.get("predicted", [])

        if not scored_dataset and not scored_validated and not scored_predicted:
            st.warning(T["no_routes"])
        else:
            c1_,c2_,c3_,c4_ = st.columns(4)
            c1_.metric("📚 Dataset",   len(scored_dataset))
            c2_.metric("✅ Validated",  len(scored_validated))
            c3_.metric("🔮 Predicted", len(scored_predicted))
            c4_.metric("Total", len(scored_dataset)+len(scored_validated)+len(scored_predicted))

            tgt_name = st.session_state.get("target_name", target_name)

            st.markdown(T["sec_dataset"])
            st.caption(T["cap_dataset"])
            if not scored_dataset:
                st.info("No dataset routes for this target.")
            else:
                if len(scored_dataset)>1:
                    fig=make_ranking_chart(scored_dataset, tgt_name)
                    st.pyplot(fig); plt.close(fig)
                st.success(T["n_found"].format(n=len(scored_dataset)))
                st.markdown("---")
                for rank,(score_total,details,route) in enumerate(scored_dataset,1):
                    display_route_card(score_total,details,route,criteria,weights,
                                       scored_dataset,rank,T["badge_dataset"],lang)

            if scored_validated:
                st.markdown("---")
                st.markdown(T["sec_validated"])
                st.caption(T["cap_validated"])
                if len(scored_validated)>1:
                    fig=make_ranking_chart(scored_validated, tgt_name)
                    st.pyplot(fig); plt.close(fig)
                for rank,(score_total,details,route) in enumerate(scored_validated,1):
                    status=route.get("validation_status","partial")
                    badge=T["badge_validated"] if status=="validated" else T["badge_partial"]
                    display_route_card(score_total,details,route,criteria,weights,
                                       scored_validated,rank,badge,lang)

            if include_predicted and RXNINSIGHT_OK and scored_predicted:
                st.markdown("---")
                st.markdown(T["sec_predicted"])
                st.caption(T["cap_predicted"])

                search_sm = st.text_input("Search by starting material SMILES",
                                          placeholder="e.g. c1ccc2[nH]ccc2c1",
                                          key="sm_search")
                all_sm_map={}
                for _,_,route in scored_predicted:
                    steps=route.get("dataset_steps",[])
                    all_prod={fi.to_canonical(s.get("product_smiles","")) for s in steps}
                    for s in steps:
                        for rsmi in s.get("reactants_smiles",[]):
                            canon=fi.to_canonical(rsmi)
                            if canon and canon not in all_prod:
                                all_sm_map.setdefault(canon,[]).append(route.get("matched_route_name","?"))

                filtered=scored_predicted
                if search_sm.strip():
                    sc=fi.to_canonical(search_sm.strip())
                    filtered=[(s,d,r) for s,d,r in scored_predicted
                              if sc in {fi.to_canonical(rsmi)
                                        for step in r.get("dataset_steps",[])
                                        for rsmi in step.get("reactants_smiles",[])}
                              or any(search_sm.lower() in rsmi.lower()
                                     for step in r.get("dataset_steps",[])
                                     for rsmi in step.get("reactants_smiles",[]))]
                    if filtered: st.success(f"{len(filtered)} route(s) found")
                    else: st.warning("No routes with this starting material")

                top30=list(all_sm_map.keys())[:30]
                with st.expander(f"📋 Browse starting materials ({len(top30)} unique)", expanded=False):
                    cols_sm=st.columns(3)
                    for i,smi in enumerate(top30):
                        col=cols_sm[i%3]
                        png=mol_png(smi,200,135)
                        if png: col.image(png,width=140)
                        col.code(smi,language=None)
                        col.caption(f"In: {', '.join(set(all_sm_map[smi]))[:50]}")

                st.markdown("---")
                for rank,(score_total,details,route) in enumerate(filtered,1):
                    display_route_card(score_total,details,route,criteria,weights,
                                       filtered,rank,T["badge_predicted"],lang)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    import pandas as pd
    results_raw = st.session_state.get("results", None)
    criteria    = st.session_state.get("criteria", list(CL.keys())[:3])
    weights     = st.session_state.get("weights", {})
    tox_index   = st.session_state.get("tox_index", {})

    if results_raw is None or not isinstance(results_raw, dict):
        st.info(T["no_analysis"])
    else:
        sc_ds  = results_raw.get("dataset",   [])
        sc_val = results_raw.get("validated", [])
        sc_pr  = results_raw.get("predicted", [])
        all_sc = sc_ds + sc_val + sc_pr

        if not all_sc:
            st.info(T["no_analysis"])
        else:
            st.subheader(T["compare_title"])

            def _route_badge_label(r):
                status=r[2].get("validation_status","dataset")
                if status=="validated": return "✅"
                if status=="partial":   return "⚡"
                if status=="predicted": return "🔮"
                return "📚"

            route_opts = {f"{_route_badge_label(r)} {r[2].get('matched_route_name','?')} "
                          f"(score {r[0]:.4f})": i for i,r in enumerate(all_sc)}
            labels = list(route_opts.keys())
            sel    = st.multiselect(T["sel_routes"], labels,
                                    default=labels[:min(3,len(labels))])

            if sel:
                sel_results = [all_sc[route_opts[s]] for s in sel]
                rows=[]
                for score,details,route in sel_results:
                    sd=route.get("dataset_steps",[]); bn_=bottleneck_yield(sd); av_=average_yield(sd)
                    yc=cumulative_yield(sd)
                    row={"Route":route.get("matched_route_name","?")[:26],
                         T["metric_score"]:f"{score:.3f}",
                         T["metric_steps"]:len(sd),
                         "Cumul. yield":f"{yc*100:.4f}%",
                         T["metric_bottleneck"]:f"{bn_:.1f}%" if bn_ else "—",
                         T["metric_avg"]:f"{av_:.1f}%" if av_ else "—"}
                    for c in criteria:
                        raw=details[c].get("raw")
                        if c == "steps":
                            continue
                        if c == "yield":
                            continue
                        row[CL[c]]=f"{raw:.4f}" if raw is not None else "N/A"
                    all_s=fi.compute_all_scores(route, tox_index)
                    for c in fi.CRITERIA_REGISTRY:
                        if c not in criteria and c not in ("steps", "yield"):
                            row[CL[c]+" ✗"]=f"{all_s[c]:.4f}"
                    rows.append(row)
                df=pd.DataFrame(rows).set_index("Route")
                st.dataframe(df,width="stretch")
                st.caption("✓ = selected criteria  ✗ = additional criteria (not used in ranking)")

                if len(sel_results)>=2:
                    st.markdown(f"**{T['radar_title']}**")
                    fig_cmp=make_comparison_chart(sel_results,criteria)
                    st.pyplot(fig_cmp); plt.close(fig_cmp)

                if len(sel_results)==2:
                    st.markdown("---")
                    st.markdown("### Reaction schemes")
                    cA,cB=st.columns(2)
                    for col,(score,details,route),lbl in zip([cA,cB],sel_results,sel[:2]):
                        with col:
                            st.markdown(f"**{route.get('matched_route_name','?')}**")
                            rk="".join(c for c in route.get("matched_route_id","r") if c.isalnum())
                            components.html(
                                build_clickable_scheme_html(route.get("dataset_steps",[]),
                                                            rk,route.get("is_predicted",False)),
                                height=480, scrolling=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def _smiles_copy_widget(smiles: str, label: str = "") -> None:
    """
    Renders a compact inline widget showing a truncated SMILES string with a
    clipboard Copy button. Used in the Dataset Explorer step-by-step view
    beneath each reactant and product image so the chemist can quickly copy
    any SMILES without manually reading the full string.
    """
    short = smiles[:38] + ("…" if len(smiles) > 38 else "")
    safe  = smiles.replace("`", "\\`").replace("\\", "\\\\")
    html_snip = f"""
<div style="
    display:flex; align-items:center; gap:6px;
    background:#f5f7fa; border:1px solid #dce3ec;
    border-radius:6px; padding:4px 8px;
    font-family:'DM Mono','Fira Mono',monospace;
    font-size:0.72rem; color:#37506e;
    margin-top:2px; margin-bottom:6px;
    overflow:hidden;
">
  <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
        title="{smiles}">{short}</span>
  <button onclick="
    navigator.clipboard.writeText(`{safe}`)
      .then(()=>{{this.textContent='✓';setTimeout(()=>this.textContent='Copy',1400)}})
      .catch(()=>{{
        var t=document.createElement('textarea');
        t.value=`{safe}`;document.body.appendChild(t);
        t.select();document.execCommand('copy');
        document.body.removeChild(t);
        this.textContent='✓';setTimeout(()=>this.textContent='Copy',1400);
      }})
  "
  style="
    background:#1a2e44; color:white; border:none;
    border-radius:4px; padding:2px 8px;
    font-size:0.68rem; font-weight:600;
    cursor:pointer; white-space:nowrap; flex-shrink:0;
  ">Copy</button>
</div>
"""
    components.html(html_snip, height=36, scrolling=False)


with tab_dataset:
    if not os.path.exists(dataset_path):
        st.warning(T["ds_not_found"].format(path=dataset_path))
    else:
        with st.spinner("Loading…"): ds=load_dataset_cached(dataset_path)
        reactions_all=ds["all"]; by_route=ds["by_route"]
        targets_uniq=sorted(set(r.get("target","?") for r in reactions_all))

        c1d,c2d,c3d=st.columns(3)
        c1d.metric(T["total_rxn"],len(reactions_all))
        c2d.metric(T["dist_routes"],len(by_route))
        c3d.metric(T["tgt_mols"],len(targets_uniq))
        st.markdown("---")

        filter_tgt=st.selectbox(T["filter_lbl"],[T["filter_all"]]+targets_uniq)

        # Route summary table — sorted numerically by cumulative yield
        rows = []
        for rid, steps in sorted(by_route.items()):
            tgt = steps[0].get("target", "?")
            if filter_tgt != T["filter_all"] and tgt != filter_tgt:
                continue
            nm  = steps[0].get("route_name", rid)
            n   = len(steps)
            ys  = [s.get("yield_percent") for s in steps]
            yo  = [y for y in ys if y is not None]
            yc  = cumulative_yield(steps)
            rows.append({
                T["col_route"]:     nm,
                T["col_target"]:    tgt,
                T["col_steps"]:     n,
                "_yc_raw":          yc,
                T["col_cumyield"]:  f"{yc * 100:.2f}%",
                T["col_missing"]:   sum(1 for y in ys if y is None),
                T["col_avg"]:       f"{sum(yo)/len(yo):.1f}%" if yo else "—",
            })

        if rows:
            import pandas as pd
            df_routes = pd.DataFrame(rows)
            df_routes = (
                df_routes
                .sort_values("_yc_raw", ascending=False)
                .drop(columns=["_yc_raw"])
                .reset_index(drop=True)
            )
            st.dataframe(df_routes, width="stretch", hide_index=True)

        st.markdown("---")
        routes_avail=[rid for rid,steps in by_route.items()
                      if filter_tgt==T["filter_all"] or steps[0].get("target","?")==filter_tgt]
        rc=st.selectbox(T["detail_sel"],routes_avail,
                        format_func=lambda x:f"{by_route[x][0].get('route_name',x)} ({by_route[x][0].get('target','?')})")
        if rc:
            sr=by_route[rc]
            n_sr=len(sr)
            st.markdown(f"**{sr[0].get('route_name')}** — {n_sr} {strip_emoji(T['col_steps']).lower()}")
            fig3=make_yield_chart(sr); st.pyplot(fig3); plt.close(fig3)

            st.markdown("### Step-by-step details")

            for step in sr:
                snum     = step.get("step_number", "?")
                rtype    = step.get("reaction_type", "—") or "—"
                yld      = step.get("yield_percent")
                cond     = step.get("conditions", {})
                prod     = step.get("product_smiles", "")
                reac     = step.get("reactants_smiles", [])
                cond_str = fmt_conditions(cond)
                is_purif = _is_purification_step(step)

                # Purification steps: override the reaction type label if unset
                if is_purif and (rtype == "—" or not rtype):
                    rtype = "Purification / Isolation"

                header_color = "#6B3E26" if is_purif else "linear-gradient(135deg, #1a2e44 0%, #2d5986 100%)"
                header_style = (
                    f"background:{header_color};"
                    if is_purif else
                    "background:linear-gradient(135deg, #1a2e44 0%, #2d5986 100%);"
                )

                st.markdown(
                    f"""
<div style="
    {header_style}
    color: white;
    padding: 10px 18px;
    border-radius: 10px 10px 0 0;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
    margin-top: 18px;
">
    Step {snum} &nbsp;·&nbsp; {rtype}
    {"&nbsp;<span style='font-size:0.75rem;background:rgba(255,255,255,0.2);border-radius:8px;padding:1px 8px;'>Purification</span>" if is_purif else ""}
</div>
""",
                    unsafe_allow_html=True,
                )

                n_reac = max(len(reac), 1)
                col_weights = [3] * n_reac + [1, 3]
                cols_rxn = st.columns(col_weights)

                for i, rsmi in enumerate(reac):
                    with cols_rxn[i]:
                        png_r = mol_png(rsmi, 240, 165)
                        if png_r:
                            st.image(png_r, use_container_width=True)
                        else:
                            st.code(rsmi, language=None)
                        _smiles_copy_widget(rsmi)

                with cols_rxn[n_reac]:
                    st.markdown(
                        """
<div style="
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    min-height: 165px;
    padding-top: 10px;
">
  <div style="display: flex; align-items: center; gap: 0;">
    <div style="
      width: 2.5em; height: 3px;
      background: #2d5986;
      border-radius: 2px 0 0 2px;
    "></div>
    <div style="
      width: 0; height: 0;
      border-top: 9px solid transparent;
      border-bottom: 9px solid transparent;
      border-left: 16px solid #2d5986;
    "></div>
  </div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                with cols_rxn[n_reac + 1]:
                    png_p = mol_png(prod, 240, 165)
                    if png_p:
                        st.image(png_p, use_container_width=True)
                    else:
                        st.code(prod, language=None)
                    _smiles_copy_widget(prod, "Product")

                info_parts = []
                if yld is not None:
                    info_parts.append(f"**Yield:** {yld}%")
                else:
                    info_parts.append("**Yield:** *not reported*")
                if cond_str:
                    info_parts.append(f"**Conditions:** {cond_str}")

                st.markdown(
                    f"""
<div style="
    background: #f0f4f8;
    border: 1px solid #dce3ec;
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 10px 18px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.86rem;
    color: #1a2e44;
    display: flex;
    gap: 32px;
    flex-wrap: wrap;
    margin-bottom: 4px;
">
    {"&nbsp;&nbsp;·&nbsp;&nbsp;".join(
        p.replace("**", "<strong>", 1).replace("**", "</strong>", 1).replace("*","<em>",1).replace("*","</em>",1)
        for p in info_parts
    )}
</div>
""",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown("**Full reaction scheme** *(click an arrow for extra details)*")
            route_ds_key = "ds_" + "".join(c for c in rc if c.isalnum())
            steps_for_scheme = [
                {
                    "step_number":      s.get("step_number"),
                    "reaction_type":    s.get("reaction_type", ""),
                    "yield_percent":    s.get("yield_percent"),
                    "reactants_smiles": s.get("reactants_smiles", []),
                    "product_smiles":   s.get("product_smiles", ""),
                    "conditions":       s.get("conditions", {}),
                    "source":           "dataset",
                }
                for s in sr
            ]
            scheme_h_ds = 320 if n_sr <= 5 else (400 if n_sr <= 12 else 480)
            components.html(
                build_clickable_scheme_html(steps_for_scheme, route_ds_key, False),
                height=scheme_h_ds,
                scrolling=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HELP
# ══════════════════════════════════════════════════════════════════════════════
with tab_help:
    st.subheader(T["help_title"])
    if lang=="en":
        st.markdown("""
**How routes are found and ranked:**

1. `reaction_dataset.json` is loaded (main dataset — full synthesis routes)
2. AiZynthFinder runs MCTS on the target (up to N routes, configurable)
3. Main dataset routes for the target are returned (all of them, unfiltered)
4. Novel AiZ routes are checked against `generic_reactions.json` step by step:
   - **Validated** (all steps found): real conditions, yield included in scoring
   - **Predicted** (no steps found): Rxn-INSIGHT conditions only, yield excluded
5. Weighted 1/i² scoring across criteria

**Purification / isolation steps:**
Steps where the reactant and product SMILES are identical (or where `reaction_type`
contains "purif", "recryst", "chroma", "isolation", or "workup") are treated as
purification steps. They are always shown in the scheme with a **brown dashed arrow**
and a "Purification" badge, even though no bond-forming chemistry occurs. This
preserves the full sequence of operations as recorded in the dataset.

**Three result sections:**
| Section | Source | Conditions | Yield in scoring |
|---------|--------|-----------|-----------------|
| 📚 Dataset | Chemistry by Design | Real | Yes |
| ✅ Validated | AiZ + generic dataset | Real (validated steps) | Yes |
| 🔮 Predicted | AiZ + Rxn-INSIGHT | Predicted | No |

**Required files:**

| File | Required | Role |
|------|----------|------|
| `reaction_dataset.json` | ✅ | Main curated routes |
| `config.yml` | ✅ | AiZynthFinder model config |
| `toxicity_dataset.json` | ✅ | Safety scores — required for Safety criterion |
| `data/uspto_rxn_insight.gzip` | ❌ | Rxn-INSIGHT USPTO database |
| `generic_reactions.json` | ❌ | Individual reactions for step validation |

**Layout tuning — quick reference:**

| Parameter | Location in code | What it controls |
|-----------|-----------------|-----------------|
| `SCHEME_H` | `build_clickable_scheme_html()` | Height of the grey molecule band |
| `PAD_TOP` | same function | Space above the band where co-reactants float |
| `MOL_W / MOL_H` | same function | Size of main molecule images |
| `CO_W / CO_H` | same function | Size of co-reactant images (above arrows) |
| `ARROW_SHAFT_W` | same function | Length of the arrow shaft |
| `ARROW_CELL_W` | same function | Width of the arrow cell (including labels) |
        """)
    else:
        st.markdown("""
**Comment les routes sont trouvées et classées :**

1. `reaction_dataset.json` est chargé (dataset principal — routes de synthèse complètes)
2. AiZynthFinder effectue une recherche MCTS sur la cible (jusqu'à N routes, configurable)
3. Les routes du dataset principal pour la cible sont retournées (toutes, sans filtre top_n)
4. Les routes AiZ nouvelles sont validées étape par étape contre `generic_reactions.json` :
   - **Validées** (toutes les étapes trouvées) : conditions réelles, yield inclus dans le score
   - **Prédites** (aucune étape) : conditions Rxn-INSIGHT uniquement, yield exclu
5. Score pondéré 1/i² sur les critères

**Étapes de purification / isolation :**
Les étapes dont le SMILES du réactif est identique au SMILES du produit (ou dont le champ
`reaction_type` contient "purif", "recryst", "chroma", "isolation" ou "workup") sont
traitées comme des étapes de purification. Elles s'affichent toujours dans le schéma avec
une **flèche pointillée marron** et un badge "Purification", même si aucune liaison
n'est formée. Cela préserve la séquence complète des opérations telle qu'enregistrée dans le dataset.

**Trois sections de résultats :**
| Section | Source | Conditions | Yield dans le score |
|---------|--------|-----------|---------------------|
| 📚 Dataset | Chemistry by Design | Réelles | Oui |
| ✅ Validées | AiZ + dataset générique | Réelles (étapes validées) | Oui |
| 🔮 Prédites | AiZ + Rxn-INSIGHT | Prédites | Non |

**Fichiers nécessaires :**

| Fichier | Obligatoire | Rôle |
|---------|-------------|------|
| `reaction_dataset.json` | ✅ | Routes de synthèse principales |
| `config.yml` | ✅ | Configuration du modèle AiZynthFinder |
| `toxicity_dataset.json` | ✅ | Scores de sécurité — requis pour le critère Sécurité |
| `data/uspto_rxn_insight.gzip` | ❌ | Base USPTO pour Rxn-INSIGHT |
| `generic_reactions.json` | ❌ | Réactions individuelles pour validation |

**Réglage de la mise en page — référence rapide :**

| Paramètre | Emplacement dans le code | Ce qu'il contrôle |
|-----------|--------------------------|-------------------|
| `SCHEME_H` | `build_clickable_scheme_html()` | Hauteur du bandeau gris dans le schéma |
| `PAD_TOP` | même fonction | Espace au-dessus du bandeau (co-réactifs) |
| `MOL_W / MOL_H` | même fonction | Taille des images de molécules principales |
| `CO_W / CO_H` | même fonction | Taille des images de co-réactifs |
| `ARROW_SHAFT_W` | même fonction | Longueur du trait de flèche |
| `ARROW_CELL_W` | même fonction | Largeur de la cellule flèche |
        """)

st.markdown("---")
st.caption(T["footer"])