# app_test.py — Retrosynthesis Interface
# ============================================
# Run:    streamlit run app_test.py
# Place next to: function_ines.py  and  reaction_dataset.json
#
# This application provides a graphical interface for retrosynthetic route
# discovery and comparison.  It relies on:
#   • function_ines.py          — backend: AiZynthFinder wrapper + dataset loading + scoring
#   • reaction_dataset.json     — curated step-by-step reaction data for 5 target molecules
#   • config.yml                — AiZynthFinder model configuration
#   • toxicity_dataset.json     (optional) — reagent/solvent safety scores

import re
import streamlit as st
import os
import io
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# CHEMISTRY BACKEND IMPORT
# Import the chemistry backend (function_ines) and RDKit.
# If any dependency is missing the app degrades gracefully:
# MODULE_OK=False triggers an error banner and stops execution.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import function_ines as fi
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    MODULE_OK  = True
    MODULE_ERR = ""
except Exception as e:
    MODULE_OK  = False
    MODULE_ERR = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call in the script)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retrosynthesis — Chemistry by Design",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  (injected into the Streamlit app iframe)
# Applies DM Serif Display / DM Sans typography, metric card styling,
# expander borders, and button gradient.  Targets Streamlit's internal
# data-testid attributes and class names.
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT STYLES  (self-contained CSS injected inside each st.html() block)
# st.html() renders into an isolated iframe, so the global CSS above does not
# reach it.  COMPONENT_STYLE is prepended to every HTML block that needs
# custom styling: score table, why-best box, animated flow.
# ─────────────────────────────────────────────────────────────────────────────
COMPONENT_STYLE = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

  /* ── Score breakdown table ─────────────────────────────────────────── */
  /* Displays raw score, weight and weighted contribution per criterion.  */
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

  /* Clickable ? tooltip on column header */
  .th-help {
    cursor: help;
    border-bottom: 1px dashed rgba(255,255,255,0.6);
    display: inline-block;
  }
  .th-info {
    display: inline-block;
    background: rgba(255,255,255,0.25);
    border-radius: 50%;
    width: 14px; height: 14px;
    font-size: 0.65rem; line-height: 14px;
    text-align: center; font-weight: 700;
    margin-left: 4px; cursor: help;
    vertical-align: middle;
  }

  /* Inline progress bar inside the table */
  .score-bar-outer {
    background:#e8edf4; border-radius:4px; height:9px;
    width:90px; display:inline-block; vertical-align:middle;
  }
  .score-bar-inner {
    background:linear-gradient(90deg,#4a86c8,#1a2e44);
    border-radius:4px; height:9px;
  }

  /* ── Why-best explanation box ──────────────────────────────────────── */
  .why-box {
    background: linear-gradient(135deg,#edf4fb,#f4f7fb);
    border-left: 4px solid #2d5986;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px; margin: 10px 0 18px 0;
    font-family:'DM Sans',sans-serif; font-size:0.89rem;
    color:#1a2e44; line-height:1.65;
  }
  .why-box strong { color:#1a2e44; font-weight:700; }
  .why-box ul { margin:6px 0 6px 20px; padding:0; }
  .why-box li { margin-bottom:3px; }
  .why-box em { font-size:0.82rem; color:#6b7a8d; }

  /* ── Animated reaction flow ────────────────────────────────────────── */
  /* Cards slide in from left with a staggered animation-delay.          */
  /* The arrow pulses subtly between cards.                              */
  @keyframes fadeSlideIn {
    from { opacity:0; transform:translateX(-14px); }
    to   { opacity:1; transform:translateX(0); }
  }
  @keyframes arrowPulse {
    0%,100% { opacity:0.55; }
    50%      { opacity:1; }
  }

  .rxn-flow-wrap {
    display: flex;
    align-items: center;
    gap: 0;
    overflow-x: auto;
    padding: 18px 12px;
    background: #f9fbfd;
    border-radius: 14px;
    border: 1px solid #dce3ec;
    margin-bottom: 14px;
  }

  /* Each reaction step card */
  .rxn-step-card {
    min-width: 165px; max-width: 205px;
    background: white;
    border-radius: 10px;
    border: 1.5px solid #dce3ec;
    padding: 10px 8px 8px 8px;
    display: flex; flex-direction: column; align-items: center;
    animation: fadeSlideIn 0.45s ease both;
    box-shadow: 0 2px 8px rgba(26,46,68,0.07);
    flex-shrink: 0;
  }
  .rxn-step-badge {
    background: linear-gradient(135deg,#1a2e44,#2d5986);
    color: white; font-size:0.65rem; font-weight:700;
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

  /* Arrow connector between cards */
  .rxn-arrow-zone {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 0 4px; min-width: 90px; flex-shrink: 0;
  }
  /* Conditions text above the arrow line — full text, no truncation */
  .rxn-arrow-cond {
    font-size: 0.59rem; color: #37506e;
    max-width: 86px; text-align: center;
    margin-bottom: 5px; line-height: 1.35;
    font-family: 'DM Sans', sans-serif;
    word-break: break-word; white-space: normal;
  }
  /* Arrow shaft + head built entirely from CSS — no misalignment */
  .rxn-arrow-shaft {
    width: 62px; height: 3px;
    background: linear-gradient(90deg,#2d5986,#1a2e44);
    border-radius: 2px 0 0 2px;
    animation: arrowPulse 2s ease-in-out infinite;
    position: relative;
    /* The head is a separate element placed right of the shaft */
  }
  .rxn-arrow-head {
    width: 0; height: 0;
    border-top: 7px solid transparent;
    border-bottom: 7px solid transparent;
    border-left: 11px solid #1a2e44;
    animation: arrowPulse 2s ease-in-out infinite;
    flex-shrink: 0;
  }
  .rxn-arrow-row {
    display: flex; align-items: center; justify-content: center;
  }

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


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: strip emoji from strings used in matplotlib labels
# Matplotlib's default font (DejaVu Sans) cannot render Unicode emoji such as
# 🔢 📈 ⚗️ — any emoji in axis/legend text causes a UserWarning.
# strip_emoji() removes them, leaving clean ASCII-safe text for all charts.
# ─────────────────────────────────────────────────────────────────────────────
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"   # misc symbols and pictographs
    "\U00002600-\U000027BF"   # misc symbols
    "\U0001F004-\U0001F0CF"
    "\U0001F1E0-\U0001F1FF"
    "\u2702-\u27B0"
    "\u24C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

def strip_emoji(text: str) -> str:
    """Return text with all emoji characters removed and whitespace normalised."""
    return _EMOJI_RE.sub("", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE STRINGS
# All user-visible strings are stored in LANG[lang][key].
# T = LANG[lang] is set once after the language radio button is read.
# Adding a new language requires only a new dict entry here.
# ─────────────────────────────────────────────────────────────────────────────
LANG = {
    "en": {
        "page_title":        "⚗️ Synthesis Route Finder",
        "page_caption":      "AiZynthFinder (MCTS)  ·  Chemistry by Design dataset",
        "sidebar_title":     "⚗️ Settings",
        "files_section":     "⚙️ Files & options",
        "ds_label":          "Dataset",
        "tox_label":         "Toxicity file",
        "cfg_label":         "AiZ config",
        "topn_label":        "Routes to display",
        "weights_exp":       "📐 View criterion weights",
        "tab_search":        "🔍 Route Search",
        "tab_analysis":      "📊 Analysis",
        "tab_dataset":       "📂 Dataset Explorer",
        "tab_help":          "❓ Help",
        "target_section":    "🎯 Target molecule",
        "mode_pre":          "Predefined",
        "mode_custom":       "Custom SMILES",
        "mol_label":         "Molecule",
        "smiles_label":      "SMILES",
        "smiles_ph":         "e.g. c1ccccc1",
        "smiles_invalid":    "Invalid SMILES",
        "smiles_valid":      "Valid SMILES ✓",
        "criteria_section":  "📊 Scoring criteria (priority order)",
        "criteria_caption":  "Criterion #1 ≈ 73% of score  ·  #2 ≈ 18%  ·  #3 ≈ 9%",
        "c1_label":          "Criterion #1 — highest weight",
        "c2_label":          "Criterion #2 — medium weight",
        "c3_label":          "Criterion #3 — lowest weight",
        "run_btn":           "🔍 Run search",
        "welcome":           "Choose a target molecule, set your criteria, and click **Run search**.",
        "avail_mols":        "### Available molecules in the dataset",
        "loading_ds":        "📂 Loading datasets…",
        "loading_aiz":       "🔬 Running AiZynthFinder (may take 1–2 min)…",
        "search_ok":         "✅ Search complete",
        "err_file":          "❌ File not found",
        "err_param":         "❌ Invalid parameter",
        "err_other":         "❌ Unexpected error",
        "no_routes":         (
            "**No routes found.**\n\n"
            "Possible reasons:\n"
            "- AiZynthFinder found no retrosynthetic path for this target\n"
            "- Proposed starting materials don't match any dataset reactants\n"
            "- Aspidospermidine is poorly covered by the USPTO model — try Galanthamine"
        ),
        "n_found":           "{n} route(s) found and ranked.",
        "chart_title":       "Route ranking — {target}",
        "score_axis":        "Total score",
        "metric_score":      "Total score",
        "metric_score_help": "Sum of (raw score × weight) across all criteria.",
        "metric_steps":      "Steps",
        "metric_bottleneck": "Bottleneck yield",
        "metric_bn_help":    "Yield of the worst single step — the main efficiency limiter.",
        "metric_avg":        "Avg step yield",
        "contrib_title":     "Score breakdown",
        "score_th_crit":     "Criterion",
        "score_th_raw":      "Raw score (0–1)",
        "score_th_raw_tip":  (
            "How well this route performs on this criterion, "
            "expressed as a value between 0 (worst) and 1 (best). "
            "Example: a yield raw score of 0.80 means the overall yield "
            "along this route is 80%."
        ),
        "score_th_weight":   "Weight",
        "score_th_contrib":  "Contribution",
        "why_best_title":    "🧠 Why is this route ranked #{r}?",
        "flow_title":        "🧬 Animated reaction flow",
        "steps_title":       "### 🧪 Detailed synthesis pathway — {n} steps",
        "yield_ok":          "**Yield:** {y}%",
        "yield_na":          "**Yield:** *not reported (50% default applied)*",
        "cond_lbl":          "**Conditions:** {c}",
        "smi_exp":           "SMILES — step {n}",
        "react_lbl":         "Reactants:",
        "prod_lbl":          "→ Product:",
        "compare_title":     "🆚 Side-by-side route comparison",
        "sel_routes":        "Select routes to compare",
        "radar_title":       "Criterion profile per route",
        "pareto_title":      "Pareto Front",
        "pareto_x":          "X axis",
        "pareto_y":          "Y axis (must differ from X)",
        "pareto_dominated":  "Dominated routes",
        "pareto_front":      "Pareto-optimal routes",
        "pareto_note":       (
            "**Pareto-optimal routes** (navy dots) cannot be improved on "
            "one criterion without worsening another — they represent the "
            "efficiency frontier.\n\n"
            "**Note on Steps axis:** because fewer steps is better, "
            "the axis shows *1 − raw_score* so that higher always means better "
            "on both axes."
        ),
        "no_analysis":       "Run a search first to unlock the Analysis tab.",
        "ds_not_found":      "Dataset `{path}` not found.",
        "filter_lbl":        "Filter by target",
        "filter_all":        "All",
        "total_rxn":         "Total reactions",
        "dist_routes":       "Distinct routes",
        "tgt_mols":          "Target molecules",
        "col_route":         "Route",
        "col_target":        "Target",
        "col_steps":         "Steps",
        "col_cumyield":      "Cumulative yield",
        "col_missing":       "Missing yields",
        "col_avg":           "Avg yield",
        "detail_sel":        "View route details",
        "help_title":        "How it works",
        "help_body": """
**How routes are found and ranked:**

1. `reaction_dataset.json` is loaded and indexed by route name and target molecule
2. AiZynthFinder runs a Monte Carlo Tree Search (MCTS) on the target to propose
   retrosynthetic disconnections, starting from a USPTO-trained neural network model
3. The starting materials suggested by AiZynthFinder are matched against reactants
   available in the dataset; only routes with matching reactants are kept
4. Matched routes are **filtered to the correct target molecule only**,
   then scored using a weighted combination of the three user-selected criteria

**Score breakdown table explained:**
- **Raw score (0–1)**: how well this route performs on a single criterion.
  For yield: raw = overall yield along the route (e.g. 0.80 = 80%).
  For steps: raw = 1/n, so fewer steps gives a higher raw score.
- **Weight**: relative importance of this criterion, derived from the priority order
  you chose (position #1 gets ~73%, #2 gets ~18%, #3 gets ~9%).
- **Contribution**: Raw × Weight — the amount actually added to the total score.
  Total score = sum of all contributions across the three criteria.

**Bottleneck yield** = yield of the single worst step in the route.
This is more informative than average or cumulative yield because it is the
step that limits the maximum obtainable amount of product.

**Pareto front**: A route is Pareto-optimal if no other route is simultaneously
better (or equal) on every criterion.  Routes on the Pareto front represent
genuine trade-offs: you cannot improve yield without adding steps, for example.

**Required files:**

| File | Required | Role |
|------|----------|------|
| `reaction_dataset.json` | ✅ | Curated reaction data |
| `config.yml` | ✅ | AiZynthFinder model configuration |
| `toxicity_dataset.json` | ❌ | Reagent/solvent safety scores (0.5 default if absent) |
""",
        "footer":            "Chemistry by Design  ·  AiZynthFinder MCTS",
        "mod_err":           "❌ Cannot load function_ines:",
        "mod_hint":          "Make sure function_ines.py is in the same folder and all dependencies are installed.",
        "searching":         "🔍 Searching…",
        "rxn_err":           "⚠️ Could not render reaction image: {e}",
        "why_prefix":        "This route is ranked <strong>#{r}</strong> because:",
        "why_steps":         "it has only <strong>{n} steps</strong> — one of the shortest pathways found",
        "why_yield":         "it achieves a bottleneck yield of <strong>{y}%</strong> on the limiting step",
        "why_avg_yield":     "its average step yield of <strong>{a}%</strong> is above the field average",
        "why_top_score":     "it dominates on <strong>{crit}</strong> (raw score {s:.2f}), the highest-weighted criterion ({w:.0%})",
        "why_balanced":      "it presents a well-balanced profile across all three selected criteria",
        "why_suffix":        "No single criterion alone determines ranking — the weighted combination reflects your stated priorities.",
    },
    "fr": {
        "page_title":        "⚗️ Recherche de routes de synthèse",
        "page_caption":      "AiZynthFinder (MCTS)  ·  Dataset Chemistry by Design",
        "sidebar_title":     "⚗️ Paramètres",
        "files_section":     "⚙️ Fichiers & options",
        "ds_label":          "Dataset",
        "tox_label":         "Fichier toxicité",
        "cfg_label":         "Config AiZ",
        "topn_label":        "Routes à afficher",
        "weights_exp":       "📐 Voir les poids des critères",
        "tab_search":        "🔍 Recherche de routes",
        "tab_analysis":      "📊 Analyse",
        "tab_dataset":       "📂 Explorer le dataset",
        "tab_help":          "❓ Aide",
        "target_section":    "🎯 Molécule cible",
        "mode_pre":          "Prédéfinie",
        "mode_custom":       "SMILES personnalisé",
        "mol_label":         "Molécule",
        "smiles_label":      "SMILES",
        "smiles_ph":         "ex : c1ccccc1",
        "smiles_invalid":    "SMILES invalide",
        "smiles_valid":      "SMILES valide ✓",
        "criteria_section":  "📊 Critères de scoring (ordre de priorité)",
        "criteria_caption":  "Critère #1 ≈ 73% du score  ·  #2 ≈ 18%  ·  #3 ≈ 9%",
        "c1_label":          "Critère #1 — poids le plus élevé",
        "c2_label":          "Critère #2 — poids moyen",
        "c3_label":          "Critère #3 — poids le plus bas",
        "run_btn":           "🔍 Lancer la recherche",
        "welcome":           "Choisissez une molécule cible, définissez vos critères et cliquez sur **Lancer la recherche**.",
        "avail_mols":        "### Molécules disponibles dans le dataset",
        "loading_ds":        "📂 Chargement des datasets…",
        "loading_aiz":       "🔬 Lancement d'AiZynthFinder (peut prendre 1–2 min)…",
        "search_ok":         "✅ Recherche terminée",
        "err_file":          "❌ Fichier introuvable",
        "err_param":         "❌ Paramètre invalide",
        "err_other":         "❌ Erreur inattendue",
        "no_routes":         (
            "**Aucune route trouvée.**\n\n"
            "Pistes possibles :\n"
            "- AiZynthFinder n'a trouvé aucun chemin rétrosynthétique pour cette cible\n"
            "- Les starting materials ne correspondent à aucun réactif du dataset\n"
            "- L'aspidospermidine est mal couverte par le modèle USPTO — essayez la galanthamine"
        ),
        "n_found":           "{n} route(s) trouvée(s) et classée(s).",
        "chart_title":       "Classement des routes — {target}",
        "score_axis":        "Score total",
        "metric_score":      "Score total",
        "metric_score_help": "Somme de (score brut × poids) sur tous les critères.",
        "metric_steps":      "Étapes",
        "metric_bottleneck": "Rendement limitant",
        "metric_bn_help":    "Rendement de l'étape la plus faible — principal facteur limitant.",
        "metric_avg":        "Rendement moyen / étape",
        "contrib_title":     "Décomposition du score",
        "score_th_crit":     "Critère",
        "score_th_raw":      "Score brut (0–1)",
        "score_th_raw_tip":  (
            "Performance de la route sur ce critère, "
            "exprimée entre 0 (pire) et 1 (meilleur). "
            "Exemple : un score brut de 0.80 sur le rendement signifie "
            "que le rendement global de la route est de 80%."
        ),
        "score_th_weight":   "Poids",
        "score_th_contrib":  "Contribution",
        "why_best_title":    "🧠 Pourquoi cette route est-elle classée #{r} ?",
        "flow_title":        "🧬 Flux de réaction animé",
        "steps_title":       "### 🧪 Chemin de synthèse détaillé — {n} étapes",
        "yield_ok":          "**Rendement :** {y}%",
        "yield_na":          "**Rendement :** *non renseigné (50% par défaut)*",
        "cond_lbl":          "**Conditions :** {c}",
        "smi_exp":           "SMILES — étape {n}",
        "react_lbl":         "Réactifs :",
        "prod_lbl":          "→ Produit :",
        "compare_title":     "🆚 Comparaison côte à côte des routes",
        "sel_routes":        "Sélectionner les routes à comparer",
        "radar_title":       "Profil par critère",
        "pareto_title":      "Front de Pareto",
        "pareto_x":          "Axe X",
        "pareto_y":          "Axe Y (doit différer de X)",
        "pareto_dominated":  "Routes dominées",
        "pareto_front":      "Routes Pareto-optimales",
        "pareto_note":       (
            "**Routes Pareto-optimales** (points bleu marine) : impossible de les améliorer "
            "sur un critère sans en dégrader un autre — elles forment la frontière d'efficacité.\n\n"
            "**Note sur l'axe Étapes :** comme moins d'étapes est préférable, "
            "l'axe affiche *1 − score_brut*, de sorte que plus haut = meilleur sur les deux axes."
        ),
        "no_analysis":       "Lancez une recherche d'abord pour accéder à l'onglet Analyse.",
        "ds_not_found":      "Dataset `{path}` introuvable.",
        "filter_lbl":        "Filtrer par cible",
        "filter_all":        "Toutes",
        "total_rxn":         "Réactions totales",
        "dist_routes":       "Routes distinctes",
        "tgt_mols":          "Molécules cibles",
        "col_route":         "Route",
        "col_target":        "Cible",
        "col_steps":         "Étapes",
        "col_cumyield":      "Rendement cumulé",
        "col_missing":       "Rendements manquants",
        "col_avg":           "Rendement moyen",
        "detail_sel":        "Voir le détail d'une route",
        "help_title":        "Comment ça marche",
        "help_body": """
**Comment les routes sont trouvées et classées :**

1. `reaction_dataset.json` est chargé et indexé par nom de route et molécule cible
2. AiZynthFinder effectue une recherche MCTS sur la cible pour proposer des
   déconnexions rétrosynthétiques, à partir d'un réseau de neurones entraîné sur USPTO
3. Les starting materials proposés par AiZynthFinder sont comparés aux réactifs
   disponibles dans le dataset ; seules les routes avec des réactifs correspondants sont conservées
4. Les routes validées sont **filtrées pour la bonne molécule cible**,
   puis scorées par combinaison pondérée des trois critères choisis

**Tableau de décomposition du score :**
- **Score brut (0–1)** : performance de la route sur un critère.
  Pour le rendement : brut = rendement global (ex. 0.80 = 80%).
  Pour les étapes : brut = 1/n, donc moins d'étapes = score plus élevé.
- **Poids** : importance relative du critère selon l'ordre de priorité
  (#1 obtient ~73%, #2 ~18%, #3 ~9%).
- **Contribution** : Brut × Poids — ce qui est ajouté au score total.

**Rendement limitant** = rendement de l'étape la plus faible.
C'est le facteur qui limite la quantité maximale de produit obtenu.

**Front de Pareto** : une route est Pareto-optimale si aucune autre route
n'est simultanément meilleure (ou égale) sur tous les critères.

**Fichiers nécessaires :**

| Fichier | Obligatoire | Rôle |
|---------|-------------|------|
| `reaction_dataset.json` | ✅ | Données de réaction |
| `config.yml` | ✅ | Configuration du modèle AiZynthFinder |
| `toxicity_dataset.json` | ❌ | Scores de sécurité (0.5 par défaut si absent) |
""",
        "footer":            "Chemistry by Design  ·  AiZynthFinder MCTS",
        "mod_err":           "❌ Impossible de charger function_ines :",
        "mod_hint":          "Vérifiez que function_ines.py est dans le même dossier et que toutes les dépendances sont installées.",
        "searching":         "🔍 Recherche en cours…",
        "rxn_err":           "⚠️ Impossible d'afficher le diagramme : {e}",
        "why_prefix":        "Cette route est classée <strong>#{r}</strong> car :",
        "why_steps":         "elle ne compte que <strong>{n} étapes</strong> — l'un des chemins les plus courts",
        "why_yield":         "elle atteint un rendement limitant de <strong>{y}%</strong> sur l'étape critique",
        "why_avg_yield":     "son rendement moyen de <strong>{a}%</strong> est au-dessus de la moyenne",
        "why_top_score":     "elle domine sur <strong>{crit}</strong> (score brut {s:.2f}), le critère le plus pondéré ({w:.0%})",
        "why_balanced":      "elle présente un profil équilibré sur les trois critères sélectionnés",
        "why_suffix":        "Aucun critère seul ne détermine le classement — la combinaison pondérée reflète vos priorités.",
    },
}

# Criterion labels: emoji versions used in UI text; plain versions for matplotlib
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

TARGETS = {
    "Galanthamine":     "OC1C=C[C@@]23c4cc(OC)ccc4CN(C)C[C@@H]2[C@@H]1O3",
    "Morphine":         "OC1=CC2=C(C=C1)[C@@H]1[C@H]3C[C@@H](O)C=C[C@@H]3N(C)CC1=C2",
    "Quinine":          "OC(c1ccnc2cc(OC)ccc12)[C@@H]1C[C@@H]2CC[N@@]1CC2/C=C",
    "Aspidospermidine": "CC[C@@]12CCCN3CC[C@@]4(C1)[C@@H](NH)c1ccccc1[C@@H]4[C@@H]23",
    "Aspidospermine":   "CC[C@@]12CCCN3CC[C@@]4(C1)[C@@H](OC(C)=O)c1ccc(OC)cc1N[C@@H]4[C@@H]23",
}
SMILES_TO_NAME = {v: k for k, v in TARGETS.items()}

# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE SELECTOR  (must run before any T[] usage)
# Placed at the very top of the sidebar so the whole app re-renders in the
# chosen language on every interaction.
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    lang_choice = st.radio(
        "🌐 Language / Langue",
        ["🇬🇧 English", "🇫🇷 Français"],
        horizontal=True,
    )
    lang = "en" if "English" in lang_choice else "fr"

T  = LANG[lang]
CL = CRITERIA_LABELS[lang]

PALETTE = ["#1a2e44", "#2d5986", "#4a86c8", "#6aabd2", "#9fc8e0"]
FIG_BG  = "#f9fbfd"


# ─────────────────────────────────────────────────────────────────────────────
# MOLECULE RENDERING HELPERS
# All molecule images are rendered via RDKit's Cairo backend (MolDraw2DCairo)
# at 2× the logical display size, giving HiDPI-sharp output with no pixelation.
#   mol_png()     → raw PNG bytes for st.image()
#   mol_pil_img() → RGBA PIL Image for compositing into reaction diagrams
#   mol_b64()     → base-64 encoded PNG for embedding in HTML <img> tags
# ─────────────────────────────────────────────────────────────────────────────

def mol_png(smiles: str, w: int = 800, h: int = 540) -> bytes | None:
    """Render a SMILES at 2× resolution via Cairo for crisp display."""
    if not MODULE_OK or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rw, rh = w * 2, h * 2
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(rw, rh)
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = True
        opts.bondLineWidth = 2.5
        opts.padding = 0.14
        opts.multipleBondOffset = 0.19
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        img = Draw.MolToImage(mol, size=(rw, rh))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def mol_pil_img(smiles: str, w: int = 480, h: int = 320):
    """Return RGBA PIL Image at 2× for compositing into reaction step diagrams."""
    from PIL import Image
    data = mol_png(smiles, w, h)
    if data is None:
        return None
    return Image.open(io.BytesIO(data)).convert("RGBA")


def mol_b64(smiles: str, w: int = 140, h: int = 100) -> str:
    """Return base-64 PNG (3× resolution) suitable for HTML <img src='data:…'>."""
    import base64
    data = mol_png(smiles, w * 3, h * 3)
    if data is None:
        return ""
    return base64.b64encode(data).decode()


# ─────────────────────────────────────────────────────────────────────────────
# REACTION STEP DIAGRAM  (static PNG rendered with PIL)
# draw_reaction_step() composes a wide PNG:  reactant(s) [+] ──cond──► product
# Molecule cells are pre-rendered at 2× via mol_pil_img(), pasted onto a 2×
# canvas, and exported at 144 dpi.  Font loader tries system paths, falls back.
# ─────────────────────────────────────────────────────────────────────────────

def draw_reaction_step(
    reactant_smiles: list,
    product_smiles: str,
    conditions_str: str,
    step_num,
    reaction_type: str,
) -> bytes | None:
    """Render one reaction step as a full-width PNG with reactants, arrow and product."""
    from PIL import Image, ImageDraw, ImageFont
    import textwrap

    MOL_W, MOL_H = 480, 320
    ARROW_W = 320
    PLUS_GAP = 60
    H_PAD = 36
    V_PAD = 28
    TITLE_H = 56
    S = 2  # molecule images are already at 2× logical size

    BG       = (250, 251, 253)
    NAVY     = (26, 46, 68)
    WHITE    = (255, 255, 255)
    COND_CLR = (55, 80, 110)
    SEP      = (210, 220, 232)

    def load_font(size, bold=False):
        """Try common font paths; fall back to PIL default if none found."""
        paths = (
            ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
             "C:/Windows/Fonts/arialbd.ttf"]
            if bold else
            ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
             "C:/Windows/Fonts/arial.ttf"]
        )
        for p in paths:
            try:
                from PIL import ImageFont as _IF
                return _IF.truetype(p, size)
            except Exception:
                pass
        from PIL import ImageFont as _IF
        return _IF.load_default()

    font_title = load_font(32, bold=True)
    font_cond  = load_font(26)
    font_plus  = load_font(52, bold=True)

    react_imgs = [mol_pil_img(s, MOL_W, MOL_H) for s in reactant_smiles]
    react_imgs = [i for i in react_imgs if i is not None]
    prod_img   = mol_pil_img(product_smiles, MOL_W, MOL_H)

    mw          = MOL_W * S
    mh          = MOL_H * S
    arrow_w_px  = ARROW_W * S
    plus_gap_px = PLUS_GAP * S
    h_pad_px    = H_PAD * S
    v_pad_px    = V_PAD * S
    title_h_px  = TITLE_H * S

    n_r = max(len(react_imgs), 1)
    react_zone_w = n_r * mw + (n_r - 1) * plus_gap_px
    total_w = h_pad_px + react_zone_w + arrow_w_px + mw + h_pad_px
    total_h = title_h_px + v_pad_px + mh + v_pad_px

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    # Title bar
    draw.rectangle([0, 0, total_w, title_h_px], fill=NAVY)
    draw.text((32, (title_h_px - 32) // 2),
              f"  Step {step_num}  ·  {reaction_type}",
              fill=WHITE, font=font_title)

    y0 = title_h_px + v_pad_px
    x  = h_pad_px

    # Paste reactant images
    for i, img in enumerate(react_imgs):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        canvas.paste(Image.alpha_composite(bg, img).convert("RGB"), (x, y0))
        x += mw
        if i < len(react_imgs) - 1:
            try:
                tw = draw.textlength("+", font=font_plus)
            except Exception:
                tw = 36
            draw.text((x + (plus_gap_px - int(tw)) // 2, y0 + mh // 2 - 28),
                      "+", fill=NAVY, font=font_plus)
            x += plus_gap_px

    # Arrow
    ax0 = x + 36
    ax1 = ax0 + arrow_w_px - 72
    ay  = y0 + mh // 2
    AH  = 28
    draw.line([(ax0, ay), (ax1 - AH + 4, ay)], fill=NAVY, width=6)
    draw.polygon([(ax1, ay), (ax1 - AH, ay - AH // 2), (ax1 - AH, ay + AH // 2)],
                 fill=NAVY)

    # Conditions text above arrow — no truncation, all lines shown
    if conditions_str:
        lines  = textwrap.wrap(conditions_str, width=30)  # all lines, no [:N] limit
        mid_x  = (ax0 + ax1) // 2
        line_h = 30
        text_y = ay - len(lines) * line_h - 20
        for line in lines:
            try:
                tw = draw.textlength(line, font=font_cond)
            except Exception:
                tw = len(line) * 16
            draw.text((mid_x - int(tw) // 2, text_y),
                      line, fill=COND_CLR, font=font_cond)
            text_y += line_h

    # Product
    px = ax1 + 36
    if prod_img:
        bg = Image.new("RGBA", prod_img.size, (255, 255, 255, 255))
        canvas.paste(Image.alpha_composite(bg, prod_img).convert("RGB"), (px, y0))

    draw.line([(0, total_h - 1), (total_w, total_h - 1)], fill=SEP, width=2)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", dpi=(144, 144))
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# ANIMATED REACTION FLOW  (HTML component rendered via st.html())
# Generates a horizontally scrollable row of step cards connected by CSS
# animated arrows.  Cards fade in with staggered delays.
# Arrow head: separate <div> border-triangle in a flex row — no ::after drift.
# Conditions text: NOT truncated; word-wrap enabled; full text always shown.
# Molecules: base-64 PNGs at 3× for crispness inside small HTML cards.
# ─────────────────────────────────────────────────────────────────────────────

def build_animated_flow_html(steps_data: list) -> str:
    """Build a scrollable HTML row of animated step cards for a synthesis route."""
    cards = []
    for i, step in enumerate(steps_data):
        snum     = step.get("step_number", i + 1)
        rtype    = step.get("reaction_type", "—")
        yld      = step.get("yield_percent")
        cond     = step.get("conditions", {})
        reac     = step.get("reactants_smiles", [])
        prod     = step.get("product_smiles", "")
        cond_str = fmt_conditions(cond)
        yld_str  = f"{yld}%" if yld is not None else "—"
        delay_ms = i * 130

        mol_smi = reac[0] if reac else prod
        b64     = mol_b64(mol_smi, 140, 100)
        img_tag = (
            f'<img class="rxn-step-mol" src="data:image/png;base64,{b64}" alt="mol"/>'
            if b64 else
            '<div class="rxn-step-mol" style="display:flex;align-items:center;'
            'justify-content:center;color:#aaa;font-size:10px">no structure</div>'
        )
        start_lbl = '<div class="rxn-start-label">Starting material</div>' if i == 0 else ""

        cards.append(f"""
<div class="rxn-step-card" style="animation-delay:{delay_ms}ms">
  {start_lbl}
  <div class="rxn-step-badge">Step {snum}</div>
  {img_tag}
  <div class="rxn-step-type">{rtype}</div>
  <div class="rxn-step-yield">&#x27F3; {yld_str}</div>
</div>""")

        if i < len(steps_data) - 1:
            # Full conditions shown — no truncation
            cards.append(f"""
<div class="rxn-arrow-zone" style="animation-delay:{delay_ms + 65}ms">
  <div class="rxn-arrow-cond">{cond_str}</div>
  <div class="rxn-arrow-row">
    <div class="rxn-arrow-shaft"></div>
    <div class="rxn-arrow-head"></div>
  </div>
</div>""")

    # Final product card — arrow carries the last step's conditions
    if steps_data:
        last_step      = steps_data[-1]
        prod_smi       = last_step.get("product_smiles", "")
        last_cond_str  = fmt_conditions(last_step.get("conditions", {}))
        b64_prod       = mol_b64(prod_smi, 140, 100)
        img_prod       = (
            f'<img class="rxn-step-mol" src="data:image/png;base64,{b64_prod}" alt="product"/>'
            if b64_prod else '<div class="rxn-step-mol"></div>'
        )
        delay_last = len(steps_data) * 130
        # Arrow before final product — includes conditions of the last step
        cards.append(f"""
<div class="rxn-arrow-zone" style="animation-delay:{delay_last - 65}ms">
  <div class="rxn-arrow-cond">{last_cond_str}</div>
  <div class="rxn-arrow-row">
    <div class="rxn-arrow-shaft"></div>
    <div class="rxn-arrow-head"></div>
  </div>
</div>
<div class="rxn-step-card" style="animation-delay:{delay_last}ms;border-color:#1a2e44;">
  <div class="rxn-end-label">Target</div>
  {img_prod}
  <div class="rxn-step-type" style="color:#1a2e44;font-weight:600;">Final product</div>
</div>""")

    inner = "\n".join(cards)
    return f"{COMPONENT_STYLE}<div class='rxn-flow-wrap'>{inner}</div>"


# ─────────────────────────────────────────────────────────────────────────────
# SCORE BREAKDOWN TABLE  (HTML rendered via st.html())
# 4-column table: Criterion | Raw score (0–1) [?] | Weight | Contribution
# "Performance" bar column removed (was always empty).
# Raw score header has a hoverable ? tooltip explaining the calculation.
# Weights passed explicitly from fi.compute_weights() to avoid recomputation.
# ─────────────────────────────────────────────────────────────────────────────

def build_score_table_html(details: dict, criteria: list, weights: dict) -> str:
    """Return self-contained HTML for the score breakdown table."""
    tip = T["score_th_raw_tip"].replace('"', '&quot;')
    rows = []
    for c in criteria:
        raw  = details[c]["raw"]
        w    = weights.get(c, 0)
        cont = details[c]["weighted"]
        label = CL[c]   # emoji label is fine in HTML
        rows.append(f"""
  <tr>
    <td>{label}</td>
    <td>{raw:.3f}</td>
    <td>{w * 100:.0f}%</td>
    <td>{cont:.3f}</td>
  </tr>""")

    rows_html = "".join(rows)
    return f"""{COMPONENT_STYLE}
<table class="score-table">
<thead>
  <tr>
    <th>{T['score_th_crit']}</th>
    <th>
      <span class="th-help" title="{tip}">{T['score_th_raw']}
        <span class="th-info">?</span>
      </span>
    </th>
    <th>{T['score_th_weight']}</th>
    <th>{T['score_th_contrib']}</th>
  </tr>
</thead>
<tbody>{rows_html}
</tbody>
</table>"""


# ─────────────────────────────────────────────────────────────────────────────
# WHY-BEST EXPLANATION BOX  (HTML rendered via st.html())
# Blue-accent box explaining why a route is ranked where it is.  Inspects:
#   - number of steps vs. all other returned routes
#   - bottleneck yield and average yield thresholds
#   - raw score on the top-weighted criterion
# At least one bullet always fires (falls back to "balanced profile").
# Uses HTML <strong> tags (not markdown **) because rendered via st.html().
# ─────────────────────────────────────────────────────────────────────────────

def build_why_best_html(rank: int, score_total: float, details: dict,
                         route: dict, criteria: list,
                         all_results: list, weights: dict) -> str:
    """Return self-contained HTML for the why-best explanation block."""
    steps_data = route.get("dataset_steps", [])
    n_steps    = len(steps_data)
    bn         = bottleneck_yield(steps_data)
    av         = average_yield(steps_data)

    bullets = []
    all_step_counts = [len(r[2].get("dataset_steps", [])) for r in all_results]
    if all_step_counts and n_steps <= min(all_step_counts):
        bullets.append(T["why_steps"].format(n=n_steps))
    if bn is not None and bn >= 70:
        bullets.append(T["why_yield"].format(y=int(bn)))
    elif av is not None and av >= 65:
        bullets.append(T["why_avg_yield"].format(a=int(av)))
    top_c   = criteria[0]
    top_raw = details[top_c]["raw"]
    top_w   = weights.get(top_c, 0)
    if top_raw >= 0.60:
        bullets.append(T["why_top_score"].format(
            crit=CL[top_c], s=top_raw, w=top_w))
    if not bullets:
        bullets.append(T["why_balanced"])

    items_html = "".join(f"<li>{b}</li>" for b in bullets)
    return f"""{COMPONENT_STYLE}
<div class="why-box">
  <strong>{T['why_best_title'].format(r=rank)}</strong><br><br>
  {T['why_prefix'].format(r=rank)}
  <ul>{items_html}</ul>
  <em>{T['why_suffix']}</em>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# GENERAL UTILITY FUNCTIONS
#   fmt_conditions()          — format a conditions dict into a readable string
#   bottleneck_yield()        — min yield across all steps (efficiency limiter)
#   average_yield()           — mean yield of steps with reported values
#   filter_routes_by_target() — post-filter search results to the correct molecule
#   load_dataset()            — cached call to fi.load_reaction_dataset()
# ─────────────────────────────────────────────────────────────────────────────

def fmt_conditions(cond: dict) -> str:
    """Format a step conditions dict into a human-readable string."""
    if not cond:
        return ""
    parts = []
    if cond.get("temperature_C"):
        parts.append(f"{cond['temperature_C']}°C")
    elif cond.get("temp_range"):
        parts.append(cond["temp_range"])
    if cond.get("solvent"):
        parts.append(cond["solvent"])
    if cond.get("co_solvent"):
        parts.append(f"/ {cond['co_solvent']}")
    if isinstance(cond.get("reagents"), list) and cond["reagents"]:
        parts.append(", ".join(cond["reagents"]))
    if cond.get("apparatus"):
        parts.append(f"({cond['apparatus']})")
    return "  ·  ".join(parts)


def bottleneck_yield(steps):
    """Return the minimum step yield (the rate-limiting step), or None."""
    ys = [s.get("yield_percent") for s in steps if s.get("yield_percent") is not None]
    return min(ys) if ys else None


def average_yield(steps):
    """Return the mean of reported step yields, or None if none are reported."""
    ys = [s.get("yield_percent") for s in steps if s.get("yield_percent") is not None]
    return sum(ys) / len(ys) if ys else None


def filter_routes_by_target(results, target_smiles: str):
    """
    Keep only routes that belong to the requested target molecule.
    Checks three fields in decreasing reliability:
      1. dataset_steps[0]['target']      — set when the dataset was built
      2. route['matched_target']         — set by the matching step
      3. route['matched_route_name']     — last resort string match
    Falls back to the full list only for custom (unknown) SMILES.
    """
    target_name = SMILES_TO_NAME.get(target_smiles, "").lower()
    if not target_name:
        return results  # custom SMILES — cannot name-filter
    filtered = []
    for item in results:
        score, details, route = item
        steps    = route.get("dataset_steps", [])
        step_tgt = (steps[0].get("target", "") if steps else "").lower()
        mt       = route.get("matched_target", "").lower()
        rn       = route.get("matched_route_name", "").lower()
        if target_name in step_tgt or target_name in mt or target_name in rn:
            filtered.append(item)
    return filtered if filtered else results


@st.cache_data(show_spinner=False)
def load_dataset(path: str):
    """Load and cache the reaction dataset via function_ines."""
    return fi.load_reaction_dataset(path)


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS  (matplotlib figures returned for st.pyplot())
# All functions strip emoji via strip_emoji() before passing text to matplotlib,
# preventing Glyph-missing UserWarnings from DejaVu Sans.
# Shared style: FIG_BG background, PALETTE colours, no spines, dashed grid.
# ─────────────────────────────────────────────────────────────────────────────

def make_ranking_chart(results, target_name: str):
    """Horizontal bar chart ranking all found routes by total score."""
    labels = [r[2].get("matched_route_name", f"Route {i+1}")[:32]
              for i, r in enumerate(results)]
    scores = [r[0] for r in results]
    n = len(scores)

    fig, ax = plt.subplots(figsize=(9, max(2.8, n * 1.0)))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(FIG_BG)
    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]
    bars   = ax.barh(np.arange(n), scores, color=colors, height=0.52, edgecolor="none")
    for bar, s in zip(bars, scores):
        ax.text(s + max(scores) * 0.013,
                bar.get_y() + bar.get_height() / 2,
                f"{s:.3f}", va="center", fontsize=9.5,
                color="#1a2e44", fontweight="600")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=10, color="#1a2e44")
    # strip_emoji not needed for score_axis / chart_title (no emoji in those strings)
    ax.set_xlabel(T["score_axis"], fontsize=10, color="#6b7a8d")
    ax.set_xlim(0, max(scores) * 1.28)
    ax.set_title(T["chart_title"].format(target=target_name),
                 fontsize=13, fontweight="bold", color="#1a2e44", pad=14)
    ax.tick_params(colors="#6b7a8d", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.grid(True, color="#dce3ec", linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.6)
    return fig


def make_yield_chart(steps_route):
    """Bar chart of step yields for a single route (dataset explorer)."""
    vals    = [s.get("yield_percent") or 50 for s in steps_route]
    is_null = [s.get("yield_percent") is None for s in steps_route]
    n = len(vals)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.85), 3.0))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(FIG_BG)
    colors = ["#c8d8e8" if null else "#1a2e44" for null in is_null]
    ax.bar(range(1, n + 1), vals, color=colors, width=0.52, edgecolor="none")
    ax.axhline(50, color="#e07b39", linestyle="--", linewidth=1.2)
    ax.set_ylim(0, 115)
    ax.set_xticks(range(1, n + 1))
    # Use plain strings in legend — no emoji
    ax.set_xlabel(strip_emoji(T["col_steps"]), fontsize=10, color="#6b7a8d")
    ax.set_ylabel("Yield (%)", fontsize=10, color="#6b7a8d")
    ax.legend(handles=[
        mpatches.Patch(color="#1a2e44",
                       label="Reported yield" if lang == "en" else "Rendement renseigne"),
        mpatches.Patch(color="#c8d8e8",
                       label="Missing -> 50%" if lang == "en" else "Manquant -> 50%"),
    ], fontsize=8, framealpha=0.8, facecolor=FIG_BG, edgecolor="#dce3ec")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color="#dce3ec", linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#6b7a8d", length=0)
    fig.tight_layout(pad=1.2)
    return fig


def make_comparison_chart(sel_results, criteria):
    """
    Horizontal grouped bar chart for side-by-side criterion comparison.
    One group of bars per criterion, one bar per route.
    All axis labels use strip_emoji() to avoid DejaVu Glyph warnings.
    """
    route_names = [r[2].get("matched_route_name", f"R{i+1}")[:20]
                   for i, r in enumerate(sel_results)]
    n_routes = len(sel_results)
    n_crit   = len(criteria)
    x        = np.arange(n_crit)
    bar_h    = 0.72 / n_routes
    offsets  = np.linspace(-(n_routes - 1) / 2,
                            (n_routes - 1) / 2, n_routes) * bar_h

    fig, ax = plt.subplots(figsize=(8, max(3.5, n_crit * 1.2)))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(FIG_BG)

    for i, (score, details, route) in enumerate(sel_results):
        vals  = [details[c]["raw"] for c in criteria]
        color = PALETTE[i % len(PALETTE)]
        name  = route_names[i]
        bars  = ax.barh(x + offsets[i], vals, bar_h * 0.88,
                        color=color, label=name, edgecolor="none", alpha=0.92)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.012, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", fontsize=8,
                    color=color, fontweight="600")

    # Strip emoji from criterion labels used as matplotlib y-ticks
    crit_labels_plain = [strip_emoji(CL[c]) for c in criteria]
    ax.set_yticks(x)
    ax.set_yticklabels(crit_labels_plain, fontsize=10, color="#1a2e44")
    ax.set_xlim(0, 1.22)
    ax.set_xlabel("Raw score (0-1)" if lang == "en" else "Score brut (0-1)",
                  fontsize=10, color="#6b7a8d")
    # Strip emoji from title too
    ax.set_title(strip_emoji(T["radar_title"]), fontsize=12,
                 fontweight="bold", color="#1a2e44", pad=12)
    ax.legend(loc="lower right", fontsize=8,
              framealpha=0.8, facecolor=FIG_BG, edgecolor="#dce3ec")
    ax.tick_params(colors="#6b7a8d", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.grid(True, color="#dce3ec", linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.4)
    return fig


def compute_pareto_front(points):
    """
    Return indices of Pareto-optimal points (maximise both axes).
    A point is dominated if another point is >= on both axes and > on at least one.
    """
    pareto = []
    for i, (x1, y1) in enumerate(points):
        dominated = any(
            (x2 >= x1 and y2 >= y1) and (x2 > x1 or y2 > y1)
            for j, (x2, y2) in enumerate(points) if j != i
        )
        if not dominated:
            pareto.append(i)
    return pareto


def make_pareto_chart(results, x_key: str, y_key: str):
    """
    Scatter plot with Pareto-optimal routes highlighted.
    For the 'steps' criterion, the raw score is inverted (1 - raw) so that
    higher always means better on both axes.
    Axis labels are plain ASCII (no emoji) to avoid DejaVu Glyph warnings.
    The title is also emoji-free.
    """
    sample = results[0][1]
    if x_key not in sample or y_key not in sample or x_key == y_key:
        return None

    xs = [r[1][x_key]["raw"] for r in results]
    ys = [r[1][y_key]["raw"] for r in results]
    xs_n = [1 - v if x_key == "steps" else v for v in xs]
    ys_n = [1 - v if y_key == "steps" else v for v in ys]

    pareto_idx = compute_pareto_front(list(zip(xs_n, ys_n)))
    dom_idx    = [i for i in range(len(results)) if i not in pareto_idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(FIG_BG)

    if dom_idx:
        ax.scatter([xs_n[i] for i in dom_idx], [ys_n[i] for i in dom_idx],
                   s=80, color="#9fc8e0", alpha=0.75, zorder=2,
                   label=strip_emoji(T["pareto_dominated"]))

    px = [xs_n[i] for i in pareto_idx]
    py = [ys_n[i] for i in pareto_idx]
    ax.scatter(px, py, s=130, color="#1a2e44", zorder=3,
               edgecolors="#4a86c8", linewidths=2.0,
               label=strip_emoji(T["pareto_front"]))

    for i in pareto_idx:
        name = results[i][2].get("matched_route_name", f"R{i+1}")[:14]
        ax.annotate(name, (xs_n[i], ys_n[i]),
                    textcoords="offset points", xytext=(7, 7),
                    fontsize=8, color="#1a2e44")

    if len(pareto_idx) > 1:
        sp_sorted = sorted(zip(px, py))
        ax.plot([p[0] for p in sp_sorted], [p[1] for p in sp_sorted],
                "--", color="#2d5986", linewidth=1.2, alpha=0.6)

    # Plain-text axis labels — note on steps inversion goes in a caption below
    def axis_label(key):
        base = strip_emoji(CL.get(key, key))
        if key == "steps":
            return base + (" (fewer = right)" if lang == "en" else " (moins = droite)")
        return base

    ax.set_xlabel(axis_label(x_key), fontsize=10, color="#6b7a8d")
    ax.set_ylabel(axis_label(y_key), fontsize=10, color="#6b7a8d")
    # No emoji in title
    ax.set_title(strip_emoji(T["pareto_title"]), fontsize=12,
                 fontweight="bold", color="#1a2e44", pad=12)
    ax.legend(fontsize=8, framealpha=0.8,
              facecolor=FIG_BG, edgecolor="#dce3ec")
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.grid(True, color="#dce3ec", linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#6b7a8d", length=0)
    fig.tight_layout(pad=1.4)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  (language already set above; this adds files, weights)
# Contains: language toggle, module error banner, file path inputs,
# top-n slider, collapsible criterion weight preview (after a search runs).
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(T["sidebar_title"])

    if not MODULE_OK:
        st.error(f"{T['mod_err']}\n\n`{MODULE_ERR}`")
        st.info(T["mod_hint"])
        st.stop()

    st.divider()
    st.subheader(T["files_section"])
    dataset_path  = st.text_input(T["ds_label"],  value="reaction_dataset.json")
    toxicity_path = st.text_input(T["tox_label"], value="toxicity_dataset.json")
    config_path   = st.text_input(T["cfg_label"], value="config.yml")
    top_n         = st.slider(T["topn_label"], 1, 5, 3)

    st.divider()
    if "criteria" in st.session_state:
        crit_for_weights = st.session_state["criteria"]
        weights_preview  = fi.compute_weights(crit_for_weights)
        with st.expander(T["weights_exp"]):
            for c in crit_for_weights:
                w = weights_preview[c]
                st.progress(float(w), text=f"{CL[c]}  —  {w * 100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA  (tabs)
# ─────────────────────────────────────────────────────────────────────────────
st.title(T["page_title"])
st.caption(T["page_caption"])

tab_search, tab_analysis, tab_dataset, tab_help = st.tabs(
    [T["tab_search"], T["tab_analysis"], T["tab_dataset"], T["tab_help"]]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ROUTE SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:

    # Target selector and criterion selectors live in the main area (not sidebar)
    top_left, top_right = st.columns([3, 2])

    with top_left:
        st.subheader(T["target_section"])
        mode = st.radio("Mode", [T["mode_pre"], T["mode_custom"]],
                        horizontal=True, label_visibility="collapsed")

        if mode == T["mode_pre"]:
            target_name   = st.selectbox(T["mol_label"], list(TARGETS.keys()))
            target_smiles = TARGETS[target_name]
        else:
            target_smiles = st.text_input(T["smiles_label"],
                                          placeholder=T["smiles_ph"])
            target_name = "Custom"
            if target_smiles:
                if Chem.MolFromSmiles(target_smiles) is None:
                    st.error(T["smiles_invalid"])
                    target_smiles = ""
                else:
                    st.success(T["smiles_valid"])

        st.subheader(T["criteria_section"])
        st.caption(T["criteria_caption"])
        all_crit = list(CL.keys())
        c1 = st.selectbox(T["c1_label"], all_crit, format_func=lambda x: CL[x])
        rest2 = [c for c in all_crit if c != c1]
        c2 = st.selectbox(T["c2_label"], rest2, format_func=lambda x: CL[x])
        rest3 = [c for c in rest2 if c != c2]
        c3 = st.selectbox(T["c3_label"], rest3, format_func=lambda x: CL[x])

        criteria = [c1, c2, c3]
        st.session_state["criteria"] = criteria

        run_search = st.button(T["run_btn"], type="primary",
                               disabled=not target_smiles)

    with top_right:
        if target_smiles:
            png = mol_png(target_smiles, 320, 220)
            if png:
                st.image(png, caption=target_name, width="stretch")
        else:
            st.markdown(T["welcome"])
            st.markdown(T["avail_mols"])
            mol_cols = st.columns(3)
            for idx, (name, smi) in enumerate(TARGETS.items()):
                with mol_cols[idx % 3]:
                    png = mol_png(smi, 200, 135)
                    if png:
                        st.image(png, caption=name, width="stretch")

    st.divider()

    # ── Run AiZynthFinder search ──────────────────────────────────────────────
    if run_search and target_smiles:
        errs = []
        if not os.path.exists(dataset_path):
            errs.append(f"`{dataset_path}`")
        if not os.path.exists(config_path):
            errs.append(f"`{config_path}`")
        for e in errs:
            st.error(f"{T['err_file']}: {e}")
        if not errs:
            with st.status(T["searching"], expanded=True) as status:
                try:
                    st.write(T["loading_ds"])
                    st.write(T["loading_aiz"])
                    raw = fi.find_best_routes(
                        target_smiles     = target_smiles,
                        criteria_priority = criteria,
                        dataset_path      = dataset_path,
                        toxicity_path     = toxicity_path,
                        config_path       = config_path,
                        top_n             = top_n * 4,
                    )
                    results_found = filter_routes_by_target(raw, target_smiles)[:top_n]
                    weights_found = fi.compute_weights(criteria)
                    st.session_state["results"]       = results_found
                    st.session_state["weights"]       = weights_found
                    st.session_state["target_name"]   = target_name
                    st.session_state["target_smiles"] = target_smiles
                    st.session_state["criteria"]      = criteria
                    status.update(label=T["search_ok"],
                                  state="complete", expanded=False)
                except FileNotFoundError as e:
                    status.update(label=T["err_file"], state="error")
                    st.error(str(e))
                except ValueError as e:
                    status.update(label=T["err_param"], state="error")
                    st.error(str(e))
                except Exception as e:
                    status.update(label=T["err_other"], state="error")
                    st.exception(e)

    # ── Display cached results ────────────────────────────────────────────────
    results  = st.session_state.get("results", [])
    weights  = st.session_state.get("weights", {})
    criteria = st.session_state.get("criteria", criteria)

    if not results:
        if run_search:
            st.warning(T["no_routes"])
    else:
        tgt_name = st.session_state.get("target_name", target_name)
        st.success(T["n_found"].format(n=len(results)))

        if len(results) > 1:
            fig = make_ranking_chart(results, tgt_name)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

        for rank, (score_total, details, route) in enumerate(results, 1):
            route_name   = route.get("matched_route_name", "?")
            route_target = route.get("matched_target", "?")
            steps_data   = route.get("dataset_steps", [])
            n_steps      = len(steps_data)
            bn           = bottleneck_yield(steps_data)
            av           = average_yield(steps_data)

            with st.expander(
                f"{medals[rank-1]}  {route_name}  ·  {route_target}"
                f"  ·  Score: **{score_total:.3f}**",
                expanded=(rank == 1),
            ):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(T["metric_score"], f"{score_total:.3f}",
                          help=T["metric_score_help"])
                m2.metric(T["metric_steps"], n_steps)
                m3.metric(T["metric_bottleneck"],
                          f"{bn:.0f}%" if bn is not None else "—",
                          help=T["metric_bn_help"])
                m4.metric(T["metric_avg"],
                          f"{av:.0f}%" if av is not None else "—")

                st.markdown(f"**{T['contrib_title']}**")
                st.html(build_score_table_html(details, criteria, weights))
                st.html(build_why_best_html(rank, score_total, details,
                                            route, criteria, results, weights))

                st.markdown("---")
                st.markdown(f"**{T['flow_title']}**")
                st.html(build_animated_flow_html(steps_data))

                st.markdown(T["steps_title"].format(n=n_steps))

                for step in steps_data:
                    snum     = step.get("step_number", "?")
                    rtype    = step.get("reaction_type", "—")
                    yld      = step.get("yield_percent")
                    cond     = step.get("conditions", {})
                    prod     = step.get("product_smiles", "")
                    reac     = step.get("reactants_smiles", [])
                    cond_str = fmt_conditions(cond)

                    try:
                        rxn = draw_reaction_step(reac, prod, cond_str, snum, rtype)
                        if rxn:
                            st.image(rxn, width="stretch")
                    except Exception as rxn_err:
                        st.caption(T["rxn_err"].format(e=rxn_err))

                    col_txt, col_smi = st.columns([2, 1])
                    with col_txt:
                        st.markdown(T["yield_ok"].format(y=yld)
                                    if yld is not None else T["yield_na"])
                        if cond_str:
                            st.markdown(T["cond_lbl"].format(c=cond_str))
                    with col_smi:
                        with st.expander(T["smi_exp"].format(n=snum), expanded=False):
                            if reac:
                                st.markdown(f"**{T['react_lbl']}**")
                                for r in reac:
                                    st.code(r, language=None)
                                st.markdown(f"**{T['prod_lbl']}**")
                            st.code(prod, language=None)
                    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYSIS  (comparison + Pareto front)
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    results  = st.session_state.get("results", [])
    criteria = st.session_state.get("criteria", list(CL.keys())[:3])
    weights  = st.session_state.get("weights", {})

    if not results:
        st.info(T["no_analysis"])
    else:
        import pandas as pd

        route_names = [r[2].get("matched_route_name", f"Route {i+1}")
                       for i, r in enumerate(results)]

        # ── Side-by-side comparison ───────────────────────────────────────────
        st.subheader(T["compare_title"])
        sel = st.multiselect(T["sel_routes"], route_names,
                             default=route_names[:min(3, len(route_names))])

        if sel:
            sel_results = [r for r in results
                           if r[2].get("matched_route_name", "") in sel]

            # Summary table
            rows = []
            for score, details, route in sel_results:
                sd  = route.get("dataset_steps", [])
                bn_ = bottleneck_yield(sd)
                av_ = average_yield(sd)
                row = {
                    "Route":               route.get("matched_route_name", "?")[:26],
                    T["metric_score"]:     f"{score:.3f}",
                    T["metric_steps"]:     len(sd),
                    T["metric_bottleneck"]:f"{bn_:.0f}%" if bn_ else "—",
                    T["metric_avg"]:       f"{av_:.0f}%" if av_ else "—",
                }
                for c in criteria:
                    row[CL[c]] = f"{details[c]['raw']:.3f}"
                rows.append(row)

            df = pd.DataFrame(rows).set_index("Route")
            st.dataframe(df, width="stretch")

            # Criterion profile: horizontal grouped bar chart
            if len(sel_results) >= 2:
                st.markdown(f"**{T['radar_title']}**")
                fig_cmp = make_comparison_chart(sel_results, criteria)
                st.pyplot(fig_cmp)
                plt.close(fig_cmp)

        st.markdown("---")

        # ── Pareto front ──────────────────────────────────────────────────────
        st.subheader(T["pareto_title"])

        sample_det = results[0][1]
        avail_axes = [c for c in ["yield", "steps", "atom_economy",
                                   "e_factor", "toxicity"]
                      if c in sample_det]

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            x_axis = st.selectbox(
                T["pareto_x"], avail_axes,
                index=0,
                format_func=lambda x: CL.get(x, x),
                key="pareto_x_sel",
            )
        with pcol2:
            # Enforce x != y: y options exclude the chosen x
            y_opts = [a for a in avail_axes if a != x_axis]
            y_axis = st.selectbox(
                T["pareto_y"], y_opts,
                index=0,
                format_func=lambda x: CL.get(x, x),
                key="pareto_y_sel",
            )

        pfig = make_pareto_chart(results, x_axis, y_axis)
        if pfig:
            st.pyplot(pfig)
            plt.close(pfig)
            # Explain Pareto and the steps-axis inversion in a caption
            st.info(T["pareto_note"])
        else:
            st.info("Select two different axes to display the Pareto front.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_dataset:
    if not os.path.exists(dataset_path):
        st.warning(T["ds_not_found"].format(path=dataset_path))
    else:
        with st.spinner("Loading…"):
            ds = load_dataset(dataset_path)

        import pandas as pd
        reactions_all = ds["all"]
        by_route      = ds["by_route"]
        targets_uniq  = sorted(set(r.get("target", "?") for r in reactions_all))

        c1d, c2d, c3d = st.columns(3)
        c1d.metric(T["total_rxn"],   len(reactions_all))
        c2d.metric(T["dist_routes"], len(by_route))
        c3d.metric(T["tgt_mols"],    len(targets_uniq))
        st.markdown("---")

        filter_tgt = st.selectbox(T["filter_lbl"],
                                  [T["filter_all"]] + targets_uniq)

        rows = []
        for rid, steps in sorted(by_route.items()):
            tgt = steps[0].get("target", "?")
            if filter_tgt != T["filter_all"] and tgt != filter_tgt:
                continue
            nm    = steps[0].get("route_name", rid)
            n     = len(steps)
            ys    = [s.get("yield_percent") for s in steps]
            ys_ok = [y for y in ys if y is not None]
            yc    = 1.0
            for s in steps:
                y = s.get("yield_percent")
                yc *= (y / 100.0) if y is not None else 0.5
            rows.append({
                T["col_route"]:    nm,
                T["col_target"]:   tgt,
                T["col_steps"]:    n,
                T["col_cumyield"]: f"{yc * 100:.1f}%",
                T["col_missing"]:  sum(1 for y in ys if y is None),
                T["col_avg"]:      f"{sum(ys_ok)/len(ys_ok):.0f}%" if ys_ok else "—",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.markdown("---")
        routes_avail = [
            rid for rid, steps in by_route.items()
            if filter_tgt == T["filter_all"]
            or steps[0].get("target", "?") == filter_tgt
        ]
        route_chosen = st.selectbox(
            T["detail_sel"], routes_avail,
            format_func=lambda x: (
                f"{by_route[x][0].get('route_name', x)} "
                f"({by_route[x][0].get('target', '?')})"
            ),
        )

        if route_chosen:
            steps_route = by_route[route_chosen]
            st.markdown(f"**{steps_route[0].get('route_name')}**"
                        f" — {len(steps_route)} {strip_emoji(T['col_steps']).lower()}")

            fig3 = make_yield_chart(steps_route)
            st.pyplot(fig3)
            plt.close(fig3)

            for step in steps_route:
                snum     = step.get("step_number", "?")
                rtype    = step.get("reaction_type", "—")
                yld      = step.get("yield_percent")
                cond     = step.get("conditions", {})
                prod     = step.get("product_smiles", "")
                reac     = step.get("reactants_smiles", [])
                cond_str = fmt_conditions(cond)

                try:
                    rxn = draw_reaction_step(reac, prod, cond_str, snum, rtype)
                    if rxn:
                        st.image(rxn, width="stretch")
                except Exception:
                    pass

                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(T["yield_ok"].format(y=yld)
                                if yld is not None else T["yield_na"])
                    if cond_str:
                        st.markdown(T["cond_lbl"].format(c=cond_str))
                with col_b:
                    with st.expander(T["smi_exp"].format(n=snum), expanded=False):
                        st.code(prod, language=None)
                st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HELP
# ══════════════════════════════════════════════════════════════════════════════
with tab_help:
    st.subheader(T["help_title"])
    st.markdown(T["help_body"])

st.markdown("---")
st.caption(T["footer"])
