# app_path_finder.py
# =============================================================================
# Retrosynthesis Interface — Streamlit front-end.
#
# Companion modules (same directory):
#   localization.py      — LANG dict, CRITERIA_LABELS, PALETTE, FIG_BG
#   molecule_rendering.py — mol_png, mol_b64_or_text_svg, fallback_data_uri,
#                           is_trivial_smiles
#   report_builder.py    — build_route_report_pdf
#
# Backend module (project root or PYTHONPATH):
#   route_engine.py — all scoring, loading, AiZynthFinder / Rxn-INSIGHT
#
# UI structure:
#   Sidebar   : language selector, file paths, search parameters
#   Tab 1     : Route Search — target input, criteria selection, run button,
#               route cards with reaction schemes
#   Tab 2     : Analysis — side-by-side route comparison, criterion profile chart
#   Tab 3     : Dataset Explorer — browse all routes in the curated dataset
#   Tab 4     : Help — methodology documentation
#
# Run:
#   streamlit run app_path_finder.py
# =============================================================================

import re
import json
import os
import sys

# Ensure sibling modules (route_engine, molecule_rendering, etc.) are always
# importable regardless of the working directory from which streamlit is launched.
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -- Local modules -------------------------------------------------------------
from localization import LANG, CRITERIA_LABELS, PALETTE, FIG_BG
from molecule_rendering import (
    mol_png,
    mol_b64_or_text_svg as _mol_b64_or_text_svg,
    fallback_data_uri   as _fallback_data_uri,
    is_trivial_smiles   as _is_trivial_smiles,
    MODULE_OK,
)
from report_builder import build_route_report_pdf

# -- Backend imports -----------------------------------------------------------
try:
    import route_engine as fi
    from rdkit import Chem
    MODULE_ERR = ""
except Exception as e:
    fi         = None
    Chem       = None
    MODULE_ERR = str(e)

try:
    from rxn_insight.reaction import Reaction as RxnInsightReaction
    RXNINSIGHT_OK = True
except ImportError:
    RXNINSIGHT_OK = False

# -- Page configuration --------------------------------------------------------
st.set_page_config(
    page_title="Retrosynthesis — Chemistry by Design",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- High-DPI matplotlib helper ------------------------------------------------
def _hires_fig(*args, dpi: int = 180, **kwargs):
    """
    Create a high-resolution matplotlib figure.

    Drop-in replacement for plt.subplots(). Sets figure and axes background
    to FIG_BG so charts blend with the page without a white box.

    Parameters
    ──────────
    *args, **kwargs : forwarded to plt.subplots()
    dpi             : int  dots per inch (default 180 for crisp display)

    Returns
    ───────
    (fig, ax) tuple, same as plt.subplots()
    """
    fig, ax = plt.subplots(*args, dpi=dpi, **kwargs)
    fig.patch.set_facecolor(FIG_BG)
    if hasattr(ax, '__iter__'):
        for a in ax:
            a.set_facecolor(FIG_BG)
    else:
        ax.set_facecolor(FIG_BG)
    return fig, ax

# =============================================================================
# Global CSS — injected once into the Streamlit page
# =============================================================================
# Uses DM Serif Display for headings and DM Sans for body text (Google Fonts).
# Metric cards, expanders, buttons, and tabs are all restyled here.
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

# =============================================================================
# CSS used inside isolated HTML components (st.html / components.html)
# =============================================================================
# Scoped to avoid leaking into the Streamlit global stylesheet.
# Includes: score breakdown table, why-best reasoning box, source badges,
# animated reaction flow cards.
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

  /* source badge shown next to step headers */
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
</style>
"""

# -- Emoji stripping for matplotlib labels ------------------------------------
# matplotlib cannot render Unicode emoji inside text — they must be removed
# before passing any label string to an axes or axis method.
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
    """
    Remove all Unicode emoji characters from a string.

    Required before passing criterion labels (which include emoji like 🔢, ♻️)
    to matplotlib, which cannot render emoji glyphs and raises a warning.

    Parameters
    ──────────
    text : str

    Returns
    ───────
    str  text with all emoji removed and leading/trailing whitespace stripped
    """
    return _EMOJI_RE.sub("", text).strip()


# =============================================================================
# Language selector (sidebar, rendered before any content)
# =============================================================================
with st.sidebar:
    lang_choice = st.radio(
        "🌐 Language / Langue",
        ["🇬🇧 English", "🇫🇷 Français"],
        horizontal=True,
        index=0,
    )
    lang = "en" if "English" in lang_choice else "fr"

# Shorthand references updated on every re-run after language selection
T  = LANG[lang]
CL = CRITERIA_LABELS[lang]


# =============================================================================
# Reaction scheme HTML builder
# =============================================================================

def _is_purification_step(step: dict) -> bool:
    """
    Determine whether a step is a purification / isolation step.

    A step is treated as a purification step when ANY of the following holds:
    • reaction_type contains 'purif', 'recryst', 'chroma', 'isolation', or 'workup'
      (case-insensitive) — explicitly labelled purification operations.
    • The product SMILES is canonically identical to one of the reactant SMILES
      (identity transform = isolation / recrystallisation with no bond-forming
      chemistry, which is the key case requested).

    Purification steps are always displayed in the reaction scheme with a brown
    dashed arrow and a "Purification" badge even when the product == reactant,
    preserving the full recorded sequence of operations.

    Parameters
    ──────────
    step : dict  step dict from dataset_steps

    Returns
    ───────
    bool  True if the step should be treated as a purification step
    """
    rtype = (step.get("reaction_type") or "").lower()
    if any(kw in rtype for kw in ("purif", "recryst", "chroma", "isolation", "workup")):
        return True
    prod  = step.get("product_smiles", "")
    reacs = step.get("reactants_smiles", [])
    # String-level identity check (fast path)
    if prod and prod in reacs:
        return True
    # Canonical comparison via RDKit (handles different SMILES notations for the same structure)
    if MODULE_OK and prod and Chem is not None:
        mol_p = Chem.MolFromSmiles(prod)
        canon_prod = Chem.MolToSmiles(mol_p) if mol_p else prod
        for r in reacs:
            mol_r = Chem.MolFromSmiles(r) if r else None
            if mol_r and Chem.MolToSmiles(mol_r) == canon_prod:
                return True
    return False


def build_clickable_scheme_html(steps_data: list, route_id: str,
                                 is_predicted: bool = False) -> str:
    """
    Build the interactive HTML reaction scheme shown in Route Search and Analysis.

    Each molecule in the synthetic sequence is rendered as a double-resolution
    Cairo PNG image. Clicking any reaction arrow opens an inline detail panel
    showing conditions, yield, reactant/product structures, and copyable SMILES.

    Layout is controlled by the named constants defined at the top of the
    function body — adjust them to resize the scheme without touching the HTML.

    Purification step handling
    ──────────────────────────
    When _is_purification_step() returns True for a step, the product molecule
    is always added to the display sequence even if its SMILES is identical to
    the preceding molecule (standard steps skip duplicates). The arrow is drawn
    with a dashed brown style and labelled "Purification / Isolation".

    Co-reactant rendering
    ─────────────────────
    Molecules in the reactants list that are not the substrate (i.e. not the
    molecule being transformed) are rendered as small structure images above
    the arrow. Trivial species (single atoms, ions — _is_trivial_smiles) are
    shown as text labels below the arrow instead to avoid cluttering the scheme.

    Parameters
    ──────────
    steps_data   : list  route["dataset_steps"]
    route_id     : str   unique identifier used as DOM element ID prefix
    is_predicted : bool  if True, arrows use orange colouring

    Returns
    ───────
    str  complete self-contained HTML document ready for components.html()

    Layout constants (edit here to resize the scheme)
    ──────────────────────────────────────────────────
    MOL_W, MOL_H     : main molecule image size in CSS pixels
    CO_W,  CO_H      : co-reactant image size (above the arrows)
    SCHEME_H         : height of the horizontal molecule band
    PAD_TOP          : space above the band where co-reactant images float
    PAD_BOTTOM       : space below the band
    ARROW_SHAFT_W    : pixel length of the horizontal arrow shaft
    ARROW_CELL_W     : total min-width of the arrow cell (shaft + labels)
    """
    if not steps_data:
        return "<p style='color:#888'>No steps.</p>"

    # -- Colour theme ----------------------------------------------------------
    # Predicted routes use orange to visually distinguish them from validated ones
    arrow_color = "#E65100" if is_predicted else "#1a2e44"
    hover_bg    = "#FFF3E0" if is_predicted else "#E8F0FC"
    panel_bg    = "#FFF8F0" if is_predicted else "#F0F7FF"
    rid_js      = "".join(c for c in route_id if c.isalnum())

    # -- LAYOUT CONSTANTS — edit these to resize the scheme -------------------
    MOL_W, MOL_H = 158, 112   # main molecule image (CSS pixels; rendered at 2×)
    CO_W,  CO_H  = 96,  70    # co-reactant image (CSS pixels; rendered at 2×)
    CELL_W       = MOL_W + 20 # total width of each molecule cell
    SCHEME_H     = 200         # height of the molecule band in pixels

    # Compute how much vertical padding is needed above the band for co-reactants
    max_co_rows = 1
    for step in steps_data:
        reactants = step.get("reactants_smiles", [])
        product   = step.get("product_smiles", "")
        cond      = step.get("conditions", {})
        co_count  = 0
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

    display_rows   = min(max_co_rows, 2)
    PAD_TOP        = display_rows * (CO_H + 2) + 30 + 20
    PAD_BOTTOM     = display_rows * (CO_H + 2) // 3 + 10
    ARROW_SHAFT_W  = 90
    ARROW_CELL_W   = 145

    # -- Build molecule sequence and per-arrow metadata -----------------------
    mol_sequence = []  # ordered list of SMILES for the horizontal display
    arrow_data   = []  # one entry per reaction arrow

    for i, step in enumerate(steps_data):
        reactants   = step.get("reactants_smiles", [])
        product     = step.get("product_smiles", "")
        cond        = step.get("conditions", {})
        rtype       = step.get("reaction_type", "") or ""
        yld         = step.get("yield_percent")
        snum        = step.get("step_number", i + 1)
        src_step    = step.get("source", "dataset")
        is_purif    = _is_purification_step(step)

        # Add first reactant on step 0; skip if already the last molecule shown
        if i == 0 and reactants:
            if reactants[0] not in mol_sequence:
                mol_sequence.append(reactants[0])

        if product:
            # Purification steps always add the product (even if same as predecessor)
            if mol_sequence and mol_sequence[-1] == product and not is_purif:
                pass  # duplicate — skip for standard steps
            else:
                mol_sequence.append(product)

        prev = mol_sequence[-2] if len(mol_sequence) >= 2 else ""
        co_draw = []   # co-reactants to draw as structure images above the arrow
        co_text = []   # trivial co-reactants to show as text below the arrow

        for r in reactants:
            if r == prev or r == product:
                continue
            if _is_trivial_smiles(r):
                co_text.append(r)
            else:
                co_draw.append(r)

        # Build the conditions display string
        all_cond_parts = []
        temp = cond.get("temperature_C")
        if temp:
            all_cond_parts.append(str(temp) + "°C")
        elif cond.get("temp_range"):
            all_cond_parts.append(cond["temp_range"])
        solv = cond.get("solvent", "")
        if solv:
            if _is_trivial_smiles(solv):
                all_cond_parts.append(solv)
            else:
                if solv not in co_draw:
                    co_draw.append(solv)
                all_cond_parts.append(solv)
        for r in (cond.get("reagents", []) or []):
            if not r:
                continue
            if _is_trivial_smiles(r):
                if r not in all_cond_parts:
                    all_cond_parts.append(r)
            else:
                if r not in co_draw:
                    co_draw.append(r)
                if r not in all_cond_parts:
                    all_cond_parts.append(r)
        if cond.get("apparatus"):
            all_cond_parts.append("(" + cond["apparatus"] + ")")
        cond_display = "  ·  ".join(all_cond_parts)

        # Purification steps: supply a reaction type label when none is set
        display_rtype = rtype
        if is_purif and not rtype:
            display_rtype = "Purification / Isolation"

        # Assemble the compact text shown below the arrow
        below_parts = []
        if yld is not None and src_step != "rxn-insight":
            below_parts.append(str(yld) + "%")
        for t in co_text[:3]:
            below_parts.append(t)
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

    # -- Pre-render all molecule images to base64 data URIs -------------------
    # Main molecules (in the horizontal band)
    mol_imgs = {smi: _mol_b64_or_text_svg(smi, MOL_W, MOL_H)
                for smi in mol_sequence if smi}
    # Co-reactant images (floating above arrows)
    co_imgs  = {}
    for a in arrow_data:
        for smi in a["co_draw"]:
            if smi and smi not in co_imgs:
                co_imgs[smi] = _mol_b64_or_text_svg(smi, CO_W, CO_H)

    # Step detail panel images (reactants and product shown when arrow is clicked)
    step_imgs = {}
    for a in arrow_data:
        r_b64s = [_mol_b64_or_text_svg(rsmi, 110, 78) for rsmi in a["reactants"][:4]]
        p_b64  = _mol_b64_or_text_svg(a["product"], 110, 78)
        step_imgs[str(a["step"])] = {
            "reactants":       r_b64s,
            "product":         p_b64,
            "reactant_smiles": a["reactants"],
            "product_smiles":  a["product"],
        }
    step_imgs_json = json.dumps(step_imgs)

    n_mols = len(mol_sequence)
    items  = []   # list of HTML fragment strings assembled into the scheme

    # -- Assemble molecule cells and arrow cells in alternating order ----------
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
                # Each co-reactant image is wrapped with its SMILES in the title attribute
                uco = co_imgs.get(s, _fallback_data_uri(s, CO_W, CO_H))
                return (
                    '<img src="' + uco + '" width="' + str(CO_W) +
                    '" height="' + str(CO_H) +
                    '" style="border:1px solid #dce3ec;border-radius:3px;background:#fff;" '
                    'title="' + s.replace('"', '&quot;') + '"/>'
                )

            # Lay out co-reactant images in rows of up to 3
            if n_co == 0:
                co_html = ""
            elif n_co == 1:
                co_html = '<div class="co-row">' + _co_img_tag(co_draw_slice[0]) + '</div>'
            else:
                top_row = '<div class="co-row">'
                for s in co_draw_slice[:-1]:
                    top_row += _co_img_tag(s)
                top_row += '</div>'
                bot_row = '<div class="co-row">' + _co_img_tag(co_draw_slice[-1]) + '</div>'
                co_html = top_row + bot_row

            # Serialise step metadata as a JSON attribute for the JS click handler
            step_json_esc = json.dumps({
                k: v for k, v in a.items() if k not in ("co_draw", "co_text", "below", "is_purif")
            }).replace('"', "&quot;")

            # Purification steps: dashed brown arrow
            purif_dash  = "stroke-dasharray:6,3;" if a["is_purif"] else ""
            purif_color = "#8B5E3C" if a["is_purif"] else arrow_color

            # Compact text labels displayed below the arrow shaft
            below_html = ""
            if a["yld"]:
                below_html += '<div class="b-yld">' + a["yld"] + '</div>'
            if a["rtype"]:
                below_html += '<div class="b-rtype">' + a["rtype"] + '</div>'
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

    # -- CSS for the self-contained scheme document ---------------------------
    css = (
        "html,body{margin:0;padding:0;background:#fff;"
        "font-family:'DM Sans',Arial,sans-serif;}"
        "*{box-sizing:border-box}"

        "#scroll-wrap{"
        "overflow-x:auto;overflow-y:visible;"
        "background:#FAFAFA;border-radius:10px;border:1px solid #dce3ec;"
        "}"

        ".band{"
        "display:flex;align-items:center;justify-content:flex-start;"
        "height:" + str(SCHEME_H) + "px;"
        "gap:2px;padding:0 8px;overflow:visible;"
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
        "padding:2px 4px;border-radius:6px;"
        "transition:background 0.12s;user-select:none;flex-shrink:0;overflow:visible;"
        "}"
        ".arrow-cell:hover{background:" + hover_bg + "}"

        ".co-above{"
        "position:absolute;bottom:100%;margin-bottom:-35px;"
        "left:50%;transform:translateX(-50%);"
        "display:flex;flex-direction:column;align-items:center;"
        "gap:2px;pointer-events:none;"
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

    # -- JavaScript: click handler + auto-resize postMessage ------------------
    # The iframe height is reported to Streamlit via postMessage so the
    # components.html() container expands/contracts when the detail panel
    # opens or closes.
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
        # Toggle: click the same arrow again to close the panel
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

    # -- Assemble the final HTML document -------------------------------------
    return (
        "<!DOCTYPE html><html><head>"
        "<style>" + css + "</style></head><body>"
        '<div id="wrap-' + route_id + '">'
        '<div id="scroll-wrap">'
        '<div class="band">' + "".join(items) + '</div>'
        '</div>'
        # Detail panel — hidden until an arrow is clicked
        '<div class="detail-panel" id="dp-' + route_id + '">'
        '<div class="dp-title" id="dpt-' + route_id + '">Step details</div>'
        '<div class="dp-grid" id="dpg-' + route_id + '"></div>'
        '<div class="step-imgs" id="dpi-' + route_id + '"></div>'
        '<div class="smi-section" id="dps-' + route_id + '"></div>'
        '</div>'
        '</div>'
        # Step image data embedded as JSON to avoid re-requesting from Python
        '<script type="application/json" id="__si_' + route_id + '">'
        + step_imgs_json + '</script>'
        '<script>' + js + '</script>'
        '</body></html>'
    )


# =============================================================================
# Score table HTML builder
# =============================================================================

# Tooltip text explaining how each raw score is computed (shown on hover in the
# score breakdown table header). Keyed by criterion name and language.
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
    Render the per-criterion score breakdown table for a route card.

    Each row shows: criterion name, raw score (0–1) with an inline tooltip
    explaining the formula, the criterion weight, and the weighted contribution
    to the total score.

    For predicted routes the yield row shows "excluded (predicted route)"
    instead of a numeric score.

    Parameters
    ──────────
    details  : dict  per-criterion detail dicts from rank_weighted()
    criteria : list  3 criterion keys in priority order
    weights  : dict  {criterion_key: weight} from fi.compute_weights()

    Returns
    ───────
    str  HTML string (includes COMPONENT_STYLE; safe for st.html())
    """
    rows = []
    csd  = CRITERIA_SCORE_DESC.get(lang, CRITERIA_SCORE_DESC["en"])
    for c in criteria:
        raw   = details[c].get("raw")
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


# =============================================================================
# Matplotlib chart helpers
# =============================================================================

def make_ranking_chart(results: list, target_name: str):
    """
    Render a horizontal bar chart ranking routes by total score.

    Displayed above the route cards in each category section (Dataset /
    Validated / Predicted) when more than one route is found.
    Score values are annotated directly on each bar.

    Parameters
    ──────────
    results     : list of (score, details, route) tuples from rank_weighted()
    target_name : str  used as the chart title suffix

    Returns
    ───────
    matplotlib.figure.Figure  (caller is responsible for plt.close())
    """
    labels = [r[2].get("matched_route_name", f"Route {i+1}")[:32]
              for i, r in enumerate(results)]
    scores = [r[0] for r in results]
    n = len(scores)

    fig, ax = _hires_fig(figsize=(7, max(2.0, n * 0.65)), dpi=180)
    colors  = [PALETTE[i % len(PALETTE)] for i in range(n)]
    bars    = ax.barh(np.arange(n), scores, color=colors, height=0.52, edgecolor="none")

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
    Render a bar chart of reported step yield for each step in a route.

    Used in the Dataset Explorer route detail view. Steps without a reported
    yield are absent from the chart (no bar drawn — no implied 0 %).
    Yield labels use integer format; change ":.0f" to ":.1f" for one decimal.

    Parameters
    ──────────
    steps_route : list of step dicts (from dataset["by_route"][route_id])

    Returns
    ───────
    matplotlib.figure.Figure  (caller is responsible for plt.close())
    """
    n          = len(steps_route)
    step_nums  = list(range(1, n + 1))
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
    Render a grouped horizontal bar chart comparing raw criterion scores across
    the routes selected in the Analysis tab.

    Each route is plotted in a distinct colour from PALETTE; each row of the
    chart corresponds to one criterion. Score values are annotated at the end
    of each bar.

    Parameters
    ──────────
    sel_results : list of (score, details, route) tuples (filtered by user selection)
    criteria    : list of 3 criterion keys

    Returns
    ───────
    matplotlib.figure.Figure  (caller is responsible for plt.close())
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
        vals  = [details[c].get("raw", 0) or 0 for c in criteria]
        color = PALETTE[i % len(PALETTE)]
        name  = route_names[i]
        bars  = ax.barh(
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

def build_why_ranked_html(rank: int, score_total: float, details: dict,
                           criteria: list, weights: dict, steps_data: list) -> str:
    """
    Build the 'Why is this route ranked #N?' reasoning box.
    Shows one bullet per top 2 criteria using relevant values, no raw scores.
    """
    bn = fi.bottleneck_yield(steps_data)
    av = fi.average_yield(steps_data)
    n  = len(steps_data)

    bullets = []

    for crit in criteria[:2]:  # only top 2 criteria
        if crit == "steps":
            bullets.append(T["why_steps"].format(n=n))
        elif crit == "yield":
            if bn is not None:
                bullets.append(T["why_yield"].format(y=f"{bn:.1f}"))
            elif av is not None:
                bullets.append(T["why_avg_yield"].format(a=f"{av:.1f}"))
        elif crit == "atom_economy":
            raw = details[crit].get("raw") or 0
            bullets.append(f"it achieves an atom economy of <strong>{raw*100:.0f}%</strong>")
        elif crit == "e_factor":
            raw = details[crit].get("raw") or 0
            bullets.append(f"it has a strong E-factor score of <strong>{raw:.2f}</strong>")
        elif crit == "toxicity":
            raw = details[crit].get("raw") or 0
            bullets.append(f"it has a safety score of <strong>{raw*100:.0f}%</strong>")

    items_html = "".join(f"<li>{b}</li>" for b in bullets)

    return f"""{COMPONENT_STYLE}
<div class="why-box">
  <div>{T["why_prefix"].format(r=rank)}</div>
  <ul>{items_html}</ul>
  <em>{T["why_suffix"]}</em>
</div>"""

# =============================================================================
# Route card display
# =============================================================================

def display_route_card(score_total, details, route, criteria, weights,
                       all_results, rank=1, badge="📚 dataset", lang="en") -> None:
    """
    Render a complete route card inside a Streamlit expander.

    Contents (top to bottom):
    1. Validation status banner (info / success / warning depending on status).
    2. Four metric columns: total score, step count, bottleneck yield, avg yield.
    3. Score breakdown table (build_score_table_html).
    4. PDF download button (build_route_report_pdf).
    5. Horizontal rule followed by the interactive reaction scheme
       (build_clickable_scheme_html inside components.html).
    6. Substances-needed expander with molecule images, solvents, reagents.

    The expander is expanded by default only for the first (rank 1) route.

    Parameters
    ──────────
    score_total : float
    details     : dict   per-criterion detail dicts from rank_weighted()
    route       : dict   enriched route dict
    criteria    : list   3 criterion keys
    weights     : dict   {criterion_key: weight}
    all_results : list   full results list for the category (unused currently)
    rank        : int    1-based rank within the category (1 = best)
    badge       : str    emoji + label shown in the expander header
    lang        : str    "en" or "fr"
    """
    route_name = route.get("matched_route_name", "?")
    tgt        = route.get("matched_target", "?")
    steps_data = route.get("dataset_steps", [])
    n_steps    = len(steps_data)
    bn         = fi.bottleneck_yield(steps_data)
    av         = fi.average_yield(steps_data)
    medals     = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
    medal      = medals[rank - 1] if rank <= 10 else f"#{rank}"
    route_key  = "".join(c for c in route.get("matched_route_id", "r") if c.isalnum())
    is_pred    = route.get("is_predicted", False)
    status     = route.get("validation_status", "dataset")

    with st.expander(
        f"{medal}  [{badge}]  {route_name}  ·  {tgt}  ·  Score: **{score_total:.3f}**",
        expanded=(rank == 1),
    ):
        # Validation status banner
        if status == "predicted":
            st.warning("⚠️ Predicted route — not experimentally validated. Yield excluded from scoring.")
        elif status == "partial":
            v = route.get("validated_steps_count", 0); t = route.get("total_steps_count", 0)
            st.info(f"⚡ Partial validation — {v}/{t} steps found in generic dataset (real conditions).")
        elif status == "validated":
            st.success("✅ Fully validated — all steps found in generic dataset (real conditions).")

        # Metric columns
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(T["metric_score"],     f"{score_total:.3f}", help=T["metric_score_help"])
        m2.metric(T["metric_steps"],     n_steps)
        m3.metric(T["metric_bottleneck"],
                  f"{bn:.1f}%" if bn is not None else "—", help=T["metric_bn_help"])
        m4.metric(T["metric_avg"],       f"{av:.1f}%" if av is not None else "—")

        # Score breakdown table
        st.markdown(f"**{T['contrib_title']}**")
        st.html(build_score_table_html(details, criteria, weights))

        # Why ranked box  
        st.markdown(f"**{T['why_best_title'].format(r=rank)}**")
        st.html(build_why_ranked_html(rank, score_total, details, criteria, weights, steps_data))   

        # PDF download button
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
            height=480,
            scrolling=True,
        )

        st.markdown("---")
        with st.expander("🧪 Substances needed", expanded=False):
            sub = fi.get_substances_list(steps_data)
            all_mols = sub["to_buy"] + sub["to_prepare"][:6]
            if all_mols:
                n_cols   = min(4, len(all_mols))
                cols_sub = st.columns(n_cols)
                for i, smi in enumerate(all_mols):
                    with cols_sub[i % n_cols]:
                        # Double-resolution PNG for substance images
                        png = mol_png(smi, 320, 220)
                        label = "🛒" if smi in sub["to_buy"] else "⚗️"
                        if png:
                            st.image(png, caption=label + " " + smi[:20], width="content")
            if sub["solvents"]:
                st.markdown("**🧴 Solvents**")
                st.write("  ·  ".join(sub["solvents"]))
            if sub["reagents"]:
                st.markdown("**🔬 Reagents**")
                st.write("  ·  ".join(sub["reagents"][:10]))


# =============================================================================
# Sidebar — file paths and search parameters
# =============================================================================

with st.sidebar:
    st.title(T["sidebar_title"])
    if not MODULE_OK:
        st.error(f"{T['mod_err']}\n\n`{MODULE_ERR}`")
        st.stop()

    st.divider()
    st.subheader(T["files_section"])

    dataset_path = st.text_input(
        T["ds_label"],
        value="../../data/reaction_dataset.json",
        help=(
            "Path to your main curated reaction dataset (JSON). "
            "Format: list of reaction dicts with fields: id, route_id, route_name, "
            "target, step_number, reactants_smiles, product_smiles, conditions, "
            "yield_percent, reaction_type."
        ),
    )
    toxicity_path = st.text_input(
        T["tox_label"] + " *",
        value="../../data/toxicity_dataset.json",
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
        value="../../data/config.yml",
        help=(
            "Path to the AiZynthFinder YAML config file. "
            "Specifies the policy network, filter, and stock files. "
            "Required — AiZynthFinder will not run without it."
        ),
    )
    rxninsight_db_path = st.text_input(
        T["rxni_label"],
        value="../../data/uspto_rxn_insight.gzip",
        help=(
            "Path to the Rxn-INSIGHT USPTO reaction database (gzip). "
            "Optional — only needed if 'Include predicted routes' is enabled. "
            "Enables reaction classification and condition prediction for novel routes."
        ),
    )
    generic_dataset_path = st.text_input(
        T["generic_label"],
        value="../../data/generic_reactions.json",
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

    # Predicted-route toggle — only shown when rxn_insight is installed
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
    # Criterion weight progress bars — live preview of the 1/i² weights
    if "criteria" in st.session_state:
        with st.expander(T["weights_exp"]):
            crit_w = fi.compute_weights(st.session_state["criteria"])
            for c in st.session_state["criteria"]:
                w = crit_w[c]
                st.progress(float(w), text=f"{CL[c]} — {w * 100:.1f}%")


# =============================================================================
# Main page layout
# =============================================================================

st.title(T["page_title"])
st.caption(T["page_caption"])

tab_search, tab_analysis, tab_dataset, tab_help = st.tabs([
    T["tab_search"], T["tab_analysis"], T["tab_dataset"], T["tab_help"]
])


# =============================================================================
# Cached dataset loaders
# =============================================================================

@st.cache_data(show_spinner=False)
def load_dataset_cached(path: str) -> dict:
    """
    Streamlit-cached wrapper around fi.load_reaction_dataset().

    Prevents reloading the JSON dataset on every re-run during the same session.
    The cache is keyed by file path so changing the path in the sidebar triggers
    a reload automatically.

    Parameters
    ──────────
    path : str  path to reaction_dataset.json

    Returns
    ───────
    dict  as returned by fi.load_reaction_dataset()
    """
    return fi.load_reaction_dataset(path)


@st.cache_data(show_spinner=False)
def get_targets_cached(path: str) -> dict:
    """
    Streamlit-cached wrapper that returns the {target_name: SMILES} mapping.

    Used to populate the predefined molecule selector in the Route Search tab.
    Caching avoids scanning all route steps on every widget interaction.

    Parameters
    ──────────
    path : str  path to reaction_dataset.json

    Returns
    ───────
    dict  {target_name: canonical_SMILES}
    """
    return fi.get_targets_from_dataset(load_dataset_cached(path))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ROUTE SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:
    top_left, top_right = st.columns([3, 2])

    with top_left:
        st.subheader(T["target_section"])
        mode = st.radio(
            "Mode",
            [T["mode_pre"], T["mode_custom"]],
            horizontal=True,
            label_visibility="collapsed",
        )

        if mode == T["mode_pre"]:
            # Load predefined targets from the dataset file
            if os.path.exists(dataset_path):
                try:
                    targets_ds = get_targets_cached(dataset_path)
                except Exception:
                    targets_ds = {}
            else:
                targets_ds = {}

            if targets_ds:
                target_name   = st.selectbox(T["mol_label"], list(targets_ds.keys()),
                                             format_func=lambda x: x.capitalize())
                target_smiles = targets_ds[target_name]
            else:
                # Fallback when the dataset file is unavailable
                target_name   = "Galanthamine"
                target_smiles = "OC1C=C[C@@]23c4cc(OC)ccc4CN(C)C[C@@H]2[C@@H]1O3"
        else:
            # Custom SMILES mode — validate on the fly
            target_smiles = st.text_input(T["smiles_label"], placeholder=T["smiles_ph"])
            target_name   = "Custom"
            if target_smiles:
                if Chem.MolFromSmiles(target_smiles) is None:
                    st.error(T["smiles_invalid"])
                    target_smiles = ""
                else:
                    st.success(T["smiles_valid"])

        st.subheader(T["criteria_section"])
        st.caption(T["criteria_caption"])
        all_crit = list(CL.keys())
        c1   = st.selectbox(T["c1_label"], all_crit, format_func=lambda x: CL[x])
        rem2 = [c for c in all_crit if c != c1]
        c2   = st.selectbox(T["c2_label"], rem2, format_func=lambda x: CL[x])
        rem3 = [c for c in rem2 if c != c2]
        c3   = st.selectbox(T["c3_label"], rem3, format_func=lambda x: CL[x])
        criteria = [c1, c2, c3]
        st.session_state["criteria"] = criteria

        run_search = st.button(T["run_btn"], type="primary", disabled=not target_smiles)

    with top_right:
        # Show target molecule preview or welcome message
        if target_smiles:
            # Double-resolution preview for the target molecule
            png = mol_png(target_smiles, 640, 440)
            if png:
                st.image(png, caption=target_name, width="stretch")
        else:
            st.markdown(T["welcome"])
            st.markdown(T["avail_mols"])
            try:
                prev = get_targets_cached(dataset_path)
                cols = st.columns(3)
                for idx, (name, smi) in enumerate(prev.items()):
                    with cols[idx % 3]:
                        # Double-resolution thumbnails for the molecule gallery
                        png = mol_png(smi, 400, 270)
                        if png:
                            st.image(png, caption=name.capitalize(), width="stretch")
            except Exception:
                pass

    st.divider()

    # -- Run search and store results in session_state ------------------------
    if run_search and target_smiles:
        errs = []
        if not os.path.exists(dataset_path): errs.append(f"`{dataset_path}`")
        if not os.path.exists(config_path):  errs.append(f"`{config_path}`")
        for e in errs:
            st.error(f"{T['err_file']}: {e}")
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
                        target_name          = target_name if mode == T["mode_pre"] else "",
                        include_predicted    = include_predicted,
                        rxninsight_db_path   = rxninsight_db_path,
                        generic_dataset_path = generic_dataset_path,
                        n_aiz_routes         = n_aiz,
                    )
                    tox_index = fi.load_toxicity_dataset(toxicity_path)
                    weights   = fi.compute_weights(criteria)
                    st.session_state.update({
                        "results":       results_raw,
                        "weights":       weights,
                        "tox_index":     tox_index,
                        "target_name":   target_name,
                        "target_smiles": target_smiles,
                        "criteria":      criteria,
                    })
                    status.update(label=T["search_ok"], state="complete", expanded=False)
                except FileNotFoundError as e:
                    status.update(label=T["err_file"],  state="error"); st.error(str(e))
                except ValueError as e:
                    status.update(label=T["err_param"], state="error"); st.error(str(e))
                except Exception as e:
                    status.update(label=T["err_other"], state="error"); st.exception(e)

    # -- Display results from session_state -----------------------------------
    results_raw = st.session_state.get("results",  None)
    weights     = st.session_state.get("weights",  {})
    criteria    = st.session_state.get("criteria", criteria)
    tox_index   = st.session_state.get("tox_index", {})

    if results_raw is None:
        pass  # No search run yet — show nothing below the divider
    elif not isinstance(results_raw, dict):
        st.warning(T["no_routes"])
    else:
        scored_dataset   = results_raw.get("dataset",   [])
        scored_validated = results_raw.get("validated", [])
        scored_predicted = results_raw.get("predicted", [])

        if not scored_dataset and not scored_validated and not scored_predicted:
            st.warning(T["no_routes"])
        else:
            # Summary counters row
            c1_, c2_, c3_, c4_ = st.columns(4)
            c1_.metric("📚 Dataset",   len(scored_dataset))
            c2_.metric("✅ Validated",  len(scored_validated))
            c3_.metric("🔮 Predicted", len(scored_predicted))
            c4_.metric("Total", len(scored_dataset) + len(scored_validated) + len(scored_predicted))

            tgt_name = st.session_state.get("target_name", target_name)

            # -- Dataset section ----------------------------------------------
            st.markdown(T["sec_dataset"])
            st.caption(T["cap_dataset"])
            if not scored_dataset:
                st.info("No dataset routes for this target.")
            else:
                if len(scored_dataset) > 1:
                    fig = make_ranking_chart(scored_dataset, tgt_name)
                    st.pyplot(fig); plt.close(fig)
                st.success(T["n_found"].format(n=len(scored_dataset)))
                st.markdown("---")
                for rank, (score_total, details, route) in enumerate(scored_dataset, 1):
                    display_route_card(score_total, details, route, criteria, weights,
                                       scored_dataset, rank, T["badge_dataset"], lang)

            # -- Validated section --------------------------------------------
            if scored_validated:
                st.markdown("---")
                st.markdown(T["sec_validated"])
                st.caption(T["cap_validated"])
                if len(scored_validated) > 1:
                    fig = make_ranking_chart(scored_validated, tgt_name)
                    st.pyplot(fig); plt.close(fig)
                for rank, (score_total, details, route) in enumerate(scored_validated, 1):
                    status = route.get("validation_status", "partial")
                    badge  = T["badge_validated"] if status == "validated" else T["badge_partial"]
                    display_route_card(score_total, details, route, criteria, weights,
                                       scored_validated, rank, badge, lang)

            # -- Predicted section --------------------------------------------
            if include_predicted and RXNINSIGHT_OK and scored_predicted:
                st.markdown("---")
                st.markdown(T["sec_predicted"])
                st.caption(T["cap_predicted"])

                # Starting-material search filter for predicted routes
                search_sm = st.text_input(
                    "Search by starting material SMILES",
                    placeholder="e.g. c1ccc2[nH]ccc2c1",
                    key="sm_search",
                )
                # Build a map of {canonical_SM_smiles: [route_names]} for the browser
                all_sm_map = {}
                for _, _, route in scored_predicted:
                    steps    = route.get("dataset_steps", [])
                    all_prod = {fi.to_canonical(s.get("product_smiles", "")) for s in steps}
                    for s in steps:
                        for rsmi in s.get("reactants_smiles", []):
                            canon = fi.to_canonical(rsmi)
                            if canon and canon not in all_prod:
                                all_sm_map.setdefault(canon, []).append(
                                    route.get("matched_route_name", "?"))

                # Filter predicted routes by starting material SMILES if a search string is given
                filtered = scored_predicted
                if search_sm.strip():
                    sc       = fi.to_canonical(search_sm.strip())
                    filtered = [
                        (s, d, r) for s, d, r in scored_predicted
                        if sc in {fi.to_canonical(rsmi)
                                  for step in r.get("dataset_steps", [])
                                  for rsmi in step.get("reactants_smiles", [])}
                        or any(
                            search_sm.lower() in rsmi.lower()
                            for step in r.get("dataset_steps", [])
                            for rsmi in step.get("reactants_smiles", [])
                        )
                    ]
                    if filtered:
                        st.success(f"{len(filtered)} route(s) found")
                    else:
                        st.warning("No routes with this starting material")

                # Collapsible starting-material browser
                top30 = list(all_sm_map.keys())[:30]
                with st.expander(f"📋 Browse starting materials ({len(top30)} unique)", expanded=False):
                    cols_sm = st.columns(3)
                    for i, smi in enumerate(top30):
                        col = cols_sm[i % 3]
                        # Double-resolution starting material thumbnails
                        png = mol_png(smi, 400, 270)
                        if png:
                            col.image(png, width=140)
                        col.code(smi, language=None)
                        col.caption(f"In: {', '.join(set(all_sm_map[smi]))[:50]}")

                st.markdown("---")
                for rank, (score_total, details, route) in enumerate(filtered, 1):
                    display_route_card(score_total, details, route, criteria, weights,
                                       filtered, rank, T["badge_predicted"], lang)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    import pandas as pd

    results_raw = st.session_state.get("results",  None)
    criteria    = st.session_state.get("criteria", list(CL.keys())[:3])
    weights     = st.session_state.get("weights",  {})
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
                # Return a single emoji indicating validation status
                status = r[2].get("validation_status", "dataset")
                if status == "validated": return "✅"
                if status == "partial":   return "⚡"
                if status == "predicted": return "🔮"
                return "📚"

            # Build option labels for the multiselect
            route_opts = {
                f"{_route_badge_label(r)} {r[2].get('matched_route_name','?')} "
                f"(score {r[0]:.4f})": i
                for i, r in enumerate(all_sc)
            }
            labels = list(route_opts.keys())
            sel    = st.multiselect(
                T["sel_routes"], labels, default=labels[:min(3, len(labels))])

            if sel:
                sel_results = [all_sc[route_opts[s]] for s in sel]

                # Build comparison dataframe
                rows = []
                for score, details, route in sel_results:
                    sd  = route.get("dataset_steps", [])
                    bn_ = fi.bottleneck_yield(sd)
                    av_ = fi.average_yield(sd)
                    yc  = fi.cumulative_yield(sd)
                    row = {
                        "Route":                  route.get("matched_route_name", "?")[:26],
                        T["metric_score"]:        f"{score:.3f}",
                        T["metric_steps"]:        len(sd),
                        "Cumul. yield":           f"{yc * 100:.4f}%",
                        T["metric_bottleneck"]:   f"{bn_:.1f}%" if bn_ else "—",
                        T["metric_avg"]:          f"{av_:.1f}%" if av_ else "—",
                    }
                    # Add raw scores for the selected criteria
                    for c in criteria:
                        if c in ("steps", "yield"):
                            continue
                        raw = details[c].get("raw")
                        row[CL[c]] = f"{raw:.4f}" if raw is not None else "N/A"
                    # Add all other criteria as informational columns (marked ✗)
                    all_s = fi.compute_all_scores(route, tox_index)
                    for c in fi.CRITERIA_REGISTRY:
                        if c not in criteria and c not in ("steps", "yield"):
                            row[CL[c] + " ✗"] = f"{all_s[c]:.4f}"
                    rows.append(row)

                df = pd.DataFrame(rows).set_index("Route")
                st.dataframe(df, width="stretch")
                st.caption("✓ = selected criteria  ✗ = additional criteria (not used in ranking)")

                # Criterion profile chart for ≥ 2 routes
                if len(sel_results) >= 2:
                    st.markdown(f"**{T['radar_title']}**")
                    fig_cmp = make_comparison_chart(sel_results, criteria)
                    st.pyplot(fig_cmp); plt.close(fig_cmp)

                # Side-by-side reaction schemes for exactly 2 routes
                if len(sel_results) == 2:
                    st.markdown("---")
                    st.markdown("### Reaction schemes")
                    cA, cB = st.columns(2)
                    for col, (score, details, route), lbl in zip([cA, cB], sel_results, sel[:2]):
                        with col:
                            st.markdown(f"**{route.get('matched_route_name','?')}**")
                            rk = "".join(c for c in route.get("matched_route_id", "r") if c.isalnum())
                            components.html(
                                build_clickable_scheme_html(
                                    route.get("dataset_steps", []),
                                    rk,
                                    route.get("is_predicted", False),
                                ),
                                height=480,
                                scrolling=True,
                            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def _smiles_copy_widget(smiles: str, label: str = "") -> None:
    """
    Render a compact inline SMILES display widget with a clipboard Copy button.

    Displayed beneath each reactant and product molecule image in the Dataset
    Explorer step-by-step view, allowing the chemist to copy any SMILES string
    without manually reading the full text.

    Parameters
    ──────────
    smiles : str  SMILES string to display and make copyable
    label  : str  optional label (currently unused — reserved for future use)
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
        with st.spinner("Loading…"):
            ds = load_dataset_cached(dataset_path)
        reactions_all = ds["all"]
        by_route      = ds["by_route"]
        targets_uniq  = sorted(set(r.get("target", "?") for r in reactions_all))

        # Summary metrics row
        c1d, c2d, c3d = st.columns(3)
        c1d.metric(T["total_rxn"],   len(reactions_all))
        c2d.metric(T["dist_routes"], len(by_route))
        c3d.metric(T["tgt_mols"],    len(targets_uniq))
        st.markdown("---")

        filter_tgt = st.selectbox(T["filter_lbl"], [T["filter_all"]] + targets_uniq)

        # Route summary table — sorted by cumulative yield descending
        rows = []
        for rid, steps in sorted(by_route.items()):
            tgt = steps[0].get("target", "?")
            if filter_tgt != T["filter_all"] and tgt != filter_tgt:
                continue
            nm  = steps[0].get("route_name", rid)
            n   = len(steps)
            ys  = [s.get("yield_percent") for s in steps]
            yo  = [y for y in ys if y is not None]
            yc  = fi.cumulative_yield(steps)
            rows.append({
                T["col_route"]:    nm,
                T["col_target"]:   tgt,
                T["col_steps"]:    n,
                "_yc_raw":         yc,
                T["col_cumyield"]: f"{yc * 100:.2f}%",
                T["col_missing"]:  sum(1 for y in ys if y is None),
                T["col_avg"]:      f"{sum(yo)/len(yo):.1f}%" if yo else "—",
            })

        if rows:
            df_routes = pd.DataFrame(rows)
            df_routes = (
                df_routes
                .sort_values("_yc_raw", ascending=False)
                .drop(columns=["_yc_raw"])
                .reset_index(drop=True)
            )
            st.dataframe(df_routes, width="stretch", hide_index=True)

        st.markdown("---")

        # Route detail selector
        routes_avail = [
            rid for rid, steps in by_route.items()
            if filter_tgt == T["filter_all"]
            or steps[0].get("target", "?") == filter_tgt
        ]
        rc = st.selectbox(
            T["detail_sel"],
            routes_avail,
            format_func=lambda x: (
                f"{by_route[x][0].get('route_name', x)} ({by_route[x][0].get('target','?')})"
            ),
        )

        if rc:
            sr    = by_route[rc]
            n_sr  = len(sr)
            st.markdown(
                f"**{sr[0].get('route_name')}** — {n_sr} {strip_emoji(T['col_steps']).lower()}"
            )
            fig3 = make_yield_chart(sr)
            st.pyplot(fig3); plt.close(fig3)

            st.markdown("### Step-by-step details")

            for step in sr:
                snum     = step.get("step_number", "?")
                rtype    = step.get("reaction_type", "—") or "—"
                yld      = step.get("yield_percent")
                cond     = step.get("conditions", {})
                prod     = step.get("product_smiles", "")
                reac     = step.get("reactants_smiles", [])
                cond_str = fi.fmt_conditions(cond)
                is_purif = _is_purification_step(step)

                # Override reaction type label for unlabelled purification steps
                if is_purif and (rtype == "—" or not rtype):
                    rtype = "Purification / Isolation"

                # Purification steps use a brown header; standard steps use navy gradient
                header_style = (
                    "background:#6B3E26;"
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

                n_reac      = max(len(reac), 1)
                col_weights = [3] * n_reac + [1, 3]
                cols_rxn    = st.columns(col_weights)

                # Reactant columns
                for i, rsmi in enumerate(reac):
                    with cols_rxn[i]:
                        # Double-resolution reactant images in the step-by-step view
                        png_r = mol_png(rsmi, 480, 330)
                        if png_r:
                            st.image(png_r, use_container_width=True)
                        else:
                            st.code(rsmi, language=None)
                        _smiles_copy_widget(rsmi)

                # Arrow column (SVG-like HTML arrow)
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

                # Product column
                with cols_rxn[n_reac + 1]:
                    # Double-resolution product image
                    png_p = mol_png(prod, 480, 330)
                    if png_p:
                        st.image(png_p, use_container_width=True)
                    else:
                        st.code(prod, language=None)
                    _smiles_copy_widget(prod, "Product")

                # Step footer: yield and conditions
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
        p.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
         .replace("*", "<em>", 1).replace("*", "</em>", 1)
        for p in info_parts
    )}
</div>
""",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown("**Full reaction scheme** *(click an arrow for extra details)*")
            route_ds_key = "ds_" + "".join(c for c in rc if c.isalnum())
            # Convert dataset step dicts to the scheme-compatible format
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
            # Adjust scheme container height based on step count
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
    if lang == "en":
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
        """)

st.markdown("---")
st.caption(T["footer"])
