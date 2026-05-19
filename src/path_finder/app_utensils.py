# app_utensils.py
# UI helper functions for the Path Finder app.
# All rendering, chart creation, HTML builders, and cached loaders live here
# so that app.py stays readable. Imported directly by app.py.

import re
import json
import base64
import io
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from localization import LANG, CRITERIA_LABELS, PALETTE, FIG_BG
from molecule_rendering import (
    mol_png,
    mol_b64_or_text_svg  as _mol_b64_or_text_svg,
    fallback_data_uri    as _fallback_data_uri,
    is_trivial_smiles    as _is_trivial_smiles,
    MODULE_OK,
)
from report_builder import build_route_report_pdf

try:
    import route_engine as rt
    from rdkit import Chem
except Exception:
    rt   = None
    Chem = None



# CSS injected inside isolated st.html() / components.html() frames.
# Scoped to avoid leaking into the Streamlit global stylesheet.

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
    border-left:4px solid #2d5986; border-radius:0 10px 10px 0;
    padding:14px 18px; margin:10px 0 18px 0;
    font-family:'DM Sans',sans-serif; font-size:0.89rem;
    color:#1a2e44; line-height:1.65;
  }
  .why-box strong { color:#1a2e44; font-weight:700; }
  .why-box ul     { margin:6px 0 6px 20px; padding:0; }
  .why-box li     { margin-bottom:3px; }
  .why-box em     { font-size:0.82rem; color:#6b7a8d; }
  .src-badge { display:inline-block; border-radius:10px; padding:1px 8px;
               font-size:0.7rem; font-weight:600; font-family:'DM Sans',sans-serif;
               margin-left:6px; }
  .src-dataset    { background:#d4edda; color:#155724; }
  .src-generic    { background:#cce5ff; color:#004085; }
  .src-rxninsight { background:#fff3cd; color:#856404; }
  @keyframes fadeSlideIn {
    from { opacity:0; transform:translateX(-14px); }
    to   { opacity:1; transform:translateX(0); }
  }
  @keyframes arrowPulse { 0%,100%{opacity:0.55;} 50%{opacity:1;} }
</style>
"""

# Per-criterion formula tooltip text shown in the score table header.
CRITERIA_SCORE_DESC = {
    "en": {
        "steps":
            "Score = 1 / number_of_steps. Fewer steps gives a score closer to 1.",
        "yield":
            "Score = cumulative yield (product of all step yields). "
            "Missing yields are treated as 100 % (neutral). "
            "Score closer to 1 means a higher overall yield.",
        "atom_economy":
            "Score = fraction of reactant atoms incorporated into the product. "
            "Score closer to 1 means less atomic waste.",
        "e_factor":
            "Score = 1 / (1 + E-factor).  E-factor = kg waste per kg product. "
            "Score closer to 1 means less waste generated.",
        "toxicity":
            "Score = 1 - (average hazard of reagents/solvents). "
            "Score closer to 1 means safer, less toxic conditions.",
    },
    "fr": {
        "steps":
            "Score = 1 / nombre_d'étapes. Moins d'étapes → score proche de 1.",
        "yield":
            "Score = rendement cumulé (produit de tous les rendements). "
            "Rendements manquants traités à 100 % (neutres). "
            "Score proche de 1 = meilleur rendement global.",
        "atom_economy":
            "Score = fraction des atomes des réactifs retrouvés dans le produit. "
            "Score proche de 1 = moins de déchets atomiques.",
        "e_factor":
            "Score = 1 / (1 + E-factor).  E-factor = kg déchets / kg produit. "
            "Score proche de 1 = moins de déchets.",
        "toxicity":
            "Score = 1 − (danger moyen des réactifs/solvants). "
            "Score proche de 1 = conditions plus sûres.",
    },
}

# Emoji regex — compiled once at module level
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

def load_banner(path: str) -> str:
    """
    Load a local image file and return a base-64 data URI for HTML embedding.

    Parameters
    ----------
    path : str
        Filesystem path to the image (PNG, JPG, SVG, …).

    Returns
    -------
    str
        A ``data:<mime>;base64,<data>`` URI, or an empty string if the file
        does not exist.
    """
    p = Path(path)
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode()
    ext  = p.suffix.lstrip(".").lower()
    mime = "image/png" if ext == "png" else f"image/{ext}"
    return f"data:{mime};base64,{data}"

def hires_fig(*args, dpi: int = 180, **kwargs):
    """
    Create a high-resolution matplotlib figure with the app background colour.

    Drop-in replacement for ``plt.subplots()``. Sets the figure and all axes
    facecolour to ``FIG_BG`` so charts blend with the page without a white box.

    Parameters
    ----------
    args : positional
        Forwarded to ``plt.subplots()``.
    dpi : int, optional
        Dots per inch for the figure (default 180).
    kwargs : keyword
        Forwarded to ``plt.subplots()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes or ndarray of Axes
    """
    fig, ax = plt.subplots(*args, dpi=dpi, **kwargs)
    fig.patch.set_facecolor(FIG_BG)
    if hasattr(ax, "__iter__"):
        for a in ax:
            a.set_facecolor(FIG_BG)
    else:
        ax.set_facecolor(FIG_BG)
    return fig, ax

def strip_emoji(text: str) -> str:
    """
    Remove all Unicode emoji characters from a string.

    Matplotlib can't render emoji so we strip them before any chart call.

    Parameters
    ----------
    text : str
        Input string, possibly containing emoji.

    Returns
    -------
    str
        *text* with all emoji removed and leading/trailing whitespace stripped.
    """
    return _EMOJI_RE.sub("", text).strip()

def is_purification_step(step: dict) -> bool:
    """True if step is a purification/isolation (keyword or identity transform)."""
    rtype = (step.get("reaction_type") or "").lower()
    if any(kw in rtype for kw in ("purif", "recryst", "chroma", "isolation", "workup")):
        return True
    prod  = step.get("product_smiles", "")
    reacs = step.get("reactants_smiles", [])
    if prod and prod in reacs:
        return True
    if MODULE_OK and prod and Chem is not None:
        mol_p      = Chem.MolFromSmiles(prod)
        canon_prod = Chem.MolToSmiles(mol_p) if mol_p else prod
        for r in reacs:
            mol_r = Chem.MolFromSmiles(r) if r else None
            if mol_r and Chem.MolToSmiles(mol_r) == canon_prod:
                return True
    return False

def build_clickable_scheme_html(
    steps_data: list,
    route_id: str,
    is_predicted: bool = False,
) -> str: # layout sizes — tweak these to resize without touching the HTML / MOL_W, MOL_H = 158, 112
    """
    Build the interactive HTML reaction scheme used in Route Search and Analysis.

    Each molecule in the synthetic sequence is rendered as a double-resolution
    Cairo PNG image.  Clicking any reaction arrow opens an inline detail panel
    showing conditions, yield, reactant/product structures, and copyable SMILES.
    Clicking the same arrow again collapses the panel.

    Purification step handling
    --------------------------
    When ``is_purification_step()`` returns ``True`` for a step, the product
    molecule is always added to the display sequence — even if its SMILES is
    identical to the preceding molecule (standard steps skip duplicates).
    The arrow is drawn with a dashed brown style and labelled
    "Purification / Isolation".

    Co-reactant rendering
    ---------------------
    Reactants that are not the substrate are rendered as small structure images
    above the arrow.  Trivial species (single atoms, ions) are shown as text
    labels below the arrow instead.

    Parameters
    ----------
    steps_data : list
        ``route["dataset_steps"]`` — list of step dicts.
    route_id : str
        Unique string identifier used as a prefix for all DOM element IDs.
    is_predicted : bool, optional
        If ``True``, arrows and detail panels use orange colouring to
        distinguish predicted routes from validated ones (default ``False``).

    Returns
    -------
    str
        A complete, self-contained HTML document ready for
        ``components.html()``.
    """
    if not steps_data:
        return "<p style='color:#888'>No steps.</p>"

    arrow_color = "#E65100" if is_predicted else "#1a2e44"
    hover_bg    = "#FFF3E0" if is_predicted else "#E8F0FC"
    panel_bg    = "#FFF8F0" if is_predicted else "#F0F7FF"
    rid_js      = "".join(c for c in route_id if c.isalnum())

    #  Layout constants 
    MOL_W, MOL_H = 158, 112
    CO_W,  CO_H  = 96,  70
    CELL_W       = MOL_W + 20
    SCHEME_H     = 200
    ARROW_SHAFT_W = 90
    ARROW_CELL_W  = 145

    # Compute padding above the band for co-reactant images
    max_co_rows = 1
    for step in steps_data:
        reactants = step.get("reactants_smiles", [])
        product   = step.get("product_smiles", "")
        cond      = step.get("conditions", {})
        co_count  = sum(
            1 for r in reactants
            if r != product and not _is_trivial_smiles(r)
        )
        if cond.get("solvent") and not _is_trivial_smiles(cond["solvent"]):
            co_count += 1
        co_count += sum(
            1 for r in (cond.get("reagents") or [])
            if r and not _is_trivial_smiles(r)
        )
        max_co_rows = max(max_co_rows, max(1, (co_count + 2) // 3))

    display_rows = min(max_co_rows, 2)
    PAD_TOP      = display_rows * (CO_H + 2) + 50
    PAD_BOTTOM   = display_rows * (CO_H + 2) // 3 + 10

    #  Build molecule sequence and per-arrow metadata 
    mol_sequence = []
    arrow_data   = []

    for i, step in enumerate(steps_data):
        reactants  = step.get("reactants_smiles", [])
        product    = step.get("product_smiles", "")
        cond       = step.get("conditions", {})
        rtype      = step.get("reaction_type", "") or ""
        yld        = step.get("yield_percent")
        snum       = step.get("step_number", i + 1)
        src_step   = step.get("source", "dataset")
        is_purif   = is_purification_step(step)

        if i == 0 and reactants and reactants[0] not in mol_sequence:
            mol_sequence.append(reactants[0])

        if product:
            if mol_sequence and mol_sequence[-1] == product and not is_purif:
                pass
            else:
                mol_sequence.append(product)

        prev    = mol_sequence[-2] if len(mol_sequence) >= 2 else ""
        co_draw = []
        co_text = []
        for r in reactants:
            if r in (prev, product):
                continue
            (co_text if _is_trivial_smiles(r) else co_draw).append(r)

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
        for r in (cond.get("reagents") or []):
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
        cond_display  = "  ·  ".join(all_cond_parts)
        display_rtype = rtype if not (is_purif and not rtype) else "Purification / Isolation"
        below_parts   = []
        if yld is not None and src_step != "rxn-insight":
            below_parts.append(str(yld) + "%")
        for t in co_text[:3]:
            below_parts.append(t)
        co_draw_set = set(co_draw)
        for t in all_cond_parts[:4]:
            if t not in below_parts and t not in co_draw_set:
                below_parts.append(t)

        arrow_data.append({
            "step": snum, "rtype": display_rtype,
            "yld":  str(yld) + "%" if (yld is not None and src_step != "rxn-insight") else "",
            "cond": cond_display, "co_draw": co_draw, "co_text": co_text,
            "below": below_parts, "reactants": reactants, "product": product,
            "source": src_step, "fg": step.get("fg_reactants", []),
            "is_purif": is_purif,
        })

    #  Pre-render molecule images 
    mol_imgs   = {s: _mol_b64_or_text_svg(s, MOL_W, MOL_H) for s in mol_sequence if s}
    co_imgs    = {}
    step_imgs  = {}
    for a in arrow_data:
        for smi in a["co_draw"]:
            if smi and smi not in co_imgs:
                co_imgs[smi] = _mol_b64_or_text_svg(smi, CO_W, CO_H)
    for a in arrow_data:
        r_b64s = [_mol_b64_or_text_svg(r, 110, 78) for r in a["reactants"][:4]]
        p_b64  = _mol_b64_or_text_svg(a["product"], 110, 78)
        step_imgs[str(a["step"])] = {
            "reactants":       r_b64s, "product":         p_b64,
            "reactant_smiles": a["reactants"], "product_smiles": a["product"],
        }
    step_imgs_json = json.dumps(step_imgs)

    #  Assemble HTML items 
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
            '<img src="' + uri + '" width="' + str(MOL_W) + '" height="' +
            str(MOL_H) + '" style="display:block;border-radius:4px;'
            'cursor:zoom-in" onclick="openFullImg(this.src)"/></div>'
        )
        if idx < len(arrow_data):
            a             = arrow_data[idx]
            co_slice      = a["co_draw"][:4]
            n_co          = len(co_slice)
            def _co_tag(s):
                u = co_imgs.get(s, _fallback_data_uri(s, CO_W, CO_H))
                return (
                    '<img src="' + u + '" width="' + str(CO_W) +
                    '" height="' + str(CO_H) +
                    '" style="border:1px solid #dce3ec;border-radius:3px;background:#fff;" '
                    'title="' + s.replace('"', "&quot;") + '"/>'
                )
            if n_co == 0:
                co_html = ""
            elif n_co == 1:
                co_html = '<div class="co-row">' + _co_tag(co_slice[0]) + '</div>'
            else:
                top = '<div class="co-row">' + "".join(_co_tag(s) for s in co_slice[:-1]) + '</div>'
                bot = '<div class="co-row">' + _co_tag(co_slice[-1]) + '</div>'
                co_html = top + bot

            step_json_esc = json.dumps(
                {k: v for k, v in a.items() if k not in ("co_draw","co_text","below","is_purif")}
            ).replace('"', "&quot;")
            purif_color = "#8B5E3C" if a["is_purif"] else arrow_color
            purif_dash  = "stroke-dasharray:6,3;" if a["is_purif"] else ""
            below_html  = (
                ('<div class="b-yld">' + a["yld"] + '</div>' if a["yld"] else "") +
                ('<div class="b-rtype">' + a["rtype"] + '</div>' if a["rtype"] else "") +
                "".join(
                    '<div class="b-cond">' + str(t)[:32] + '</div>'
                    for t in a["below"] if t != a["yld"]
                )
            )
            items.append(
                '<div class="arrow-cell" data-step="' + step_json_esc +
                '" onclick="showStepFn' + rid_js + '(' + str(idx) + ',this)">'
                '<div class="co-above">' + co_html + '</div>'
                '<div class="arrow-line">'
                '<div class="a-shaft" style="background:' + purif_color + ';' + purif_dash + '"></div>'
                '<div class="a-head" style="border-left-color:' + purif_color + ';"></div>'
                '</div>'
                '<div class="step-lbl" style="color:' + purif_color + ';">Step ' + str(a["step"]) + '</div>'
                + below_html + '</div>'
            )

    # inline CSS for the reaction scheme iframe
    css = (
        "html,body{margin:0;padding:0;background:#fff;font-family:'DM Sans',Arial,sans-serif;}"
        "*{box-sizing:border-box}"
        "#scroll-wrap{overflow-x:auto;overflow-y:visible;background:#FAFAFA;"
        "border-radius:10px;border:1px solid #dce3ec;}"
        ".band{display:flex;align-items:center;justify-content:flex-start;"
        "height:" + str(SCHEME_H) + "px;gap:2px;padding:0 8px;overflow:visible;"
        "padding-top:" + str(PAD_TOP) + "px;padding-bottom:" + str(PAD_BOTTOM) + "px;"
        "box-sizing:content-box;}"
        ".mol-cell{display:flex;flex-direction:column;align-items:center;justify-content:center;"
        "min-width:" + str(CELL_W) + "px;max-width:" + str(CELL_W) + "px;"
        "height:" + str(SCHEME_H) + "px;"
        "background:#fff;border:1px solid #dce3ec;border-radius:8px;padding:4px;"
        "box-shadow:0 1px 4px rgba(26,46,68,0.06);flex-shrink:0;}"
        ".mol-role{font-size:0.55rem;font-weight:700;letter-spacing:0.06em;"
        "text-transform:uppercase;border-radius:10px;padding:1px 6px;margin-bottom:3px;white-space:nowrap;}"
        ".sm-role{background:#cce5ff;color:#004085}.tgt-role{background:#d4edda;color:#155724}"
        ".arrow-cell{position:relative;display:flex;flex-direction:column;align-items:center;"
        "justify-content:center;cursor:pointer;min-width:" + str(ARROW_CELL_W) + "px;"
        "height:" + str(SCHEME_H) + "px;padding:2px 4px;border-radius:6px;"
        "transition:background 0.12s;user-select:none;flex-shrink:0;overflow:visible;}"
        ".arrow-cell:hover{background:" + hover_bg + "}"
        ".co-above{position:absolute;bottom:100%;margin-bottom:-35px;left:50%;"
        "transform:translateX(-50%);display:flex;flex-direction:column;"
        "align-items:center;gap:2px;pointer-events:none;}"
        ".co-row{display:flex;flex-direction:row;gap:2px;align-items:center;justify-content:center;}"
        ".arrow-line{display:flex;align-items:center;width:" + str(ARROW_SHAFT_W + 10) + "px;flex-shrink:0;}"
        ".a-shaft{width:" + str(ARROW_SHAFT_W) + "px;height:2.5px;background:" + arrow_color + ";"
        "border-radius:2px 0 0 2px;flex-shrink:0;}"
        ".a-head{width:0;height:0;border-top:6px solid transparent;border-bottom:6px solid transparent;"
        "border-left:10px solid " + arrow_color + ";flex-shrink:0;}"
        ".step-lbl{font-size:10px;font-weight:700;color:" + arrow_color + ";text-align:center;margin-top:2px;}"
        ".b-yld{font-size:12px;font-weight:700;color:" + arrow_color + ";text-align:center;}"
        ".b-rtype{font-size:8.5px;color:#6b7a8d;text-align:center;word-break:break-word;"
        "white-space:normal;line-height:1.2;max-width:136px;font-style:italic;}"
        ".b-cond{font-size:8px;color:#37506e;text-align:center;word-break:break-word;"
        "max-width:136px;line-height:1.15;}"
        ".detail-panel{margin-top:6px;padding:12px 16px;background:" + panel_bg + ";"
        "border-left:4px solid " + arrow_color + ";border-radius:0 8px 8px 0;display:none;}"
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
        ".copy-btn{background:" + arrow_color + ";color:white;border:none;border-radius:3px;"
        "padding:2px 6px;font-size:9px;cursor:pointer;white-space:nowrap;flex-shrink:0}"
        ".copy-btn:hover{opacity:0.8}"
    )

    #  JavaScript 
    js = (
        "var __si=JSON.parse(document.getElementById('__si_" + route_id + "').textContent);"
        "function notifyResize(){"
        "var h=document.getElementById('wrap-" + route_id + "').scrollHeight;"
        "try{window.parent.postMessage({isStreamlitMessage:true,"
        "type:'streamlit:setFrameHeight',height:h+8},'*');}catch(e){}}"
        "function openFullImg(src){"
        "var w=window.open('','_blank','width=900,height=700');"
        "w.document.write('<html><body style=\"margin:0;background:#111;display:flex;"
        "align-items:center;justify-content:center;height:100vh\">')"
        ";w.document.write('<img src=\"'+src+'\" style=\"max-width:100%;max-height:100%;"
        "object-fit:contain\"/>');"
        "w.document.write('</body></html>');w.document.close();}"
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
        "if(d.source==='rxn-insight'&&d.fg&&d.fg.length>0)rows.push(['Functional groups',d.fg.join(', ')]);"
        "grid.innerHTML=rows.map(function(r){"
        "return '<span class=\"dk\">'+r[0]+'</span><span class=\"dv\">'+r[1]+'</span>';}).join('');"
        "var si=__si[String(d.step)]||{};"
        "var imgDiv=document.getElementById('dpi-" + route_id + "');"
        "var ih='';"
        "(si.reactants||[]).forEach(function(b64){"
        "if(b64)ih+='<img src=\"'+b64+'\" width=\"105\" height=\"74\" "
        "style=\"cursor:zoom-in\" onclick=\"openFullImg(this.src)\"/>';});"
        "if((si.reactants||[]).some(Boolean))ih+='<span class=\"step-arrow-txt\">&#8594;</span>';"
        "if(si.product)ih+='<img src=\"'+si.product+'\" width=\"105\" height=\"74\" "
        "style=\"cursor:zoom-in\" onclick=\"openFullImg(this.src)\"/>';"
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
        "<!DOCTYPE html><html><head><style>" + css + "</style></head><body>"
        '<div id="wrap-' + route_id + '">'
        '<div id="scroll-wrap"><div class="band">' + "".join(items) + '</div></div>'
        '<div class="detail-panel" id="dp-' + route_id + '">'
        '<div class="dp-title" id="dpt-' + route_id + '">Step details</div>'
        '<div class="dp-grid" id="dpg-' + route_id + '"></div>'
        '<div class="step-imgs" id="dpi-' + route_id + '"></div>'
        '<div class="smi-section" id="dps-' + route_id + '"></div>'
        '</div></div>'
        '<script type="application/json" id="__si_' + route_id + '">'
        + step_imgs_json + '</script>'
        '<script>' + js + '</script>'
        '</body></html>'
    )

def build_score_table_html(
    details: dict,
    criteria: list,
    weights: dict,
    lang: str = "en",
) -> str:
    """
    Render the per-criterion score breakdown table for a route card.

    Each row displays: criterion name, raw score 0–1 with a hoverable
    tooltip explaining the formula, the criterion weight, and the weighted
    contribution to the total score.  For predicted routes the yield row
    shows "excluded (predicted route)" instead of a numeric score.

    Parameters
    ----------
    details : dict
        Per-criterion detail dicts from ``rank_weighted()`` with keys
        ``raw``, ``weight``, ``weighted``, and optionally ``excluded``.
    criteria : list of str
        Three criterion keys in descending priority order.
    weights : dict
        ``{criterion_key: float}`` from ``rt.compute_weights()``.
    lang : str, optional
        Language code ``"en"`` or ``"fr"`` (default ``"en"``).

    Returns
    -------
    str
        HTML string that includes ``COMPONENT_STYLE``; safe to pass to
        ``st.html()``.
    """
    T   = LANG[lang]
    CL  = CRITERIA_LABELS[lang]
    csd = CRITERIA_SCORE_DESC.get(lang, CRITERIA_SCORE_DESC["en"])
    rows = []
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
        rows.append(
            f"<tr>"
            f"<td>{CL[c]}</td>"
            f"<td><span class='th-help' title='{tip_c}' style='cursor:help;"
            f"border-bottom:1px dashed #888'>{raw:.3f}"
            f"<span class='th-info' style='margin-left:3px'>?</span></span></td>"
            f"<td>{w*100:.0f}%</td>"
            f"<td>{cont:.3f}</td>"
            f"</tr>"
        )
    return (
        COMPONENT_STYLE +
        f"<table class='score-table'>"
        f"<thead><tr>"
        f"<th>{T['score_th_crit']}</th>"
        f"<th>{T['score_th_raw']}</th>"
        f"<th>{T['score_th_weight']}</th>"
        f"<th>{T['score_th_contrib']}</th>"
        f"</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table>"
    )

def make_ranking_chart(results: list, target_name: str, lang: str = "en"):
    """
    Render a horizontal bar chart ranking routes by total score.

    Displayed above the route cards whenever more than one route is found
    in a given category.  Score values are annotated directly on each bar.

    Parameters
    ----------
    results : list
        List of ``(score, details, route)`` tuples from ``rank_weighted()``.
    target_name : str
        Target molecule name — used as the chart title suffix.
    lang : str, optional
        Language code for axis / title strings (default ``"en"``).

    Returns
    -------
    matplotlib.figure.Figure
        Caller is responsible for calling ``plt.close(fig)``.
    """
    T      = LANG[lang]
    labels = [r[2].get("matched_route_name", f"Route {i+1}")[:32]
              for i, r in enumerate(results)]
    scores = [r[0] for r in results]
    n      = len(scores)
    fig, ax = hires_fig(figsize=(7, max(2.0, n * 0.65)))
    colors  = [PALETTE[i % len(PALETTE)] for i in range(n)]
    bars    = ax.barh(np.arange(n), scores, color=colors, height=0.52, edgecolor="none")
    for bar, s in zip(bars, scores):
        ax.text(s + max(scores) * 0.013, bar.get_y() + bar.get_height() / 2,
                f"{s:.4f}", va="center", fontsize=9.5, color="#1a2e44", fontweight="600")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=10, color="#1a2e44")
    ax.set_xlabel(T["score_axis"], fontsize=10, color="#6b7a8d")
    ax.set_xlim(0, max(scores) * 1.30)
    ax.set_title(T["chart_title"].format(target=target_name),
                 fontsize=13, fontweight="bold", color="#1a2e44", pad=14)
    ax.tick_params(colors="#6b7a8d", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.grid(True, color="#dce3ec", linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.6)
    return fig

def make_yield_chart(steps_route: list, lang: str = "en"):
    """
    Render a bar chart of reported step yields for a single route.

    Used in the Dataset Explorer route detail view.  Steps without a reported
    yield are omitted from the chart entirely (no bar, no implied 0 %).
    Yield labels use integer format; change ``":.0f"`` to ``":.1f"`` for one
    decimal place.

    Parameters
    ----------
    steps_route : list
        List of step dicts from ``dataset["by_route"][route_id]``.
    lang : str, optional
        Language code for axis labels (default ``"en"``).

    Returns
    -------
    matplotlib.figure.Figure
        Caller is responsible for calling ``plt.close(fig)``.
    """
    T  = LANG[lang]
    n  = len(steps_route)
    reported_vals  = []
    reported_steps = []
    for i, s in enumerate(steps_route):
        y = s.get("yield_percent")
        if y is not None:
            reported_vals.append(y)
            reported_steps.append(i + 1)
    fig, ax = hires_fig(figsize=(max(4.5, n * 0.7), 2.6))
    if reported_vals:
        bars = ax.bar(reported_steps, reported_vals,
                      color="#1a2e44", width=0.52, edgecolor="none")
        for bar, v in zip(bars, reported_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"{v:.0f}%", ha="center", va="bottom",
                    fontsize=7.5, color="#1a2e44", fontweight="600")
    ax.set_ylim(0, 125)
    ax.set_xticks(list(range(1, n + 1)))
    ax.set_xlabel(strip_emoji(T["col_steps"]), fontsize=10, color="#6b7a8d")
    ax.set_ylabel("Yield (%)", fontsize=10, color="#6b7a8d")
    ax.legend(handles=[mpatches.Patch(color="#1a2e44", label="Reported yield")],
              fontsize=8, framealpha=0.8, facecolor=FIG_BG, edgecolor="#dce3ec")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color="#dce3ec", linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#6b7a8d", length=0)
    fig.tight_layout(pad=1.2)
    return fig

def make_comparison_chart(sel_results: list, criteria: list, lang: str = "en"):
    """
    Render a grouped horizontal bar chart comparing raw criterion scores.

    Each route is plotted in a distinct colour from ``PALETTE``; each row
    corresponds to one criterion.  Score values are annotated at the end of
    each bar.  Used in the Analysis tab for side-by-side route comparison.

    Parameters
    ----------
    sel_results : list
        Subset of ``(score, details, route)`` tuples selected by the user
        in the Analysis tab multiselect.
    criteria : list of str
        Three criterion keys in priority order.
    lang : str, optional
        Language code for axis and title labels (default ``"en"``).

    Returns
    -------
    matplotlib.figure.Figure
        Caller is responsible for calling ``plt.close(fig)``.
    """
    T           = LANG[lang]
    CL          = CRITERIA_LABELS[lang]
    route_names = [r[2].get("matched_route_name", f"R{i+1}")[:20]
                   for i, r in enumerate(sel_results)]
    n_routes = len(sel_results)
    n_crit   = len(criteria)
    x        = np.arange(n_crit)
    bar_h    = 0.72 / n_routes
    offsets  = np.linspace(-(n_routes - 1) / 2, (n_routes - 1) / 2, n_routes) * bar_h
    fig, ax  = hires_fig(figsize=(6.5, max(2.8, n_crit * 1.0)))
    for i, (score, details, route) in enumerate(sel_results):
        vals  = [details[c].get("raw", 0) or 0 for c in criteria]
        color = PALETTE[i % len(PALETTE)]
        bars  = ax.barh(x + offsets[i], vals, bar_h * 0.88,
                        color=color, label=route_names[i], edgecolor="none", alpha=0.92)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.012, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=8, color=color, fontweight="600")
    ax.set_yticks(x)
    ax.set_yticklabels([strip_emoji(CL[c]) for c in criteria], fontsize=10, color="#1a2e44")
    ax.set_xlim(0, 1.28)
    ax.set_xlabel("Raw score (0–1)", fontsize=10, color="#6b7a8d")
    ax.set_title(strip_emoji(T["radar_title"]),
                 fontsize=12, fontweight="bold", color="#1a2e44", pad=12)
    ax.legend(loc="lower right", fontsize=8,
              framealpha=0.8, facecolor=FIG_BG, edgecolor="#dce3ec")
    ax.tick_params(colors="#6b7a8d", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.grid(True, color="#dce3ec", linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.4)
    return fig

def build_why_ranked_html(
    rank: int,
    score_total: float,
    details: dict,
    criteria: list,
    weights: dict,
    steps_data: list,
    lang: str = "en",
) -> str:
    """
    Build the "Why is this route ranked #N?" reasoning box.

    Generates one plain-language bullet per top-2 criterion using domain-relevant
    values (step count, bottleneck yield, atom economy score, etc.) rather than
    raw numeric scores.  Always produces at least one bullet.

    Parameters
    ----------
    rank : int
        1-based rank of the route within its category.
    score_total : float
        Total weighted score.
    details : dict
        Per-criterion detail dicts from ``rank_weighted()``.
    criteria : list of str
        Three criterion keys in descending priority order.
    weights : dict
        ``{criterion_key: float}`` from ``fi.compute_weights()``.
    steps_data : list
        ``route["dataset_steps"]`` used to compute yield statistics.
    lang : str, optional
        Language code ``"en"`` or ``"fr"`` (default ``"en"``).

    Returns
    -------
    str
        HTML string that includes ``COMPONENT_STYLE``; safe for ``st.html()``.
    """
    T   = LANG[lang]
    CL  = CRITERIA_LABELS[lang]
    bn  = rt.bottleneck_yield(steps_data)
    av  = rt.average_yield(steps_data)
    n   = len(steps_data)
    bullets = []
    for crit in criteria[:2]:
        if crit == "steps":
            bullets.append(T["why_steps"].format(n=n))
        elif crit == "yield":
            if bn is not None:
                bullets.append(T["why_yield"].format(y=f"{bn:.1f}"))
            elif av is not None:
                bullets.append(T["why_avg_yield"].format(a=f"{av:.1f}"))
        elif crit == "atom_economy":
            raw = details[crit].get("raw") or 0
            bullets.append(
                f"it achieves an atom economy of <strong>{raw*100:.0f}%</strong>"
            )
        elif crit == "e_factor":
            raw = details[crit].get("raw") or 0
            bullets.append(
                f"it has a strong E-factor score of <strong>{raw:.2f}</strong>"
            )
        elif crit == "toxicity":
            raw = details[crit].get("raw") or 0
            bullets.append(
                f"it has a safety score of <strong>{raw*100:.0f}%</strong>"
            )
    items_html = "".join(f"<li>{b}</li>" for b in bullets)
    return (
        COMPONENT_STYLE +
        "<div class='why-box'>"
        "<div>" + T["why_prefix"].format(r=rank) + "</div>"
        "<ul>" + items_html + "</ul>"
        "<em>" + T["why_suffix"] + "</em>"
        "</div>"
    )

def smiles_copy_widget(smiles: str, label: str = "") -> None:
    """
    Render a compact SMILES display widget with a clipboard Copy button.

    Displayed beneath each reactant and product molecule image in the Dataset
    Explorer step-by-step view, allowing the chemist to copy any SMILES string
    without manually transcribing it.

    Parameters
    ----------
    smiles : str
        SMILES string to display and make copyable via the browser clipboard.
    label : str, optional
        Reserved for future use (currently unused).

    Returns
    -------
    None
        Renders directly into the current Streamlit column via
        ``components.html()``.
    """
    short = smiles[:38] + ("…" if len(smiles) > 38 else "")
    safe  = smiles.replace("`", "\\`").replace("\\", "\\\\")
    html_snip = (
        "<div style='display:flex;align-items:center;gap:6px;background:#f5f7fa;"
        "border:1px solid #dce3ec;border-radius:6px;padding:4px 8px;"
        "font-family:\"DM Mono\",\"Fira Mono\",monospace;font-size:0.72rem;"
        "color:#37506e;margin-top:2px;margin-bottom:6px;overflow:hidden;'>"
        "<span style='flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'"
        f" title='{smiles}'>{short}</span>"
        "<button onclick=\""
        f"navigator.clipboard.writeText(`{safe}`)"
        ".then(()=>{this.textContent='✓';setTimeout(()=>this.textContent='Copy',1400)})"
        ".catch(()=>{var t=document.createElement('textarea');"
        f"t.value=`{safe}`;document.body.appendChild(t);t.select();"
        "document.execCommand('copy');document.body.removeChild(t);"
        "this.textContent='✓';setTimeout(()=>this.textContent='Copy',1400);})\""
        " style='background:#1a2e44;color:white;border:none;border-radius:4px;"
        "padding:2px 8px;font-size:0.68rem;font-weight:600;cursor:pointer;"
        "white-space:nowrap;flex-shrink:0;'>Copy</button>"
        "</div>"
    )
    components.html(html_snip, height=36, scrolling=False)

def display_route_card(
    score_total: float,
    details: dict,
    route: dict,
    criteria: list,
    weights: dict,
    all_results: list,
    rank: int = 1,
    badge: str = "📚 dataset",
    lang: str = "en",
) -> None:
    """
    Render a complete route card inside a Streamlit expander.

    The card contains (top to bottom):

    1. Validation status banner (success / info / warning).
    2. Four metric columns: total score, step count, bottleneck yield, avg yield.
    3. Score breakdown table (``build_score_table_html``).
    4. Why-ranked reasoning box (``build_why_ranked_html``).
    5. PDF download button (``build_route_report_pdf``).
    6. Horizontal rule followed by the interactive reaction scheme
       (``build_clickable_scheme_html`` inside ``components.html``).
    7. Substances-needed expander with molecule thumbnails, solvents, reagents.

    The expander is expanded by default only for the first (rank 1) route.

    Parameters
    ----------
    score_total : float
        Total weighted score of the route.
    details : dict
        Per-criterion detail dicts from ``rank_weighted()``.
    route : dict
        Enriched route dict (output of the matching / enrichment pipeline).
    criteria : list of str
        Three criterion keys in priority order.
    weights : dict
        ``{criterion_key: float}`` from ``fi.compute_weights()``.
    all_results : list
        Full ranked result list for the category (reserved for future use).
    rank : int, optional
        1-based rank within the category (default 1).
    badge : str, optional
        Emoji + label shown in the expander header (default ``"📚 dataset"``).
    lang : str, optional
        Language code ``"en"`` or ``"fr"`` (default ``"en"``).

    Returns
    -------
    None
        Renders directly into the current Streamlit context.
    """
    T          = LANG[lang]
    route_name = route.get("matched_route_name", "?")
    tgt        = route.get("matched_target", "?")
    steps_data = route.get("dataset_steps", [])
    n_steps    = len(steps_data)
    bn         = rt.bottleneck_yield(steps_data)
    av         = rt.average_yield(steps_data)
    medals     = ["🥇","🥈","🥉","4️.","5️.","6️.","7️.","8️.","9️.","10."]
    medal      = medals[rank - 1] if rank <= 10 else f"#{rank}"
    route_key  = "".join(c for c in route.get("matched_route_id", "r") if c.isalnum())
    is_pred    = route.get("is_predicted", False)
    status     = route.get("validation_status", "dataset")
    v          = route.get("validated_steps_count", 0)
    t_         = route.get("total_steps_count", 0)

    with st.expander(
        f"{medal}  [{badge}]  {route_name}  ·  {tgt}  ·  Score: **{score_total:.3f}**",
        expanded=(rank == 1),
    ):
        # Status banner
        if status == "validated":
            st.success("✅ Fully validated — all steps found in generic dataset (real conditions).")
        elif status == "partial" and v > 0:
            st.info(f"🧪 {T['partial_badge'].format(v=v, t=t_)}")
        elif status == "predicted":
            st.warning(
                "⚠️ Predicted route — no steps found in experimental literature. "
                "Yield excluded from scoring."
                if lang == "en" else
                "⚠️ Route prédite — aucune étape trouvée dans la littérature. "
                "Le rendement est exclu du score."
            )

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(T["metric_score"],     f"{score_total:.3f}", help=T["metric_score_help"])
        m2.metric(T["metric_steps"],     n_steps)
        m3.metric(T["metric_bottleneck"],
                  f"{bn:.1f}%" if bn is not None else "—", help=T["metric_bn_help"])
        m4.metric(T["metric_avg"],       f"{av:.1f}%" if av is not None else "—")

        # Score table
        st.markdown(f"**{T['contrib_title']}**")
        st.html(build_score_table_html(details, criteria, weights, lang))

        # Why ranked
        st.markdown(f"**{T['why_best_title'].format(r=rank)}**")
        st.html(build_why_ranked_html(rank, score_total, details, criteria, weights, steps_data, lang))

        # PDF download
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
            sub      = rt.get_substances_list(steps_data)
            all_mols = sub["to_buy"] + sub["to_prepare"][:6]
            if all_mols:
                n_cols   = min(4, len(all_mols))
                cols_sub = st.columns(n_cols)
                for i, smi in enumerate(all_mols):
                    with cols_sub[i % n_cols]:
                        png = mol_png(smi, 320, 220)
                        if png:
                            st.image(png, width="stretch")
                        smiles_copy_widget(smi)
            if sub["solvents"]:
                st.markdown("**🧴 Solvents**")
                st.write("  ·  ".join(sub["solvents"]))
            if sub["reagents"]:
                st.markdown("**🔬 Reagents**")
                st.write("  ·  ".join(sub["reagents"][:10]))

@st.cache_data(show_spinner=False)
def load_dataset_cached(path: str) -> dict:
    """
    Streamlit-cached wrapper around ``rt.load_reaction_dataset()``.

    Prevents reloading the JSON dataset on every widget interaction during the
    same session.  The cache is keyed by *path* so changing the sidebar input
    automatically triggers a fresh load.

    Parameters
    ----------
    path : str
        Filesystem path to ``reaction_dataset.json``.

    Returns
    -------
    dict
        As returned by ``rt.load_reaction_dataset()`` — keys ``all``,
        ``by_product``, ``by_reactant``, ``by_route``, ``metadata``.
    """
    return rt.load_reaction_dataset(path)

@st.cache_data(show_spinner=False)
def get_targets_cached(path: str) -> dict:
    """
    Streamlit-cached wrapper returning the ``{target_name: SMILES}`` mapping.

    Used to populate the predefined molecule selector in the Route Search tab.
    Caching avoids iterating over all route steps on every widget interaction.

    Parameters
    ----------
    path : str
        Filesystem path to ``reaction_dataset.json``.

    Returns
    -------
    dict
        ``{target_name: canonical_SMILES}`` for all targets in the dataset.
    """
    return rt.get_targets_from_dataset(load_dataset_cached(path))
