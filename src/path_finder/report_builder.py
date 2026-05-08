# report_builder.py
# =============================================================================
# PDF report generation for the Retrosynthesis Interface.
#
# This module has NO Streamlit dependency and can be called from any context
# (CLI, tests, notebook) as long as PIL and RDKit are available.
#
# Public API:
#   build_route_report_pdf(score_total, details, route, criteria) → bytes
#       Generates a multi-page A4 PDF for a single synthesis route.
#       Returns raw bytes suitable for st.download_button() or file.write().
#
# Page layout:
#   Page 1  : Navy header band (route name, target, status, total score),
#             four metric cards (steps, cumulative yield, bottleneck, avg yield),
#             score breakdown table with per-criterion raw / weight / contribution,
#             starting-material structure images (up to 5, rendered with Cairo).
#   Pages 2+: Up to STEPS_PER_PAGE (= 3) reaction steps per page, each with:
#             • Coloured header band (orange = Rxn-INSIGHT source, navy = dataset)
#             • Conditions bar
#             • Reactant structure images → arrow → product structure image
#
# Molecule images inside the PDF are rendered with RDKit Cairo at 2× the
# layout pixel size and then PIL-resampled back to the target size, giving
# clean bond lines at 200 DPI without excessive file size.
#
# Dependencies:
#   - Pillow (PIL)  — page composition, font loading, PDF serialisation
#   - RDKit         — molecule rendering (rdMolDraw2D Cairo)
#   - route_engine (fi) — bottleneck_yield, average_yield,
#                                 cumulative_yield, get_substances_list
# =============================================================================

# -- RDKit imports with graceful degradation ----------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    MODULE_OK = True
except Exception:
    MODULE_OK = False

import route_engine as fi


def build_route_report_pdf(score_total: float, details: dict, route: dict,
                            criteria: list) -> bytes:
    """
    Generate a multi-page PDF report for a single synthesis route using PIL only
    (no reportlab dependency required).

    Page layout
    ───────────
    Page 1  : Summary header (route name, target, validation status, total score),
              metric cards (steps, cumulative yield, bottleneck, avg yield),
              score breakdown table, and starting material structure images.
    Pages 2+: Up to 3 reaction steps per page, each with Cairo molecule images
              at 200 DPI.

    All molecule images are rendered with RDKit Cairo at 2× the display size
    for sharpness in the PDF.

    Parameters
    ──────────
    score_total : float  total weighted score (shown in the header)
    details     : dict   per-criterion detail dicts from rank_weighted()
    route       : dict   enriched route dict (must have "dataset_steps")
    criteria    : list   3 criterion keys in priority order

    Returns
    ───────
    bytes  PDF file content ready for st.download_button() or open(..., "wb")
    """
    from PIL import Image as _PI, ImageDraw as _PID, ImageFont as _PIF
    import io as _io

    # -- PDF page geometry (A4 at 200 DPI) ------------------------------------
    DPI  = 200
    PW   = int(8.27 * DPI)   # page width  = 1654 px
    PH   = int(11.69 * DPI)  # page height = 2338 px
    MG   = int(0.55 * DPI)   # margin      =  110 px
    CW   = PW - 2 * MG       # usable content width

    # -- Colour palette (RGB tuples) ------------------------------------------
    NAVY   = (26,  46,  68)
    WHITE  = (255, 255, 255)
    LIGHT  = (235, 242, 250)
    LGREY  = (245, 248, 252)
    GREY   = (107, 122, 141)
    ORANGE = (230,  81,   0)
    GREEN  = ( 21,  87,  36)
    SEP    = (220, 227, 236)
    BG     = (249, 251, 253)

    # -- Font loader -----------------------------------------------------------
    def _fnt(size, bold=False):
        # Try system font paths (Linux first, then macOS); fall back to PIL default
        candidates = (
            ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]
            if bold else
            ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/System/Library/Fonts/Supplemental/Arial.ttf"]
        )
        for p in candidates:
            try:
                return _PIF.truetype(p, size)
            except Exception:
                pass
        return _PIF.load_default()

    # -- Molecule renderer (PIL Image at target size) -------------------------
    def _mol_pil(smi, w, h):
        # Render a molecule to a PIL Image at 2× the requested size for crispness
        if not smi or not MODULE_OK:
            return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        try:
            if mol.GetNumAtoms() > 1:
                rdDepictor.Compute2DCoords(mol)
            drw = rdMolDraw2D.MolDraw2DCairo(w * 2, h * 2)
            drw.drawOptions().bondLineWidth       = 1.0
            drw.drawOptions().padding             = 0.1
            drw.drawOptions().addStereoAnnotation = True
            drw.DrawMolecule(mol)
            drw.FinishDrawing()
            img = _PI.open(_io.BytesIO(drw.GetDrawingText())).convert("RGBA")
            bg  = _PI.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            return bg.convert("RGB").resize((w, h), _PI.LANCZOS)
        except Exception:
            return None

    # -- Text helpers ---------------------------------------------------------
    def _trunc(txt, n):
        return txt[:n] + "…" if len(txt) > n else txt

    def _text_w(draw, txt, font):
        try:
            return int(draw.textlength(txt, font=font))
        except Exception:
            return len(txt) * max(6, font.size // 2)

    def _wrap(draw, txt, max_px, font):
        # Word-wrap a string to fit within max_px pixels wide
        words = txt.split(); lines = []; line = ""
        for w in words:
            test = (line + " " + w).strip()
            if _text_w(draw, test, font) <= max_px:
                line = test
            else:
                if line:
                    lines.append(line)
                line = w
        if line:
            lines.append(line)
        return lines or [""]

    # -- Drawing primitives ---------------------------------------------------
    def _draw_band(draw, y, h, color=NAVY):
        draw.rectangle([0, y, PW, y + h], fill=color)

    def _draw_rounded(draw, x, y, w, h, fill, outline=None, r=5):
        draw.rounded_rectangle([x, y, x + w, y + h], radius=r,
                                fill=fill, outline=outline or fill)

    # -- Extract route metadata -----------------------------------------------
    steps_data = route.get("dataset_steps", [])
    rname      = route.get("matched_route_name", "Unknown route")
    target     = route.get("matched_target", "?")
    status     = route.get("validation_status", "dataset")
    status_lbl = {
        "dataset":   "Dataset",
        "validated": "Validated",
        "partial":   "Partial",
        "predicted": "Predicted",
    }.get(status, status)
    sub   = fi.get_substances_list(steps_data)
    bn_   = fi.bottleneck_yield(steps_data)
    av_   = fi.average_yield(steps_data)
    cumyl = fi.cumulative_yield(steps_data)

    pages = []

    # =========================================================================
    # Page 1 — summary
    # =========================================================================
    p1 = _PI.new("RGB", (PW, PH), BG)
    d1 = _PID.Draw(p1)

    BH = int(0.72 * DPI)  # header band height
    _draw_band(d1, 0, BH)
    d1.text((MG, 20), _trunc(rname, 60), fill=WHITE, font=_fnt(38, True))
    d1.text((MG, 64),
            f"Target: {target}   ·   {status_lbl}   ·   Score: {score_total:.4f}",
            fill=(180, 200, 220), font=_fnt(22))

    y1 = BH + 22

    # Metric cards row
    mets = [
        ("Steps",        str(len(steps_data))),
        ("Cumul. yield", f"{cumyl * 100:.4f}%"),
        ("Bottleneck",   f"{bn_:.4f}%" if bn_ else "—"),
        ("Avg yield",    f"{av_:.4f}%"  if av_ else "—"),
    ]
    mc = CW // len(mets); mx = MG
    for lbl, val in mets:
        _draw_rounded(d1, mx, y1, mc - 8, 64, LIGHT, SEP)
        d1.text((mx + 10, y1 + 5),  lbl, fill=GREY, font=_fnt(18))
        d1.text((mx + 10, y1 + 28), val, fill=NAVY, font=_fnt(26, True))
        mx += mc
    y1 += 78

    d1.line([(MG, y1), (PW - MG, y1)], fill=SEP, width=2); y1 += 18

    # Score breakdown table
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
        row  = [
            c,
            "excluded" if excl else (f"{raw:.4f}" if raw is not None else "—"),
            "0%"       if excl else f"{(dtl.get('weight') or 0) * 100:.0f}%",
            "—"        if excl else f"{dtl.get('weighted') or 0:.4f}",
        ]
        bg_c = LIGHT if ci % 2 == 0 else LGREY
        hx = MG
        for txt, cw in zip(row, col_ws):
            _draw_rounded(d1, hx, y1, cw - 4, 28, bg_c, SEP, 3)
            d1.text((hx + 7, y1 + 5), txt, fill=NAVY, font=_fnt(16))
            hx += cw
        y1 += 28
    y1 += 16

    d1.line([(MG, y1), (PW - MG, y1)], fill=SEP, width=2); y1 += 16

    # Starting materials grid (up to 5 structures)
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

    # =========================================================================
    # Pages 2+: step-by-step
    # =========================================================================
    def _render_step_at(draw, page, step, x0, y0, avail_w, avail_h):
        """
        Render one reaction step into a rectangular region of a PIL page.

        Layout within the region (top to bottom):
        1. Coloured header band: step number, reaction type, yield
        2. Conditions bar
        3. Reactant structures → arrow → product structure
        """
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
        if t_:
            cond_parts.append(f"{t_}°C")
        s_ = cond.get("solvent", "")
        if s_:
            cond_parts.append(s_)
        rr = cond.get("reagents", []) or []
        if rr:
            cond_parts.append(", ".join(str(r) for r in rr[:2]))
        cond_s = "  ·  ".join(cond_parts) or "—"

        y = y0
        HDR_H  = int(avail_h * 0.22)
        COND_H = int(avail_h * 0.16)

        _draw_band(draw, y, HDR_H, hdr_c)
        lbl = f"Step {snum} — {_trunc(rtype, 30)}"
        draw.text((L, y + 5),  lbl,               fill=WHITE,          font=_fnt(16, True))
        draw.text((L, y + 24), f"Yield: {yld_s}",  fill=(200, 215, 230), font=_fnt(13))
        y += HDR_H

        _draw_rounded(draw, L, y, W, COND_H - 4, LIGHT, SEP, 3)
        clines = _wrap(draw, "Cond: " + cond_s, W - 8, _fnt(13))[:2]
        cy = y + 4
        for cl in clines:
            draw.text((L + 5, cy), cl, fill=NAVY, font=_fnt(13))
            cy += 17
        y += COND_H

        mol_zone = avail_h - HDR_H - COND_H - 18
        mol_h    = max(50, min(mol_zone, int(avail_h * 0.45)))
        n_reac   = max(len(reac), 1)
        ARW      = int(W * 0.22)
        PLS      = int(W * 0.08)
        mol_w    = max(40, (W - ARW - (n_reac - 1) * PLS) // (n_reac + 1))
        mol_w    = min(mol_w, int(mol_h * 1.4))
        mol_h    = min(mol_h, int(mol_w * 0.68))

        y += 6
        mx = L
        for i, rsmi in enumerate(reac[:3]):
            img = _mol_pil(rsmi, mol_w, mol_h)
            if img:
                page.paste(img, (mx, y))
                lbl2 = _trunc(rsmi, 14)
                tw   = _text_w(draw, lbl2, _fnt(11))
                draw.text((mx + (mol_w - tw) // 2, y + mol_h + 2),
                          lbl2, fill=GREY, font=_fnt(11))
            mx += mol_w
            if i < len(reac) - 1:
                draw.text((mx + 4, y + mol_h // 2 - 10), "+", fill=NAVY, font=_fnt(22, True))
                mx += PLS
        ay  = y + mol_h // 2
        ax0 = mx + 4; ax1 = mx + ARW - 10
        draw.line([(ax0, ay), (ax1, ay)], fill=NAVY, width=3)
        draw.polygon([(ax1, ay - 6), (ax1 + 9, ay), (ax1, ay + 6)], fill=NAVY)
        mx += ARW
        p_img = _mol_pil(prod, mol_w, mol_h)
        if p_img:
            page.paste(p_img, (mx, y))
            lbl3 = _trunc(prod, 14)
            tw   = _text_w(draw, lbl3, _fnt(11))
            draw.text((mx + (mol_w - tw) // 2, y + mol_h + 2),
                      lbl3, fill=GREEN, font=_fnt(11))

        draw.line([(x0, y0 + avail_h - 2), (x0 + avail_w, y0 + avail_h - 2)],
                  fill=SEP, width=1)

    STEPS_PER_PAGE = 3
    STEP_H = (PH - 2 * MG - int(0.15 * DPI) * (STEPS_PER_PAGE - 1)) // STEPS_PER_PAGE
    STEP_W = CW
    it = iter(steps_data)
    while True:
        chunk = [next(it, None) for _ in range(STEPS_PER_PAGE)]
        if not any(chunk):
            break
        pg  = _PI.new("RGB", (PW, PH), BG)
        drw = _PID.Draw(pg)
        for slot, step in enumerate(chunk):
            if step is None:
                continue
            y0 = MG + slot * (STEP_H + int(0.15 * DPI))
            _render_step_at(drw, pg, step, MG, y0, STEP_W, STEP_H)
        pages.append(pg)

    # -- Serialise all pages into a single multi-page PDF ---------------------
    buf = _io.BytesIO()
    if pages:
        pages[0].save(buf, format="PDF", save_all=True,
                      append_images=pages[1:], resolution=DPI)
    return buf.getvalue()
