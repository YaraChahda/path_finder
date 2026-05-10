# report_builder.py
# =============================================================================
# PDF report generation for the Retrosynthesis Interface.
#
# Public API:
#   build_route_report_pdf(score_total, details, route, criteria) → bytes
#
# Page layout:
#   Page 1  : Navy header, metric cards, score table, starting materials.
#   Pages 2+: Up to 3 steps per page — header band, conditions, mol images.
#
# Molecule images: RDKit Cairo at 3× display size, bondLineWidth=6,
#   then PIL Lanczos downsample → crisp thick bonds at any zoom.
#
# Text policy: NO truncation anywhere. SMILES and reaction types are always
#   displayed in full. Long strings are word-wrapped across multiple lines.
# =============================================================================

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
    Generate a multi-page A4 PDF for a single synthesis route using PIL.

    Page 1  : Summary header, metric cards, score breakdown table,
              starting-material structure images with full SMILES labels.
    Pages 2+: Up to 3 reaction steps per page. Each step shows a coloured
              header band (navy=dataset, orange=predicted), a conditions bar,
              and molecule images (reactants → arrow → product) centred
              horizontally. Full SMILES labels use identical font and style
              for reactants and product. Reaction types are never truncated.

    Molecule rendering: Cairo at 3× display size, bondLineWidth=6,
    then Lanczos-downsampled to display size — gives thick, crisp bonds
    that are clearly visible at normal viewing and remain sharp when zoomed.

    Text policy: nothing is ever truncated with ellipsis. Long SMILES and
    reaction type strings are word-wrapped across multiple lines using
    character-boundary splitting so the full string is always visible.

    Parameters
    ──────────
    score_total : float
    details     : dict   from rank_weighted()
    route       : dict   enriched route dict (must have "dataset_steps")
    criteria    : list   3 criterion keys

    Returns
    ───────
    bytes  PDF content
    """
    from PIL import Image as _PI, ImageDraw as _PID, ImageFont as _PIF
    import io as _io

    # ── Page geometry (A4 at 200 DPI) ─────────────────────────────────────────
    DPI = 200
    PW  = int(8.27  * DPI)   # 1654 px
    PH  = int(11.69 * DPI)   # 2338 px
    MG  = int(0.55  * DPI)   # 110 px  (left/right margin)
    CW  = PW - 2 * MG        # usable width

    # ── Colours ───────────────────────────────────────────────────────────────
    NAVY   = (26,  46,  68)
    WHITE  = (255, 255, 255)
    LIGHT  = (235, 242, 250)
    LGREY  = (245, 248, 252)
    GREY   = (107, 122, 141)
    ORANGE = (230,  81,   0)
    GREEN  = ( 21,  87,  36)
    SEP    = (220, 227, 236)
    BG     = (249, 251, 253)

    # ── Fonts ─────────────────────────────────────────────────────────────────
    def _fnt(size, bold=False):
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

    # Font size constants — adjust here to rescale all text uniformly
    F_TITLE    = 52   # route name in page-1 header
    F_SUBTITLE = 30   # target / status / score line
    F_SECTION  = 28   # section headings ("Score breakdown", "Starting materials")
    F_CARD_LBL = 20   # metric card label
    F_CARD_VAL = 34   # metric card value
    F_TH       = 20   # table header text
    F_TD       = 20   # table cell text
    F_ROW_H    = 32   # pixel height of each table row (must comfortably fit F_TD)
    F_COND     = 22   # conditions text in step pages
    F_STEP_HDR = 28   # step header ("Step N · reaction type")
    F_YIELD    = 22   # yield text in step header (right-aligned)
    F_SMILES   = 19   # SMILES label under molecule images

    # ── Molecule renderer ─────────────────────────────────────────────────────
    def _mol_pil(smi, w, h):
        """
        Render smi to a PIL RGB Image of size (w, h).

        Cairo draws at 3× then Lanczos-downsamples so bond lines are thick
        and atom labels are readable. bondLineWidth=6 at 3× gives ~2 pt bonds
        on the page — clearly visible without being cartoonish.
        """
        if not smi or not MODULE_OK:
            return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        try:
            if mol.GetNumAtoms() > 1:
                rdDepictor.Compute2DCoords(mol)
            scale = 3
            drw = rdMolDraw2D.MolDraw2DCairo(w * scale, h * scale)
            opts = drw.drawOptions()
            opts.bondLineWidth       = 6.0   # thick bonds
            opts.padding             = 0.15
            opts.addStereoAnnotation = True
            opts.atomLabelFontSize   = 0.55  # readable atom labels
            drw.DrawMolecule(mol)
            drw.FinishDrawing()
            img = _PI.open(_io.BytesIO(drw.GetDrawingText())).convert("RGBA")
            bg  = _PI.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            return bg.convert("RGB").resize((w, h), _PI.LANCZOS)
        except Exception:
            return None

    # ── Text helpers ──────────────────────────────────────────────────────────
    def _text_w(draw, txt, font):
        """Return the pixel width of txt rendered with font."""
        try:
            return int(draw.textlength(txt, font=font))
        except Exception:
            return len(txt) * max(6, font.size // 2)

    def _wrap_words(draw, txt, max_px, font):
        """
        Word-wrap txt to fit within max_px pixels wide (word boundary).
        Used for natural-language text (conditions, reaction type).
        Returns a list of line strings.
        """
        words = txt.split()
        lines = []
        line  = ""
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

    def _wrap_smiles(draw, txt, max_px, font):
        """
        Wrap a SMILES string (no spaces) across lines by character count.
        Splits at the longest prefix that fits within max_px, then continues.
        Never truncates — the full SMILES is always emitted.
        Returns a list of line strings.
        """
        if not txt:
            return [""]
        lines = []
        while txt:
            # Binary-search for the longest prefix that fits
            lo, hi = 1, len(txt)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if _text_w(draw, txt[:mid], font) <= max_px:
                    lo = mid
                else:
                    hi = mid - 1
            lines.append(txt[:lo])
            txt = txt[lo:]
        return lines

    def _draw_text_lines(draw, lines, x, y, font, fill, line_gap=4):
        """
        Draw a list of text lines starting at (x, y).
        Returns the y coordinate immediately after the last line.
        """
        try:
            lh = font.size + line_gap
        except Exception:
            lh = 14 + line_gap
        for ln in lines:
            draw.text((x, y), ln, fill=fill, font=font)
            y += lh
        return y

    def _wrap(draw, txt, max_px, font):
        """Alias kept for conditions (word-wrap)."""
        return _wrap_words(draw, txt, max_px, font)

    def _draw_band(draw, y, h, color=NAVY):
        draw.rectangle([0, y, PW, y + h], fill=color)

    def _draw_rounded(draw, x, y, w, h, fill, outline=None, r=4):
        draw.rounded_rectangle([x, y, x + w, y + h], radius=r,
                                fill=fill, outline=outline or fill)

    # ── Route metadata ────────────────────────────────────────────────────────
    steps_data = route.get("dataset_steps", [])
    rname      = route.get("matched_route_name", "Unknown route")
    target     = route.get("matched_target", "?")
    status     = route.get("validation_status", "dataset")
    status_lbl = {"dataset": "Dataset", "validated": "Validated",
                  "partial": "Partial", "predicted": "Predicted"}.get(status, status)
    sub   = fi.get_substances_list(steps_data)
    bn_   = fi.bottleneck_yield(steps_data)
    av_   = fi.average_yield(steps_data)
    cumyl = fi.cumulative_yield(steps_data)

    pages = []

    # =========================================================================
    # PAGE 1 — Summary
    # =========================================================================
    p1 = _PI.new("RGB", (PW, PH), BG)
    d1 = _PID.Draw(p1)

    # Header band — wraps route name if very long
    rname_lines = _wrap_words(d1, rname, CW, _fnt(F_TITLE, True))
    n_title_lines = len(rname_lines)
    BH = n_title_lines * (F_TITLE + 6) + F_SUBTITLE + 48
    _draw_band(d1, 0, BH)

    ty = 16
    for ln in rname_lines:
        d1.text((MG, ty), ln, fill=WHITE, font=_fnt(F_TITLE, True))
        ty += F_TITLE + 6
    subtitle = (f"Target: {target}   ·   {status_lbl}   ·   "
                f"Score: {score_total:.4f}")
    d1.text((MG, ty + 4), subtitle, fill=(180, 200, 220), font=_fnt(F_SUBTITLE))

    y1 = BH + 24

    # Metric cards
    CARD_H = F_CARD_LBL + F_CARD_VAL + 24
    mets = [
        ("Steps",        str(len(steps_data))),
        ("Cumul. yield", f"{cumyl * 100:.2f}%"),
        ("Bottleneck",   f"{bn_:.1f}%" if bn_ else "—"),
        ("Avg yield",    f"{av_:.1f}%" if av_ else "—"),
    ]
    mc = CW // len(mets)
    mx = MG
    for lbl, val in mets:
        _draw_rounded(d1, mx, y1, mc - 10, CARD_H, LIGHT, SEP)
        d1.text((mx + 12, y1 + 8), lbl, fill=GREY, font=_fnt(F_CARD_LBL))
        d1.text((mx + 12, y1 + 8 + F_CARD_LBL + 6), val,
                fill=NAVY, font=_fnt(F_CARD_VAL, True))
        mx += mc
    y1 += CARD_H + 20

    d1.line([(MG, y1), (PW - MG, y1)], fill=SEP, width=2)
    y1 += 16

    # Score breakdown table
    d1.text((MG, y1), "Score breakdown", fill=NAVY, font=_fnt(F_SECTION, True))
    y1 += F_SECTION + 12

    col_ws = [int(CW * p) for p in [0.36, 0.21, 0.17, 0.26]]
    hdrs   = ["Criterion", "Raw (0–1)", "Weight", "Contribution"]
    hx = MG
    for txt, cw in zip(hdrs, col_ws):
        _draw_rounded(d1, hx, y1, cw - 4, F_ROW_H, NAVY, NAVY, 3)
        d1.text((hx + 8, y1 + (F_ROW_H - F_TH) // 2), txt,
                fill=WHITE, font=_fnt(F_TH, True))
        hx += cw
    y1 += F_ROW_H + 2

    for ci, c in enumerate(criteria):
        dtl  = details.get(c, {})
        raw  = dtl.get("raw")
        excl = dtl.get("excluded", False)
        row  = [
            c,
            "excluded" if excl else (f"{raw:.4f}" if raw is not None else "—"),
            "0%"       if excl else f"{(dtl.get('weight') or 0) * 100:.0f}%",
            "—"        if excl else f"{dtl.get('weighted') or 0:.4f}",
        ]
        bg_c = LIGHT if ci % 2 == 0 else LGREY
        hx = MG
        for txt, cw in zip(row, col_ws):
            _draw_rounded(d1, hx, y1, cw - 4, F_ROW_H, bg_c, SEP, 3)
            d1.text((hx + 8, y1 + (F_ROW_H - F_TD) // 2), txt,
                    fill=NAVY, font=_fnt(F_TD))
            hx += cw
        y1 += F_ROW_H + 2
    y1 += 18

    d1.line([(MG, y1), (PW - MG, y1)], fill=SEP, width=2)
    y1 += 16

    # Starting materials grid (up to 5)
    sm_list = sub["to_buy"]
    if sm_list:
        d1.text((MG, y1), "Starting materials", fill=NAVY, font=_fnt(F_SECTION, True))
        y1 += F_SECTION + 12
        n_sm = min(len(sm_list), 5)
        sm_w = min(int(CW / n_sm) - 12, int(1.2 * DPI))
        sm_h = int(sm_w * 0.70)
        # SMILES labels may wrap — measure max lines to reserve space
        max_smi_lines = 1
        for smi in sm_list[:n_sm]:
            lines = _wrap_smiles(d1, smi, sm_w, _fnt(F_SMILES))
            max_smi_lines = max(max_smi_lines, len(lines))
        smi_block_h = max_smi_lines * (F_SMILES + 4)

        sx = MG
        for smi in sm_list[:n_sm]:
            img = _mol_pil(smi, sm_w, sm_h)
            if img:
                p1.paste(img, (sx, y1))
            # Full SMILES, wrapped to fit within molecule column width
            smi_lines = _wrap_smiles(d1, smi, sm_w, _fnt(F_SMILES))
            _draw_text_lines(d1, smi_lines, sx, y1 + sm_h + 4,
                             _fnt(F_SMILES), GREY)
            sx += sm_w + 12
        y1 += sm_h + smi_block_h + 12

    pages.append(p1)

    # =========================================================================
    # PAGES 2+ — Step-by-step
    # =========================================================================
    def _render_step_at(draw, page, step, x0, y0, avail_w, avail_h):
        """
        Draw one reaction step in a rectangular slot on a page.

        Layout (top → bottom):
          1. Coloured header band — full reaction type (word-wrapped if long),
             yield right-aligned on the first line.
          2. Light conditions bar — full conditions text, word-wrapped.
          3. Molecule row — reactants + "+" + arrow + product, centred.
             Full SMILES labels (character-wrapped) below each image in the
             same GREY colour and F_SMILES font for reactants and product.

        Nothing is ever truncated.
        """
        L = x0 + 6
        W = avail_w - 12

        snum  = step.get("step_number", "?")
        rtype = step.get("reaction_type", "—") or "—"
        yld   = step.get("yield_percent")
        src_s = step.get("source", "dataset")
        cond  = step.get("conditions", {}) or {}
        reac  = step.get("reactants_smiles", [])
        prod  = step.get("product_smiles", "")

        hdr_c = ORANGE if src_s == "rxn-insight" else NAVY
        yld_s = (f"{yld}%" if yld is not None else
                 "not shown" if src_s == "rxn-insight" else "—")

        # Conditions string (full, no truncation)
        cond_parts = []
        if cond.get("temperature_C"):
            cond_parts.append(f"{cond['temperature_C']}°C")
        if cond.get("solvent"):
            cond_parts.append(cond["solvent"])
        rr = cond.get("reagents", []) or []
        if rr:
            cond_parts.append(", ".join(str(r) for r in rr))
        cond_s = "  ·  ".join(cond_parts) or "—"

        # ── Measure how many lines the reaction type needs in the header ──────
        # Reserve right side for yield text
        yld_txt  = f"Yield: {yld_s}"
        yld_w    = _text_w(draw, yld_txt, _fnt(F_YIELD))
        hdr_text_w = W - yld_w - 20   # width available for "Step N · rtype"
        step_prefix = f"Step {snum}  ·  "
        prefix_w    = _text_w(draw, step_prefix, _fnt(F_STEP_HDR, True))
        rtype_w     = hdr_text_w - prefix_w

        if rtype_w > 20:
            rtype_lines = _wrap_words(draw, rtype, rtype_w, _fnt(F_STEP_HDR, True))
        else:
            # Not enough room beside yield — wrap full width below prefix
            rtype_lines = _wrap_words(draw, rtype, W, _fnt(F_STEP_HDR, True))

        n_hdr_lines = max(len(rtype_lines), 1)
        HDR_H  = n_hdr_lines * (F_STEP_HDR + 6) + 20
        HDR_H  = max(HDR_H, F_STEP_HDR + F_YIELD + 28)

        # ── Measure conditions lines ──────────────────────────────────────────
        cond_lines = _wrap_words(draw, "Conditions: " + cond_s, W - 16, _fnt(F_COND))
        n_cond     = len(cond_lines)
        COND_H     = n_cond * (F_COND + 4) + 16
        COND_H     = max(COND_H, F_COND + 24)

        # ── Draw header band ──────────────────────────────────────────────────
        _draw_band(draw, y0, HDR_H, hdr_c)

        # Step + reaction type (left side)
        hx_txt = L
        hy_txt = y0 + 10
        draw.text((hx_txt, hy_txt), step_prefix,
                  fill=WHITE, font=_fnt(F_STEP_HDR, True))
        hx_rtype = hx_txt + prefix_w
        for i, ln in enumerate(rtype_lines):
            draw.text((hx_rtype, hy_txt + i * (F_STEP_HDR + 6)),
                      ln, fill=WHITE, font=_fnt(F_STEP_HDR, True))

        # Yield (right side, top line)
        draw.text((x0 + avail_w - yld_w - 12, y0 + 12),
                  yld_txt, fill=(200, 215, 230), font=_fnt(F_YIELD))

        y = y0 + HDR_H

        # ── Draw conditions bar ───────────────────────────────────────────────
        _draw_rounded(draw, L, y, W, COND_H - 4, LIGHT, SEP, 3)
        cy = y + 8
        for cl in cond_lines:
            draw.text((L + 8, cy), cl, fill=NAVY, font=_fnt(F_COND))
            cy += F_COND + 4
        y += COND_H

        # ── Molecule row ──────────────────────────────────────────────────────
        # How many SMILES lines does each molecule label need?
        # We use the same max_px = mol_w for all labels.
        # First pass: compute mol dimensions.
        mol_zone = avail_h - HDR_H - COND_H - 16

        n_reac_shown = max(min(len(reac), 3), 1)
        PLUS_W = 32   # gap for "+" between reactants
        ARW_W  = 64   # gap for → arrow

        n_mols_total = n_reac_shown + 1
        total_gap    = max(0, n_reac_shown - 1) * PLUS_W + ARW_W
        mol_w = max(50, (W - total_gap) // n_mols_total)
        mol_w = min(mol_w, int(mol_zone * 1.5))

        # Reserve space for SMILES lines — measure worst case
        all_smiles = [rsmi for rsmi in reac[:n_reac_shown]] + [prod]
        max_smi_lines = 1
        for s in all_smiles:
            if s:
                lns = _wrap_smiles(draw, s, mol_w, _fnt(F_SMILES))
                max_smi_lines = max(max_smi_lines, len(lns))
        smi_block_h = max_smi_lines * (F_SMILES + 4)

        mol_h = max(50, mol_zone - smi_block_h - 12)
        mol_h = min(mol_h, int(mol_w * 0.72))
        mol_w = min(mol_w, int(mol_h * 1.5))

        # Recompute row width with final mol_w and centre
        row_w = n_reac_shown * mol_w + max(0, n_reac_shown - 1) * PLUS_W + ARW_W + mol_w
        mx    = x0 + (avail_w - row_w) // 2
        my    = y + 6

        # Draw each reactant
        for i, rsmi in enumerate(reac[:n_reac_shown]):
            img = _mol_pil(rsmi, mol_w, mol_h)
            if img:
                page.paste(img, (mx, my))

            # Full SMILES, character-wrapped to mol_w
            smi_lines = _wrap_smiles(draw, rsmi, mol_w, _fnt(F_SMILES))
            _draw_text_lines(draw, smi_lines, mx, my + mol_h + 4,
                             _fnt(F_SMILES), GREY)

            mx += mol_w

            if i < n_reac_shown - 1:
                plus_lbl = "+"
                pw = _text_w(draw, plus_lbl, _fnt(F_STEP_HDR, True))
                draw.text(
                    (mx + (PLUS_W - pw) // 2,
                     my + mol_h // 2 - F_STEP_HDR // 2),
                    plus_lbl, fill=NAVY, font=_fnt(F_STEP_HDR, True))
                mx += PLUS_W

        # Arrow →
        ay  = my + mol_h // 2
        ax0 = mx + 4
        ax1 = mx + ARW_W - 12
        draw.line([(ax0, ay), (ax1, ay)], fill=NAVY, width=4)
        draw.polygon([(ax1, ay - 8), (ax1 + 10, ay), (ax1, ay + 8)], fill=NAVY)
        mx += ARW_W

        # Product — identical label style as reactants
        p_img = _mol_pil(prod, mol_w, mol_h)
        if p_img:
            page.paste(p_img, (mx, my))

        prod_lines = _wrap_smiles(draw, prod, mol_w, _fnt(F_SMILES))
        _draw_text_lines(draw, prod_lines, mx, my + mol_h + 4,
                         _fnt(F_SMILES), GREY)

        # Separator at bottom of slot
        draw.line([(x0, y0 + avail_h - 2), (x0 + avail_w, y0 + avail_h - 2)],
                  fill=SEP, width=1)

    # ── Paginate steps ────────────────────────────────────────────────────────
    STEPS_PER_PAGE = 3
    GAP_H  = int(0.12 * DPI)
    STEP_H = (PH - 2 * MG - GAP_H * (STEPS_PER_PAGE - 1)) // STEPS_PER_PAGE

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
            y0 = MG + slot * (STEP_H + GAP_H)
            _render_step_at(drw, pg, step, MG, y0, CW, STEP_H)
        pages.append(pg)

    # ── Serialise to multi-page PDF ───────────────────────────────────────────
    buf = _io.BytesIO()
    if pages:
        pages[0].save(buf, format="PDF", save_all=True,
                      append_images=pages[1:], resolution=DPI)
    return buf.getvalue()