# molecule_rendering.py
# =============================================================================
# RDKit-based molecule image rendering for the Retrosynthesis Interface.
#
# This module has No Streamlit dependency and can be imported independently
# for testing or reuse outside the app.
#
# Public API:
#   mol_png(smiles, w, h)              → bytes | None
#       High-resolution PNG for st.image() calls throughout the app.
#       Draws at 4× the requested pixel size via Cairo.
#
#   mol_b64_or_text_svg(smiles, w, h)  → str
#       Base64 PNG data-URI for embedding inside self-contained HTML schemes.
#       Draws at 8× then Lanczos-downsamples to 2× for maximum sharpness.
#
#   fallback_data_uri(text, w, h)      → str
#       Grey placeholder rectangle with a text label; returned when RDKit
#       cannot parse the SMILES so HTML <img> tags never show broken icons.
#
#   is_trivial_smiles(smiles)          → bool
#       Returns True for single atoms, ions, or salt mixtures with ≤ 2 heavy
#       atoms per fragment — used to decide whether to draw or label a species.
#
# Resolution strategy (why 4× / 8×):
#   cairo renders vector-quality lines at any integer scale.  Drawing at 4×
#   (mol_png) or 8× (scheme images) then displaying at 1× or 2× CSS pixels
#   produces crisp bond lines on HiDPI / Retina screens without blurring.
#
# Dependencies:
#   - RDKit  (rdkit-pypi or rdkit conda package)
#   - Pillow (PIL) — only for _mol_b64_or_text_svg and _fallback_data_uri
# =============================================================================

import io
import base64

# -- RDKit imports with graceful degradation ----------------------------------
# If RDKit is unavailable all functions return None / a tiny placeholder URI
# so the app can still start and show error messages rather than crashing.
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    MODULE_OK = True
except Exception:
    MODULE_OK = False


# =============================================================================
# Public rendering functions
# =============================================================================

def mol_png(smiles: str, w: int = 800, h: int = 540) -> bytes | None:
    """
    Render a molecule as a high-resolution PNG using RDKit Cairo.

    Draws at 8× the requested pixel size and returns raw PNG bytes.
    Passing the result to ``st.image()`` displays at the correct size
    because Streamlit scales by CSS width, not pixel count.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule to render.
    w : int, optional
        Display width in pixels (default 800). Cairo draws at 8×.
    h : int, optional
        Display height in pixels (default 540).

    Returns
    -------
    bytes or None
        Raw PNG data, or ``None`` if the SMILES is invalid or RDKit
        is unavailable.

    Notes
    -----
    Used by the target molecule preview, Dataset Explorer step columns,
    substances-needed expander, and starting-material browser.
    """
    if not MODULE_OK or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        # Draw at 8× the target dimensions for high-DPI sharpness
        drawer = rdMolDraw2D.MolDraw2DCairo(w * 8, h * 8)
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = True
        opts.bondLineWidth = 2.5
        opts.padding = 0.14
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        # Fallback to PIL-based rendering if Cairo is unavailable
        img = Draw.MolToImage(mol, size=(w * 8, h * 8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def mol_b64_or_text_svg(smiles: str, w: int, h: int) -> str:
    """
    Render a molecule as a base64-encoded PNG data-URI for HTML embedding.

    Draws at 8× the requested display size via Cairo, then keeps the full
    8× resolution so the browser can display at CSS width while retaining
    full resolution for popup zoom.

    Parameters
    ----------
    smiles : str
        SMILES of the molecule to render.
    w : int
        Target display width in CSS pixels.
    h : int
        Target display height in CSS pixels.

    Returns
    -------
    str
        A ``data:image/png;base64,...`` URI ready for ``<img src="...">``.
        Falls back to ``fallback_data_uri()`` if the SMILES cannot be parsed.

    Notes
    -----
    Used exclusively inside ``build_clickable_scheme_html()`` for both the
    main molecule sequence and co-reactant images above arrows.
    Single atoms and ions skip ``Compute2DCoords`` to avoid RDKit errors.
    """
    if not smiles or not MODULE_OK:
        return fallback_data_uri(smiles or "?", w, h)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return fallback_data_uri(smiles, w, h)
    try:
        from PIL import Image as _PI
        import io as _io
        # Only compute 2-D coordinates for multi-atom molecules
        if mol.GetNumAtoms() > 1:
            rdDepictor.Compute2DCoords(mol)
        # Draw at 8× the display size for crisp rendering at any DPI
        drawer = rdMolDraw2D.MolDraw2DCairo(w * 8, h * 8)
        drawer.drawOptions().bondLineWidth       = 1.0
        drawer.drawOptions().padding             = 0.15
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        data = drawer.GetDrawingText()
        # Keep full 8× resolution — the HTML img tag uses CSS width/height
        # so the browser displays at the correct size while retaining full
        # resolution for popup zoom.
        buf = _io.BytesIO(data)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return "data:image/png;base64," + b64
    except Exception:
        return fallback_data_uri(smiles, w, h)


def fallback_data_uri(text: str, w: int, h: int) -> str:
    """
    Generate a grey placeholder PNG data-URI with a centred text label.

    Ensures every ``<img>`` tag in the reaction scheme has a valid ``src``
    so the browser never shows a broken-image icon.

    Parameters
    ----------
    text : str
        Label to display, truncated to 18 characters with an ellipsis.
    w : int
        Image width in pixels.
    h : int
        Image height in pixels.

    Returns
    -------
    str
        A ``data:image/png;base64,...`` URI. Falls back to a 1×1 transparent
        PNG if Pillow is unavailable.
    """
    try:
        from PIL import Image as _PI, ImageDraw as _PID
        import io as _io
        img  = _PI.new("RGB", (w, h), (248, 248, 248))
        draw = _PID.Draw(img)
        display = text[:18] + "…" if len(text) > 18 else text
        draw.rectangle([2, 2, w - 3, h - 3], outline=(200, 200, 200))
        try:
            bbox = draw.textbbox((0, 0), display)
            tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        except Exception:
            tw = len(display) * 6; th = 12
        draw.text(((w - tw) // 2, (h - th) // 2), display, fill=(80, 80, 80))
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return "data:image/png;base64," + b64
    except Exception:
        # Last-resort 1×1 transparent PNG — always a valid data URI
        return (
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
            "AAAADULEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )


def is_trivial_smiles(smiles: str) -> bool:
    """
    Return True for single atoms, ions, or salts with very few heavy atoms.

    Determines whether a co-reactant should be drawn as a 2-D structure
    image above the reaction arrow or shown as plain text below it.

    Parameters
    ----------
    smiles : str
        SMILES string to evaluate.

    Returns
    -------
    bool
        ``True`` if every fragment of the molecule has 2 or fewer heavy
        atoms, making it too small to be worth rendering as a structure.

    Notes
    -----
    Catches single atoms ([Pd]), ions ([Na+]), and diatomics (Cl2, H2).
    Returns ``True`` also when RDKit is unavailable or the SMILES is empty.
    """
    if not smiles or not MODULE_OK:
        return True
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True
    if mol.GetNumAtoms() <= 2:
        return True
    # Check each fragment of a salt / mixture separately
    frags = smiles.split('.')
    if all(
        Chem.MolFromSmiles(f) is not None and Chem.MolFromSmiles(f).GetNumAtoms() <= 2
        for f in frags if f
    ):
        return True
    return False
