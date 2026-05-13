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

    Resolution: draws at 8× the requested pixel size then returns the raw PNG
    bytes. Passing the result directly to st.image() displays it at the correct
    size because Streamlit scales by CSS width, not pixel count.

    Used by:
    • Target molecule preview (top-right of Route Search tab)
    • Dataset Explorer step-by-step reactant / product columns
    • Substances-needed expander molecule grid
    • Starting-material browser for predicted routes

    Parameters
    ──────────
    smiles : str  SMILES string of the molecule to render
    w      : int  display width in pixels (Cairo draws at 4×, default 800)
    h      : int  display height in pixels (default 540)

    Returns
    ───────
    bytes  raw PNG data, or None if SMILES is invalid or RDKit unavailable
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

    Used exclusively inside build_clickable_scheme_html() (ui_components.py)
    to embed molecule images inside the self-contained HTML reaction scheme —
    both for the main molecule sequence and for co-reactants shown above arrows.

    Resolution strategy:
    • Cairo draws at 8× the requested display size.
    • PIL Lanczos-downsamples to 2× the display size.
    • The HTML <img> tag uses the original w/h CSS values.
    → Result: 2× pixel density, equivalent to a Retina/HiDPI asset.

    Edge cases handled:
    • Single atoms and ions ([Pd], [Na+] etc.) — Compute2DCoords is skipped
      for molecules with ≤ 1 heavy atom because it raises errors.
    • Unparseable SMILES — falls back to fallback_data_uri() (grey rectangle).

    Parameters
    ──────────
    smiles : str  SMILES of the molecule to render
    w      : int  target display width in CSS pixels
    h      : int  target display height in CSS pixels

    Returns
    ───────
    str  "data:image/png;base64,..." URI ready for <img src="...">
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

    Returned by mol_b64_or_text_svg() when RDKit cannot parse the SMILES
    (e.g. ions, malformed strings, single atoms like [Pd]).
    Ensures every HTML <img> tag in the reaction scheme has a valid src so
    the browser never shows a broken-image icon.

    Parameters
    ──────────
    text : str  label to display (truncated to 18 chars with ellipsis)
    w    : int  image width in pixels
    h    : int  image height in pixels

    Returns
    ───────
    str  "data:image/png;base64,..." URI
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
    Return True for single atoms, ions, or salts where every fragment has
    2 or fewer heavy atoms.

    Used in build_clickable_scheme_html() (ui_components.py) to decide whether
    a co-reactant should be rendered as a 2-D structure image above the arrow
    or shown as abbreviated plain text below it. Single-atom or diatomic species
    (e.g. [Pd], [Na+], Cl2, H2) produce uninformative molecule images and are
    better represented as short text labels.

    Parameters
    ──────────
    smiles : str

    Returns
    ───────
    bool  True if the molecule is too small to be worth rendering as a 2-D image
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
