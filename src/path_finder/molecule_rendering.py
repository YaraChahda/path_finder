# molecule_rendering.py
# RDKit molecule rendering for the Path Finder app.
# No Streamlit dependency — can be imported and tested independently.
#
# mol_png()               -> bytes  : high-res PNG for st.image()
# mol_b64_or_text_svg()   -> str    : base64 PNG for embedded HTML schemes
# fallback_data_uri()     -> str    : grey placeholder when SMILES is invalid
# is_trivial_smiles()     -> bool   : True for single atoms / tiny ions
# Dependencies:
#   - RDKit  (rdkit-pypi or rdkit conda package)
#   - Pillow (PIL) — only for _mol_b64_or_text_svg and _fallback_data_uri

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
    Render a molecule as PNG using RDKit Cairo at 8x the display size.

    Parameters
    ----------
    smiles : str
        SMILES of the molecule to render.
    w, h : int
        Display size in pixels — Cairo renders at 8x for HiDPI sharpness.

    Returns
    -------
    bytes or None
        Raw PNG bytes, or None if the SMILES is invalid.
    """
    if not MODULE_OK or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        # 8x upscale — crisp on HiDPI
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
    Return a base64 PNG data-URI for embedding in the reaction scheme HTML.

    Parameters
    ----------
    smiles : str
        SMILES of the molecule.
    w, h : int
        Target display size in CSS pixels (renders at 8x internally).

    Returns
    -------
    str
        data:image/png;base64,... URI, or fallback_data_uri() if parsing fails.
    """
    if not smiles or not MODULE_OK:
        return fallback_data_uri(smiles or "?", w, h)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return fallback_data_uri(smiles, w, h)
    try:
        from PIL import Image as _PI
        import io as _io
        # skip 2D coords for single atoms, they cause rdkit errors
        if mol.GetNumAtoms() > 1:
            rdDepictor.Compute2DCoords(mol)
        # 8x upscale — crisp on HiDPI
        drawer = rdMolDraw2D.MolDraw2DCairo(w * 8, h * 8)
        drawer.drawOptions().bondLineWidth       = 1.0
        drawer.drawOptions().padding             = 0.15
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        data = drawer.GetDrawingText()
        # keep 8x resolution, CSS width handles display size
        # so the browser displays at the correct size while retaining full
        # resolution for popup zoom.
        buf = _io.BytesIO(data)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return "data:image/png;base64," + b64
    except Exception:
        return fallback_data_uri(smiles, w, h)


def fallback_data_uri(text: str, w: int, h: int) -> str:
    """
    Grey placeholder PNG with a text label, for when SMILES can't be parsed.

    Parameters
    ----------
    text : str
        Label to show (truncated to 18 chars).
    w, h : int
        Image dimensions in pixels.

    Returns
    -------
    str
        data:image/png;base64,... URI (falls back to 1x1 if Pillow unavailable).
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
    Return True if the SMILES is too small to render as a structure image
    (single atoms, ions, salts with ≤2 heavy atoms per fragment).

    Parameters
    ----------
    smiles : str
        SMILES to check.

    Returns
    -------
    bool
        True if every fragment has 2 or fewer heavy atoms.
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
