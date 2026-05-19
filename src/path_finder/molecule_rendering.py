# molecule_rendering.py
# RDKit rendering helpers — PNG and base64 for the app

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



# Public rendering functions


def mol_png(smiles: str, w: int = 800, h: int = 540) -> bytes | None:
    """
    Return a PNG image of the molecule for display in the route scheme.
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
    Return a data URI of the molecule image for display in the route scheme.
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
        drawer.drawOptions().bondLineWidth = 1.0
        drawer.drawOptions().padding = 0.15
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
    Return a data URI of a placeholder image with the given text.
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
