# molecule_rendering.py
# RDKit → PNG/base64 for the app. If RDKit is missing every function
# returns None or a placeholder so the app can still start.

import io
import base64

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    MODULE_OK = True
except Exception:
    MODULE_OK = False

# Public rendering functions
def mol_png(smiles: str, w: int = 800, h: int = 540) -> bytes | None:
    """PNG bytes of the molecule, or None if the SMILES is invalid."""
    if not MODULE_OK or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        # 8× so it stays sharp on HiDPI screens; CSS handles the display size
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
    """Data URI (PNG base64) of the molecule, or a text placeholder if invalid."""
    if not smiles or not MODULE_OK:
        return fallback_data_uri(smiles or "?", w, h)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return fallback_data_uri(smiles, w, h)
    try:
        from PIL import Image as _PI
        import io as _io
        # single atoms crash rdDepictor, skip them
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
        # full 8× resolution in the file; CSS width controls display size
        buf = _io.BytesIO(data)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return "data:image/png;base64," + b64
    except Exception:
        return fallback_data_uri(smiles, w, h)


def fallback_data_uri(text: str, w: int, h: int) -> str:
    """PNG placeholder with the given text. Falls back to a 1×1 transparent PNG if PIL is also missing."""
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
    """True if every fragment has <=2 heavy atoms (single atoms, ions, diatomics)."""
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