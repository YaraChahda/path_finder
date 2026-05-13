"""
path_finder/cli.py
──────────────────
Console entry points:
    path-finder          → launches the Streamlit app
    path-finder-setup    → interactive setup wizard

These commands are available after `pip install path-finder-retrosynthesis`.
"""
import subprocess
import sys
import shutil
from pathlib import Path


def _pkg_root() -> Path:
    """Absolute path to the installed path_finder package directory."""
    return Path(__file__).parent


def _data_dir() -> Path:
    """
    Working data directory — where config.yml and model files live.
    Defaults to ./data relative to the current working directory.
    Override with the PATH_FINDER_DATA environment variable.
    """
    import os
    return Path(os.environ.get("PATH_FINDER_DATA", "data"))


def main():
    """
    Launch the Streamlit app.
    Console script: path-finder
    """
    app = _pkg_root() / "app_path_finder.py"
    if not app.exists():
        print(f"[ERROR] App not found at {app}")
        print("Try reinstalling: pip install --force-reinstall path-finder-retrosynthesis")
        sys.exit(1)

    data = _data_dir()
    if not (data / "config.yml").exists():
        print("⚠️  config.yml not found.")
        print(f"   Run `path-finder-setup` first, then edit {data / 'config.yml'}\n")

    print("Launching Path Finder — Retrosynthesis Interface")
    print("Open http://localhost:8501 in your browser.\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app)],
        check=True,
    )


def setup():
    """
    Interactive setup wizard.
    Console script: path-finder-setup

    What it does:
    1. Checks RDKit / AiZynthFinder / Rxn-INSIGHT are installed
    2. Creates the working data/ directory
    3. Copies bundled datasets from the installed package into data/
    4. Copies config_template.yml → data/config.yml
    5. Guides the user to fill in AiZynthFinder model paths
    6. Validates the installation
    """
    print("\n" + "═" * 62)
    print("  Path Finder — Setup Wizard  v1.0.0")
    print("  Yara Chahda · Corentin Postmann · Inès Ouchen")
    print("═" * 62 + "\n")

    pkg  = _pkg_root()
    data = _data_dir()
    data.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {data.resolve()}\n")

    # ── Step 1: Check dependencies ─────────────────────────────────────────────
    print("Step 1/5 — Checking dependencies…")
    try:
        from rdkit import Chem  # noqa: F401
        print("  ✓ RDKit")
    except ImportError:
        print("  ✗ RDKit not found")
        print("    → conda install -c conda-forge rdkit")
        print("    → Then rerun: path-finder-setup")

    try:
        from aizynthfinder.aizynthfinder import AiZynthFinder  # noqa: F401
        print("  ✓ AiZynthFinder")
    except ImportError:
        print("  ✗ AiZynthFinder — pip install aizynthfinder")

    try:
        from rxn_insight.reaction import Reaction  # noqa: F401
        print("  ✓ Rxn-INSIGHT  (predicted routes enabled)")
    except ImportError:
        print("  ✗ Rxn-INSIGHT not installed  (predicted routes disabled)")
        print("    → pip install path-finder-retrosynthesis[predicted]")

    # ── Step 2: Copy bundled datasets into data/ ───────────────────────────────
    print("\nStep 2/5 — Copying bundled datasets…")
    for fname in ["reaction_dataset.json", "toxicity_dataset.json", "generic_reactions.json"]:
        src  = pkg / "data" / fname
        dest = data / fname
        if dest.exists():
            print(f"  ✓ {fname} already present — skipped")
        elif src.exists():
            shutil.copy2(src, dest)
            print(f"  ✓ {fname} copied")
        else:
            print(f"  ✗ {fname} missing from package — try reinstalling")

    # ── Step 3: Guide AiZynthFinder model download ────────────────────────────
    print("\nStep 3/5 — AiZynthFinder model files")
    print("  These files are too large to bundle — download them manually.")
    print("  URL: https://github.com/MolecularAI/aizynthfinder/releases\n")
    print("  Files needed:")
    print("    • uspto_model.onnx")
    print("    • uspto_templates.csv.gz")
    print("    • uspto_filter_model.onnx  (optional)")
    print("    • zinc_stock.hdf5")
    aiz_dir = data / "aizynthfinder"
    aiz_dir.mkdir(exist_ok=True)
    print(f"\n  Place them in: {aiz_dir.resolve()}")

    # ── Step 4: Create config.yml ──────────────────────────────────────────────
    print("\nStep 4/5 — AiZynthFinder config…")
    config_out  = data / "config.yml"
    config_tmpl = pkg / "data" / "config_template.yml"

    if config_out.exists():
        ans = input("  config.yml already exists. Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("  Kept existing config.yml")
        else:
            _write_config(config_tmpl, config_out, aiz_dir)
    else:
        _write_config(config_tmpl, config_out, aiz_dir)

    print(f"\n  ⚠️  Open {config_out.resolve()}")
    print("     Replace /PATH/TO/AIZYNTHFINDER/ with the absolute path")
    print("     to the folder where you placed the model files.")

    # ── Step 5: Validate ───────────────────────────────────────────────────────
    print("\nStep 5/5 — Validation…")
    checks = [
        (data / "reaction_dataset.json",  "Main reaction dataset"),
        (data / "toxicity_dataset.json",  "Toxicity dataset"),
        (data / "generic_reactions.json", "Generic reactions (USPTO)"),
        (data / "config.yml",             "AiZynthFinder config"),
    ]
    all_ok = True
    for path, label in checks:
        if path.exists():
            print(f"  ✓ {label}")
        else:
            print(f"  ✗ {label} — missing")
            all_ok = False

    print()
    if all_ok:
        print("✅ Setup complete!\n")
        print("Next steps:")
        print(f"  1. Edit {config_out.resolve()}")
        print("     Replace /PATH/TO/AIZYNTHFINDER/ with your model folder path.")
        print("  2. Launch the app:\n\n     path-finder\n")
    else:
        print("⚠️  Setup incomplete. Fix the issues above then run path-finder-setup again.\n")


def _write_config(template: Path, output: Path, aiz_dir: Path) -> None:
    if template.exists():
        content = template.read_text()
    else:
        content = _fallback_template()
    content = content.replace("/PATH/TO/AIZYNTHFINDER/", str(aiz_dir) + "/")
    output.write_text(content)
    print(f"  ✓ config.yml written to {output}")


def _fallback_template() -> str:
    return """\
# AiZynthFinder configuration — generated by path-finder-setup
# Replace /PATH/TO/AIZYNTHFINDER/ with the folder containing your model files.

expansion:
  uspto:
    - /PATH/TO/AIZYNTHFINDER/uspto_model.onnx
    - /PATH/TO/AIZYNTHFINDER/uspto_templates.csv.gz

filter:
  uspto:
    - /PATH/TO/AIZYNTHFINDER/uspto_filter_model.onnx

stock:
  zinc:
    - /PATH/TO/AIZYNTHFINDER/zinc_stock.hdf5

search:
  algorithm: mcts
  max_transforms: 6
  iteration_limit: 100
  return_first: false
"""
