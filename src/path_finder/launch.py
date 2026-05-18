"""
Entry points for the path-finder CLI.
Registered in pyproject.toml under [project.scripts].
"""

import subprocess
import sys
import shutil
from pathlib import Path


def _pkg_root() -> Path:
    return Path(__file__).parent


def _data_dir() -> Path:
    import os
    return Path(os.environ.get("PATH_FINDER_DATA", "data"))


def main():
    """Start the Streamlit app (called by the path-finder command)."""
    app = _pkg_root() / "app.py"
    if not app.exists():
        print(f"[ERROR] App not found at {app}")
        print("Try reinstalling: pip install --force-reinstall path-finder-retrosynthesis")
        sys.exit(1)

    data = _data_dir()
    if not (data / "config.yml").exists():
        print("WARNING: config.yml not found — run `path-finder-setup` first.\n")

    print("\nPath Finder — Retrosynthesis Interface")
    print("AiZynthFinder · Rxn-INSIGHT · Chemistry by Design")
    print("Yara Chahda · Corentin Postmann · Inès Ouchen Laksiri — EPFL 2026")
    print("\nOpen http://localhost:8501 in your browser.\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app)],
        check=True,
    )


def setup():
    """
    First-time setup wizard (called by path-finder-setup).
    Checks dependencies, copies bundled data, downloads AiZ models, creates config.yml.
    """
    print("\n" + "=" * 62)
    print("  Path Finder - Setup Wizard  v1.0.1")
    print("  Yara Chahda - Corentin Portmann - Ines Ouchen - EPFL 2025")
    print("=" * 62 + "\n")

    pkg  = _pkg_root()
    data = _data_dir()
    data.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {data.resolve()}\n")

    # Step 1: Check dependencies
    print("Step 1/5 - Checking dependencies...")

    try:
        from rdkit import Chem  # noqa: F401
        print("  OK RDKit")
    except ImportError:
        print("  MISSING RDKit")
        print("    -> conda install -c conda-forge rdkit")
        print("    -> Then rerun: path-finder-setup\n")

    aiz_ok = False
    try:
        from aizynthfinder.aizynthfinder import AiZynthFinder  # noqa: F401
        print("  OK AiZynthFinder")
        aiz_ok = True
    except ImportError:
        print("  MISSING AiZynthFinder")
        print("    -> pip install aizynthfinder")

    try:
        from rxn_insight.reaction import Reaction  # noqa: F401
        print("  OK Rxn-INSIGHT (predicted routes enabled)")
    except ImportError:
        print("  MISSING Rxn-INSIGHT (predicted routes disabled)")
        print("    -> pip install path-finder-retrosynthesis[predicted]")

    # Step 2: Copy bundled datasets
    print("\nStep 2/5 - Copying bundled datasets...")
    for fname in [
        "reaction_dataset.json",
        "toxicity_dataset.json",
        "generic_reactions.json",
    ]:
        src  = pkg / "data" / fname
        dest = data / fname
        if dest.exists():
            print(f"  OK {fname} already present - skipped")
        elif src.exists():
            shutil.copy2(src, dest)
            print(f"  OK {fname} copied")
        else:
            print(f"  MISSING {fname} - try reinstalling the package")

    # Step 3: Download AiZynthFinder models
    print("\nStep 3/5 - Downloading AiZynthFinder model files...")
    aiz_dir = data / "aizynthfinder"
    aiz_dir.mkdir(exist_ok=True)
    config_from_aiz = False

    if not aiz_ok:
        print("  SKIPPED - AiZynthFinder not installed")
        print("    -> pip install aizynthfinder && path-finder-setup")
    else:
        model_files = list(aiz_dir.glob("*.onnx")) + list(aiz_dir.glob("*.hdf5"))
        if model_files:
            print(f"  OK Models already present ({len(model_files)} files) - skipped")
            config_from_aiz = (aiz_dir / "config.yml").exists()
        else:
            print(f"  Downloading to {aiz_dir}  (~500 MB, please wait...)")
            downloaded = False

            # Try the standard module path (AiZynthFinder >= 4.x)
            try:
                subprocess.run(
                    [sys.executable, "-m",
                     "aizynthfinder.tools.download_public_data",
                     str(aiz_dir)],
                    check=True,
                )
                downloaded = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            # Fallback: older API
            if not downloaded:
                try:
                    subprocess.run(
                        [sys.executable, "-c",
                         "from aizynthfinder.tools.download_public_data import main; "
                         f"main(['{aiz_dir}'])"],
                        check=True,
                    )
                    downloaded = True
                except Exception:
                    pass

            if downloaded:
                print("  OK Models downloaded successfully")
                config_from_aiz = (aiz_dir / "config.yml").exists()
            else:
                print("  FAILED - download did not complete")
                print("  -> Download manually from:")
                print("     https://github.com/MolecularAI/aizynthfinder/releases")
                print(f"  -> Place the files in: {aiz_dir.resolve()}")

    # Step 4: Create config.yml
    print("\nStep 4/5 - Creating config.yml...")
    config_out = data / "config.yml"

    if config_out.exists():
        print(f"  OK config.yml already exists - skipped")
    elif config_from_aiz:
        shutil.copy2(aiz_dir / "config.yml", config_out)
        print("  OK config.yml copied from AiZynthFinder download")
        print("  OK No manual editing required - paths set automatically")
    else:
        _write_config_from_template(pkg, config_out, aiz_dir)
        print(f"\n  ATTENTION: open {config_out.resolve()}")
        print("  Replace /PATH/TO/AIZYNTHFINDER/ with:")
        print(f"  {aiz_dir.resolve()}/")

    # Step 5: Validate
    print("\nStep 5/5 - Validation...")
    checks = [
        (data / "reaction_dataset.json",  "Main reaction dataset"),
        (data / "toxicity_dataset.json",  "Toxicity dataset"),
        (data / "generic_reactions.json", "Generic reactions (USPTO)"),
        (data / "config.yml",             "AiZynthFinder config"),
    ]
    all_ok = True
    for path, label in checks:
        if path.exists():
            print(f"  OK {label}")
        else:
            print(f"  MISSING {label}")
            all_ok = False

    model_files = list(aiz_dir.glob("*.onnx")) + list(aiz_dir.glob("*.hdf5"))
    if model_files:
        print(f"  OK AiZynthFinder models ({len(model_files)} files)")
    else:
        print(f"  MISSING AiZynthFinder models in {aiz_dir}")
        all_ok = False

    print()
    if all_ok:
        print("Setup complete!\n")
        print("Launch the app with:\n\n    path-finder\n")
    else:
        print("Setup incomplete - fix the issues above then rerun:\n\n    path-finder-setup\n")


def _write_config_from_template(pkg: Path, output: Path, aiz_dir: Path) -> None:
    template = pkg / "data" / "config_template.yml"
    content  = template.read_text() if template.exists() else _fallback_template()
    content  = content.replace(
        "/PATH/TO/AIZYNTHFINDER/",
        str(aiz_dir.resolve()) + "/",
    )
    output.write_text(content)
    print(f"  OK config.yml written to {output}")


def _fallback_template() -> str:
    return """\
expansion:
  uspto:
    - /PATH/TO/AIZYNTHFINDER/uspto_model.onnx
    - /PATH/TO/AIZYNTHFINDER/uspto_templates.csv.gz
  ringbreaker:
    - /PATH/TO/AIZYNTHFINDER/uspto_ringbreaker_model.onnx
    - /PATH/TO/AIZYNTHFINDER/uspto_ringbreaker_templates.csv.gz
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
