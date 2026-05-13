# Path Finder

**AiZynthFinder · Chemistry by Design · Rxn-INSIGHT**

*Yara Chahda · Corentin Postmann · Inès Ouchen — EPFL 2025*

Retrosynthesis route finder: find, rank and compare synthesis routes for any target molecule.

---

## Installation

### Step 1 — Install RDKit (conda only)

```bash
conda install -c conda-forge rdkit
```

> RDKit cannot be installed via pip — this single conda step is required.

### Step 2 — Install Path Finder

```bash
pip install path-finder-retrosynthesis
```

With predicted routes (Rxn-INSIGHT):
```bash
pip install path-finder-retrosynthesis[predicted]
```

### Step 3 — First-time setup

```bash
path-finder-setup
```

The wizard copies the bundled datasets, creates `data/config.yml`, and tells you exactly which files to download.

### Step 4 — Download AiZynthFinder model files

Go to https://github.com/MolecularAI/aizynthfinder/releases and download:

| File | Purpose |
|------|---------|
| `uspto_model.onnx` | Expansion policy network |
| `uspto_templates.csv.gz` | Reaction templates |
| `uspto_filter_model.onnx` | Filter policy (optional) |
| `zinc_stock.hdf5` | Purchasable building blocks |

Place them in `data/aizynthfinder/`.

### Step 5 — Edit config.yml

Open `data/config.yml` and replace `/PATH/TO/AIZYNTHFINDER/` with the absolute path to your model files.

**macOS / Linux:**
```yaml
expansion:
  uspto:
    - /Users/alice/data/aizynthfinder/uspto_model.onnx
    - /Users/alice/data/aizynthfinder/uspto_templates.csv.gz
```

**Windows:**
```yaml
expansion:
  uspto:
    - C:/Users/alice/data/aizynthfinder/uspto_model.onnx
    - C:/Users/alice/data/aizynthfinder/uspto_templates.csv.gz
```

### Step 6 — Launch

```bash
path-finder
```

Open http://localhost:8501 in your browser.

---

## Quick summary

```
conda install -c conda-forge rdkit
pip install path-finder-retrosynthesis
path-finder-setup
# edit data/config.yml
path-finder
```

---

## Data files

| File | Bundled | Description |
|------|---------|-------------|
| `reaction_dataset.json` | ✅ | Curated synthesis routes |
| `toxicity_dataset.json` | ✅ | Safety scores |
| `generic_reactions.json` | ✅ | 10 000 USPTO reactions |
| `config.yml` | ❌ | Created by setup wizard |
| `data/aizynthfinder/` | ❌ | Model files — download separately |
| `uspto_rxn_insight.gzip` | ❌ | Rxn-INSIGHT database (optional) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `config.yml not found` | Run `path-finder-setup` |
| AiZynthFinder crash | Check paths in `config.yml` are absolute |
| `No routes found` | Try Galanthamine or Morphine |
| Rxn-INSIGHT disabled | `pip install path-finder-retrosynthesis[predicted]` |
| Slow search (~2 min) | Normal — AiZynthFinder MCTS is intensive |

---

## Developer setup

```bash
git clone https://github.com/YaraChahda/path_finder.git
cd path_finder
conda install -c conda-forge rdkit
pip install -e ".[predicted]"
path-finder-setup
path-finder
```

---

## Citation

- AiZynthFinder: Genheden et al., J. Cheminf. 2020
- Rxn-INSIGHT: Thakkar et al., J. Cheminf. 2023
- Open Reaction Database: Kearnes et al., JACS 2021
