![Project Logo](assets/banner.png)

![Coverage Status](assets/coverage-badge.svg)

<h1 align="center">
path_finder
</h1>

<br>

# Retrosynthesis Interface

AiZynthFinder · Chemistry by Design · Rxn-INSIGHT

## What this app does ?

This Streamlit application finds and ranks synthesis routes for a target molecule using:
- **AiZynthFinder** (MCTS retrosynthetic search)
- A curated reaction dataset (Chemistry by Design)
- **Rxn-INSIGHT** for reaction classification and condition prediction

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-org/retrosynthesis-interface.git
cd retrosynthesis-interface
```

### 2. Create and activate the conda environment
```bash
conda env create -f environment.yml
conda activate retrosynthesis
```
#### USPTO Rxn-INSIGHT database
Download `uspto_rxn_insight.gzip` from:
> [https://zenodo.org/records/10171745]

Place it at:
```
data/uspto_rxn_insight.gzip
```

#### AiZynthFinder model files
[[Download the pre-trained USPTO models from the AiZynthFinder releases page:
> https://github.com/MolecularAI/aizynthfinder/releases

You need:
- `uspto_model.onnx`
- `uspto_templates.csv.gz`

Place them anywhere on your machine (e.g. `data/aizynthfinder/`)]]

### 4. Configure AiZynthFinder

Copy the template config and edit it with your local paths:

```bash
cp data/config_template.yml data/config.yml
```

Edit `data/config.yml` and replace every path with the **absolute path** on your machine:

```yaml
expansion:
  uspto:
    - /absolute/path/to/uspto_model.onnx
    - /absolute/path/to/uspto_templates.csv.gz

stock:
  zinc:
    - /absolute/path/to/zinc_stock.hdf5

filter:
  uspto:
    - /absolute/path/to/uspto_filter_model.onnx
```

> **Important:** use absolute paths (starting with `/` on macOS/Linux or `C:\` on Windows).
> Relative paths cause silent failures in AiZynthFinder.

You can find the correct paths for the stock and filter files in the AiZynthFinder documentation:
> https://molecularai.github.io/aizynthfinder/

### 5. Run the app

```bash
streamlit run app/app_path_finder.py
```

The app will open at `http://localhost:8501`.

## File structure

```
retrosynthesis-interface/
├── app/
│   ├── app_path_finder.py      # Streamlit front-end
│   ├── route_engine.py         # Backend — scoring, AiZ, Rxn-INSIGHT
│   ├── molecule_rendering.py   # RDKit Cairo rendering
│   ├── localization.py         # EN/FR UI strings
│   └── report_builder.py       # PDF generation
├── data/
│   ├── reaction_dataset.json   # Main curated routes (included)
│   ├── toxicity_dataset.json   # Safety scores (included)
│   ├── generic_reactions.json  # Individual reactions (included)
│   ├── config_template.yml     # AiZ config template (edit → config.yml)
│   └── config.yml              # ← YOUR local config (not committed to git)
├── assets/
│   └── banner.png
├── environment.yml
└── README.md
```
## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: config.yml` | Check that `data/config.yml` exists and all paths inside are absolute |
| AiZynthFinder hangs or crashes | Verify the `.onnx` and `.csv.gz` paths in `config.yml` are correct |
| `No routes found` | The target SMILES may not match the dataset; try Galanthamine or Morphine |
| Rxn-INSIGHT disabled | `uspto_rxn_insight.gzip` must be present at `data/` and `rxn_insight` installed |
| Slow search (~2 min) | Normal — AiZynthFinder MCTS is computationally intensive |




This package aims to find the best retro synthesis pathways for some drugs, based on criteria selected by the user.

## 🔥 Usage

```python
from mypackage import main_func

# One line to rule them all
result = main_func(data)
```

This usage example shows how to quickly leverage the package's main functionality with just one line of code (or a few lines of code). 
After importing the `main_func` (to be renamed by you), you simply pass in your `data` and get the `result` (this is just an example, your package might have other inputs and outputs). 
Short and sweet, but the real power lies in the detailed documentation.

## 👩‍💻 Installation

Create a new environment, you may also give the environment a different name. 

```
conda create -n path_finder python=3.10 
```

```
conda activate path_finder
(conda_env) $ pip install .
```

If you need jupyter lab, install it 

```
(path_finder) $ pip install jupyterlab
```


## 🛠️ Development installation

Initialize Git (only for the first time). 

Note: You should have create an empty repository on `https://github.com:YaraChahda/path_finder`.

```
git init
git add * 
git add .*
git commit -m "Initial commit" 
git branch -M main
git remote add origin git@github.com:YaraChahda/path_finder.git 
git push -u origin main
```

Then add and commit changes as usual. 

To install the package, run

```
(path_finder) $ pip install -e ".[test,doc]"
```

### Run tests and coverage

```
(conda_env) $ pip install tox
(conda_env) $ tox
```



