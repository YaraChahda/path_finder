import json, glob, pandas as pd
from rdkit import Chem

PARQUET_GLOB = "data/orderly/uspto/extracted_ords/*.parquet"
OUTPUT_JSON = "data/generic_reactions.json"
MAX_REACTIONS = 10000

def canonical(smi):
    if not smi or not isinstance(smi, str) or smi.strip() == "":
        return None
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        return Chem.MolToSmiles(mol) if mol else None
    except:
        return None

def safe_float(val):
    try:
        v = float(val)
        return v if 0.0 < v <= 100.0 else None
    except:
        return None

def safe_str(val):
    if val is None: return None
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none", "", "<missing>", "missing") else None

files = sorted(glob.glob(PARQUET_GLOB))
print(f"Found {len(files)} parquet files")
if not files:
    print("ERROR: no parquet files found — check the path")
    exit(1)

df0 = pd.read_parquet(files[0])
print(f"Sample columns: {list(df0.columns)}")

reactions, skipped = [], 0

for fpath in files:
    if len(reactions) >= MAX_REACTIONS:
        break
    try:
        df = pd.read_parquet(fpath)
    except Exception as e:
        print(f"  skip {fpath}: {e}")
        continue

    cols = df.columns.tolist()
    reactant_cols = sorted([c for c in cols if c.startswith("reactant_")])
    product_cols = sorted([c for c in cols if c.startswith("product_")])
    solvent_cols = sorted([c for c in cols if c.startswith("solvent_")])
    agent_cols = sorted([c for c in cols if c.startswith("agent_") or
                            c.startswith("reagent_") or c.startswith("catalyst_")])
    yield_col = next((c for c in cols if "yield" in c.lower()), None)
    rxn_col = next((c for c in cols if any(k in c.lower() for k in
                          ["rxn_class","reaction_type","name_action","rxn_str"])), None)

    for _, row in df.iterrows():
        if len(reactions) >= MAX_REACTIONS:
            break
        reacs = [c for c in (canonical(str(row.get(rc, ""))) for rc in reactant_cols) if c]
        if not reacs:
            skipped += 1; continue
        prod = next((canonical(str(row.get(pc, ""))) for pc in product_cols
                     if canonical(str(row.get(pc, "")))), None)
        if not prod:
            skipped += 1; continue
        if len(reacs) == 1 and reacs[0] == prod:
            skipped += 1; continue
        reactions.append({
            "id": f"ORD-{len(reactions):07d}",
            "route_id": f"ORD-{len(reactions):07d}",
            "step_number": 1,
            "reactants_smiles": reacs,
            "product_smiles": prod,
            "yield_percent": safe_float(row.get(yield_col)) if yield_col else None,
            "reaction_type": safe_str(row.get(rxn_col)) if rxn_col else "unknown",
            "source": "ORD-USPTO",
            "conditions": {
                "temperature_C": None,
                "solvent": safe_str(row.get(solvent_cols[0])) if solvent_cols else None,
                "co_solvent": safe_str(row.get(solvent_cols[1])) if len(solvent_cols) > 1 else None,
                "reagents": [r for r in (safe_str(row.get(ac)) for ac in agent_cols) if r],
                "apparatus": None,
            },
        })

print(f"Converted: {len(reactions)}   Skipped: {skipped}")
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(reactions, f, indent=2, ensure_ascii=False)
print(f"Written: {OUTPUT_JSON}")
