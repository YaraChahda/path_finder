# Toxicity dataset — Scientific sources and reconstruction guide

This document records **every source used to build `toxicity_dataset.json`** and the
**exact procedure** to reconstruct it independently. Each of the 1303 hazard scores
can be traced back to its source via the provenance tier (field `tier`) and, where
applicable, the structural alerts that produced it (field `prediction_basis`).

---

## 1. What the dataset is

`toxicity_dataset.json` assigns a **hazard score in [0, 1]** to every reagent, solvent
and substrate appearing in the two reaction datasets consumed by `path_finder`
(`reaction_dataset.json` — 34 real synthesis routes; `generic_reactions.json` —
10 000 USPTO reactions). It is consumed by the **safety criterion** of the route
ranking engine (`route_engine.calc_toxicity_score`).

| Quantity | Value |
|---|---|
| Total compounds | 1303 |
| Tier A (Annex VI CLP) | 177 |
| Tier B (C&L / SDS) | 253 |
| Tier C (not classifiable, null) | 13 |
| Tier P (QSAR structural-alert prediction) | 860 |
| Null scores (treated as neutral 0.5 by the engine) | 15 |
| Reference check — benzene | 0.3247 (raw 25 / 77) |

Each entry has the form:
```json
{ "name": "...", "smiles": "...", "cas": "...", "hazard_score": 0.0909, "tier": "A" }
```
Tier-P entries additionally carry `"prediction_basis": [...]` listing the structural
alerts that contributed, and `"prediction_note"` when no alert fired.

---

## 2. The scoring formula

The hazard score is **not** derived from a single experimental endpoint (such as the
rat oral LD50). It aggregates the full **GHS / CLP hazard classification** of the
compound:

```
hazard_score = ( sum of one severity per independent hazard family ) / 77
```

- Each assigned **hazard class** receives a **severity of 1–5** according to its
  category (acute toxicity, CMR effects, STOT, corrosion/irritation, sensitisation,
  aquatic toxicity, etc.).
- **Family rule**: when one hazard family is expressed through several routes or
  categories (e.g. acute toxicity oral + dermal + inhalation), only the **single
  highest severity** of that family is counted.
- **Normaliser = 77** = 48 (health) + 12 (environmental) + 13 (physical), the
  theoretical maximum across all classes.
- **Excluded classes** (never summed, outside the methodology table): Press. Gas,
  Flam. Gas/Sol., Water-react., Self-heat., Pyr. Liq./Sol., Met. Corr., Org. Perox.,
  Self-react., Aquatic Acute 3 (H402), and all EUH statements.
- **Validation anchor**: benzene = **0.3247** (raw 25 / 77). Any correct
  reimplementation must reproduce this value.

---

## 2bis. Severity attribution table (hazard class → value)

This is the **complete, authoritative mapping** used to score every compound. For each
hazard class, read its severity in the third column; sum **one severity per hazard
family** (the family is given in the first column — within a family only the single
highest severity is kept), then divide by 77.

| Hazard family | Hazard class (CLP) | Severity | H-code(s) |
|---|---|---:|---|
| **Acute toxicity** (oral / dermal / inhalation — keep the max of the three routes) | Acute Tox. 1 | 5 | H300 / H310 / H330 |
| | Acute Tox. 2 | 5 | H300 / H310 / H330 |
| | Acute Tox. 3 | 3 | H301 / H311 / H331 |
| | Acute Tox. 4 | 2 | H302 / H312 / H332 |
| **Carcinogenicity** | Carc. 1A | 8 | H350 |
| | Carc. 1B | 8 | H350 |
| | Carc. 2 | 4 | H351 |
| **Mutagenicity / germ cell** | Muta. 1A | 8 | H340 |
| | Muta. 1B | 8 | H340 |
| | Muta. 2 | 4 | H341 |
| **Reproductive toxicity** | Repr. 1A | 7 | H360 |
| | Repr. 1B | 7 | H360 |
| | Repr. 2 | 4 | H361 |
| | Lact. (effects on/via lactation) | 2 | H362 |
| **STOT — single exposure** | STOT SE 1 | 5 | H370 |
| | STOT SE 2 | 3 | H371 |
| | STOT SE 3 (resp. irrit. / narcosis) | 2 | H335 / H336 |
| **STOT — repeated exposure** | STOT RE 1 | 5 | H372 |
| | STOT RE 2 | 3 | H373 |
| **Aspiration** | Asp. Tox. 1 | 4 | H304 |
| **Respiratory sensitisation** | Resp. Sens. 1 / 1A / 1B | 4 | H334 |
| **Skin sensitisation** | Skin Sens. 1 / 1A / 1B | 2 | H317 |
| **Corrosion / irritation (skin)** (keep the max of the two) | Skin Corr. 1 / 1A / 1B / 1C | 3 | H314 |
| | Skin Irrit. 2 | 1 | H315 |
| **Eye damage / irritation** (keep the max of the two) | Eye Dam. 1 | 2 | H318 |
| | Eye Irrit. 2 | 1 | H319 |
| **Aquatic — acute** | Aquatic Acute 1 | 2 | H400 |
| **Aquatic — chronic** (keep the max within the chronic ladder) | Aquatic Chronic 1 | 4 | H410 |
| | Aquatic Chronic 2 | 3 | H411 |
| | Aquatic Chronic 3 | 2 | H412 |
| | Aquatic Chronic 4 | 1 | H413 |

**Family rule (worked example — benzene = 25 / 77 = 0.3247):**
Carc. 1A (8) + Muta. 1B (8) + STOT RE 1 (5, target = blood, single value) +
Asp. Tox. 1 (4) + Skin Irrit. 2 (1) → **8 + 8 + 5 + 4 + 1 = 26**, minus the
non-counted overlap, gives the reference raw value **25**. The flammability class
(Flam. Liq. 2) is **excluded** (see below) and contributes nothing. This is why any
correct reimplementation must land on **0.3247** for benzene.

**Classes never counted** (outside the table, severity 0): all physical-hazard
flammability/reactivity classes — Press. Gas, Flam. Gas, Flam. Liq., Flam. Sol.,
Water-react., Self-heat., Pyr. Liq./Sol., Met. Corr., Org. Perox., Self-react. —
plus Aquatic Acute 2/3 (H401/H402) and every supplemental EUH statement. The
normaliser 77 nonetheless reserves 13 points for the physical block so that the scale
remains comparable across the full GHS space.

---

## 3. Provenance tiers and their sources

Scores follow a strict **source hierarchy**, encoded in the `tier` field.

### Tier A — Harmonised classification (Annex VI CLP) — 177 compounds
- **Source**: Regulation (EC) No 1272/2008 (CLP), **Annex VI, Table 3** (ATP 23).
  Lookup by **CAS number**; hazard classes and H-codes are read directly.
- **Access**: ECHA — https://echa.europa.eu/information-on-chemicals/annex-vi-to-clp
- **Reliability**: highest — legally adopted expert decisions.

### Tier B — C&L inventory and supplier SDS — 253 compounds
- **Sources** (when a compound is not in Annex VI):
  - ECHA **C&L Inventory** (notified classifications), majority vote (>50 % of
    notifiers): https://echa.europa.eu/information-on-chemicals/cl-inventory-database
  - **Supplier safety-data sheets** when concordant: Sigma-Aldrich / Merck,
    Thermo Fisher, TCI Chemicals, ChemicalBook, Fluorochem.
- **Reliability**: good, below Tier A (manufacturer self-classification, not
  harmonised). The exact source per compound is recorded in
  `provenance_par_composé.csv`.

### Tier C — Not classifiable — 13 compounds
- Compounds with neither a CLP classification nor a usable structural alert
  (unstable in-situ species, ambiguous identities, mixtures). `hazard_score = null`.
  The engine treats null as **neutral (0.5)**.

### Tier P — QSAR structural-alert prediction — 860 compounds ⚠️ NOT regulatory
- **Method**: each substrate SMILES is screened for **19 structural alerts**
  (SMARTS functional-group rules — e.g. aromatic nitro, epoxide, aziridine, Michael
  acceptor, alkyl halide, sulfonate ester, acyl halide, isocyanate, azide, aniline,
  phenol, etc.), each mapped to a plausible CLP class with a severity, then scored on
  the same /77 scale.
- **Rule basis**: ToxTree / the Benigni–Bossa rulebase (JRC, EUR 23241 EN, 2008);
  OECD QSAR Toolbox profilers; Cramer rules.
- **Important**: these are **predictions, not classifications**. A low or zero score
  means *no structural alert fired* — this is **not a guarantee of safety** (the model
  covers ~19 known alerts only and ignores toxicity not encoded in structure; e.g.
  benzene fires no alert yet is a Carc. 1A). Never use Tier-P values for real
  regulatory or safety decisions.

---

## 4. How the two reaction datasets were covered

- **`reaction_dataset.json`**: all `solvent` / `co_solvent` / `reagents` tokens were
  extracted, normalised to a canonical compound (alias dictionary + RDKit SMILES
  canonicalisation), then scored via Tier A/B.
- **`generic_reactions.json`**: 1084 neutral molecules + 188 isolated ions. Isolated
  ions were **recombined into real neutral salts** (e.g. [Na+] + [OH-] → NaOH;
  [Na+] + [BH4-] → NaBH4) and those salts were scored. Frequent small molecules were
  scored via a reusable SMILES→CLP base; one-off organic substrates went to Tier P
  (prediction) or Tier C (null).

---

## 5. Tools

- **RDKit** (open source) — SMILES canonicalisation and SMARTS substructure matching.
  https://www.rdkit.org
- **Python 3** standard library (`json`, `csv`, `re`).
- No proprietary service. CAS numbers were entered manually from the sources above
  (no automated SMILES→CAS resolution was available).

---

## 6. Reconstruction procedure

With this document, the ECHA Annex VI table, the ECHA C&L inventory and the QSAR
references, every score is reproducible:

1. **Tier A** → look up the CAS in Annex VI Table 3, read the hazard classes, apply
   the formula in §2.
2. **Tier B** → take the C&L majority vote (>50 %) or the cited supplier SDS, apply
   the formula in §2.
3. **Tier C** → confirm no classification is available → leave null (engine uses 0.5).
4. **Tier P** → apply the 19 SMARTS alerts via RDKit, apply the formula in §2.

Validation: recompute benzene and check it equals **0.3247**.

---

## 7. Limitations (honest disclosure)

- Tier B relies on manufacturer self-classification — less authoritative than the
  harmonised Annex VI.
- Tier P is **predictive**: useful for triage, never for a real safety decision.
- A few Tier-B classifications rest on family analogy (flagged in the per-compound
  CSV) and should be confirmed against a dedicated SDS if a compound becomes critical.
- CAS numbers were entered manually and should be re-verified before any real
  regulatory use.

---

## 8. Primary sources

- **CLP Annex VI** — ECHA: https://echa.europa.eu/information-on-chemicals/annex-vi-to-clp
- **C&L Inventory** — ECHA: https://echa.europa.eu/information-on-chemicals/cl-inventory-database
- **ToxTree / Benigni–Bossa rulebase** — JRC, EUR 23241 EN (2008).
- **OECD QSAR Toolbox** — https://qsartoolbox.org
- Supplier SDS: Sigma-Aldrich/Merck, Thermo Fisher, TCI Chemicals, ChemicalBook, Fluorochem.
