"""
app_streamlit.py — Interface de rétrosynthèse
=============================================
Lancer :  streamlit run app_streamlit.py
Dossier : place ce fichier à côté de function_ines.py et reaction_dataset.json
"""

import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT DE FUNCTION_INES — sans aucun mock
# AiZynthFinder doit être importé tel quel pour que le vrai modèle tourne
# ─────────────────────────────────────────────────────────────────────────────
try:
    import function_ines as fi
    from rdkit import Chem
    from rdkit.Chem import Draw
    MODULE_OK = True
    MODULE_ERR = ""
except Exception as e:
    MODULE_OK = False
    MODULE_ERR = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rétrosynthèse — Chemistry by Design",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

CIBLES = {
    "Galanthamine":     "OC1C=C[C@@]23c4cc(OC)ccc4CN(C)C[C@@H]2[C@@H]1O3",
    "Morphine":         "OC1=CC2=C(C=C1)[C@@H]1[C@H]3C[C@@H](O)C=C[C@@H]3N(C)CC1=C2",
    "Quinine":          "OC(c1ccnc2cc(OC)ccc12)[C@@H]1C[C@@H]2CC[N@@]1CC2/C=C",
    "Aspidospermidine": "CC[C@@]12CCCN3CC[C@@]4(C1)[C@@H](NH)c1ccccc1[C@@H]4[C@@H]23",
    "Aspidospermine":   "CC[C@@]12CCCN3CC[C@@]4(C1)[C@@H](OC(C)=O)c1ccc(OC)cc1N[C@@H]4[C@@H]23",
}

CRITERES_INFO = {
    "steps":        {"label": "🔢 Nombre d'étapes",   "aide": "Moins d'étapes = meilleur score (1/n)."},
    "yield":        {"label": "📈 Rendement global",   "aide": "Produit des rendements. Yield null → 50% par défaut."},
    "atom_economy": {"label": "⚗️ Économie atomique",  "aide": "Part des atomes des réactifs dans le produit."},
    "e_factor":     {"label": "♻️ E-factor",            "aide": "Moins de déchets = score plus haut."},
    "toxicity":     {"label": "☣️ Sécurité",            "aide": "Score de sécurité réactifs + solvants."},
}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def mol_image(smiles, w=280, h=180):
    if not MODULE_OK or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=(w, h))

def fmt_conditions(cond):
    if not cond:
        return "—"
    parts = []
    if cond.get("temperature_C"):
        parts.append(f"{cond['temperature_C']}°C")
    elif cond.get("temp_range"):
        parts.append(cond["temp_range"])
    if cond.get("solvent"):
        parts.append(f"Solvant : {cond['solvent']}")
    if cond.get("co_solvent"):
        parts.append(f"/ {cond['co_solvent']}")
    if cond.get("reagents"):
        reag = cond["reagents"]
        if isinstance(reag, list) and reag:
            parts.append(", ".join(reag))
    if cond.get("apparatus"):
        parts.append(f"({cond['apparatus']})")
    return "  ·  ".join(parts) if parts else "—"

def yield_cumule(steps):
    r = 1.0
    for s in steps:
        y = s.get("yield_percent")
        r *= (y / 100.0) if y is not None else 0.5
    return r

@st.cache_data(show_spinner=False)
def charger_dataset(path):
    return fi.load_reaction_dataset(path)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚗️ Paramètres")

    if not MODULE_OK:
        st.error(f"❌ Impossible de charger function_ines :\n\n`{MODULE_ERR}`")
        st.info("Vérifie que function_ines.py est dans le même dossier et que toutes les dépendances sont installées.")
        st.stop()

    st.subheader("🎯 Molécule cible")
    mode = st.radio("Mode", ["Prédéfinie", "SMILES personnalisé"],
                    label_visibility="collapsed")

    if mode == "Prédéfinie":
        nom_cible    = st.selectbox("Molécule", list(CIBLES.keys()))
        smiles_cible = CIBLES[nom_cible]
    else:
        smiles_cible = st.text_input("SMILES", placeholder="ex: c1ccccc1")
        nom_cible    = "Personnalisée"
        if smiles_cible:
            if Chem.MolFromSmiles(smiles_cible) is None:
                st.error("SMILES invalide")
                smiles_cible = ""
            else:
                st.success("SMILES valide ✓")

    if smiles_cible:
        img = mol_image(smiles_cible, 240, 150)
        if img:
            st.image(img, caption=nom_cible, use_container_width=True)

    st.divider()

    st.subheader("📊 Critères (ordre de priorité)")
    st.caption("Critère #1 = ~73% du score  ·  #2 = ~18%  ·  #3 = ~9%")

    tous   = list(CRITERES_INFO.keys())
    c1     = st.selectbox("Critère #1 — priorité haute",   tous,
                          format_func=lambda x: CRITERES_INFO[x]["label"])
    reste2 = [c for c in tous if c != c1]
    c2     = st.selectbox("Critère #2 — priorité moyenne", reste2,
                          format_func=lambda x: CRITERES_INFO[x]["label"])
    reste3 = [c for c in reste2 if c != c2]
    c3     = st.selectbox("Critère #3 — priorité basse",   reste3,
                          format_func=lambda x: CRITERES_INFO[x]["label"])

    criteres = [c1, c2, c3]
    poids    = fi.compute_weights(criteres)

    with st.expander("📐 Voir les poids"):
        for c in criteres:
            w = poids[c]
            st.progress(float(w), text=f"{CRITERES_INFO[c]['label']}  —  {w*100:.1f}%")

    st.divider()

    st.subheader("⚙️ Fichiers & options")
    dataset_path  = st.text_input("Dataset",    value="reaction_dataset.json")
    toxicity_path = st.text_input("Toxicité",   value="toxicity_dataset.json")
    config_path   = st.text_input("Config AiZ", value="config.yml")
    top_n         = st.slider("Routes à afficher", 1, 5, 3)

    st.divider()
    lancer = st.button(
        "🔍 Lancer la recherche",
        type="primary",
        use_container_width=True,
        disabled=not smiles_cible,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONTENU PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

st.title("⚗️ Recherche de routes de synthèse")
st.caption("AiZynthFinder (MCTS)  ·  Dataset Chemistry by Design  ·  Scoring pondéré 1/i²")

tab_recherche, tab_dataset, tab_aide = st.tabs(
    ["🔍 Recherche de routes", "📂 Explorer le dataset", "❓ Aide"]
)

# ── TAB 1 : RECHERCHE ────────────────────────────────────────────────────────
with tab_recherche:

    if not smiles_cible:
        st.info("👈 Choisis une molécule cible et tes 3 critères dans la barre latérale, "
                "puis clique **Lancer la recherche**.")
        st.markdown("### Molécules disponibles dans le dataset")
        cols = st.columns(len(CIBLES))
        for col, (nom, smi) in zip(cols, CIBLES.items()):
            with col:
                img = mol_image(smi, 200, 140)
                if img:
                    st.image(img, caption=nom, use_container_width=True)

    elif lancer:
        erreurs = []
        if not os.path.exists(dataset_path):
            erreurs.append(f"Dataset introuvable : `{dataset_path}`")
        if not os.path.exists(config_path):
            erreurs.append(f"Config AiZynthFinder introuvable : `{config_path}`")
        if erreurs:
            for e in erreurs:
                st.error(e)
            st.stop()

        col_info, col_mol = st.columns([3, 1])
        with col_info:
            st.markdown(f"**Cible :** {nom_cible}")
            st.code(smiles_cible, language=None)
            st.markdown(
                f"**Critères :** {CRITERES_INFO[c1]['label']} › "
                f"{CRITERES_INFO[c2]['label']} › {CRITERES_INFO[c3]['label']}"
            )
        with col_mol:
            img = mol_image(smiles_cible, 260, 160)
            if img:
                st.image(img, use_container_width=True)

        # ── Pipeline — pas de cache, AiZynthFinder doit toujours tourner ──
        with st.status("🔍 Recherche en cours...", expanded=True) as status:
            try:
                st.write("📂 Chargement des datasets...")
                st.write("🔬 Lancement d'AiZynthFinder (peut prendre 1–2 min)...")
                results = fi.find_best_routes(
                    target_smiles     = smiles_cible,
                    criteria_priority = criteres,
                    dataset_path      = dataset_path,
                    toxicity_path     = toxicity_path,
                    config_path       = config_path,
                    top_n             = top_n,
                )
                status.update(
                    label    = "✅ Recherche terminée",
                    state    = "complete",
                    expanded = False,
                )
            except FileNotFoundError as e:
                status.update(label="❌ Fichier introuvable", state="error")
                st.error(str(e))
                st.stop()
            except ValueError as e:
                status.update(label="❌ Paramètre invalide", state="error")
                st.error(str(e))
                st.stop()
            except Exception as e:
                status.update(label="❌ Erreur inattendue", state="error")
                st.exception(e)
                st.stop()

        # ── Résultats ─────────────────────────────────────────────────────
        if not results:
            st.warning(
                "**Aucune route trouvée.**\n\n"
                "Pistes possibles :\n"
                "- La molécule doit être dans le dataset "
                "(galanthamine, morphine, quinine, aspidospermidine, aspidospermine)\n"
                "- AiZynthFinder n'a peut-être pas trouvé de routes pour cette cible\n"
                "- Les starting materials proposés ne matchent pas les réactifs du dataset"
            )
        else:
            st.success(f"**{len(results)} route(s)** trouvée(s) et classée(s).")

            if len(results) > 1:
                noms   = [r[2].get("matched_route_name", f"Route {i+1}")[:20]
                          for i, r in enumerate(results)]
                scores = [r[0] for r in results]
                fig, ax = plt.subplots(figsize=(8, 2.5))
                colors  = ["#1565C0" if i == 0 else "#90CAF9"
                           for i in range(len(scores))]
                bars = ax.barh(noms[::-1], scores[::-1],
                               color=colors[::-1], height=0.5)
                for bar, s in zip(bars, scores[::-1]):
                    ax.text(s + 0.001, bar.get_y() + bar.get_height() / 2,
                            f"{s:.4f}", va="center", fontsize=9)
                ax.set_xlabel("Score total")
                ax.set_xlim(0, max(scores) * 1.3 if scores else 1)
                ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("---")

            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
            for rang, (score_total, details, route) in enumerate(results, 1):
                nom_route   = route.get("matched_route_name", "?")
                cible_route = route.get("matched_target", "?")
                steps_data  = route.get("dataset_steps", [])
                n_steps     = len(steps_data)
                yld_cum     = yield_cumule(steps_data)

                with st.expander(
                    f"{medals[rang-1]}  {nom_route}  —  {cible_route}"
                    f"  —  Score : **{score_total:.4f}**",
                    expanded=(rang == 1),
                ):
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Score total",      f"{score_total:.4f}")
                    m2.metric("Étapes",           n_steps)
                    m3.metric("Rendement cumulé", f"{yld_cum * 100:.1f}%")
                    m4.metric("Réactifs communs", route.get("coverage", "—"))

                    st.markdown("**Contribution de chaque critère**")
                    fig2, ax2 = plt.subplots(figsize=(7, 1.8))
                    cats = [CRITERES_INFO[c]["label"] for c in criteres]
                    raws = [details[c]["raw"]      for c in criteres]
                    weis = [details[c]["weighted"] for c in criteres]
                    x    = np.arange(len(criteres))
                    ax2.bar(x - 0.18, raws, 0.35,
                            label="Score brut",            color="#90CAF9")
                    ax2.bar(x + 0.18, weis, 0.35,
                            label="Contribution pondérée", color="#1565C0")
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(cats, fontsize=9)
                    ax2.set_ylim(0, 1.05)
                    ax2.legend(fontsize=8)
                    ax2.spines[["top", "right"]].set_visible(False)
                    fig2.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)

                    st.markdown("---")
                    st.markdown(f"**Étapes de la synthèse ({n_steps} étapes)**")

                    for step in steps_data:
                        snum  = step.get("step_number", "?")
                        rtype = step.get("reaction_type", "—")
                        yld   = step.get("yield_percent")
                        cond  = step.get("conditions", {})
                        prod  = step.get("product_smiles", "")
                        reac  = step.get("reactants_smiles", [])

                        col_txt, col_img = st.columns([3, 1])
                        with col_txt:
                            yld_str = (f"**{yld}%**" if yld is not None
                                       else "*non renseigné (50% par défaut)*")
                            st.markdown(
                                f"**Étape {snum}** — {rtype}  \n"
                                f"Rendement : {yld_str}  \n"
                                f"Conditions : {fmt_conditions(cond)}"
                            )
                            with st.expander(f"SMILES — étape {snum}",
                                             expanded=False):
                                for rsmi in reac:
                                    st.code(rsmi, language=None)
                                if reac:
                                    st.markdown("→ **produit :**")
                                st.code(prod, language=None)
                        with col_img:
                            img_prod = mol_image(prod, 180, 120)
                            if img_prod:
                                st.image(img_prod,
                                         caption=f"Produit étape {snum}",
                                         use_container_width=True)
                            else:
                                st.caption("⚠️ SMILES non affichable")
                        st.divider()


# ── TAB 2 : EXPLORER LE DATASET ──────────────────────────────────────────────
with tab_dataset:
    st.subheader("Explorer le dataset")

    if not os.path.exists(dataset_path):
        st.warning(f"Dataset `{dataset_path}` introuvable.")
    else:
        with st.spinner("Chargement..."):
            ds = charger_dataset(dataset_path)

        reactions_all = ds["all"]
        by_route      = ds["by_route"]

        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Réactions totales", len(reactions_all))
        col_s2.metric("Routes distinctes", len(by_route))
        targets_uniq = sorted(set(r.get("target", "?") for r in reactions_all))
        col_s3.metric("Molécules cibles",  len(targets_uniq))

        st.markdown("---")

        filtre_cible = st.selectbox("Filtrer par cible",
                                    ["Toutes"] + targets_uniq)

        import pandas as pd
        rows_table = []
        for rid, steps in sorted(by_route.items()):
            target = steps[0].get("target", "?")
            if filtre_cible != "Toutes" and target != filtre_cible:
                continue
            nom       = steps[0].get("route_name", rid)
            n         = len(steps)
            yields    = [s.get("yield_percent") for s in steps]
            yields_ok = [y for y in yields if y is not None]
            yld_cum   = yield_cumule(steps)
            rows_table.append({
                "Route":        nom,
                "Cible":        target,
                "Étapes":       n,
                "Yield cumulé": f"{yld_cum * 100:.1f}%",
                "Yield null":   sum(1 for y in yields if y is None),
                "Yield moyen":  (f"{sum(yields_ok)/len(yields_ok):.0f}%"
                                 if yields_ok else "—"),
            })

        if rows_table:
            st.dataframe(pd.DataFrame(rows_table),
                         use_container_width=True, hide_index=True)

        st.markdown("---")

        routes_dispo = [
            rid for rid, steps in by_route.items()
            if filtre_cible == "Toutes"
            or steps[0].get("target", "?") == filtre_cible
        ]
        route_choisie = st.selectbox(
            "Voir le détail d'une route",
            routes_dispo,
            format_func=lambda x: (
                f"{by_route[x][0].get('route_name', x)} "
                f"({by_route[x][0].get('target', '?')})"
            ),
        )

        if route_choisie:
            steps_route = by_route[route_choisie]
            st.markdown(f"**{steps_route[0].get('route_name')}** "
                        f"— {len(steps_route)} étapes")

            yields_plot = [s.get("yield_percent") or 50 for s in steps_route]
            is_null     = [s.get("yield_percent") is None for s in steps_route]
            fig3, ax3   = plt.subplots(
                figsize=(max(6, len(yields_plot) * 0.65), 2.5))
            ax3.bar(range(1, len(yields_plot) + 1), yields_plot,
                    color=["#BDBDBD" if n else "#1565C0" for n in is_null],
                    width=0.6)
            ax3.axhline(50, color="#E65100", linestyle="--", linewidth=0.8)
            ax3.set_xlabel("Étape")
            ax3.set_ylabel("Yield (%)")
            ax3.set_ylim(0, 110)
            ax3.set_xticks(range(1, len(yields_plot) + 1))
            ax3.legend(handles=[
                mpatches.Patch(color="#1565C0", label="Yield renseigné"),
                mpatches.Patch(color="#BDBDBD", label="Yield null → 50% par défaut"),
            ], fontsize=8)
            ax3.spines[["top", "right"]].set_visible(False)
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            for step in steps_route:
                snum  = step.get("step_number", "?")
                rtype = step.get("reaction_type", "—")
                yld   = step.get("yield_percent")
                cond  = step.get("conditions", {})
                prod  = step.get("product_smiles", "")

                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(
                        f"**Étape {snum}** — {rtype}  \n"
                        f"Rendement : {f'{yld}%' if yld is not None else '*(non renseigné)*'}  \n"
                        f"Conditions : {fmt_conditions(cond)}"
                    )
                with col_b:
                    img = mol_image(prod, 160, 110)
                    if img:
                        st.image(img, caption=f"Produit {snum}",
                                 use_container_width=True)
                    else:
                        st.caption("⚠️ SMILES non affichable")
                st.divider()


# ── TAB 3 : AIDE ─────────────────────────────────────────────────────────────
with tab_aide:
    st.subheader("Comment ça marche")
    st.markdown("""
**Pipeline en 4 étapes :**
1. Le dataset `reaction_dataset.json` est chargé et indexé
2. AiZynthFinder explore la molécule cible (MCTS + modèle USPTO)
3. Les starting materials d'AiZynthFinder sont comparés aux réactifs du dataset
4. Les routes validées sont scorées selon les 3 critères pondérés en 1/i²

**Poids des critères :**
- Critère #1 → ~73% du score final
- Critère #2 → ~18%
- Critère #3 → ~9%

**Fichiers nécessaires :**

| Fichier | Obligatoire | Rôle |
|---------|-------------|------|
| `reaction_dataset.json` | ✅ | Données des réactions |
| `config.yml` | ✅ | Configuration AiZynthFinder |
| `toxicity_dataset.json` | ❌ | Scores de sécurité (sinon 0.5 par défaut) |

**Pourquoi 0 routes parfois ?**
- AiZynthFinder doit proposer au moins un starting material présent dans le dataset
- L'aspidospermidine peut donner 0 routes car le modèle USPTO ne connaît pas bien ce squelette
- Essaie avec la galanthamine qui fonctionne bien
    """)

st.markdown("---")
st.caption("Pipeline de rétrosynthèse  ·  Dataset Chemistry by Design  ·  AiZynthFinder MCTS")