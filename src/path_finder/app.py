# app.py
# Streamlit UI for Path Finder. Layout and event handling only — all rendering logic lives in app_utensils.py.
# Run: streamlit run app.py

import os
import sys

# needed when launched from a different working directory
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from app_layout import LANG, CRITERIA_LABELS, PALETTE, FIG_BG
from molecule_rendering import mol_png, MODULE_OK
from app_utensils import (
    load_banner,
    hires_fig,
    strip_emoji,
    is_purification_step,
    build_clickable_scheme_html,
    build_score_table_html,
    make_ranking_chart,
    make_yield_chart,
    make_comparison_chart,
    build_why_ranked_html,
    smiles_copy_widget,
    display_route_card,
    load_dataset_cached,
    get_targets_cached,
)

try:
    import route_engine as rt
    from rdkit import Chem
    MODULE_ERR = ""
except Exception as e:
    rt = None
    Chem = None
    MODULE_ERR = str(e)

try:
    from rxn_insight.reaction import Reaction as RxnInsightReaction
    RXNINSIGHT_OK = True
except ImportError:
    RXNINSIGHT_OK = False

# must be the first Streamlit call
st.set_page_config(
    page_title="Retrosynthesis — Chemistry by Design",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# typography, metric cards, expanders, buttons
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; color: #1a2e44; }
[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif; font-size: 1.45rem !important; color: #1a2e44;
}
[data-testid="stMetricLabel"] {
    font-size: 0.76rem !important; color: #6b7a8d;
    text-transform: uppercase; letter-spacing: 0.06em;
}
div[data-testid="stExpander"] { border: 1px solid #dce3ec; border-radius: 12px; background: #f9fbfd; }
.stButton > button {
    background: linear-gradient(135deg, #1a2e44 0%, #2d5986 100%);
    color: white !important; border: none; border-radius: 8px;
    font-family: 'DM Sans', sans-serif; font-weight: 600; letter-spacing: 0.04em; transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
.stTabs [data-baseweb="tab"] { font-family: 'DM Sans', sans-serif; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

def main() -> None:
    """Four-tab UI: Route Search, Analysis, Dataset Explorer, Help."""
    banner_uri = load_banner("../../assets/banner.png")
    if banner_uri:
        st.markdown(
            f'<div style="text-align:center;padding:18px 0 8px 0;">'
            f'<img src="{banner_uri}" style="width:420px;max-width:90%;'
            f'margin:0 auto;display:block;"/></div>',
            unsafe_allow_html=True,
        )

    with st.sidebar:
        lang_choice = st.radio(
            "🌐 Language / Langue",
            ["🇬🇧 English", "🇫🇷 Français"],
            horizontal=True,
            index=0,
        )
    lang = "en" if "English" in lang_choice else "fr"
    T    = LANG[lang]
    CL   = CRITERIA_LABELS[lang]

    with st.sidebar:
        st.title(T["sidebar_title"])
        if not MODULE_OK:
            st.error(f"{T['mod_err']}\n\n`{MODULE_ERR}`")
            st.stop()

        st.divider()
        st.subheader(T["files_section"])

        dataset_path = st.text_input(T["ds_label"], value="data/reaction_dataset.json", help=T["help_ds"])
        toxicity_path = st.text_input(T["tox_label"]+" *", value="data/toxicity_dataset.json", help=T["help_tox"])
        config_path = st.text_input(T["cfg_label"], value="data/config.yml", help=T["help_cfg"])
        rxninsight_db_path = st.text_input(T["rxni_label"], value="data/uspto_rxn_insight.gzip", help=T["help_rxni"])
        generic_dataset_path = st.text_input(T["generic_label"], value="data/generic_reactions.json", help=T["help_generic"])

        if not os.path.exists(toxicity_path):
            st.warning(f"⚠️ `{toxicity_path}` not found — Safety scores will default to 0.5")

        st.divider()
        top_n = st.slider(T["topn_label"], 1, 5, 3, help=T["help_topn"])
        n_aiz = st.slider(T["naiz_label"], 5, 50, 25, help=T["help_naiz"])

        include_predicted = False
        if RXNINSIGHT_OK:
            include_predicted = st.toggle(
                "Include predicted routes (Rxn-INSIGHT)", value=True,
                help=(
                    "AiZynthFinder routes with no match in the generic dataset are "
                    "classified as predicted and annotated by Rxn-INSIGHT. "
                    "Yield is excluded from their score."
                ),
            )
        else:
            st.caption("🤖 Predicted routes disabled — add `data/uspto_rxn_insight.gzip` to enable (see README)")

        st.divider()
        if "criteria" in st.session_state:
            with st.expander(T["weights_exp"]):
                crit_w = rt.compute_weights(st.session_state["criteria"])
                for c in st.session_state["criteria"]:
                    w = crit_w[c]
                    st.progress(float(w), text=f"{strip_emoji(CL[c])} — {w * 100:.1f}%")
    # Page header
    st.title(T["page_title"])
    st.caption(T["page_caption"])

    tab_search, tab_analysis, tab_dataset, tab_help = st.tabs([
        T["tab_search"], T["tab_analysis"], T["tab_dataset"], T["tab_help"],
    ])

    with tab_search:
        top_left, top_right = st.columns([3, 2])

        with top_left:
            st.subheader(T["target_section"])
            mode = st.radio("Mode", [T["mode_pre"], T["mode_custom"]],
                            horizontal=True, label_visibility="collapsed")

            if mode == T["mode_pre"]:
                targets_ds = {}
                if os.path.exists(dataset_path):
                    try:
                        targets_ds = get_targets_cached(dataset_path)
                    except Exception:
                        pass
                if targets_ds:
                    target_name = st.selectbox(T["mol_label"], list(targets_ds.keys()),
                                                 format_func=lambda x: x.capitalize())
                    target_smiles = targets_ds[target_name]
                else:
                    target_name = "Galanthamine"
                    target_smiles = "OC1C=C[C@@]23c4cc(OC)ccc4CN(C)C[C@@H]2[C@@H]1O3"
            else:
                target_smiles = st.text_input(T["smiles_label"], placeholder=T["smiles_ph"])
                target_name = "Custom"
                if target_smiles:
                    if Chem.MolFromSmiles(target_smiles) is None:
                        st.error(T["smiles_invalid"])
                        target_smiles = ""
                    else:
                        st.success(T["smiles_valid"])

            st.subheader(T["criteria_section"])
            st.caption(T["criteria_caption"])
            all_crit = list(CL.keys())
            c1 = st.selectbox(T["c1_label"], all_crit, format_func=lambda x: CL[x])
            rem2 = [c for c in all_crit if c != c1]
            c2 = st.selectbox(T["c2_label"], rem2, format_func=lambda x: CL[x])
            rem3 = [c for c in rem2 if c != c2]
            c3 = st.selectbox(T["c3_label"], rem3, format_func=lambda x: CL[x])
            criteria = [c1, c2, c3]
            st.session_state["criteria"] = criteria

            run_search = st.button(T["run_btn"], type="primary", disabled=not target_smiles)

        with top_right:
            if target_smiles:
                png = mol_png(target_smiles, 640, 440)
                if png:
                    st.image(png, caption=target_name, use_container_width=True)
            else:
                st.markdown(T["welcome"])
                st.markdown(T["avail_mols"])
                try:
                    prev = get_targets_cached(dataset_path)
                    cols = st.columns(3)
                    for idx, (name, smi) in enumerate(prev.items()):
                        with cols[idx % 3]:
                            png = mol_png(smi, 400, 270)
                            if png:
                                st.image(png, caption=name.capitalize(), use_container_width=True)
                except Exception:
                    pass

        st.divider()

        if run_search and target_smiles:
            errs = []
            if not os.path.exists(dataset_path): errs.append(f"`{dataset_path}`")
            if not os.path.exists(config_path):  errs.append(f"`{config_path}`")
            for e in errs:
                st.error(f"{T['err_file']}: {e}")
            if not errs:
                with st.status(T["searching"], expanded=True) as status:
                    try:
                        st.write(T["loading_ds"])
                        st.write(T["loading_aiz"])
                        if include_predicted and RXNINSIGHT_OK:
                            st.write(T["loading_rxni"])
                        results_raw = rt.find_best_routes(
                            target_smiles = target_smiles,
                            criteria_priority = criteria,
                            dataset_path = dataset_path,
                            toxicity_path = toxicity_path,
                            config_path = config_path,
                            top_n = top_n,
                            target_name = target_name if mode == T["mode_pre"] else "",
                            include_predicted = include_predicted,
                            rxninsight_db_path = rxninsight_db_path,
                            generic_dataset_path = generic_dataset_path,
                            n_aiz_routes = n_aiz,
                        )
                        tox_index = rt.load_toxicity_dataset(toxicity_path)
                        weights = rt.compute_weights(criteria)
                        st.session_state.update({
                            "results": results_raw,
                            "weights": weights,
                            "tox_index": tox_index,
                            "target_name": target_name,
                            "target_smiles": target_smiles,
                            "criteria": criteria,
                        })
                        status.update(label=T["search_ok"], state="complete", expanded=False)
                    except FileNotFoundError as e:
                        status.update(label=T["err_file"],  state="error"); st.error(str(e))
                    except ValueError as e:
                        status.update(label=T["err_param"], state="error"); st.error(str(e))
                    except Exception as e:
                        status.update(label=T["err_other"], state="error"); st.exception(e)

        results_raw = st.session_state.get("results", None)
        weights = st.session_state.get("weights", {})
        criteria = st.session_state.get("criteria", criteria)

        if results_raw is not None and isinstance(results_raw, dict):
            scored_dataset = results_raw.get("dataset", [])
            scored_validated = results_raw.get("validated", [])
            scored_predicted = results_raw.get("predicted", [])

            if not scored_dataset and not scored_validated and not scored_predicted:
                st.warning(T["no_routes"])
            else:
                c1_, c2_, c3_, c4_ = st.columns(4)
                c1_.metric("📚 Dataset", len(scored_dataset))
                c2_.metric("✅ Validated", len(scored_validated))
                c3_.metric("🤖 Predicted", len(scored_predicted))
                c4_.metric("Total", len(scored_dataset) + len(scored_validated) + len(scored_predicted))

                tgt_name = st.session_state.get("target_name", target_name)

                st.markdown(T["sec_dataset"])
                st.caption(T["cap_dataset"])
                if not scored_dataset:
                    st.info("No dataset routes for this target.")
                else:
                    if len(scored_dataset) > 1:
                        fig = make_ranking_chart(scored_dataset, tgt_name, lang)
                        st.pyplot(fig); plt.close(fig)
                    st.success(T["n_found"].format(n=len(scored_dataset)))
                    st.markdown("---")
                    for rank, (score_total, details, route) in enumerate(scored_dataset, 1):
                        display_route_card(score_total, details, route, criteria, weights,
                                           scored_dataset, rank, T["badge_dataset"], lang)

                if scored_validated:
                    st.markdown("---")
                    st.markdown(T["sec_validated"])
                    st.caption(T["cap_validated"])
                    if len(scored_validated) > 1:
                        fig = make_ranking_chart(scored_validated, tgt_name, lang)
                        st.pyplot(fig); plt.close(fig)
                    for rank, (score_total, details, route) in enumerate(scored_validated, 1):
                        display_route_card(score_total, details, route, criteria, weights,
                                           scored_validated, rank, T["badge_validated"], lang)

                if include_predicted and RXNINSIGHT_OK and scored_predicted:
                    st.markdown("---")
                    st.markdown(T["sec_predicted"])
                    st.caption(T["cap_predicted"])

                    search_sm = st.text_input(
                        "Search by starting material SMILES",
                        placeholder="e.g. c1ccc2[nH]ccc2c1",
                        key="sm_search",
                    )
                    all_sm_map = {}
                    for _, _, route in scored_predicted:
                        steps    = route.get("dataset_steps", [])
                        all_prod = {rt.to_canonical(s.get("product_smiles", "")) for s in steps}
                        for s in steps:
                            for rsmi in s.get("reactants_smiles", []):
                                canon = rt.to_canonical(rsmi)
                                if canon and canon not in all_prod:
                                    all_sm_map.setdefault(canon, []).append(
                                        route.get("matched_route_name", "?"))

                    filtered = scored_predicted
                    if search_sm.strip():
                        sc = rt.to_canonical(search_sm.strip())
                        filtered = [
                            (s, d, r) for s, d, r in scored_predicted
                            if sc in {rt.to_canonical(rsmi)
                                      for step in r.get("dataset_steps", [])
                                      for rsmi in step.get("reactants_smiles", [])}
                            or any(
                                search_sm.lower() in rsmi.lower()
                                for step in r.get("dataset_steps", [])
                                for rsmi in step.get("reactants_smiles", [])
                            )
                        ]
                        st.success(f"{len(filtered)} route(s) found") if filtered else \
                            st.warning("No routes with this starting material")

                    top30 = list(all_sm_map.keys())[:30]
                    with st.expander(f"🗂️ Browse starting materials ({len(top30)} unique)", expanded=False):
                        cols_sm = st.columns(3)
                        for i, smi in enumerate(top30):
                            col = cols_sm[i % 3]
                            png = mol_png(smi, 400, 270)
                            if png:
                                col.image(png, width=140)
                            col.code(smi, language=None)
                            col.caption(f"In: {', '.join(set(all_sm_map[smi]))[:50]}")

                    st.markdown("---")
                    for rank, (score_total, details, route) in enumerate(filtered, 1):
                        display_route_card(score_total, details, route, criteria, weights,
                                           filtered, rank, T["badge_predicted"], lang)

    # ANALYSIS tab
    with tab_analysis:
        results_raw = st.session_state.get("results", None)
        criteria = st.session_state.get("criteria", list(CL.keys())[:3])
        weights = st.session_state.get("weights", {})
        tox_index = st.session_state.get("tox_index", {})

        if results_raw is None or not isinstance(results_raw, dict):
            st.info(T["no_analysis"])
        else:
            sc_ds = results_raw.get("dataset", [])
            sc_val = results_raw.get("validated", [])
            sc_pr = results_raw.get("predicted", [])
            all_sc = sc_ds + sc_val + sc_pr

            if not all_sc:
                st.info(T["no_analysis"])
            else:
                st.subheader(T["compare_title"])

                def _badge_label(r):
                    s = r[2].get("validation_status", "dataset")
                    return {"validated":"✅","partial":"🔬","predicted":"🤖"}.get(s,"📚")

                route_opts = {
                    f"{_badge_label(r)} {r[2].get('matched_route_name','?')} "
                    f"(score {r[0]:.4f})": i
                    for i, r in enumerate(all_sc)
                }
                labels = list(route_opts.keys())
                sel = st.multiselect(T["sel_routes"], labels,
                                        default=labels[:min(3, len(labels))])

                if sel:
                    sel_results = [all_sc[route_opts[s]] for s in sel]
                    rows = []
                    for score, details, route in sel_results:
                        sd  = route.get("dataset_steps", [])
                        bn_ = rt.bottleneck_yield(sd)
                        av_ = rt.average_yield(sd)
                        yc  = rt.cumulative_yield(sd)
                        row = {
                            "Route": route.get("matched_route_name","?")[:26],
                            T["metric_score"]: f"{score:.3f}",
                            T["metric_steps"]: len(sd),
                            "Cumul. yield": f"{yc * 100:.4f}%",
                            T["metric_bottleneck"]:f"{bn_:.1f}%" if bn_ else "—",
                            T["metric_avg"]: f"{av_:.1f}%" if av_ else "—",
                        }
                        for c in criteria:
                            if c in ("steps", "yield"):
                                continue
                            raw = details[c].get("raw")
                            row[CL[c]] = f"{raw:.4f}" if raw is not None else "N/A"
                        all_s = rt.compute_all_scores(route, tox_index)
                        for c in rt.CRITERIA_REGISTRY:
                            if c not in criteria and c not in ("steps","yield"):
                                row[CL[c]+" ✗"] = f"{all_s[c]:.4f}"
                        rows.append(row)

                    df = pd.DataFrame(rows).set_index("Route")
                    st.dataframe(
                        df, width="stretch",
                        column_config={col: st.column_config.TextColumn(col) for col in df.columns},
                    )
                    st.caption("✓ = selected criteria  ✗ = additional criteria (not used in ranking)")

                    if len(sel_results) >= 2:
                        st.markdown(f"**{T['radar_title']}**")
                        fig_cmp = make_comparison_chart(sel_results, criteria, lang)
                        st.pyplot(fig_cmp); plt.close(fig_cmp)

                    if len(sel_results) == 2:
                        st.markdown("---")
                        st.markdown("### Reaction schemes")
                        cA, cB = st.columns(2)
                        for col, (score, details, route), lbl in zip(
                                [cA, cB], sel_results, sel[:2]):
                            with col:
                                st.markdown(f"**{route.get('matched_route_name','?')}**")
                                rk = "".join(
                                    c for c in route.get("matched_route_id","r")
                                    if c.isalnum()
                                )
                                components.html(
                                    build_clickable_scheme_html(
                                        route.get("dataset_steps",[]),
                                        rk,
                                        route.get("is_predicted", False),
                                    ),
                                    height=480, scrolling=True,
                                )

    # DATASET EXPLORER    
    with tab_dataset:
        if not os.path.exists(dataset_path):
            st.warning(T["ds_not_found"].format(path=dataset_path))
        else:
            with st.spinner("Loading…"):
                ds = load_dataset_cached(dataset_path)
            reactions_all = ds["all"]
            by_route = ds["by_route"]
            targets_uniq = sorted(set(r.get("target","?") for r in reactions_all))

            c1d, c2d, c3d = st.columns(3)
            c1d.metric(T["total_rxn"], len(reactions_all))
            c2d.metric(T["dist_routes"], len(by_route))
            c3d.metric(T["tgt_mols"], len(targets_uniq))
            st.markdown("---")

            filter_tgt = st.selectbox(T["filter_lbl"], [T["filter_all"]] + targets_uniq)

            rows = []
            for rid, steps in sorted(by_route.items()):
                tgt = steps[0].get("target", "?")
                if filter_tgt != T["filter_all"] and tgt != filter_tgt:
                    continue
                nm = steps[0].get("route_name", rid)
                n = len(steps)
                ys = [s.get("yield_percent") for s in steps]
                yo = [y for y in ys if y is not None]
                yc = rt.cumulative_yield(steps)
                rows.append({
                    T["col_route"]: nm,
                    T["col_target"]: tgt,
                    T["col_steps"]: n,
                    T["col_cumyield"]: round(yc * 100, 4),
                    T["col_missing"]: sum(1 for y in ys if y is None),
                    T["col_avg"]: round(sum(yo)/len(yo), 1) if yo else None,
                })

            if rows:
                df_routes = (
                    pd.DataFrame(rows)
                    .sort_values(T["col_cumyield"], ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(
                    df_routes, width="stretch", hide_index=True,
                    column_config={
                        T["col_route"]: st.column_config.TextColumn(T["col_route"]),
                        T["col_target"]: st.column_config.TextColumn(T["col_target"]),
                        T["col_steps"]: st.column_config.NumberColumn(T["col_steps"], format="%d"),
                        T["col_cumyield"]: st.column_config.NumberColumn(T["col_cumyield"], format="%.2f %%"),
                        T["col_missing"]: st.column_config.NumberColumn(T["col_missing"], format="%d"),
                        T["col_avg"]: st.column_config.NumberColumn(T["col_avg"], format="%.1f %%"),
                    },
                )

            st.markdown("---")
            routes_avail = [
                rid for rid, steps in by_route.items()
                if filter_tgt == T["filter_all"]
                or steps[0].get("target","?") == filter_tgt
            ]
            rc = st.selectbox(
                T["detail_sel"], routes_avail,
                format_func=lambda x: (
                    f"{by_route[x][0].get('route_name',x)} ({by_route[x][0].get('target','?')})"
                ),
            )

            if rc:
                sr = by_route[rc]
                n_sr = len(sr)
                st.markdown(
                    f"**{sr[0].get('route_name')}** — {n_sr} "
                    f"{strip_emoji(T['col_steps']).lower()}"
                )
                fig3 = make_yield_chart(sr, lang)
                st.pyplot(fig3); plt.close(fig3)

                st.markdown("### Step-by-step details")
                for step in sr:
                    snum = step.get("step_number", "?")
                    rtype = step.get("reaction_type", "—") or "—"
                    yld = step.get("yield_percent")
                    cond = step.get("conditions", {})
                    prod = step.get("product_smiles", "")
                    reac = step.get("reactants_smiles", [])
                    cond_str = rt.fmt_conditions(cond)
                    is_purif = is_purification_step(step)
                    if is_purif and (rtype == "—" or not rtype):
                        rtype = "Purification / Isolation"

                    header_style = (
                        "background:#6B3E26;" if is_purif else
                        "background:linear-gradient(135deg,#1a2e44 0%,#2d5986 100%);"
                    )
                    st.markdown(
                        f'<div style="{header_style}color:white;padding:10px 18px;'
                        f'border-radius:10px 10px 0 0;font-family:\'DM Sans\',sans-serif;'
                        f'font-weight:700;font-size:0.95rem;letter-spacing:0.04em;margin-top:18px;">'
                        f'Step {snum}&nbsp;·&nbsp;{rtype}'
                        + (f'&nbsp;<span style="font-size:0.75rem;background:rgba(255,255,255,0.2);'
                           f'border-radius:8px;padding:1px 8px;">Purification</span>'
                           if is_purif else "") +
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    n_reac = max(len(reac), 1)
                    cols_rxn = st.columns([3] * n_reac + [1, 3])

                    for i, rsmi in enumerate(reac):
                        with cols_rxn[i]:
                            png_r = mol_png(rsmi, 480, 330)
                            if png_r:
                                st.image(png_r, use_container_width=True)
                            else:
                                st.code(rsmi, language=None)
                            smiles_copy_widget(rsmi)

                    with cols_rxn[n_reac]:
                        st.markdown(
                            '<div style="display:flex;align-items:center;justify-content:center;'
                            'height:330px;"><div style="display:flex;align-items:center;gap:0;">'
                            '<div style="width:2.5em;height:3px;background:#2d5986;'
                            'border-radius:2px 0 0 2px;"></div>'
                            '<div style="width:0;height:0;border-top:9px solid transparent;'
                            'border-bottom:9px solid transparent;'
                            'border-left:16px solid #2d5986;"></div>'
                            '</div></div>',
                            unsafe_allow_html=True,
                        )

                    with cols_rxn[n_reac + 1]:
                        png_p = mol_png(prod, 480, 330)
                        if png_p:
                            st.image(png_p, use_container_width=True)
                        else:
                            st.code(prod, language=None)
                        smiles_copy_widget(prod)

                    info_parts = []
                    if yld is not None:
                        info_parts.append(f"**Yield:** {yld}%")
                    else:
                        info_parts.append("**Yield:** *not reported*")
                    if cond_str:
                        info_parts.append(f"**Conditions:** {cond_str}")

                    st.markdown(
                        '<div style="background:#f0f4f8;border:1px solid #dce3ec;border-top:none;'
                        'border-radius:0 0 10px 10px;padding:10px 18px;'
                        'font-family:\'DM Sans\',sans-serif;font-size:0.86rem;color:#1a2e44;'
                        'display:flex;gap:32px;flex-wrap:wrap;margin-bottom:4px;">' +
                        "&nbsp;&nbsp;·&nbsp;&nbsp;".join(
                            p.replace("**","<strong>",1).replace("**","</strong>",1)
                             .replace("*","<em>",1).replace("*","</em>",1)
                            for p in info_parts
                        ) + '</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("---")
                st.markdown("**Full reaction scheme** *(click an arrow for extra details)*")
                route_ds_key     = "ds_" + "".join(c for c in rc if c.isalnum())
                steps_for_scheme = [
                    {
                        "step_number": s.get("step_number"),
                        "reaction_type": s.get("reaction_type",""),
                        "yield_percent": s.get("yield_percent"),
                        "reactants_smiles": s.get("reactants_smiles",[]),
                        "product_smiles": s.get("product_smiles",""),
                        "conditions": s.get("conditions",{}),
                        "source": "dataset",
                    }
                    for s in sr
                ]
                scheme_h = 320 if n_sr <= 5 else (400 if n_sr <= 12 else 480)
                components.html(
                    build_clickable_scheme_html(steps_for_scheme, route_ds_key, False),
                    height=scheme_h, scrolling=True,
                )

    # HELP    
    with tab_help:
        st.subheader(T["help_title"])
        if lang == "en":
            st.markdown("""
**How routes are found and ranked:**

1. `reaction_dataset.json` is loaded (main dataset — full synthesis routes)
2. AiZynthFinder runs MCTS on the target (up to N routes, configurable)
3. Main dataset routes for the target are returned (all of them, unfiltered)
4. Novel AiZ routes are checked against `generic_reactions.json` step by step:
   - **Validated** (all steps found): real conditions, yield included in scoring
   - **Predicted** (no steps found): Rxn-INSIGHT conditions only, yield excluded
5. Weighted 1/i² scoring across criteria

**Purification / isolation steps:**
Steps where the reactant and product SMILES are identical (or where `reaction_type`
contains "purif", "recryst", "chroma", "isolation", or "workup") are treated as
purification steps. They are always shown in the scheme with a **brown dashed arrow**
and a "Purification" badge, even though no bond-forming chemistry occurs. This
preserves the full sequence of operations as recorded in the dataset.

**Three result sections:**
| Section | Source | Conditions | Yield in scoring |
|---------|--------|-----------|-----------------|
| 📚 Dataset | Chemistry by Design | Real | Yes |
| ✅ Validated | AiZ + generic dataset | Real (validated steps) | Yes |
| 🤖 Predicted | AiZ + Rxn-INSIGHT | Predicted | No |

**Required files:**

| File | Required | Role |
|------|----------|------|
| `reaction_dataset.json` | ✅ | Main curated routes |
| `config.yml` | ✅ | AiZynthFinder model config |
| `toxicity_dataset.json` | ✅ | Safety scores — required for Safety criterion |
| `data/uspto_rxn_insight.gzip` | ❌ | Rxn-INSIGHT USPTO database |
| `generic_reactions.json` | ❌ | Individual reactions for step validation |
            """)
        else:
            st.markdown("""
**Comment les routes sont trouvées et classées :**

1. `reaction_dataset.json` est chargé (dataset principal — routes de synthèse complètes)
2. AiZynthFinder effectue une recherche MCTS sur la cible (jusqu'à N routes, configurable)
3. Les routes du dataset principal pour la cible sont retournées (toutes, sans filtre)
4. Les routes AiZ nouvelles sont validées étape par étape contre `generic_reactions.json` :
   - **Validées** (toutes les étapes trouvées) : conditions réelles, yield inclus dans le score
   - **Prédites** (aucune étape) : conditions Rxn-INSIGHT uniquement, yield exclu
5. Score pondéré 1/i² sur les critères

**Étapes de purification / isolation :**
Les étapes dont le SMILES du réactif est identique au SMILES du produit sont traitées
comme des étapes de purification. Elles s'affichent avec une **flèche pointillée marron**
et un badge "Purification", même si aucune liaison n'est formée.

**Trois sections de résultats :**
| Section | Source | Conditions | Yield dans le score |
|---------|--------|-----------|---------------------|
| 📚 Dataset | Chemistry by Design | Réelles | Oui |
| ✅ Validées | AiZ + dataset générique | Réelles (étapes validées) | Oui |
| 🤖 Prédites | AiZ + Rxn-INSIGHT | Prédites | Non |

**Fichiers nécessaires :**

| Fichier | Obligatoire | Rôle |
|---------|-------------|------|
| `reaction_dataset.json` | ✅ | Routes de synthèse principales |
| `config.yml` | ✅ | Configuration du modèle AiZynthFinder |
| `toxicity_dataset.json` | ✅ | Scores de sécurité |
| `data/uspto_rxn_insight.gzip` | ❌ | Base USPTO pour Rxn-INSIGHT |
| `generic_reactions.json` | ❌ | Réactions individuelles pour validation |
            """)

    st.markdown("---")
    st.caption(T["footer"])


if __name__ == "__main__":
    main()
