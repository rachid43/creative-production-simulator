"""Lean Creative Production Simulator - simplified view for the Head of Creative and Lead Creative.

Run:  streamlit run app.py        (full 9-page analyst version: streamlit run app_advanced.py)
"""
from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

from analytics.report import build_analysis
from config.defaults import baseline_config
from simulation.scenarios import run_replications
from visualization import simple_pages as SP

st.set_page_config(page_title="Creative Production Simulator", page_icon="🎬", layout="wide")
PAGES = ["Overview", "PODs", "Waste & Flow", "What-if"]
B = baseline_config()
DEFAULTS = dict(target=180, pods=6, editors=3, avail=80, complex=20, simple=30, fpa=85, review="strategist",
                cross=False, strat_diff=False, strat_avail=70, horizon=5, warmup=1, seed=42, reps=5)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
ss = st.session_state


def reset():
    for k in list(ss.keys()):
        if k in DEFAULTS or k.startswith("pod_tbl_"):
            del ss[k]
    for k in ("run", "wi"):
        ss.pop(k, None)


sb = st.sidebar
sb.title("🎬 Creative Simulator")
c1, c2 = sb.columns(2)
run_clicked = c1.button("▶ Run", type="primary", width="stretch")
c2.button("↺ Reset", width="stretch", on_click=reset)
sb.subheader("Demand & team")
sb.number_input("Weekly target (ads)", 20, 600, key="target")
sb.number_input("PODs", 1, 12, key="pods")
sb.number_input("Editors per POD", 1, 8, key="editors")
sb.slider("Productive availability %", 50, 100, key="avail", help="Share of the shift spent on production (rest: meetings, asset search). Playbook: 80%.")
sb.subheader("Work & quality")
sb.slider("Complex ads %", 0, 70, key="complex", help="Standard edit hours: Simple 1.5 · Medium 3 · Complex 5.")
sb.slider("Simple ads %", 0, 70, key="simple")
sb.slider("First-pass approval %", 40, 100, key="fpa", help="Ads approved without rework (assumption).")
sb.selectbox("Review done by", ["strategist", "dedicated", "lead_creative"], key="review")
sb.checkbox("Cross-POD balancing", key="cross", help="Idle editors pull ready briefs from busier PODs.")
with sb.expander("Advanced: roles & PODs", expanded=False):
    st.caption("Leave blank/off to use the organisation values above.")
    st.checkbox("Strategists' availability differs from editors", key="strat_diff",
                help="Applies to strategists (briefing + review), dedicated reviewers and the Lead Creative.")
    if ss["strat_diff"]:
        st.slider("Strategist productive availability %", 30, 100, key="strat_avail")
    st.caption("Per-POD demand share, complexity mix and first-pass approval: edit the table in the main area (\"POD assumptions\").")
sb.subheader("Simulation")
sb.slider("Weeks", 2, 12, key="horizon")
sb.slider("Warm-up weeks", 0, 4, key="warmup", help="Excluded from KPIs so the system is measured in steady state.")
sb.slider("Replications", 1, 15, key="reps")
sb.number_input("Random seed", 0, 99999, key="seed")
sb.caption("All other assumptions (brief/review time, rework, WIP limits…) use the playbook baseline. "
           "Change them in the full version: `streamlit run app_advanced.py`.")

try:
    if ss["simple"] + ss["complex"] > 100:
        raise ValueError("Simple % + Complex % must not exceed 100%.")
    s, cx = ss["simple"] / 100, ss["complex"] / 100
    n_p = int(ss["pods"])
    tbl = ss.get(f"pod_tbl_{n_p}", {})
    edited = tbl.get("edited_rows", {}) if isinstance(tbl, dict) else {}

    def cell(i, col, default=None):
        v = edited.get(i, {}).get(col, default)
        return default if v is None or v != v else v

    shares = [float(cell(i, "Demand share", 1.0)) for i in range(n_p)]
    pod_mix, pod_fp = [], []
    for i in range(n_p):
        c_, s_ = cell(i, "Complex %"), cell(i, "Simple %")
        if c_ is None and s_ is None:
            pod_mix.append(None)
        else:
            cc = (cx if c_ is None else c_ / 100)
            ss_ = (s if s_ is None else s_ / 100)
            if cc + ss_ > 1 + 1e-9:
                raise ValueError(f"POD {i+1}: Simple % + Complex % must not exceed 100%.")
            pod_mix.append({"Simple": ss_, "Medium": 1 - cc - ss_, "Complex": cc})
        f_ = cell(i, "First-pass %")
        pod_fp.append(None if f_ is None else f_ / 100)
    cfg = baseline_config(
        horizon_weeks=int(ss["horizon"]), warmup_weeks=min(int(ss["warmup"]), int(ss["horizon"]) - 1), seed=int(ss["seed"]), replications=int(ss["reps"]),
        org=dict(n_pods=n_p, editors_per_pod=int(ss["editors"]), productive_availability=ss["avail"] / 100,
                 strategist_availability=ss["strat_avail"] / 100 if ss["strat_diff"] else None),
        demand=dict(weekly_target_ads=float(ss["target"]), pod_share=shares if len(set(shares)) > 1 else None),
        complexity=dict(mix={"Simple": s, "Medium": 1 - s - cx, "Complex": cx}, pod_mix=pod_mix if any(m is not None for m in pod_mix) else None),
        quality=dict(first_pass_approval_rate=ss["fpa"] / 100, review_mode=ss["review"],
                     pod_first_pass=pod_fp if any(f is not None for f in pod_fp) else None),
        lean=dict(cross_pod_balancing=bool(ss["cross"]), cross_pod_review_sharing=bool(ss["cross"])))
except Exception as e:
    st.error(f"Invalid inputs: {e}")
    st.stop()

sig = hashlib.md5(cfg.model_dump_json().encode()).hexdigest()
if run_clicked or "run" not in ss:
    with st.spinner("Simulating…"):
        first, reps = run_replications(cfg, cfg.replications, cfg.seed)
        ss["run"] = dict(a=build_analysis(first, reps), sig=sig)
a = ss["run"]["a"]

st.title("Creative Production Simulator")
n_p = int(ss["pods"])
with st.expander("⚙ POD assumptions (optional) - leave a cell blank to use the organisation value", expanded=False):
    df0 = pd.DataFrame({"POD": [f"POD {p+1}" for p in range(n_p)], "Demand share": [1.0] * n_p,
                        "Complex %": [None] * n_p, "Simple %": [None] * n_p, "First-pass %": [None] * n_p})
    st.data_editor(df0, key=f"pod_tbl_{n_p}", hide_index=True, disabled=["POD"], width="stretch",
                   column_config={"Demand share": st.column_config.NumberColumn(help="Relative share of weekly demand", min_value=0.0, max_value=10.0, step=0.1),
                                  "Complex %": st.column_config.NumberColumn(min_value=0, max_value=100, step=5),
                                  "Simple %": st.column_config.NumberColumn(min_value=0, max_value=100, step=5),
                                  "First-pass %": st.column_config.NumberColumn(min_value=0, max_value=100, step=5)})
    st.caption("Medium % is the remainder. Press **▶ Run** after editing.")
if ss["run"]["sig"] != sig:
    st.warning("Inputs changed - press **▶ Run** to update. Showing the previous results.")
page = st.radio("View", PAGES, horizontal=True, label_visibility="collapsed", key="page")
st.divider()
if page == "Overview":
    SP.page_overview(a)
elif page == "PODs":
    SP.page_pods(a)
elif page == "Waste & Flow":
    SP.page_waste(a)
else:
    SP.page_whatif(a["cfg"])
