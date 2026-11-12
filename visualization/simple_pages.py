"""Simplified pages for the Head of Creative and Lead Creative: few KPIs, plain-language insights."""
from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from analytics import lean_metrics as lm
from simulation.scenarios import Scenario, comparison_table, mean_ci, preset_scenarios, run_scenarios

from . import charts as C
from .pod_network import pod_network_figure, snapshot_times

NOTE = "Simulated from the assumptions in the sidebar - not actual operational data."
ICON = {"green": "🟢 On track", "amber": "🟠 At risk", "red": "🔴 Overloaded"}


def _plot(fig, key):
    st.plotly_chart(fig, width="stretch", key=key)


def _band(a: Dict) -> str:
    k = a["kpis"]
    reps = a.get("reps")
    if reps is not None and len(reps) > 1:
        m, hw = mean_ci(reps["target_attainment_pct"])
        txt = f"{m:.0f}% ± {hw:.0f}"
        ok = m - hw >= 97
    else:
        txt, ok = f"{k['target_attainment_pct']:.0f}%", k["target_attainment_pct"] >= 97
    return txt, ok


# ----------------------------------------------------------------------------- Overview (Head of Creative)
def page_overview(a: Dict):
    k, cfg, b, ts = a["kpis"], a["cfg"], a["bott"], a["takt"]
    st.caption(NOTE)
    txt, ok = _band(a)
    load = ts["planned_load_ratio"] * 100
    msg = (f"**Can we deliver {k['weekly_target']:.0f} ads/week?** Simulated attainment: **{txt}** of target "
           f"({k['accepted_per_week']:.0f} accepted/week). Planned editor load is {load:.0f}% of productive capacity "
           f"({ts['required_hours_per_editor_week']:.1f} h needed vs {ts['available_productive_h_per_editor_week']:.1f} h available per editor-week).")
    if ok:
        st.success(msg)
    else:
        st.warning(msg + (" Above ~90% load, variability builds queues, so output falls short even when average capacity looks sufficient." if load > 90 else ""))
    c = st.columns(6)
    c[0].metric("Accepted / week", f"{k['accepted_per_week']:.0f}", f"{k['accepted_per_week']-k['weekly_target']:+.0f} vs target",
                help="Only ads approved after QA count. An exported ad waiting for review is still work in progress.")
    c[1].metric("Lead time", f"{k['lead_time_h']:.1f} h", help="Working hours from demand arrival to approval, including all waiting.")
    c[2].metric("On-time delivery", f"{k['on_time_delivery_pct']:.0f}%", help=f"Approved within the {cfg.demand.sla_days:g}-day SLA (assumption).")
    c[3].metric("First-pass approval", f"{k['first_pass_yield_pct']:.0f}%", help="Ads approved at first review, no rework.")
    c[4].metric("Editor utilization", f"{k['editor_utilization_pct']:.0f}%", help="Busy editing hours / available productive hours. Above 95% = critical load.")
    c[5].metric("Work in progress", f"{k['wip_avg']:.0f} ads", help="Average ads started (brief begun) but not yet approved.")
    l, r = st.columns(2)
    with l:
        _plot(C.cumulative_plan_vs_actual(a["daily"], k["weekly_target"], lm.window(a["res"])["d_w"], cfg.org.days_per_week), "o_cum")
    with r:
        _plot(C.pod_target_vs_actual(a["pods"], a["pod_status"]), "o_pod")
    st.subheader("What is holding output back?")
    with st.container(border=True):
        st.markdown(f"**{b['primary']}**")
        for e in b["explanation"][:4]:
            st.markdown(f"- {e}")
    st.subheader("Top 3 actions to test")
    for r_ in a["recs"][:3]:
        with st.container(border=True):
            st.markdown(f"**{r_['priority']}. {r_['title']}**  \n{r_['observed']}  \n*Try:* {r_['intervention']}")
    st.caption("Open the **What-if** page to test these as scenarios. Directions are hypotheses, not guaranteed gains.")


def _clock(cfg, t, t_end):
    H, D = cfg.org.hours_per_day, cfg.org.days_per_week
    if t >= t_end - 1e-6:
        return f"End of week {cfg.horizon_weeks}"
    return f"Week {int(t // (H * D)) + 1}, day {int(t // H) % D + 1}, hour {t % H:.0f}"


# ----------------------------------------------------------------------------- PODs (Lead Creative)
def page_pods(a: Dict):
    res, cfg, pods = a["res"], a["cfg"], a["pods"]
    st.caption(NOTE + " Use this view for the daily question: who is overloaded, who has spare capacity, what to rebalance?")
    times = snapshot_times(res)
    i = st.select_slider("Time in simulation", options=list(range(len(times))), value=len(times) - 1, key="p_i",
                         format_func=lambda x: _clock(cfg, times[x], res.t_end))
    with st.expander("How to read this view", expanded=False):
        st.markdown("- **Boxes** left to right: Backlog → Briefing → Ready → Editing → Review(+rework) → Approved. The big number = ads in that stage now.\n"
                    "- **Fill** = how full the stage is versus its limit (ready buffer 6, editing limit 3, review limit +2). **Red fill = over the limit** (pile-up).\n"
                    "- **Circles**: strategist (S) and editors (E). Blue = editing, orange = rework, grey = idle, red = blocked by review, yellow = waiting for assets, white = absent.\n"
                    "- **Border colour**: green on track · amber at risk · red = behind target, editors ≥95% loaded, or ≥3 ads waiting per editor.\n"
                    "- Week 1 is warm-up (excluded from KPIs). The last slider position is the end of the run.")
    _plot(pod_network_figure(res, times[i], None, height=680), "p_net")
    st.subheader("POD scorecard (measured window)")
    t = pods.copy()
    t["Status"] = t["pod"].map(lambda p: ICON[a["pod_status"][int(p)]])
    t["Attainment %"] = (t["attainment"] * 100).round(0)
    t["Editor util %"] = (t["utilization"] * 100).round(0)
    st.dataframe(t[["name", "Status", "accepted", "target", "Attainment %", "Editor util %", "queue", "lead_time_h", "fpy", "rework_h"]]
                 .rename(columns={"name": "POD", "accepted": "Accepted", "target": "Target", "queue": "Waiting for editor", "lead_time_h": "Lead time h",
                                  "fpy": "First-pass %", "rework_h": "Rework h"}).round(1), hide_index=True, width="stretch")
    hi, lo = t.sort_values("utilization").iloc[-1], t.sort_values("utilization").iloc[0]
    if hi["utilization"] - lo["utilization"] > 0.08:
        st.info(f"**Rebalancing hint:** {hi['name']} runs at {hi['Editor util %']:.0f}% with {int(hi['queue'])} ads waiting, while {lo['name']} runs at {lo['Editor util %']:.0f}%. "
                "Turn on **Cross-POD balancing** in the sidebar to test lending ready briefs between PODs.")
    else:
        st.info("POD loads are within 8 points of each other - no strong rebalancing signal.")
    pick = st.selectbox("Editor detail for", list(range(cfg.org.n_pods)), format_func=lambda p: f"POD {p+1}", key="p_pick")
    e = a["editors"]
    e = e[e["pod"] == pick]
    st.dataframe(e.assign(**{"Util %": (e["utilization"] * 100).round(0)})[["worker", "Util %", "ads_accepted", "avg_cycle_h", "rework_h"]]
                 .rename(columns={"worker": "Editor", "ads_accepted": "Accepted", "avg_cycle_h": "Avg edit h", "rework_h": "Rework h"}).round(2),
                 hide_index=True, width="stretch")


# ----------------------------------------------------------------------------- Waste & Flow
def page_waste(a: Dict):
    k, res, cfg = a["kpis"], a["res"], a["cfg"]
    mura, muri = a["mura"], a["muri"]
    w = lm.window(res)
    st.caption(NOTE)
    c = st.columns(4)
    c[0].metric("Mura (unevenness)", f"{k['mura_index']:.0f}/100", mura["level"], delta_color="off", help="Variation in workload, demand, output and queues. LOWER is better.")
    c[1].metric("Muri (overburden)", f"{k['muri_index']:.0f}/100", help="Share of person-days at high or critical utilization. LOWER is better.")
    c[2].metric("Muda (waste)", f"{k['muda_waste_share_pct']:.0f}%", help="Share of all ad time spent waiting, blocked or reworking. LOWER is better.")
    c[3].metric("Flow efficiency", f"{k['flow_efficiency_pct']:.0f}%", help="Value-adding time / total lead time. HIGHER is better.")
    st.caption("Direction: **Mura ↓ · Muri ↓ · Muda ↓ · Flow efficiency ↑**.  Zero is not a realistic target for any of them - watch the trend and the biggest contributor.")
    with st.expander("How are Mura, Muri, Muda and flow efficiency calculated?", expanded=False):
        comp = mura["components"].copy()
        comp["cv"], comp["score"] = comp["cv"].round(2), comp["score"].round(0)
        st.markdown(f"**Mura (0-100, lower = more level).** Six measures of variation, each as a coefficient of variation (CV = standard deviation / mean). "
                    f"Each is scored `min(1, CV / {cfg.thresholds.mura_cv_reference:g}) x 100`, and the index is the plain average of the six. "
                    f"Bands: below {cfg.thresholds.mura_low:g} low, below {cfg.thresholds.mura_high:g} moderate, else high. Current components:")
        st.dataframe(comp.rename(columns={"component": "Measure", "cv": "CV", "score": "Score (0-100)"}), hide_index=True, width="stretch")
        st.markdown(f"**Muri (0-100, lower = less overburden).** Every person-day (editors and strategists) is classed by utilization = busy productive hours / available productive hours. "
                    f"`Muri = 100 x (0.5 x share of days at {cfg.thresholds.util_normal_upto*100:.0f}-{cfg.thresholds.util_high_upto*100:.0f}% + 1.0 x share of days above {cfg.thresholds.util_high_upto*100:.0f}%)`. "
                    f"Now: {muri['share_high']*100:.0f}% of person-days are in the high band and {muri['share_critical']*100:.0f}% are critical. Thresholds are management settings, not universal Lean standards.")
        st.markdown("**Muda (%, lower = less waste).** Every working hour of every ad's lead time is assigned to one category as it happens. "
                    "`Muda = waste hours / all hours`, where waste = waiting for a strategist, an editor, review or assets, being blocked by the review limit, and rework editing and re-review. "
                    "Necessary review time and the availability overhead (meetings) are *not* counted as waste.")
        st.markdown("**Flow efficiency (%, higher = better).** `value-adding hours (briefing + editing) / total lead time x 100`. "
                    "100% would mean an ad never waits. The classification can be changed in the full version.")
    l, r = st.columns(2)
    with l:
        _plot(C.lead_time_breakdown(a["decomposition"], cfg), "w_dec")
    with r:
        _plot(C.pareto_horizontal(a["waste"], "Biggest causes of delay"), "w_par")
    top = a["waste"].iloc[0] if len(a["waste"]) else None
    if top is not None:
        st.info(f"**Largest single loss:** {top['cause']} - {top['share_pct']:.0f}% of all waste hours. Fix this first.")
    st.subheader("Load by person and day")
    t = cfg.thresholds
    st.markdown(f"Each row is one person (strategist S, then the editors E of each POD); each column is one working day of the **measured window** "
                f"(weeks {cfg.warmup_weeks+1}-{cfg.horizon_weeks}; the {cfg.warmup_weeks} warm-up week(s) are left out, which is why the days start at week {cfg.warmup_weeks+1}). "
                f"**Green = normal ({t.util_spare_below*100:.0f}-{t.util_normal_upto*100:.0f}%) is the target.** Blue means spare capacity (below {t.util_spare_below*100:.0f}%) - good only if there is no queue; "
                f"amber and red mean overburden.")
    _plot(C.load_heatmap(res.worker_day, cfg, w["d_w"]), "w_heat")


# ----------------------------------------------------------------------------- What-if
def page_whatif(cfg):
    st.caption("Compares the current setup (A) with alternatives using the same random numbers. " + NOTE)
    presets = preset_scenarios(cfg)
    names = {s.key: s.name for s in presets}
    pick = st.multiselect("Scenarios", [s.key for s in presets[1:]], default=["B", "E", "F"], format_func=lambda x: names[x])
    reps = st.slider("Replications", 3, 15, 5)
    if st.button("Run what-if", type="primary"):
        sel = [presets[0]] + [s for s in presets if s.key in pick]
        with st.spinner("Simulating…"):
            runs = run_scenarios(sel, cfg, reps, cfg.seed)
        st.session_state["wi"] = (comparison_table(runs, "A", {s.key: s.name for s in sel}), [s.key for s in sel], reps)
    if "wi" not in st.session_state:
        st.info("Pick scenarios and press **Run what-if**.")
        return
    tab, keys, reps = st.session_state["wi"]
    show = {"accepted_per_week": "Accepted / week", "lead_time_h": "Lead time (h)", "on_time_delivery_pct": "On-time %",
            "editor_utilization_pct": "Editor util %", "backlog_end": "Backlog at end"}
    rows = []
    for key in keys:
        row = {"Scenario": names.get(key, key)}
        for m, lab in show.items():
            g = tab[(tab.scenario == key) & (tab.metric == m)].iloc[0]
            row[lab] = f"{g['mean']:.1f}" + ("" if key == "A" else f" ({g['delta']:+.1f}{'' if g['distinguishable'] else ', noise'})")
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Means over {reps} replications. Brackets = change vs A; 'noise' = the change is within replication variation and should not be trusted.")
    _plot(C.scenario_metric_chart(tab, "accepted_per_week", {k_: names.get(k_, k_) for k_ in keys}), "wi_acc")
    _plot(C.scenario_metric_chart(tab, "lead_time_h", {k_: names.get(k_, k_) for k_ in keys}), "wi_lead")
