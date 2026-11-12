"""Streamlit page renderers (presentation only; all numbers come from analytics/ and simulation/)."""
from __future__ import annotations

import time
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

from analytics import lean_metrics as lm
from analytics.report import executive_pdf, kpi_ci_table, results_zip
from config.schema import SimConfig
from simulation.entities import WAIT_CATEGORIES
from simulation.scenarios import (
    COMPARE_METRICS, Scenario, comparison_table, mean_ci, preset_scenarios, run_replications, run_scenarios,
)
from simulation.scheduling import plan_week

from . import charts as C
from .pod_network import pod_network_figure, pod_state_at, snapshot_times

SIM_NOTE = "Simulated output from the stated assumptions - not actual operational data."
D_ = lm.DEFINITIONS


def f(v, d=1, suffix=""):
    return "n/a" if v is None or v != v else f"{v:,.{d}f}{suffix}"


def plot(fig, key=None):
    st.plotly_chart(fig, width="stretch", key=key)


def ci_note(a: Dict, col: str, d=1) -> str:
    reps = a.get("reps")
    if reps is None or len(reps) < 2:
        return ""
    m, hw = mean_ci(reps[col])
    return f"{len(reps)} reps: {m:,.{d}f} ± {hw:,.{d}f}"


def card(col, label, value, help_, delta=None, delta_color="normal"):
    col.metric(label, value, delta=delta, delta_color=delta_color, help=help_)


# =========================================================================== executive
def page_executive(a: Dict):
    k, cfg, b = a["kpis"], a["cfg"], a["bott"]
    st.caption(SIM_NOTE + f"  Horizon {cfg.horizon_weeks} wk (warm-up {cfg.warmup_weeks}), seed {a['res'].seed}.")
    ts = a["takt"]
    # feasibility verdict
    reps = a.get("reps")
    att = k["target_attainment_pct"]
    verdict = f"Simulated attainment {att:.0f}% of the {k['weekly_target']:.0f}/week target"
    if reps is not None and len(reps) > 1:
        m, hw = mean_ci(reps["target_attainment_pct"])
        verdict = f"Simulated attainment {m:.0f}% ± {hw:.0f} (95% CI over {len(reps)} replications) of the {k['weekly_target']:.0f}/week target"
        ok = m - hw >= 97
    else:
        ok = att >= 97
    on_paper = f"On paper each editor needs {ts['required_hours_per_editor_week']:.1f} h/week (standard + expected rework) vs {ts['available_productive_h_per_editor_week']:.1f} h available = {ts['planned_load_ratio']*100:.0f}% planned load."
    (st.success if ok else st.warning)(f"**Is the weekly target achievable?** {verdict}. {on_paper}"
                                         + ("" if ok else " Variability and queues, not average capacity alone, erode output near 100% load."))

    r1 = st.columns(4)
    card(r1[0], "Weekly target", f(k["weekly_target"], 0), "Customer demand: accepted ads required per week.")
    card(r1[1], "Simulated accepted / week", f(k["accepted_per_week"]), D_["Throughput"] + " " + ci_note(a, "accepted_per_week"),
         delta=f"{k['accepted_per_week']-k['weekly_target']:+.1f} vs target")
    card(r1[2], "Target attainment", f(att, 1, "%"), D_["Takt attainment"] + " " + ci_note(a, "target_attainment_pct"))
    card(r1[3], "Estimated editing capacity", f(k["estimated_capacity_ads_per_week"], 0, " ads/wk"),
         "Analytic: available productive editor hours / (mean standard hours + expected rework). An estimate, not a simulation output.")
    r2 = st.columns(4)
    card(r2[0], "Avg editing cycle time", f(k["cycle_time_edit_h"], 2, " h"), D_["Cycle time"] + f" Including rework: {k['cycle_time_edit_incl_rework_h']:.2f} h. Elapsed incl. overhead: {k['edit_elapsed_h']:.2f} h.")
    card(r2[1], "Avg end-to-end lead time", f(k["lead_time_h"], 1, " h"), D_["Lead time"] + f" Median {k['lead_time_median_h']:.1f} h, P90 {k['lead_time_p90_h']:.1f} h. " + ci_note(a, "lead_time_h"))
    card(r2[2], "First-pass approval", f(k["first_pass_yield_pct"], 1, "%"), D_["First-pass yield"])
    card(r2[3], "Avg editor utilization", f(k["editor_utilization_pct"], 1, "%"), D_["Editor utilization"])
    r3 = st.columns(4)
    card(r3[0], "Total WIP (avg / end)", f"{f(k['wip_avg'], 0)} / {k['wip_end']}", D_["WIP"] + f" Backlog awaiting briefs at end: {k['backlog_end']}.")
    card(r3[1], "On-time delivery", f(k["on_time_delivery_pct"], 1, "%"), f"Accepted on or before due date / (accepted + open ads already past due). SLA = {cfg.demand.sla_days:g} working days.")
    card(r3[2], "Flow efficiency", f(k["flow_efficiency_pct"], 1, "%"), D_["Flow efficiency"])
    card(r3[3], "Likely bottleneck", b["primary"].split(" (")[0], "Combined evidence score - see explanation below. Not simply the most utilised resource.")

    c1, c2 = st.columns(2)
    with c1:
        plot(C.cumulative_plan_vs_actual(a["daily"], k["weekly_target"], lm.window(a["res"])["d_w"], cfg.org.days_per_week), "ex_cum")
    with c2:
        plot(C.pod_target_vs_actual(a["pods"], a["pod_status"]), "ex_pod")
    c3, c4 = st.columns(2)
    with c3:
        plot(C.pareto(a["waste"], "cause", "hours", "Pareto of delay causes (job-hours of waste)"), "ex_pareto")
    with c4:
        plot(C.lead_time_decomposition(a["decomposition"]), "ex_dec")
        with st.container(border=True):
            st.markdown(f"**Constraint analysis: {b['primary']}**")
            for e in b["explanation"]:
                st.markdown(f"- {e}")
    with st.expander("Mura / Muri / Muda summary"):
        m = st.columns(3)
        card(m[0], "Mura index", f(k["mura_index"], 0, f" ({a['mura']['level']})"), D_["Mura"])
        card(m[1], "Muri index", f(k["muri_index"], 0), D_["Muri"])
        card(m[2], "Muda waste share", f(k["muda_waste_share_pct"], 0, "%"), D_["Muda"])
    ex = st.columns(2)
    ex[0].download_button("Download all results (CSV zip)", results_zip(a), "lean_sim_results.zip", "application/zip")
    try:
        ex[1].download_button("Download executive summary (PDF)", executive_pdf(a), "executive_summary.pdf", "application/pdf")
    except Exception as e:  # pragma: no cover
        ex[1].caption(f"PDF export unavailable: {e}")


# =========================================================================== network
def page_network(a: Dict):
    res, cfg = a["res"], a["cfg"]
    st.caption(SIM_NOTE + "  Colour = status per configurable thresholds (sidebar > Status thresholds).")
    times = snapshot_times(res)
    step = max(1, int(round(cfg.org.hours_per_day / cfg.snapshot_interval_hours / 2)))  # half-day playback steps
    idx = list(range(0, len(times), 1))
    c1, c2, c3 = st.columns([5, 1.2, 1.6])
    with c1:
        i = st.select_slider("Simulation time", options=idx, value=idx[-1], key="net_i",
                             format_func=lambda x: f"wk {int(times[x]//(cfg.org.hours_per_day*cfg.org.days_per_week))+1} · d {int(times[x]//cfg.org.hours_per_day)%cfg.org.days_per_week+1} · {times[x]%cfg.org.hours_per_day:.0f}h")
    with c2:
        play = st.button("▶ Play", help="Steps through the time-stamped snapshots (half-day steps).")
    with c3:
        sel = st.selectbox("Inspect POD", list(range(cfg.org.n_pods)), format_func=lambda p: f"POD {p+1}", key="net_pod")
    holder = st.empty()
    if play:
        for j in range(i, len(times), step):
            holder.plotly_chart(pod_network_figure(res, times[j], sel), width="stretch", key=f"play_{j}")
            time.sleep(0.12)
    else:
        holder.plotly_chart(pod_network_figure(res, times[i], sel), width="stretch", key="net_static")
    stt = pod_state_at(res, times[i])
    st.markdown("**Status rules** — "
                f"Red: attainment < {cfg.thresholds.attain_amber*100:.0f}% or utilization ≥ {cfg.thresholds.pod_util_red*100:.0f}% or waiting work ≥ {cfg.thresholds.pod_queue_red_per_editor:g}/editor; "
                f"Amber: attainment < {cfg.thresholds.attain_green*100:.0f}% or utilization ≥ {cfg.thresholds.pod_util_amber*100:.0f}%; otherwise Green. Playback utilization is sampled hourly up to the selected time.")
    st.subheader(f"POD {sel+1} — editor-level performance (full measured window)")
    ed = a["editors"]
    e = ed[ed["pod"] == sel].copy()
    show = e[["worker", "utilization", "ads_accepted", "avg_cycle_h", "edit_h", "rework_h", "capacity_h", "speed", "can_complex"]].copy()
    show["utilization"] = (show["utilization"] * 100).round(1)
    show.columns = ["Editor", "Utilization %", "Ads accepted", "Avg edit cycle h", "Edit h", "Rework h", "Capacity h", "Speed factor", "Complex-certified"]
    st.dataframe(show.round(2), hide_index=True, width="stretch")
    plot(C.editor_timeline(res.jobs, sel, res.t_end - cfg.org.hours_per_day * cfg.org.days_per_week, res.t_end), "net_tl")
    with st.expander("All POD details"):
        for _, r in a["pods"].iterrows():
            with st.expander(f"{r['name']} — {a['pod_status'][int(r['pod'])].upper()} · accepted {r['accepted']:.0f}/{r['target']:.0f} · util {r['utilization']*100:.0f}%"):
                st.write({"Backlog": int(r["backlog"]), "Ready": int(r["ready"]), "Editing": int(r["editing"]), "Review WIP": int(r["review_wip"]),
                          "Rework queue": int(r["rework_q"]), "Lead time (h)": round(r["lead_time_h"], 1), "First-pass yield %": round(r["fpy"], 1),
                          "Rework hours": round(r["rework_h"], 1), "Ads transferred out": int(r["transfers_out"])})


# =========================================================================== Mura
def page_mura(a: Dict):
    res, cfg, m, k = a["res"], a["cfg"], a["mura"], a["kpis"]
    st.caption("**Mura = unevenness.** " + D_["Mura"] + " " + SIM_NOTE)
    cols = st.columns(4)
    card(cols[0], "Mura index", f(m["index"], 0, f" · {m['level']}"), f"{D_['Mura']} Bands: < {cfg.thresholds.mura_low:g} low, < {cfg.thresholds.mura_high:g} moderate, else high. Reference CV = {cfg.thresholds.mura_cv_reference:g}.")
    card(cols[1], "Workload Balance Index", f(m["workload_balance_index"], 2), D_["Workload Balance Index"])
    card(cols[2], "Editor utilization CV", f(m["components"].set_index("component").cv.iloc[0], 3), "Coefficient of variation of editor utilization across editors.")
    card(cols[3], "Max / min editor workload", f(m["busy_ratio"], 2, "x"), f"Busiest editor {m['busy_max']:.0f} h vs least busy {m['busy_min']:.0f} h in the window.")
    with st.expander("How the Mura index is calculated", expanded=False):
        st.markdown("Index = mean of component scores. Each component score = **min(1, CV / reference CV) × 100**, where CV = standard deviation / mean "
                    f"(reference CV = {cfg.thresholds.mura_cv_reference:g}, configurable). Components are weighted equally. Higher = more uneven.")
        st.dataframe(m["components"].round(3), hide_index=True, width="stretch")
    c1, c2 = st.columns([1, 1])
    with c1:
        plot(C.mura_components(m["components"]), "mu_comp")
    with c2:
        plot(C.pod_workload_heatmap(a["muri"]["pod_week"]), "mu_hm")
    ed = lm.editor_day_table(res)
    plot(C.editor_util_heatmap(ed), "mu_ed")
    c3, c4 = st.columns(2)
    with c3:
        plot(C.daily_variability(a["daily"], lm.window(res)["d_w"], k["weekly_target"] / cfg.org.days_per_week), "mu_daily")
    with c4:
        plot(C.workload_box(ed), "mu_box")
    # demand vs capacity per day (standard hours)
    w = lm.window(res)
    j = a["cohort"]
    rw = lm.expected_rework_hours_per_ad(cfg)
    dd = j.assign(day=(j["arrival"] // w["H"]).astype(int)).groupby("day").agg(n=("job_id", "count"), h=("est_hours", "sum"))
    demand = (dd["h"] + dd["n"] * rw).reindex(range(w["d_w"], w["n_days"]), fill_value=0)
    sp = res.workers.set_index("worker")["speed"]
    capd = ed.assign(c=ed["capacity_h"] * ed["worker"].map(sp)).groupby("day")["c"].sum()
    plot(C.demand_vs_capacity(a["daily"], demand, capd), "mu_dvc")
    st.subheader("Idle capacity beside overload")
    imb = m["imbalance"]
    if len(imb) and imb["concurrent"].any():
        x = imb[imb["concurrent"]].copy()
        x["overloaded_pods"] = x["overloaded_pods"].map(", ".join)
        x["underused_pods"] = x["underused_pods"].map(", ".join)
        st.warning(f"{int(imb['concurrent'].sum())} of {len(imb)} days had a POD in the spare band (<{cfg.thresholds.util_spare_below*100:.0f}% utilization) while another POD had queued work and high load. "
                   f"Idle hours in the under-used PODs on those days: {x['idle_hours_underused'].sum():.0f} h. "
                   + ("Cross-POD balancing is ON." if cfg.lean.cross_pod_balancing else "Cross-POD balancing is OFF - enable it in Lean controls to test."))
        st.dataframe(x.assign(day=x["day"] + 1).rename(columns={"day": "Day"}).round(1), hide_index=True, width="stretch")
    else:
        st.success("No day shows an idle POD beside an overloaded POD.")


# =========================================================================== Muri
def page_muri(a: Dict):
    res, cfg, mu = a["res"], a["cfg"], a["muri"]
    t = cfg.thresholds
    st.caption("**Muri = overburden.** " + D_["Muri"] + " " + SIM_NOTE)
    st.info(f"Load bands (management thresholds, not universal Lean standards): spare < {t.util_spare_below*100:.0f}% · normal {t.util_spare_below*100:.0f}–{t.util_normal_upto*100:.0f}% · high {t.util_normal_upto*100:.0f}–{t.util_high_upto*100:.0f}% · critical > {t.util_high_upto*100:.0f}%. "
            "**Utilization** (busy hours / available productive hours) measures what actually happened; **planned workload overload** (assigned standard hours / capacity) shows what was demanded — they differ when work queues up.")
    cols = st.columns(4)
    card(cols[0], "Muri index", f(mu["index"], 0), D_["Muri"])
    card(cols[1], "Capacity shortfall", f(mu["capacity_shortfall_h"], 0, " h"), "Planned standard hours (incl. expected rework) above editor capacity, summed over POD-weeks.")
    card(cols[2], "Overtime required", f(mu["overtime_required_per_week"], 0, " h/wk"), "Clock-hours/week needed to cover the shortfall (forecast). Actual overtime used: " + f"{mu['overtime_used_per_week']:.0f} h/wk (cap {mu['overtime_cap_per_day']:g} h/day).")
    card(cols[3], "Critical-load share", f(mu["share_critical"] * 100, 0, "%"), "Share of resource-days above the critical threshold.")
    wd = mu["worker_day"]
    tabs = st.tabs(["Editors", "Strategists", "Reviewers"])
    with tabs[0]:
        plot(C.worker_util_heatmap(wd, "editor"), "mr_ed")
    with tabs[1]:
        plot(C.worker_util_heatmap(wd, "strategist"), "mr_st")
        st.caption("Strategists share one capacity between briefing and (in the default mode) review - never counted twice.")
    with tabs[2]:
        if (wd["role"].isin(["reviewer", "lead_creative"])).any():
            plot(C.worker_util_heatmap(wd[wd["role"].isin(["reviewer", "lead_creative"])].assign(role="reviewer"), "reviewer"), "mr_rv")
        else:
            st.info("Review is performed by strategists in this configuration (see Quality & review → Review performed by).")
    c1, c2 = st.columns(2)
    with c1:
        plot(C.capacity_vs_assigned(mu["per_worker"][mu["per_worker"]["role"] == "editor"]), "mr_cap")
    with c2:
        plot(C.overload_distribution(wd), "mr_dist")
    c3, c4 = st.columns(2)
    with c3:
        plot(C.overtime_forecast(mu["pod_week"]), "mr_ot")
    with c4:
        plot(C.bottleneck_ranking(a["bott"]["table"]), "mr_bn")
    st.subheader("Consecutive high-utilization days")
    s = mu["streaks"].sort_values("longest_high_streak_days", ascending=False)
    n_long = int((s["longest_high_streak_days"] >= t.consecutive_high_days).sum())
    st.write(f"{n_long} of {len(s)} resources had ≥ {t.consecutive_high_days} consecutive days at/above {t.util_normal_upto*100:.0f}% utilization.")
    st.dataframe(s.head(12), hide_index=True, width="stretch")
    # why near-100% matters
    rho = np.linspace(0.5, 0.99, 50)
    ca2, cs2 = cfg.demand.demand_cv ** 2 + 0.05, cfg.complexity.editing_time_cv ** 2 + 0.2
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=rho * 100, y=rho / (1 - rho) * (ca2 + cs2) / 2, mode="lines", line=dict(color=C.BLUE, width=3)))
    u = a["kpis"]["editor_utilization_pct"] / 100
    fig.add_vline(x=u * 100, line_dash="dash", line_color=C.ORANGE, annotation_text="simulated editor utilization")
    fig.update_xaxes(title="Utilization %")
    fig.update_yaxes(title="Queue wait / processing time")
    plot(C._style(fig, "Why ~100% utilization is dangerous (Kingman approximation, illustrative)", legend=False, h=300), "mr_king")
    st.caption("With variable demand and processing times, average waiting grows roughly as ρ/(1−ρ): moving from 85% to 95% load more than triples queueing time, and lead time rises faster than output. This curve is a textbook approximation for one server, shown for intuition; the simulation results above are the evidence for this system.")


# =========================================================================== Muda
def page_muda(a: Dict):
    res, cfg, k = a["res"], a["cfg"], a["kpis"]
    coh = a["cohort"]
    st.caption("**Muda = waste.** " + D_["Muda"] + " Waste is accumulated from simulation events (each hour of every ad's lead time is assigned to exactly one category). " + SIM_NOTE)
    cols = st.columns(4)
    card(cols[0], "Waste share of job-hours", f(k["muda_waste_share_pct"], 0, "%"), D_["Muda"])
    card(cols[1], "Flow efficiency", f(k["flow_efficiency_pct"], 1, "%"), D_["Flow efficiency"])
    ll = a["little"]
    raw = float((coh["T_brief"] + coh["T_edit"] + coh["T_review"] + coh["overhead"]).mean()) if len(coh) else 0
    cw = ll["throughput_per_hour"] * raw
    card(cols[2], "Excess WIP vs critical WIP", f(max(0, ll["L_observed"] - cw), 0, " ads"), D_["Critical WIP"] + f" Critical WIP ≈ {cw:.0f}; observed {ll['L_observed']:.0f}.")
    dv = lm.delivered_in_window(res)
    card(cols[3], "Handoffs per ad (unnecessary)", f"{dv['handoffs'].mean():.1f} ({dv['unnecessary_handoffs'].mean():.2f})",
         "Handoffs: strategist→editor, editor→review, review→rework. 'Unnecessary' = rework returns, editor changes on rework and cross-POD transfers.")
    c1, c2 = st.columns(2)
    with c1:
        plot(C.lead_time_decomposition(a["decomposition"]), "md_dec")
    with c2:
        plot(C.pareto(a["waste"], "cause", "hours", "Waiting-time & waste Pareto (job-hours)"), "md_par")
    # capacity loss (person-hours) from events
    ed = lm.editor_day_table(res)
    idle = lambda c: float(ed.get(f"idle_{c}", pd.Series([0.0])).sum())
    cap = pd.DataFrame([
        ("Rework editing", float(ed["busy_rework"].sum()), "loss"),
        ("No ready brief (briefs queued, strategist busy)", idle("starved_brief_supply"), "loss"),
        ("Blocked by review WIP limit", idle("blocked_review_wip"), "loss"),
        ("Waiting for assets", idle("asset_wait"), "loss"),
        ("Edit WIP limit reached", idle("blocked_wip_limit"), "loss"),
        ("Skill mismatch", idle("skill_mismatch"), "loss"),
        ("Interruptions", idle("interruption"), "loss"),
        ("Brief in progress (necessary process time)", idle("brief_in_progress"), "necessary"),
        ("No demand in POD", idle("no_demand"), "demand-limited"),
    ], columns=["cause", "hours", "kind"]).sort_values("hours", ascending=False)
    plot(C.capacity_loss_bar(cap), "md_cap")
    c3, c4 = st.columns(2)
    with c3:
        plot(C.rework_by_pod(a["pods"]), "md_rw")
    with c4:
        plot(C.flow_eff_by_pod(a["flow_eff_pod"]), "md_fe")
    plot(C.wip_trend(res.pod_snapshots[res.pod_snapshots["t"] >= lm.window(res)["t_w"]]), "md_wip")
    st.subheader("Value-added classification (configurable in sidebar)")
    cl = cfg.classification
    rows = [("Touch: " + n, c) for n, c in cl.touch.items()] + [("Wait: " + n.replace("wait_", ""), c) for n, c in cl.waits.items()]
    st.dataframe(pd.DataFrame(rows, columns=["Activity", "Class"]), hide_index=True, width="stretch")
    st.caption("Review time is classed as necessary non-value-adding (a required quality gate); waiting for review is classed as waste. Change the classes in the sidebar to see flow efficiency respond.")


# =========================================================================== takt & flow
def page_takt(a: Dict):
    res, cfg, k, ts = a["res"], a["cfg"], a["kpis"], a["takt"]
    st.caption("Correct Lean definitions: **takt** is demand-driven pace; **cycle/processing time** is measured work; **lead time** includes waiting; **throughput** counts only accepted ads. " + SIM_NOTE)
    c = st.columns(4)
    card(c[0], "Takt (gross, per editor)", f(ts["gross_takt_h_per_ad_per_editor"], 2, " h/ad"), D_["Takt time"])
    card(c[1], "Takt (effective, productive)", f(ts["effective_takt_h_per_ad_per_editor"], 2, " h/ad"), "Gross takt x productive availability: the productive hours an editor can spend per ad to meet demand.")
    card(c[2], "Org takt (between accepted ads)", f(ts["org_takt_minutes_between_ads"], 1, " min"), "Available weekly minutes / weekly demand across the whole organisation.")
    card(c[3], "Required vs available", f"{ts['required_hours_per_editor_week']:.1f} / {ts['available_productive_h_per_editor_week']:.1f} h", "Per editor per week: (standard hours + expected rework) x ads, vs productive hours available.")
    st.dataframe(pd.DataFrame([
        ("Customer-demand takt", f"{ts['gross_takt_h_per_ad_per_editor']:.2f} h per ad per editor", "Available production time / demand"),
        ("Effective available editing time", f"{ts['available_productive_h_per_editor_week']:.1f} h/editor/week", "Hours x availability x (1-absence)"),
        ("Actual processing cycle time", f"{k['cycle_time_edit_h']:.2f} h (first pass), {k['cycle_time_edit_incl_rework_h']:.2f} h incl. rework", "Simulated touch time"),
        ("End-to-end lead time", f"{k['lead_time_h']:.1f} working h", "Approval - arrival"),
        ("Throughput", f"{k['accepted_per_week']:.1f} accepted ads/week ({k['throughput_per_day']:.1f}/day)", "Accepted only"),
        ("First-pass yield / rework-adjusted capacity", f"{k['first_pass_yield_pct']:.0f}% / {ts['capacity_ads_per_week']:.0f} ads/week", "Capacity after expected rework"),
        ("Weighted production units", f"{k['weighted_units_per_week']:.0f} standard h/week", "Playbook open question: weighting ads by complexity avoids unfair allocation"),
    ], columns=["Concept", "Value", "Definition"]), hide_index=True, width="stretch")
    if ts["mean_standard_hours"] > ts["effective_takt_h_per_ad_per_editor"]:
        st.error("Mean standard editing hours exceed the effective takt: the target is infeasible without changing the mix.")
    else:
        st.info(f"Mean standard editing hours ({ts['mean_standard_hours']:.2f} h) fit inside the effective takt ({ts['effective_takt_h_per_ad_per_editor']:.2f} h) "
                f"with only {(ts['effective_takt_h_per_ad_per_editor']-ts['required_hours_per_ad'])/ts['effective_takt_h_per_ad_per_editor']*100:.1f}% headroom once expected rework ({ts['expected_rework_hours']:.2f} h) is included.")
    c1, c2 = st.columns(2)
    with c1:
        plot(C.lead_time_hist(lm.delivered_in_window(res), cfg.demand.sla_days * cfg.org.hours_per_day), "tf_hist")
    with c2:
        plot(C.daily_variability(a["daily"], lm.window(res)["d_w"], k["weekly_target"] / cfg.org.days_per_week), "tf_daily")
    st.subheader("Little's Law  (L = λW)")
    ll = a["little"]
    cc = st.columns(4)
    card(cc[0], "L observed (ads in system)", f(ll["L_observed"], 1), D_["Little's Law"])
    card(cc[1], "λ (accepted per hour)", f(ll["throughput_per_hour"], 2), "Accepted ads per working hour in the window.")
    card(cc[2], "W (mean lead time)", f(ll["mean_lead_time"], 1, " h"), "Mean lead time of ads accepted in the window.")
    card(cc[3], "λ × W", f(ll["lambda_times_W"], 1), f"Ratio L / λW = {ll['ratio']:.2f}")
    plot(C.littles_law_chart(ll["wip_series"], ll["L_observed"], ll["lambda_times_W"]), "tf_ll")
    if ll["stable"]:
        st.success("The window looks approximately stationary, so L ≈ λW is a meaningful consistency check.")
    else:
        st.warning("The comparison is NOT a reliable steady-state check: " + "; ".join(ll["caveats"]) + ". Use a longer horizon with a warm-up period and steady demand.")
    st.caption("Limitations: Little's Law needs a stable system; short windows, an initial backlog, non-uniform demand, or WIP still growing/shrinking at the horizon all break the equality (λ here counts only accepted ads, while L includes ads that never finish inside the window).")
    with st.expander("Metric definitions"):
        for n, d in D_.items():
            st.markdown(f"**{n}** — {d}")


# =========================================================================== Heijunka
def page_heijunka(cfg: SimConfig, a: Dict):
    st.caption("Heijunka levels the *plan*: standard editing hours are distributed in proportion to available capacity. " + SIM_NOTE)
    c1, c2, c3 = st.columns(3)
    cross = c1.checkbox("Allow cross-POD placement (capacity-aware only)", False, key="hj_cross")
    seed = c2.number_input("Plan seed", 0, 9999, int(cfg.seed), key="hj_seed")
    method = c3.selectbox("Show plan for", ["capacity_aware", "equal_hours", "equal_count"], format_func=lambda m: LABELS[m], key="hj_m")
    plans = {m: plan_week(cfg, m, seed=int(seed), allow_cross_pod=cross) for m in LABELS}
    sm = pd.DataFrame([p["summary"] for p in plans.values()])
    sm["Method"] = sm["method"].map(LABELS)
    best = sm.sort_values(["balance_index", "overflow_hours"], ascending=[False, True]).iloc[0]
    st.dataframe(sm[["Method", "balance_index", "load_ratio_cv", "max_load_ratio", "min_load_ratio", "overloaded_editors", "overflow_hours", "ads_fit_in_week"]]
                 .rename(columns={"balance_index": "Workload Balance Index", "load_ratio_cv": "Load-ratio CV", "max_load_ratio": "Max load/capacity",
                                  "min_load_ratio": "Min load/capacity", "overloaded_editors": "Editors over 100%", "overflow_hours": "Overflow h",
                                  "ads_fit_in_week": "Ads finishing in week"}).round(3), hide_index=True, width="stretch")
    eq_c = sm.set_index("method").loc["equal_count"]
    st.success(f"**Best balance: {best['Method']}** (Balance Index {best['balance_index']:.2f}, peak load {best['max_load_ratio']*100:.0f}% of capacity). "
               f"Equal ad-count ignores that Complex ads take 3.3x the hours of Simple ones, so editors drawing complex ads exceed capacity (peak {eq_c['max_load_ratio']*100:.0f}%). "
               "Equal standard hours fixes the mix but still treats every editor as equally fast and available; capacity-aware allocation divides hours by each editor's speed and availability.")
    plot(C.heijunka_load_chart({m: p["editors"] for m, p in plans.items()}, LABELS), "hj_load")
    p = plans[method]
    t1, t2, t3 = st.tabs(["Weekly plan by POD & editor", "Output & brief release schedule", "Validate by simulation"])
    with t1:
        pod = p["pods"].copy()
        pod["load_ratio"] = (pod["load_ratio"] * 100).round(0)
        st.dataframe(pod[["name" if "name" in pod else "pod_name", "planned_ads", "simple", "medium", "complex", "load_h", "capacity_h", "load_ratio"]]
                     .rename(columns={"pod_name": "POD", "planned_ads": "Planned ads", "simple": "Simple", "medium": "Medium", "complex": "Complex",
                                      "load_h": "Planned workload h", "capacity_h": "Available productive h", "load_ratio": "Load/capacity %"}).round(1),
                     hide_index=True, width="stretch")
        ed = p["editors"]
        for pp, g in ed.groupby("pod"):
            with st.expander(f"POD {pp+1} editors"):
                x = g[["editor", "planned_ads", "simple", "medium", "complex", "load_h", "capacity_h", "load_ratio", "ads_fit_in_week"]].copy()
                x["load_ratio"] = (x["load_ratio"] * 100).round(0)
                st.dataframe(x.round(1), hide_index=True, width="stretch")
        plot(C.heijunka_plan_heatmap(ed), "hj_hm")
    with t2:
        d = p["daily_output"].reset_index()
        d.columns = ["day", "ads"]
        d["Day"] = [C.WEEKDAYS[i] for i in d["day"]]
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(x=d["Day"], y=d["ads"], marker_color=C.BLUE))
        fig.add_hline(y=cfg.demand.weekly_target_ads / cfg.org.days_per_week, line_dash="dash", line_color=C.GREY, annotation_text="daily pitch")
        plot(C._style(fig, "Planned completed ads by day (capacity-limited schedule)", legend=False, h=300), "hj_day")
        rel = p["brief_release"].copy()
        rel.index = [C.WEEKDAYS[i] for i in rel.index]
        rel.columns = [f"POD {c+1}" for c in rel.columns]
        st.markdown("**Brief release schedule** (briefs to release, so each is ready by the day before its planned editing start)")
        st.dataframe(rel, width="stretch")
    with t3:
        st.write("Runs the discrete-event simulation with each assignment policy (same random numbers) to test whether better planned balance translates into flow. "
                 "'Pull' is the playbook default: editors take the next ready brief when free.")
        reps = st.slider("Replications", 1, 10, 4, key="hj_reps")
        if st.button("Run policy comparison", key="hj_run"):
            rows = []
            bar = st.progress(0.0)
            pols = {"pull": "Pull (default)", "equal_count": "Equal ad-count", "equal_hours": "Equal standard hours", "capacity_aware": "Capacity-aware"}
            for i, (pol, lab) in enumerate(pols.items()):
                c = cfg.copy_with(lean={"assignment_policy": pol, "cross_pod_balancing": cross and pol == "capacity_aware"})
                _, df = run_replications(c, reps, cfg.seed, keep_first=False)
                m1, h1 = mean_ci(df["accepted_per_week"]); m2, h2 = mean_ci(df["lead_time_h"]); m3, h3 = mean_ci(df["workload_balance_index"])
                m4, _ = mean_ci(df["mura_index"])
                rows.append({"Policy": lab, "Accepted/week": f"{m1:.1f} ± {h1:.1f}", "Lead time h": f"{m2:.1f} ± {h2:.1f}",
                             "Balance Index (simulated)": f"{m3:.2f} ± {h3:.2f}", "Mura": round(m4, 1)})
                bar.progress((i + 1) / len(pols))
            st.session_state["hj_res"] = pd.DataFrame(rows)
        if "hj_res" in st.session_state:
            st.dataframe(st.session_state["hj_res"], hide_index=True, width="stretch")
            st.caption("± = 95% CI half-width over replications (Student t). Differences smaller than the CI should not be read as real.")


LABELS = {"equal_count": "Equal ad-count", "equal_hours": "Equal standard hours", "capacity_aware": "Capacity-aware"}


# =========================================================================== scenario lab
def _custom_scenario_ui(base: SimConfig):
    with st.expander("➕ Custom scenario", expanded=False):
        c = st.columns(4)
        name = c[0].text_input("Name", "Custom", key="cs_name")
        dem = c[1].number_input("Weekly demand", 20, 600, int(base.demand.weekly_target_ads), key="cs_dem")
        cx = c[2].slider("Complex share %", 0, 80, int(base.complexity.mix["Complex"] * 100), key="cs_cx")
        fp = c[3].slider("First-pass approval %", 30, 100, int(base.quality.first_pass_approval_rate * 100), key="cs_fp")
        c = st.columns(4)
        rem = c[0].number_input("Editors removed", 0, 10, 0, key="cs_rem")
        cross = c[1].checkbox("Cross-POD balancing", base.lean.cross_pod_balancing, key="cs_cross")
        std = c[2].checkbox("Brief standardization", base.lean.brief_standardization, key="cs_std")
        revx = c[3].slider("Review speed x", 0.5, 2.0, 1.0, 0.1, key="cs_rev")
        if st.button("Add custom scenario", key="cs_add"):
            s_ = base.complexity.mix["Simple"]
            cxf = cx / 100
            simple = min(s_, 1 - cxf)
            mix = {"Simple": simple, "Medium": 1 - cxf - simple, "Complex": cxf}
            counts = [base.org.editors_in_pod(p) for p in range(base.org.n_pods)]
            left = rem
            for p in range(base.org.n_pods - 1, -1, -1):
                while left > 0 and counts[p] > 1:
                    counts[p] -= 1
                    left -= 1
            ov = {"demand": {"weekly_target_ads": float(dem)}, "complexity": {"mix": mix},
                  "quality": {"first_pass_approval_rate": fp / 100, "review_capacity_multiplier": revx},
                  "lean": {"cross_pod_balancing": cross, "brief_standardization": std}, "org": {"editor_counts": counts}}
            try:
                sc = Scenario(f"X{len(st.session_state.get('custom', []))+1}", name, "Custom scenario", ov)
                sc.build(base)
                st.session_state.setdefault("custom", []).append(sc)
                st.success(f"Added '{name}'.")
            except Exception as e:
                st.error(f"Invalid scenario: {e}")


def page_scenarios(cfg: SimConfig):
    st.caption("Scenarios branch from the **current sidebar configuration** (Scenario A = that baseline). All scenarios use the same random seeds (common random numbers) so differences are attributable to the change. " + SIM_NOTE)
    presets = preset_scenarios(cfg)
    _custom_scenario_ui(cfg)
    allsc = presets + st.session_state.get("custom", [])
    names = {s.key: s.name for s in allsc}
    chosen = st.multiselect("Scenarios to run (A is always included)", [s.key for s in allsc[1:]], default=[s.key for s in presets[1:]],
                            format_func=lambda k: names[k], key="sc_sel")
    cc = st.columns(3)
    reps = cc[0].slider("Replications", 2, 30, min(cfg.replications, 10), key="sc_reps")
    seed0 = cc[1].number_input("Base seed", 0, 10 ** 6, int(cfg.seed), key="sc_seed")
    if cc[2].button("Run scenarios", type="primary", key="sc_run"):
        sel = [allsc[0]] + [s for s in allsc if s.key in chosen]
        bar = st.progress(0.0, text="Running scenarios…")
        runs = run_scenarios(sel, cfg, reps, int(seed0), progress=lambda x: bar.progress(x, text=f"Running scenarios… {x*100:.0f}%"))
        st.session_state["sc_runs"] = dict(runs=runs, names={s.key: s.name for s in sel}, reps=reps)
        bar.empty()
    if "sc_runs" not in st.session_state:
        st.info("Choose scenarios and press **Run scenarios**.")
        return
    S = st.session_state["sc_runs"]
    tab = comparison_table(S["runs"], "A", S["names"])
    st.subheader(f"Comparison ({S['reps']} replications, 95% CI)")
    piv = tab.pivot_table(index=["label"], columns="name", values="mean", sort=False)
    piv = piv[[S["names"][k] for k in S["runs"]]]
    st.dataframe(piv.round(2), width="stretch")
    st.markdown("**Change versus baseline** (paired differences over the same seeds; 'better/worse' appears only when the 95% CI of the paired difference excludes zero — otherwise the change is within replication noise.)")
    d = tab[tab["scenario"] != "A"].copy()
    d["Δ (CI)"] = d.apply(lambda r: f"{r['delta']:+.2f} ± {r['delta_ci95']:.2f}" if r["delta_ci95"] == r["delta_ci95"] else f"{r['delta']:+.2f}", axis=1)
    d["Δ %"] = d["delta_pct"].round(1)
    st.dataframe(d.pivot_table(index="label", columns="name", values="Δ (CI)", aggfunc="first", sort=False)[[S["names"][k] for k in S["runs"] if k != "A"]], width="stretch")
    v = d.pivot_table(index="label", columns="name", values="verdict", aggfunc="first", sort=False)[[S["names"][k] for k in S["runs"] if k != "A"]]
    st.dataframe(v, width="stretch")
    keyset = ["accepted_per_week", "lead_time_h", "wip_avg", "editor_utilization_pct", "overtime_required_per_week", "on_time_delivery_pct", "mura_index", "muri_index", "muda_waste_share_pct"]
    plot(C.scenario_delta_chart(tab, S["names"], keyset), "sc_delta")
    metric = st.selectbox("Metric detail", [m[0] for m in COMPARE_METRICS], format_func=lambda m: dict((x[0], x[1]) for x in COMPARE_METRICS)[m], key="sc_metric")
    plot(C.scenario_metric_chart(tab, metric, S["names"]), "sc_metric_chart")
    st.download_button("Download comparison (CSV)", tab.to_csv(index=False), "scenario_comparison.csv", "text/csv")


# =========================================================================== recommendations
def page_recommendations(a: Dict, cfg: SimConfig):
    st.caption("Recommendations are generated from this run's KPIs. They are hypotheses: use **Test** to run the proposed change as a paired simulation. " + SIM_NOTE)
    k = a["kpis"]
    st.markdown(f"**Current state:** attainment {k['target_attainment_pct']:.0f}% · lead time {k['lead_time_h']:.1f} h · editor utilization {k['editor_utilization_pct']:.0f}% · likely constraint: **{a['bott']['primary']}**")
    for r in a["recs"]:
        with st.expander(f"#{r['priority']}  {r['title']}", expanded=r["priority"] <= 2):
            st.markdown(f"**Observed problem:** {r['observed']}")
            st.markdown(f"**Supporting KPI:** {r['kpi']}")
            st.markdown(f"**Proposed intervention:** {r['intervention']}")
            st.markdown(f"**Expected direction:** {r['direction']} *(a direction to verify, not a guaranteed improvement)*")
            st.markdown(f"**Estimated impact if tested:** {r['estimated_impact']}")
            if st.button("Test this change (paired simulation)", key=f"rec_{r['key']}"):
                reps = min(max(cfg.replications, 4), 10)
                with st.spinner("Simulating baseline vs change…"):
                    sc = Scenario("T", "Change", r["title"], r["scenario"])
                    base_s = Scenario("A", "Current", "")
                    runs = run_scenarios([base_s, sc], cfg, reps, cfg.seed)
                    tab = comparison_table(runs, "A", {"A": "Current", "T": "Change"})
                t = tab[(tab["scenario"] == "T") & tab["metric"].isin(["accepted_per_week", "lead_time_h", "wip_avg", "editor_utilization_pct", "on_time_delivery_pct", "mura_index", "muri_index", "muda_waste_share_pct"])]
                out = t[["label", "delta", "delta_ci95", "delta_pct", "verdict"]].rename(columns={"label": "Metric", "delta": "Δ", "delta_ci95": "± 95% CI", "delta_pct": "Δ %", "verdict": "Verdict"})
                st.dataframe(out.round(2), hide_index=True, width="stretch")
                st.caption(f"{reps} paired replications. 'within replication noise' means the change cannot be distinguished from random variation.")
