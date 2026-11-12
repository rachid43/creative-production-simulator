"""Lean Creative Production Simulator - Streamlit command center.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from pydantic import ValidationError

from analytics.report import build_analysis
from config.defaults import ASSUMPTION_NOTES, baseline_config
from config.schema import SimConfig
from simulation.scenarios import run_replications
from visualization import dashboards as DB

st.set_page_config(page_title="Lean Creative Production Simulator", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

PAGES = ["Executive Command Center", "POD Production Network", "Mura Analysis", "Muri Analysis", "Muda Analysis",
         "Takt & Flow Performance", "Heijunka Workload Balancing", "Scenario Laboratory", "Improvement Recommendations"]
B = baseline_config()

WIDGET_DEFAULTS = {
    "n_pods": B.org.n_pods, "editors_per_pod": B.org.editors_per_pod, "strategists_per_pod": B.org.strategists_per_pod,
    "editors_removed": 0, "hours_per_day": B.org.hours_per_day, "days_per_week": B.org.days_per_week,
    "availability": int(B.org.productive_availability * 100), "absenteeism": 0, "overtime": 0.0,
    "weekly_target": int(B.demand.weekly_target_ads), "pattern": B.demand.arrival_pattern, "demand_cv": int(B.demand.demand_cv * 100),
    "backlog": 0, "ready_start": 0, "sla_days": B.demand.sla_days, "urgent_share": int(B.demand.urgent_share * 100),
    "mix_simple": 30, "mix_complex": 20, "h_simple": 1.5, "h_medium": 3.0, "h_complex": 5.0, "edit_cv": int(B.complexity.editing_time_cv * 100),
    "speed_cv": int(B.org.editor_speed_cv * 100), "complex_capable": 3,
    "fpa": int(B.quality.first_pass_approval_rate * 100), "repeat_rw": int(B.quality.rework_repeat_probability * 100),
    "rw_frac": int(B.quality.rework_hours_fraction * 100), "max_loops": B.quality.max_rework_loops, "on_max": B.quality.on_max_loops,
    "review_h": B.quality.review_hours, "review_delay": 0.0, "review_mode": B.quality.review_mode, "dedicated_n": 2, "lead_share": 50,
    "review_mult": 1.0, "brief_h": B.quality.brief_hours, "asset_p": int(B.quality.asset_delay_probability * 100), "asset_h": B.quality.asset_delay_hours,
    "buffer": B.lean.ready_brief_buffer, "edit_wip": B.lean.editing_wip_limit, "rev_wip": B.lean.review_wip_limit, "enforce": True,
    "cross": False, "cross_rev": False, "priority": False, "standard": False, "policy": "pull",
    "attain_green": 95, "attain_amber": 85, "util_spare": 70, "util_normal": 85, "util_high": 95, "pod_red": 95, "pod_amber": 85,
    "horizon": B.horizon_weeks, "warmup": B.warmup_weeks, "seed": B.seed, "reps": B.replications,
}
CLASS_KEYS_T = list(B.classification.touch)
CLASS_KEYS_W = list(B.classification.waits)
for k, v in WIDGET_DEFAULTS.items():
    st.session_state.setdefault(k, v)
for k in CLASS_KEYS_T:
    st.session_state.setdefault(f"clsT_{k}", B.classification.touch[k])
for k in CLASS_KEYS_W:
    st.session_state.setdefault(f"clsW_{k}", B.classification.waits[k])

ss = st.session_state


def reset_to_baseline():
    for k in list(ss.keys()):
        if k in WIDGET_DEFAULTS or k.startswith(("clsT_", "clsW_", "share_")):
            del ss[k]
    for k in ("run", "sc_runs", "hj_res", "custom"):
        ss.pop(k, None)


# --------------------------------------------------------------------------- sidebar
sb = st.sidebar
sb.title("🎬 Lean Creative Simulator")
sb.caption("Configure → Run. Simulated outputs, not operational data.")
b1, b2 = sb.columns(2)
run_clicked = b1.button("▶ Run simulation", type="primary", width="stretch")
b2.button("↺ Reset to baseline", width="stretch", on_click=reset_to_baseline)

with sb.expander("A. Organization", expanded=False):
    st.number_input("Number of PODs", 1, 12, key="n_pods", help="Playbook: 6 PODs.")
    st.number_input("Editors per POD", 1, 10, key="editors_per_pod", help="Playbook: 3 video editors per POD.")
    st.number_input("Strategists per POD", 1, 4, key="strategists_per_pod")
    st.number_input("Editors removed from workforce", 0, 20, key="editors_removed", help="Removed one at a time from the last PODs (staffing shortage).")
    st.number_input("Working hours per day", 4.0, 12.0, step=0.5, key="hours_per_day")
    st.number_input("Working days per week", 1, 7, key="days_per_week")
    st.slider("Productive availability %", 40, 100, key="availability", help="Share of shift that is productive (rest: meetings, asset search, coordination). Playbook: 80%.")
    st.slider("Planned absenteeism % (per person-day)", 0, 30, key="absenteeism")
    st.number_input("Overtime hours / editor-day (cap)", 0.0, 4.0, step=0.5, key="overtime", help="Used only when a POD has waiting work; lifts productive rate up to 100% of the shift.")
    st.slider("Editor speed variability % (CV)", 0, 30, key="speed_cv")
    st.number_input("Complex-certified editors per POD", 0, 10, key="complex_capable", help="Skill limit used for cross-POD sharing and allocation.")

with sb.expander("B. Demand", expanded=False):
    st.number_input("Weekly target ads", 10, 1000, key="weekly_target")
    st.selectbox("Daily arrival pattern", ["uniform", "front_loaded", "back_loaded", "monday_peak", "midweek_peak"], key="pattern")
    st.slider("Demand variability % (CV of daily arrivals)", 0, 100, key="demand_cv")
    st.number_input("Backlog at simulation start (ads)", 0, 1000, key="backlog")
    st.number_input("Ready briefs per POD at start (carry-in WIP)", 0, 20, key="ready_start")
    st.number_input("SLA (working days after arrival)", 0.5, 10.0, step=0.5, key="sla_days")
    st.slider("Urgent ads %", 0, 50, key="urgent_share")
    with st.expander("Demand by POD (relative shares)"):
        for p in range(int(ss["n_pods"])):
            ss.setdefault(f"share_{p}", 1.0)
            st.number_input(f"POD {p+1}", 0.0, 10.0, step=0.1, key=f"share_{p}")

with sb.expander("C. Ad complexity", expanded=False):
    st.caption("Illustrative mix - replace with measured data.")
    st.slider("Simple %", 0, 100, key="mix_simple")
    st.slider("Complex %", 0, 100, key="mix_complex")
    st.caption(f"Medium % = {100 - ss['mix_simple'] - ss['mix_complex']}")
    st.number_input("Simple standard hours", 0.25, 20.0, step=0.25, key="h_simple")
    st.number_input("Medium standard hours", 0.25, 20.0, step=0.25, key="h_medium")
    st.number_input("Complex standard hours", 0.25, 40.0, step=0.25, key="h_complex")
    st.slider("Editing-time variability % (CV)", 0, 100, key="edit_cv")

with sb.expander("D. Quality & review", expanded=False):
    st.slider("First-pass approval %", 30, 100, key="fpa")
    st.caption(f"→ rework probability at first review = {100 - ss['fpa']}% (derived, kept consistent)")
    st.slider("Repeat-rework probability % (2nd+ reviews)", 0, 100, key="repeat_rw")
    st.slider("Rework duration (% of standard editing hours)", 0, 200, key="rw_frac")
    st.number_input("Maximum rework loops", 0, 10, key="max_loops")
    st.selectbox("After max loops", ["accept", "cancel"], key="on_max", help="accept = escalated approval; cancel = scrap the ad")
    st.number_input("Review processing time (h)", 0.05, 4.0, step=0.05, key="review_h")
    st.number_input("External review waiting / feedback delay (h)", 0.0, 24.0, step=0.5, key="review_delay", help="Elapsed latency before an ad becomes reviewable; consumes no resource.")
    st.selectbox("Review performed by", ["strategist", "lead_creative", "dedicated"], key="review_mode",
                 help="strategist: shares capacity with briefing. lead_creative: the single Lead Creative (50% of time). dedicated: separate reviewers.")
    st.number_input("Dedicated reviewers", 1, 10, key="dedicated_n")
    st.slider("Lead Creative review share %", 10, 100, key="lead_share")
    st.slider("Review capacity adjustment (speed ×)", 0.5, 3.0, step=0.1, key="review_mult")
    st.number_input("Brief time per ad (h)", 0.1, 4.0, step=0.1, key="brief_h")
    st.slider("Missing-asset probability %", 0, 50, key="asset_p")
    st.number_input("Mean asset delay (h)", 0.0, 8.0, step=0.5, key="asset_h")

with sb.expander("E. Lean controls", expanded=False):
    st.number_input("Ready brief buffer per POD", 1, 30, key="buffer")
    st.number_input("Editing WIP limit per POD", 1, 20, key="edit_wip")
    st.number_input("Review WIP limit per POD", 1, 20, key="rev_wip")
    st.checkbox("Enforce WIP limits (pull) — off = push", key="enforce")
    st.checkbox("Cross-POD balancing (editing)", key="cross")
    st.checkbox("Cross-POD review sharing", key="cross_rev")
    st.checkbox("Priority scheduling (urgent / due date first)", key="priority")
    st.checkbox("Brief standardization", key="standard", help="Assumed effects: -25% brief time, +5 pts first-pass approval, -50% asset delays (editable in config).")
    st.selectbox("Work assignment policy", ["pull", "equal_count", "equal_hours", "capacity_aware"], key="policy",
                 help="pull = editors take next ready brief; others pre-assign each ad to a named editor.")

with sb.expander("Status thresholds (POD colours & load bands)", expanded=False):
    st.slider("Green if attainment ≥ %", 50, 100, key="attain_green")
    st.slider("Red if attainment < %", 30, 100, key="attain_amber")
    st.slider("POD util amber ≥ %", 50, 100, key="pod_amber")
    st.slider("POD util red ≥ %", 50, 100, key="pod_red")
    st.slider("Spare capacity below %", 30, 90, key="util_spare")
    st.slider("Normal band up to %", 50, 99, key="util_normal")
    st.slider("High band up to %", 60, 100, key="util_high")

with sb.expander("Lean classification (value-add vs waste)", expanded=False):
    for k in CLASS_KEYS_T:
        st.selectbox(f"Touch: {k}", ["VA", "NNVA", "WASTE"], key=f"clsT_{k}")
    for k in CLASS_KEYS_W:
        st.selectbox(f"Wait: {k.replace('wait_', '')}", ["VA", "NNVA", "WASTE"], key=f"clsW_{k}")

with sb.expander("F. Simulation controls", expanded=True):
    st.slider("Horizon (weeks)", 1, 12, key="horizon")
    st.slider("Warm-up (weeks, excluded from KPIs)", 0, 6, key="warmup", help="Use ≥1 warm-up week for steady-state analysis; use 0 to analyse the complete initial backlog / first week.")
    st.number_input("Random seed", 0, 10 ** 6, key="seed")
    st.slider("Replications", 1, 30, key="reps")
with sb.expander("Baseline assumptions"):
    for n in ASSUMPTION_NOTES:
        st.caption("• " + n)


def build_config():
    s = ss
    n = int(s["n_pods"])
    counts = [int(s["editors_per_pod"])] * n
    left = int(s["editors_removed"])
    for p in range(n - 1, -1, -1):
        while left > 0 and counts[p] > 1:
            counts[p] -= 1
            left -= 1
    if left > 0:
        raise ValueError("Cannot remove that many editors (each POD keeps at least one).")
    simple, cx = s["mix_simple"] / 100, s["mix_complex"] / 100
    med = 1 - simple - cx
    if med < -1e-9:
        raise ValueError("Simple % + Complex % must not exceed 100%.")
    shares = [float(s.get(f"share_{p}", 1.0)) for p in range(n)]
    cap = int(s["complex_capable"])
    data = dict(
        horizon_weeks=int(s["horizon"]), warmup_weeks=int(s["warmup"]), seed=int(s["seed"]), replications=int(s["reps"]),
        org=dict(n_pods=n, editors_per_pod=int(s["editors_per_pod"]), strategists_per_pod=int(s["strategists_per_pod"]),
                 editor_counts=counts if s["editors_removed"] else None, hours_per_day=float(s["hours_per_day"]), days_per_week=int(s["days_per_week"]),
                 productive_availability=s["availability"] / 100, planned_absenteeism=s["absenteeism"] / 100, overtime_hours_per_day=float(s["overtime"]),
                 editor_speed_cv=s["speed_cv"] / 100, complex_capable_per_pod=None if cap >= int(s["editors_per_pod"]) else cap),
        demand=dict(weekly_target_ads=float(s["weekly_target"]), pod_share=shares if len(set(shares)) > 1 else None, arrival_pattern=s["pattern"],
                    demand_cv=s["demand_cv"] / 100, initial_backlog_ads=int(s["backlog"]), initial_ready_briefs_per_pod=int(s["ready_start"]),
                    sla_days=float(s["sla_days"]), urgent_share=s["urgent_share"] / 100),
        complexity=dict(mix={"Simple": simple, "Medium": max(med, 0.0), "Complex": cx},
                        standard_hours={"Simple": s["h_simple"], "Medium": s["h_medium"], "Complex": s["h_complex"]}, editing_time_cv=s["edit_cv"] / 100),
        quality=dict(first_pass_approval_rate=s["fpa"] / 100, rework_probability=(100 - s["fpa"]) / 100, rework_repeat_probability=s["repeat_rw"] / 100,
                     rework_hours_fraction=s["rw_frac"] / 100, max_rework_loops=int(s["max_loops"]), on_max_loops=s["on_max"], review_hours=float(s["review_h"]),
                     review_response_delay_hours=float(s["review_delay"]), review_mode=s["review_mode"], dedicated_reviewers=int(s["dedicated_n"]),
                     lead_creative_review_share=s["lead_share"] / 100, review_capacity_multiplier=float(s["review_mult"]), brief_hours=float(s["brief_h"]),
                     asset_delay_probability=s["asset_p"] / 100, asset_delay_hours=float(s["asset_h"])),
        lean=dict(ready_brief_buffer=int(s["buffer"]), editing_wip_limit=int(s["edit_wip"]), review_wip_limit=int(s["rev_wip"]),
                  enforce_wip_limits=bool(s["enforce"]), cross_pod_balancing=bool(s["cross"]), cross_pod_review_sharing=bool(s["cross_rev"]),
                  priority_scheduling=bool(s["priority"]), brief_standardization=bool(s["standard"]), assignment_policy=s["policy"]),
        thresholds=dict(attain_green=s["attain_green"] / 100, attain_amber=s["attain_amber"] / 100, pod_util_red=s["pod_red"] / 100,
                        pod_util_amber=s["pod_amber"] / 100, util_spare_below=s["util_spare"] / 100, util_normal_upto=s["util_normal"] / 100,
                        util_high_upto=s["util_high"] / 100),
        classification=dict(touch={k: s[f"clsT_{k}"] for k in CLASS_KEYS_T}, waits={k: s[f"clsW_{k}"] for k in CLASS_KEYS_W}),
    )
    return SimConfig.model_validate(data)


try:
    cfg = build_config()
    cfg_error = None
except (ValidationError, ValueError) as e:
    cfg, cfg_error = None, str(e)

sig = hashlib.md5(cfg.model_dump_json().encode()).hexdigest() if cfg else None


def run_now(cfg: SimConfig):
    bar = st.progress(0.0, text="Simulating…")
    first, reps = run_replications(cfg, cfg.replications, cfg.seed, progress=lambda x: bar.progress(x * 0.8, text=f"Replication {int(x*cfg.replications)}/{cfg.replications}"))
    bar.progress(0.85, text="Analysing (bottleneck sensitivity re-runs)…")
    a = build_analysis(first, reps)
    bar.empty()
    ss["run"] = dict(a=a, sig=sig)


# --------------------------------------------------------------------------- main
st.title("Lean Creative Production Simulator")
if cfg_error:
    st.error("Invalid inputs — fix them in the sidebar:\n\n" + cfg_error)
    st.stop()
if run_clicked or "run" not in ss:
    run_now(cfg)
a = ss["run"]["a"]
if ss["run"]["sig"] != sig:
    st.warning("Inputs changed since the last run — showing the previous results. Press **▶ Run simulation** to update.")

page = st.radio("Page", PAGES, horizontal=True, label_visibility="collapsed", key="page")
st.divider()
if page == PAGES[0]:
    DB.page_executive(a)
elif page == PAGES[1]:
    DB.page_network(a)
elif page == PAGES[2]:
    DB.page_mura(a)
elif page == PAGES[3]:
    DB.page_muri(a)
elif page == PAGES[4]:
    DB.page_muda(a)
elif page == PAGES[5]:
    DB.page_takt(a)
elif page == PAGES[6]:
    DB.page_heijunka(a["cfg"], a)
elif page == PAGES[7]:
    DB.page_scenarios(a["cfg"])
else:
    DB.page_recommendations(a, a["cfg"])
