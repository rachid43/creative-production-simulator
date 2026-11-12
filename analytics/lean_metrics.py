"""Lean metrics with documented formulas: takt, Mura, Muri, Muda, flow efficiency, Little's Law.

All durations are *working hours* on the simulation clock.  Every function takes a
``SimulationResult`` (or a config) - nothing here invents data that the simulation did not produce.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.schema import SimConfig
from simulation.entities import TOUCH_CATEGORIES, WAIT_CATEGORIES
from simulation.engine import SimulationResult

# --------------------------------------------------------------------------- #
# Definitions (shown as tooltips in the UI)
# --------------------------------------------------------------------------- #
DEFINITIONS: Dict[str, str] = {
    "Takt time": "Available production time / customer demand. Sets the required pace; it is NOT a measurement of how fast work is done. Baseline: 40 h x 18 editors / 180 ads = 4.0 editor-hours per ad (gross); x 80 % availability = 3.2 productive hours.",
    "Cycle time": "Actual processing (touch) time per ad - productive hours an editor spends on it. Excludes queues and waiting.",
    "Lead time": "Approval timestamp - demand arrival timestamp (end-to-end, working hours). Includes every wait.",
    "Throughput": "Accepted ads per working day / week. An exported ad waiting for review is still WIP, not throughput.",
    "Takt attainment": "Accepted ads / target ads x 100.",
    "First-pass yield": "Ads approved without rework / ads first reviewed x 100.",
    "WIP": "Ads started (brief in progress or later) but not yet accepted. Demand waiting for a brief is backlog, not WIP.",
    "Flow efficiency": "Value-adding processing time / total lead time x 100. Classification of value-adding vs non-value-adding is configurable.",
    "Editor utilization": "Busy editing hours (new edits + rework) / available productive editing hours x 100. Can exceed 100 % only with overtime.",
    "Workload Balance Index": "1 - CV of (planned standard workload hours / available productive capacity) across editors; 1 = perfectly level, 0 = CV of 100 % or worse.",
    "Little's Law": "L = lambda x W: average number in the system equals throughput times average time in system. Valid only for a stable, stationary regime.",
    "Mura": "Unevenness: variation in workload, demand, output and queues. Index = mean over components of min(1, CV / reference CV) x 100.",
    "Muri": "Overburden: resources loaded above a sustainable level. Index = 100 x (0.5 x share of resource-days in the high band + 1.0 x share in the critical band).",
    "Muda": "Waste: non-value-adding time that is not necessary - waiting, blocking, rework, missing assets. Share = waste hours / total lead-time hours.",
    "Heijunka": "Production levelling: smoothing the mix and volume of released work so that no resource is overloaded while another is idle.",
    "Critical WIP": "Minimum WIP needed to sustain the observed throughput with zero waiting: throughput x raw processing time. WIP above it adds waiting, not output.",
}


# --------------------------------------------------------------------------- #
# Window helpers
# --------------------------------------------------------------------------- #
def window(res: SimulationResult) -> dict:
    cfg = res.cfg
    D, H = cfg.org.days_per_week, cfg.org.hours_per_day
    d_w = cfg.warmup_weeks * D
    return dict(d_w=d_w, t_w=d_w * H, weeks=cfg.horizon_weeks - cfg.warmup_weeks, n_days=cfg.n_days, H=H, D=D)


def cv(x) -> float:
    x = np.asarray([v for v in x if v == v], dtype=float)
    if len(x) < 2 or x.mean() == 0:
        return 0.0
    return float(x.std(ddof=0) / abs(x.mean()))


# --------------------------------------------------------------------------- #
# Takt and capacity
# --------------------------------------------------------------------------- #
def expected_rework_hours_per_ad(cfg: SimConfig, pod: Optional[int] = None) -> float:
    """E[rework editing hours per ad] (geometric rework loops). pod=None -> demand-share-weighted across PODs."""
    if pod is None:
        return float(sum(s * expected_rework_hours_per_ad(cfg, p) for p, s in enumerate(cfg.pod_shares)))
    q = cfg.quality
    p1 = 1 - cfg.first_pass_for(pod)
    total, p_reach = 0.0, p1
    for k in range(q.max_rework_loops):
        total += p_reach * q.rework_hours_fraction * cfg.mean_std_hours_for(pod)
        p_reach *= q.rework_repeat_probability
    return total


def expected_review_hours_per_ad(cfg: SimConfig, pod: Optional[int] = None) -> float:
    if pod is None:
        return float(sum(s * expected_review_hours_per_ad(cfg, p) for p, s in enumerate(cfg.pod_shares)))
    q = cfg.quality
    p1 = 1 - cfg.first_pass_for(pod)
    n_reviews, p_reach = 1.0, p1
    for k in range(q.max_rework_loops):
        n_reviews += p_reach
        p_reach *= q.rework_repeat_probability
    return n_reviews * q.review_hours / q.review_capacity_multiplier


def takt_summary(cfg: SimConfig, weekly_target: Optional[float] = None) -> dict:
    """Takt time and capacity arithmetic (playbook section 1), separating the Lean concepts."""
    org = cfg.org
    target = weekly_target if weekly_target is not None else cfg.demand.weekly_target_ads
    n_ed = org.total_editors
    weekly_h = org.weekly_hours
    std = float(sum(s * cfg.mean_std_hours_for(p) for p, s in enumerate(cfg.pod_shares)))
    rw = expected_rework_hours_per_ad(cfg)
    eff_hours_per_ad = std + rw
    avail_per_editor = weekly_h * org.productive_availability * (1 - org.planned_absenteeism) * org.editor_speed_multiplier
    capacity_ads = n_ed * avail_per_editor / eff_hours_per_ad if eff_hours_per_ad else 0.0
    return dict(
        weekly_target=target,
        daily_target=target / org.days_per_week,
        editors=n_ed,
        ads_per_editor_week=target / n_ed if n_ed else float("nan"),
        gross_takt_h_per_ad_per_editor=(weekly_h * n_ed / target) if target else float("nan"),
        effective_takt_h_per_ad_per_editor=(weekly_h * n_ed * org.productive_availability / target) if target else float("nan"),
        org_takt_minutes_between_ads=(weekly_h * 60 / target) if target else float("nan"),
        mean_standard_hours=std,
        expected_rework_hours=rw,
        required_hours_per_ad=eff_hours_per_ad,
        available_productive_h_per_editor_week=avail_per_editor,
        required_hours_per_editor_week=(target / n_ed * eff_hours_per_ad) if n_ed else float("nan"),
        capacity_ads_per_week=capacity_ads,
        capacity_vs_target=capacity_ads / target if target else float("nan"),
        planned_load_ratio=(target * eff_hours_per_ad) / (n_ed * avail_per_editor) if n_ed and avail_per_editor else float("nan"),
        expected_review_h_per_ad=expected_review_hours_per_ad(cfg),
    )


def workload_balance_index(load_ratios) -> float:
    """WBI = max(0, 1 - CV(load ratios)); load ratio = planned standard hours / available productive capacity."""
    return float(max(0.0, 1.0 - cv(load_ratios)))


# --------------------------------------------------------------------------- #
# Classification, flow efficiency, lead-time decomposition
# --------------------------------------------------------------------------- #
def cohort(res: SimulationResult) -> pd.DataFrame:
    """Ads that arrived after warm-up (delivered or still open) - the population for waste analysis."""
    w = window(res)
    j = res.jobs
    return j[j["arrival"] >= w["t_w"] - 1e-9].copy()


def delivered_in_window(res: SimulationResult) -> pd.DataFrame:
    w = window(res)
    j = res.jobs
    return j[(j["status"] == "delivered") & (j["approval"] >= w["t_w"] - 1e-9)].copy()


def classified_hours(df: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    """Per-ad hours by Lean class: VA, NNVA, WASTE (columns), using the configurable classification."""
    cl = cfg.classification
    out = pd.DataFrame(index=df.index)
    for c in ("VA", "NNVA", "WASTE"):
        out[c] = 0.0
    for k in TOUCH_CATEGORIES:
        out[cl.touch.get(k, "NNVA")] += df.get(f"T_{k}", 0.0)
    out[cl.touch.get("overhead", "NNVA")] += df["overhead"]
    for k in WAIT_CATEGORIES:
        out[cl.waits.get(k, "WASTE")] += df.get(f"L_{k}", 0.0)
    return out


def flow_efficiency(df: pd.DataFrame, cfg: SimConfig) -> float:
    """Flow efficiency (%) = sum(value-adding hours) / sum(lead time) x 100 over delivered ads."""
    d = df[df["status"] == "delivered"]
    if d.empty or d["lead_time"].sum() <= 0:
        return float("nan")
    return float(classified_hours(d, cfg)["VA"].sum() / d["lead_time"].sum() * 100)


def lead_time_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Mean hours per delivered ad in each component; the components sum to the mean lead time."""
    d = df[df["status"] == "delivered"]
    labels = {
        "T_brief": "Briefing (touch)", "T_edit": "Editing (touch)", "T_review": "Review (touch)",
        "T_rework": "Rework editing (touch)", "T_rework_review": "Rework review (touch)",
        "overhead": "In-process overhead (availability loss)",
        "L_wait_brief_queue": "Wait: brief queue", "L_wait_ready_queue": "Wait: editor capacity",
        "L_wait_asset": "Wait: assets", "L_wait_blocked_review": "Blocked: review WIP limit",
        "L_wait_review_external": "Wait: external feedback", "L_wait_review_queue": "Wait: review capacity",
        "L_wait_rework_queue": "Wait: rework pickup", "L_wait_unavailable": "Wait: resource absent",
    }
    rows = [dict(component=lab, hours=float(d[c].mean()) if len(d) else 0.0, key=c) for c, lab in labels.items()]
    return pd.DataFrame(rows)


def waste_pareto(df: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    """Job-hours of WASTE-classified time by cause (cohort = delivered + open ads), sorted descending."""
    labels = {
        "L_wait_brief_queue": "Waiting for brief capacity", "L_wait_ready_queue": "Waiting for editor capacity",
        "L_wait_asset": "Waiting for assets", "L_wait_blocked_review": "Blocked by review WIP limit",
        "L_wait_review_external": "Waiting for external feedback", "L_wait_review_queue": "Waiting for review",
        "L_wait_rework_queue": "Waiting for rework pickup", "L_wait_unavailable": "Waiting: resource absent",
        "T_rework": "Rework processing", "T_rework_review": "Rework review processing",
        "overhead": "In-process overhead",
    }
    cl = cfg.classification
    cls_of = {f"L_{k}": v for k, v in cl.waits.items()}
    cls_of.update({f"T_{k}": v for k, v in cl.touch.items()})
    cls_of["overhead"] = cl.touch.get("overhead", "NNVA")
    rows = []
    for c, lab in labels.items():
        if cls_of.get(c) == "WASTE" and c in df:
            rows.append(dict(cause=lab, hours=float(df[c].sum())))
    out = pd.DataFrame(rows).sort_values("hours", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["cause", "hours"])
    tot = out["hours"].sum()
    out["share_pct"] = out["hours"] / tot * 100 if tot > 0 else 0.0
    out["cum_pct"] = out["share_pct"].cumsum()
    return out


def waste_share(df: pd.DataFrame, cfg: SimConfig) -> float:
    """Muda index (%) = WASTE-classified job-hours / total job-hours accrued (delivered + open ads)."""
    if df.empty:
        return float("nan")
    cls = classified_hours(df, cfg)
    total = cls.sum().sum()
    return float(cls["WASTE"].sum() / total * 100) if total > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Mura / Muri
# --------------------------------------------------------------------------- #
def util_band(u: float, cfg: SimConfig) -> str:
    t = cfg.thresholds
    if u < t.util_spare_below:
        return "Spare capacity"
    if u < t.util_normal_upto:
        return "Normal"
    if u < t.util_high_upto:
        return "High load"
    return "Critical load"


def editor_day_table(res: SimulationResult) -> pd.DataFrame:
    w = window(res)
    wd = res.worker_day
    e = wd[(wd["role"] == "editor") & (wd["day"] >= w["d_w"])].copy()
    e["util"] = np.where(e["capacity_h"] > 0, e["busy_h"] / e["capacity_h"].replace(0, np.nan), np.nan)
    return e


def pod_week_workload(res: SimulationResult) -> pd.DataFrame:
    """Planned (standard-hour) workload arriving per POD-week vs. available productive capacity."""
    cfg, w = res.cfg, window(res)
    j = res.jobs
    j = j[j["arrival"] >= w["t_w"] - 1e-9].copy()
    j["week"] = (j["arrival"] // (w["H"] * w["D"])).astype(int)
    e = res.worker_day[(res.worker_day["role"] == "editor") & (res.worker_day["day"] >= w["d_w"])]
    sp = res.workers.set_index("worker")["speed"]
    e = e.assign(cap_std=e["capacity_h"] * e["worker"].map(sp))
    cap = e.groupby(["pod", "week"]).agg(capacity_h=("capacity_h", "sum"), cap_std=("cap_std", "sum"), busy_h=("busy_h", "sum"),
                                         ot_clock_h=("ot_clock_h", "sum")).reset_index()
    wl = j.groupby(["pod", "week"]).agg(arrived=("job_id", "count"), std_hours=("est_hours", "sum")).reset_index()
    out = cap.merge(wl, on=["pod", "week"], how="left").fillna({"arrived": 0, "std_hours": 0})
    rw = out["pod"].map(lambda p: expected_rework_hours_per_ad(cfg, int(p)))
    out["planned_hours"] = out["std_hours"] + out["arrived"] * rw
    out["load_ratio"] = out["planned_hours"] / out["cap_std"].replace(0, np.nan)
    out["util"] = out["busy_h"] / out["capacity_h"].replace(0, np.nan)
    out["shortfall_h"] = (out["planned_hours"] - out["cap_std"]).clip(lower=0)
    a = cfg.org.productive_availability
    out["ot_required_clock_h"] = out["shortfall_h"] / a
    return out


def mura_analysis(res: SimulationResult) -> dict:
    cfg, w = res.cfg, window(res)
    ref = cfg.thresholds.mura_cv_reference
    ed = editor_day_table(res)
    per_ed = ed.groupby("worker").agg(busy=("busy_h", "sum"), cap=("capacity_h", "sum"), pod=("pod", "first")).reset_index()
    per_ed["util"] = per_ed["busy"] / per_ed["cap"].replace(0, np.nan)
    pw = pod_week_workload(res)
    pod_tot = pw.groupby("pod").agg(planned=("planned_hours", "sum"), cap=("cap_std", "sum")).reset_index()
    pod_tot["load_ratio"] = pod_tot["planned"] / pod_tot["cap"].replace(0, np.nan)
    j = res.jobs
    d_arr = j[j["arrival"] >= w["t_w"] - 1e-9].copy()
    d_arr["day"] = (d_arr["arrival"] // w["H"]).astype(int)
    days = range(w["d_w"], w["n_days"])
    daily_arr = d_arr.groupby("day").size().reindex(days, fill_value=0)
    dv = delivered_in_window(res)
    dv["day"] = (dv["approval"] // w["H"]).clip(upper=w["n_days"] - 1).astype(int)
    daily_out = dv.groupby("day").size().reindex(days, fill_value=0)
    ps = res.pod_snapshots
    ps = ps[ps["t"] >= w["t_w"] - 1e-9]
    q_series = ps.assign(q=ps["ready"] + ps["rework_q"]).groupby("t")["q"].sum()
    ratio = (dv["actual_editing_hours"] / dv["est_hours"]).replace([np.inf, -np.inf], np.nan).dropna()
    comps = {
        "Editor utilization (CV across editors)": cv(per_ed["util"]),
        "POD workload / capacity (CV across PODs)": cv(pod_tot["load_ratio"]),
        "Daily accepted output (CV across days)": cv(daily_out.values),
        "Daily demand arrivals (CV across days)": cv(daily_arr.values),
        "Editing queue length (CV over time)": cv(q_series.values),
        "Editing time vs standard (CV of actual/standard)": cv(ratio.values),
    }
    table = pd.DataFrame([dict(component=k, cv=v, score=min(1.0, v / ref) * 100) for k, v in comps.items()])
    index = float(table["score"].mean())
    lo, hi = cfg.thresholds.mura_low, cfg.thresholds.mura_high
    level = "Low" if index < lo else ("Moderate" if index < hi else "High")
    umax, umin = (per_ed["util"].max(), per_ed["util"].min()) if len(per_ed) else (np.nan, np.nan)
    bmax, bmin = (per_ed["busy"].max(), per_ed["busy"].min()) if len(per_ed) else (np.nan, np.nan)
    # editor workload in standard hours
    stdh = d_arr.groupby("first_editor")["est_hours"].sum()
    sp = res.workers.set_index("worker")
    ed_ratio = []
    for _, r in per_ed.iterrows():
        cap_std = r["cap"] * sp.loc[r["worker"], "speed"]
        ed_ratio.append(stdh.get(r["worker"], 0.0) / cap_std if cap_std > 0 else np.nan)
    per_ed["std_workload_ratio"] = ed_ratio
    wbi = workload_balance_index(per_ed["std_workload_ratio"].dropna())
    return dict(index=index, level=level, components=table, per_editor=per_ed, pod_load=pod_tot,
                daily_arrivals=daily_arr, daily_output=daily_out, queue_series=q_series,
                util_max=umax, util_min=umin, busy_max=bmax, busy_min=bmin,
                busy_ratio=(bmax / bmin) if bmin and bmin > 0 else float("inf"),
                workload_balance_index=wbi, imbalance=imbalance_days(res))


def imbalance_days(res: SimulationResult) -> pd.DataFrame:
    """Days when one POD sits in the spare band while another is overloaded (idle capacity next to a queue)."""
    cfg, w = res.cfg, window(res)
    t = cfg.thresholds
    ed = editor_day_table(res)
    pod_day = ed.groupby(["day", "pod"]).agg(cap=("capacity_h", "sum"), busy=("busy_h", "sum")).reset_index()
    pod_day["util"] = pod_day["busy"] / pod_day["cap"].replace(0, np.nan)
    pod_day["spare_h"] = (pod_day["cap"] - pod_day["busy"]).clip(lower=0)
    ps = res.pod_snapshots
    # end-of-day queue: last snapshot of each day
    last = ps.sort_values("t").groupby(["day", "pod"]).tail(1).set_index(["day", "pod"])
    rows = []
    for d, g in pod_day.groupby("day"):
        over = []
        under = []
        for _, r in g.iterrows():
            q = last["ready"].get((d, r["pod"]), 0) + last["rework_q"].get((d, r["pod"]), 0) + last["backlog"].get((d, r["pod"]), 0)
            if (r["util"] >= t.util_high_upto - 0.05 and q >= 1) or q >= cfg.org.editors_per_pod * t.pod_queue_red_per_editor / 3:
                over.append((int(r["pod"]), q))
            if r["util"] < t.util_spare_below:
                under.append((int(r["pod"]), r["spare_h"]))
        rows.append(dict(day=d, overloaded_pods=[f"POD {p+1}" for p, _ in over], queued_ads=int(sum(q for _, q in over)),
                         underused_pods=[f"POD {p+1}" for p, _ in under], idle_hours_underused=float(sum(h for _, h in under)),
                         concurrent=bool(over and under)))
    out = pd.DataFrame(rows)
    return out


def muri_analysis(res: SimulationResult) -> dict:
    cfg, w = res.cfg, window(res)
    t = cfg.thresholds
    wd = res.worker_day[res.worker_day["day"] >= w["d_w"]].copy()
    wd = wd[wd["capacity_h"] > 0]
    wd["util"] = wd["busy_h"] / wd["capacity_h"]
    wd["band"] = wd["util"].map(lambda u: util_band(u, cfg))
    per_w = wd.groupby(["worker", "role", "pod"]).agg(busy=("busy_h", "sum"), cap=("capacity_h", "sum"),
                                                      ot_clock_h=("ot_clock_h", "sum")).reset_index()
    per_w["util"] = per_w["busy"] / per_w["cap"]
    per_w["band"] = per_w["util"].map(lambda u: util_band(u, cfg))
    f_high = float(((wd["util"] >= t.util_normal_upto) & (wd["util"] < t.util_high_upto)).mean()) if len(wd) else 0.0
    f_crit = float((wd["util"] >= t.util_high_upto).mean()) if len(wd) else 0.0
    index = 100 * (0.5 * f_high + 1.0 * f_crit)
    # consecutive high-utilisation days per resource
    streaks = []
    for wid, g in wd.sort_values("day").groupby("worker"):
        best = cur = 0
        for u in g["util"]:
            cur = cur + 1 if u >= t.util_normal_upto else 0
            best = max(best, cur)
        streaks.append(dict(worker=wid, role=g["role"].iloc[0], pod=g["pod"].iloc[0], longest_high_streak_days=best))
    streaks = pd.DataFrame(streaks)
    pw = pod_week_workload(res)
    ot_cap = cfg.org.overtime_hours_per_day
    ed_wd = wd[wd["role"] == "editor"]
    return dict(index=index, share_high=f_high, share_critical=f_crit, per_worker=per_w, worker_day=wd,
                streaks=streaks, pod_week=pw,
                overtime_used_clock_h=float(ed_wd["ot_clock_h"].sum()),
                overtime_used_per_week=float(ed_wd["ot_clock_h"].sum() / max(w["weeks"], 1)),
                overtime_required_clock_h=float(pw["ot_required_clock_h"].sum()),
                overtime_required_per_week=float(pw["ot_required_clock_h"].sum() / max(w["weeks"], 1)),
                capacity_shortfall_h=float(pw["shortfall_h"].sum()),
                overtime_cap_per_day=ot_cap)


# --------------------------------------------------------------------------- #
# Little's Law
# --------------------------------------------------------------------------- #
def littles_law(res: SimulationResult, step: float = 1.0) -> dict:
    """Compare time-average system WIP (arrival -> approval) with throughput x mean lead time.

    L_obs   = time average of ads in system (arrived, not yet approved/cancelled) over the measured window
    lambda  = accepted ads per hour in the window;  W = mean lead time of ads accepted in the window
    The identity L = lambda x W holds exactly only for a stationary system observed long enough that
    start/end effects vanish; we also report whether the window looks stationary.
    """
    cfg, w = res.cfg, window(res)
    j = res.jobs
    t0, t1 = w["t_w"], res.t_end
    grid = np.arange(t0, t1 + 1e-9, step)
    arr = np.sort(j["arrival"].values)
    done = j[j["status"] != "open"]
    end_t = np.sort(np.where(done["status"] == "delivered", done["approval"], done["review_end"].fillna(t1)).astype(float))
    n_sys = np.searchsorted(arr, grid, side="right") - np.searchsorted(end_t, grid, side="right")
    L = float(n_sys.mean())
    dv = delivered_in_window(res)
    hours = t1 - t0
    lam = len(dv) / hours if hours > 0 else float("nan")
    W = float(dv["lead_time"].mean()) if len(dv) else float("nan")
    third = max(1, len(n_sys) // 3)
    first, last = n_sys[:third].mean(), n_sys[-third:].mean()
    drift = (last - first) / max(L, 1e-9)
    stable = (w["weeks"] >= cfg.thresholds.little_min_weeks) and abs(drift) < 0.25
    reasons = []
    if w["weeks"] < cfg.thresholds.little_min_weeks:
        reasons.append(f"window is only {w['weeks']} week(s) (< {cfg.thresholds.little_min_weeks}); start-up and end effects dominate")
    if abs(drift) >= 0.25:
        reasons.append(f"WIP drifts {drift*100:+.0f}% between first and last third of the window (non-stationary)")
    if cfg.demand.arrival_pattern != "uniform" or cfg.demand.demand_cv > 0.3:
        reasons.append("demand is non-uniform/highly variable, so averages hide the within-week cycle")
    if cfg.demand.initial_backlog_ads > 0:
        reasons.append("an initial backlog was loaded, so the system starts away from steady state")
    return dict(L_observed=L, throughput_per_hour=lam, mean_lead_time=W, lambda_times_W=lam * W if W == W else float("nan"),
                ratio=(L / (lam * W)) if lam and W == W and lam * W > 0 else float("nan"),
                stable=bool(stable), drift_pct=float(drift * 100), caveats=reasons,
                wip_series=pd.Series(n_sys, index=grid))
