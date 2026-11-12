"""Bottleneck identification.

The most utilised resource is *not* assumed to be the constraint.  Each candidate constraint is
scored from several independent pieces of evidence, each normalised to 0-1 with the named
reference values below, then averaged:

* utilisation of the resource group (ramp from ``UTIL_FLOOR`` to 100 %)
* queue waiting: share of all job-hours spent waiting in front of / blocked by that stage
* starvation: share of *downstream* editor capacity idled because of that stage
* queue growth: backlog growth per week relative to the weekly target
* throughput sensitivity: extra accepted ads/week obtained in a what-if re-run that adds capacity
  to that stage only (same random numbers, so the difference is attributable to the lever)

A candidate with strong utilisation but no waiting, no starvation and no sensitivity therefore
scores low - e.g. a busy-but-not-limiting strategist.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from simulation.engine import SimulationResult, run_simulation
from . import lean_metrics as lm

# Named reference values used to normalise evidence (documented, not hidden)
UTIL_FLOOR = 0.70          # utilisation below this contributes 0
WAIT_SHARE_REF = 0.25      # waiting = 25 % of all job-hours -> full score
STARVE_REF = 0.10          # 10 % of editor capacity idled by upstream/downstream stage -> full score
GROWTH_REF = 0.05          # backlog growing by 5 % of weekly target per week -> full score
SENS_REF = 0.03            # +3 % of weekly target in accepted ads/week from a +15 % lever -> full score
REWORK_REF = 0.15          # rework = 15 % of busy hours -> full score
SPREAD_REF = 0.60          # (max-min)/mean POD planned load/capacity of 60 % -> full score (random noise alone is ~10-20 %)
NO_CONSTRAINT_SCORE = 0.35  # below this the best candidate is too weak to call a binding constraint
LEVER_DELTA = 0.15

LABELS = {
    "editing": "Editing capacity (overloaded editors)",
    "brief": "Brief supply (strategist capacity / ready briefs)",
    "review": "Review capacity / turnaround",
    "rework": "Excessive rework",
    "imbalance": "Unbalanced workload across PODs",
    "assets": "Unavailable assets",
    "external": "External feedback delay",
}


def _ramp(x: float, lo: float, hi: float) -> float:
    if x != x:
        return 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def throughput_sensitivity(cfg, seed: int, n_seeds: int = 3, delta: float = LEVER_DELTA) -> Dict[str, float]:
    """Extra accepted ads/week for each what-if lever, averaged over paired seeds (common random numbers)."""
    from .kpis import compute_kpis

    base_cfg = cfg.copy_with(record_events=False, snapshot_interval_hours=4.0)
    q = base_cfg.quality
    fp = base_cfg.quality.first_pass_approval_rate
    levers = {
        "editing": base_cfg.copy_with(org={"editor_speed_multiplier": base_cfg.org.editor_speed_multiplier * (1 + delta)}),
        "brief": base_cfg.copy_with(quality={"brief_hours": q.brief_hours / (1 + delta)}),
        "review": base_cfg.copy_with(quality={"review_capacity_multiplier": q.review_capacity_multiplier * (1 + delta)}),
        "rework": base_cfg.copy_with(quality={"first_pass_approval_rate": 1 - (1 - fp) * 0.5,
                                              "rework_repeat_probability": q.rework_repeat_probability * 0.5,
                                              "pod_first_pass": None if q.pod_first_pass is None else
                                              [None if x is None else 1 - (1 - x) * 0.5 for x in q.pod_first_pass]}),
        "imbalance": base_cfg.copy_with(lean={"cross_pod_balancing": True, "cross_pod_review_sharing": True}),
        "assets": base_cfg.copy_with(quality={"asset_delay_probability": 0.0}),
        "external": base_cfg.copy_with(quality={"review_response_delay_hours": 0.0}),
    }
    out = {k: [] for k in levers}
    for s in range(n_seeds):
        sd = seed + s
        base = compute_kpis(run_simulation(base_cfg, sd))["accepted_per_week"]
        for k, c in levers.items():
            out[k].append(compute_kpis(run_simulation(c, sd))["accepted_per_week"] - base)
    return {k: float(np.mean(v)) for k, v in out.items()}


def _slope_per_week(series: pd.Series, hours_per_week: float) -> float:
    if len(series) < 3:
        return 0.0
    x = series.index.values.astype(float) / hours_per_week
    return float(np.polyfit(x, series.values.astype(float), 1)[0])


def detect_bottleneck(res: SimulationResult, sensitivity: Optional[Dict[str, float]] = None) -> dict:
    cfg, w = res.cfg, lm.window(res)
    coh = lm.cohort(res)
    wk_h = w["H"] * w["D"]
    cols = [c for c in coh.columns if c.startswith("T_") or c.startswith("L_")] + ["overhead"]
    total = float(coh[cols].sum().sum()) or 1.0
    share = lambda keys: float(sum(coh[k].sum() for k in keys)) / total

    ed = lm.editor_day_table(res)
    ed_cap = float(ed["capacity_h"].sum()) or 1.0
    ed_util = float(ed["busy_h"].sum() / ed_cap)
    idle = lambda c: float(ed[f"idle_{c}"].sum()) / ed_cap if f"idle_{c}" in ed else 0.0
    wd = res.worker_day[(res.worker_day["day"] >= w["d_w"])]
    st = wd[wd["role"] == "strategist"]
    st_cap = float(st["capacity_h"].sum()) or 1.0
    st_util = float(st["busy_h"].sum() / st_cap)
    brief_util = float(st["busy_brief"].sum() / st_cap)
    mode = cfg.quality.review_mode
    if mode == "strategist":
        rev_util = float((st["busy_review"] + st["busy_rework_review"]).sum() / st_cap)
        rev_util_total = st_util  # shared with briefing
    else:
        rv = wd[wd["role"].isin(["reviewer", "lead_creative"])]
        rv_cap = float(rv["capacity_h"].sum()) or 1.0
        rev_util = rev_util_total = float(rv["busy_h"].sum() / rv_cap)

    ps = res.pod_snapshots[res.pod_snapshots["t"] >= w["t_w"] - 1e-9]
    backlog_slope = _slope_per_week(ps.groupby("t")["backlog"].sum(), wk_h)
    review_slope = _slope_per_week(ps.groupby("t")["review_wip"].sum(), wk_h)
    ready_avg = float(ps.groupby("t")["ready"].sum().mean() / max(cfg.org.n_pods * cfg.lean.ready_brief_buffer, 1))
    growth_ref = GROWTH_REF * cfg.demand.weekly_target_ads
    backlog_growth_editing = _ramp(backlog_slope, 0, growth_ref) if ready_avg >= 0.5 else 0.0
    backlog_growth_brief = _ramp(backlog_slope, 0, growth_ref) if ready_avg < 0.5 else 0.0

    # Imbalance is judged on the *structural* load (planned standard hours / capacity per POD), not on resulting
    # utilisation: uneven utilisation can be a symptom of some other shared constraint (e.g. one reviewer).
    pw = lm.pod_week_workload(res).groupby("pod").agg(p=("planned_hours", "sum"), c=("cap_std", "sum"))
    lr = (pw["p"] / pw["c"].replace(0, np.nan)).dropna()
    spread = float((lr.max() - lr.min()) / lr.mean()) if len(lr) > 1 and lr.mean() > 0 else 0.0
    imb = lm.imbalance_days(res)
    concurrent_share = float(imb["concurrent"].mean()) if len(imb) else 0.0

    rework_hours = float((coh["T_rework"] + coh["T_rework_review"]).sum())
    busy_all = float(wd["busy_h"].sum()) or 1.0
    rework_share = rework_hours / busy_all

    sens = sensitivity or {}
    sref = SENS_REF * cfg.demand.weekly_target_ads
    s_norm = lambda k: _ramp(sens[k], 0, sref) if k in sens else None

    cand: Dict[str, dict] = {}

    def add(key, comps, ev):
        vals = [v for v in comps.values() if v is not None]
        cand[key] = dict(candidate=LABELS[key], score=float(np.mean(vals)) if vals else 0.0, components=comps, evidence=ev)

    wait_edit = share(["L_wait_ready_queue", "L_wait_rework_queue"])
    add("editing", dict(utilization=_ramp(ed_util, UTIL_FLOOR, 1.0), queue_wait=_ramp(wait_edit, 0, WAIT_SHARE_REF),
                        queue_growth=backlog_growth_editing, sensitivity=s_norm("editing")),
        [f"editor utilization {ed_util*100:.0f}%", f"{wait_edit*100:.1f}% of job-hours waiting for an editor",
         f"backlog trend {backlog_slope:+.1f} ads/week with ready buffer {ready_avg*100:.0f}% full"]
        + ([f"+{LEVER_DELTA*100:.0f}% editing speed adds {sens['editing']:+.1f} ads/week"] if "editing" in sens else []))
    wait_brief = share(["L_wait_brief_queue"])
    starve_b = idle("starved_brief_supply")
    add("brief", dict(utilization=_ramp(st_util, UTIL_FLOOR, 1.0), queue_wait=_ramp(wait_brief, 0, WAIT_SHARE_REF),
                      starvation=_ramp(starve_b, 0, STARVE_REF), queue_growth=backlog_growth_brief, sensitivity=s_norm("brief")),
        [f"strategist utilization {st_util*100:.0f}% (briefing {brief_util*100:.0f}%)", f"{wait_brief*100:.1f}% of job-hours waiting for a brief",
         f"{starve_b*100:.1f}% of editor capacity idle with no ready brief"]
        + ([f"+{LEVER_DELTA*100:.0f}% briefing speed adds {sens['brief']:+.1f} ads/week"] if "brief" in sens else []))
    wait_rev = share(["L_wait_review_queue", "L_wait_blocked_review"])
    blocked = idle("blocked_review_wip")
    add("review", dict(utilization=_ramp(rev_util_total, UTIL_FLOOR, 1.0),
                       queue_wait=_ramp(wait_rev, 0, WAIT_SHARE_REF), starvation=_ramp(blocked, 0, STARVE_REF),
                       queue_growth=_ramp(review_slope, 0, 0.5), sensitivity=s_norm("review")),
        [f"review resource utilization {rev_util_total*100:.0f}% ({mode} reviews)", f"{wait_rev*100:.1f}% of job-hours waiting for / blocked by review",
         f"{blocked*100:.1f}% of editor capacity blocked by the review WIP limit"]
        + ([f"+{LEVER_DELTA*100:.0f}% review speed adds {sens['review']:+.1f} ads/week"] if "review" in sens else []))
    wait_rw = share(["L_wait_rework_queue"])
    add("rework", dict(rework_hours=_ramp(rework_share, 0, REWORK_REF), queue_wait=_ramp(wait_rw, 0, WAIT_SHARE_REF / 2),
                       sensitivity=s_norm("rework")),
        [f"rework + rework review = {rework_share*100:.1f}% of all busy hours", f"{wait_rw*100:.1f}% of job-hours waiting for rework pickup"]
        + ([f"halving rework adds {sens['rework']:+.1f} ads/week"] if "rework" in sens else []))
    add("imbalance", dict(pod_load_spread=_ramp(spread, 0, SPREAD_REF), sensitivity=s_norm("imbalance")),
        [f"POD planned load/capacity spread (max-min)/mean = {spread*100:.0f}%",
         f"({concurrent_share*100:.0f}% of days show an idle POD beside an overloaded one - see Mura page)"]
        + ([f"cross-POD balancing adds {sens['imbalance']:+.1f} ads/week"] if "imbalance" in sens else []))
    asset_share = share(["L_wait_asset"])
    add("assets", dict(editor_idle=_ramp(idle("asset_wait"), 0, STARVE_REF), queue_wait=_ramp(asset_share, 0, WAIT_SHARE_REF),
                       sensitivity=s_norm("assets")),
        [f"{idle('asset_wait')*100:.1f}% of editor capacity idle waiting for assets", f"{asset_share*100:.1f}% of job-hours waiting for assets"]
        + ([f"eliminating asset delays adds {sens['assets']:+.1f} ads/week"] if "assets" in sens else []))
    ext = share(["L_wait_review_external"])
    add("external", dict(queue_wait=_ramp(ext, 0, WAIT_SHARE_REF), sensitivity=s_norm("external")),
        [f"{ext*100:.1f}% of job-hours waiting for external feedback"] if cfg.quality.review_response_delay_hours > 0 else ["no external feedback delay configured"])

    table = pd.DataFrame([dict(key=k, candidate=v["candidate"], score=v["score"], evidence="; ".join(v["evidence"]))
                          for k, v in cand.items()]).sort_values("score", ascending=False).reset_index(drop=True)
    top = table.iloc[0]
    from .kpis import compute_kpis
    att = compute_kpis(res)["target_attainment_pct"]
    queue_end = float(res.pod_snapshots[res.pod_snapshots["t"] == res.pod_snapshots["t"].max()][["backlog", "ready", "rework_q"]].sum().sum())
    no_constraint = bool(top["score"] < NO_CONSTRAINT_SCORE or (att >= 98 and queue_end <= cfg.org.n_pods * cfg.lean.ready_brief_buffer))
    note = ""
    if not no_constraint and top["key"] != "editing" and ed_util >= cfg.thresholds.util_high_upto:
        note = (f"Editors are heavily utilised ({ed_util*100:.0f}%), yet the strongest combined evidence points to "
                f"'{top['candidate']}': high utilisation alone does not make a resource the constraint.")
    elif not no_constraint and top["key"] == "editing":
        note = "Editor utilisation, waiting for editors and throughput sensitivity agree: extra editing capacity limits output."
    primary = "No binding constraint (demand-limited)" if no_constraint else top["candidate"]
    explanation = (["Output is meeting target with small queues - the system is demand-limited, not capacity-limited."] if no_constraint else
                   [f"Top candidate: {top['candidate']} (evidence score {top['score']:.2f}).", *top["evidence"].split("; ")])
    if note:
        explanation.append(note)
    if not sens:
        explanation.append("Throughput-sensitivity evidence was not computed; scores use utilisation, waiting, starvation and queue growth only.")
    return dict(primary=primary, primary_key=None if no_constraint else top["key"], table=table, candidates=cand,
                explanation=explanation, no_constraint=no_constraint, editor_utilization=ed_util, strategist_utilization=st_util)
