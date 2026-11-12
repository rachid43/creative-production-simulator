"""Rule-based Lean improvement recommendations, each tied to observed simulation KPIs.

Every recommendation carries a ``scenario`` override that can be run in the Scenario Laboratory (or the
quick-test button) so the *direction* of improvement is verified by simulation rather than asserted.
Nothing here promises a specific improvement.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import lean_metrics as lm

# Named trigger thresholds (documented, not hidden)
WAIT_SHARE_TRIGGER = 0.05      # >5 % of job-hours waiting in a queue
IDLE_SHARE_TRIGGER = 0.02      # >2 % of editor capacity idle for one avoidable cause
FPY_TRIGGER = 90.0             # first-pass yield below 90 %
REWORK_SHARE_TRIGGER = 0.05    # rework >5 % of busy hours
WIP_EXCESS_RATIO = 1.5         # observed WIP > 1.5 x critical WIP
DAILY_CV_TRIGGER = 0.20


def build_recommendations(res, kpis: dict, bott: dict, mura: dict, muri: dict) -> List[dict]:
    cfg = res.cfg
    coh = lm.cohort(res)
    cols = [c for c in coh.columns if c.startswith(("T_", "L_"))] + ["overhead"]
    total = float(coh[cols].sum().sum()) or 1.0
    share = lambda keys: float(sum(coh[k].sum() for k in keys)) / total
    ed = lm.editor_day_table(res)
    cap = float(ed["capacity_h"].sum()) or 1.0
    idle = lambda c: float(ed.get(f"idle_{c}", ed["busy_h"] * 0).sum()) / cap
    score = {k: v["score"] for k, v in bott["candidates"].items()}
    recs: List[dict] = []

    def add(key, strength, title, observed, kpi, intervention, direction, scenario, impact):
        recs.append(dict(key=key, strength=float(strength), title=title, observed=observed, kpi=kpi, intervention=intervention,
                         direction=direction, scenario=scenario, estimated_impact=impact))

    # 1. imbalance
    pod_cv = float(mura["components"].set_index("component").cv["POD workload / capacity (CV across PODs)"])
    imb = mura["imbalance"]
    conc = float(imb["concurrent"].mean()) if len(imb) else 0.0
    if (pod_cv > 0.10 or conc > 0.10) and not cfg.lean.cross_pod_balancing:
        add("imbalance", max(score.get("imbalance", 0), conc),
            "Rebalance editing hours across PODs",
            f"POD planned load/capacity CV is {pod_cv:.2f}; {conc*100:.0f}% of days have an idle POD beside an overloaded POD.",
            f"Workload Balance Index {mura['workload_balance_index']:.2f}; max/min editor busy-hours ratio {mura['busy_ratio']:.2f}",
            "Enable cross-POD pull of ready briefs (skill-matched) and capacity-aware weekly allocation (Heijunka page).",
            "Lower Mura; higher throughput and shorter lead time in the overloaded PODs.",
            {"lean": {"cross_pod_balancing": True, "cross_pod_review_sharing": True}}, "Test as a scenario: cross-POD balancing on.")
    # 2. review
    w_rev = share(["L_wait_review_queue", "L_wait_blocked_review"])
    blocked = idle("blocked_review_wip")
    if w_rev > WAIT_SHARE_TRIGGER or blocked > IDLE_SHARE_TRIGGER:
        add("review", max(score.get("review", 0), w_rev * 4),
            "Reduce review turnaround",
            f"{w_rev*100:.1f}% of job-hours are spent waiting for / blocked by review; {blocked*100:.1f}% of editor capacity is blocked by the review WIP limit.",
            f"Review wait {coh['L_wait_review_queue'].mean():.2f} h/ad; blocked-for-review {coh['L_wait_blocked_review'].mean():.2f} h/ad",
            "Give review a dedicated time-box or reviewer, raise review speed (templates/checklists) or share review across PODs.",
            "Shorter lead time and less editor blocking; throughput rises only if review was limiting.",
            {"quality": {"review_capacity_multiplier": cfg.quality.review_capacity_multiplier * 1.3},
             "lean": {"cross_pod_review_sharing": True}}, "Scenario: review 30% faster + cross-POD review sharing.")
    # 3. brief supply
    w_brief = share(["L_wait_brief_queue"])
    starve = idle("starved_brief_supply")
    if (w_brief > WAIT_SHARE_TRIGGER or starve > IDLE_SHARE_TRIGGER) and not cfg.lean.brief_standardization:
        add("brief", max(score.get("brief", 0), starve * 8),
            "Standardise briefs",
            f"Editors are idle with no ready brief for {starve*100:.1f}% of capacity; {w_brief*100:.1f}% of job-hours wait for a brief.",
            f"Strategist utilization {bott['strategist_utilization']*100:.0f}%; ready-brief buffer {cfg.lean.ready_brief_buffer}",
            "Use reusable brief templates, batch briefing the day before, and protect strategist time for briefing.",
            "Fewer starvation gaps; shorter briefing queue; possibly higher first-pass approval.",
            {"lean": {"brief_standardization": True}}, "Scenario: brief standardisation (assumed effects are editable in Lean controls).")
    # 4. rework
    rw_share = float((coh["T_rework"] + coh["T_rework_review"]).sum()) / (float(res.worker_day["busy_h"].sum()) or 1.0)
    fpy = kpis.get("first_pass_yield_pct", np.nan)
    if rw_share > REWORK_SHARE_TRIGGER or (fpy == fpy and fpy < FPY_TRIGGER):
        add("rework", max(score.get("rework", 0), rw_share * 3),
            "Improve first-pass quality to cut rework",
            f"First-pass yield is {fpy:.0f}%; rework consumes {rw_share*100:.1f}% of all busy hours.",
            f"Rework rate {kpis['rework_rate_pct']:.0f}% of ads; rework hours {float((coh['T_rework']+coh['T_rework_review']).sum()):.0f} h",
            "Clearer briefs, pre-review checklists, early creative alignment before full edits.",
            "Capacity released from rework; shorter lead time; lower Muda.",
            {"quality": {"first_pass_approval_rate": min(0.97, cfg.quality.first_pass_approval_rate + 0.07)}}, "Scenario: first-pass approval +7 points.")
    # 5. editing capacity
    if kpis["editor_utilization_pct"] >= cfg.thresholds.util_high_upto * 100 - 3 or muri["capacity_shortfall_h"] > 0:
        add("capacity", max(score.get("editing", 0), kpis["editor_utilization_pct"] / 100 - 0.6),
            "Adjust staffing or overtime",
            f"Editors run at {kpis['editor_utilization_pct']:.0f}% utilization; planned workload exceeds capacity by {muri['capacity_shortfall_h']:.0f} h in the window.",
            f"Target attainment {kpis['target_attainment_pct']:.0f}%; overtime need {muri['overtime_required_per_week']:.0f} clock-h/week",
            "Add editing capacity (staffing, speed/tooling) or simplify the complexity mix; use overtime only as a short-term buffer.",
            "Higher throughput and shorter queues if editing is the constraint; near-100% loads still queue under variability.",
            {"org": {"editor_speed_multiplier": cfg.org.editor_speed_multiplier * 1.10}}, "Scenario: +10% effective editing capacity.")
    # 6. WIP limits
    ll = lm.littles_law(res)
    raw = float((coh["T_brief"] + coh["T_edit"] + coh["T_review"] + coh["overhead"]).mean()) if len(coh) else 0.0
    crit_wip = ll["throughput_per_hour"] * raw
    if crit_wip > 0 and ll["L_observed"] > WIP_EXCESS_RATIO * crit_wip and cfg.lean.ready_brief_buffer > 3:
        add("wip", min(1.0, ll["L_observed"] / crit_wip / 5),
            "Adjust WIP limits",
            f"Average ads in system ({ll['L_observed']:.0f}) is {ll['L_observed']/crit_wip:.1f}x the critical WIP ({crit_wip:.0f}) needed for the same throughput.",
            f"Little's Law: L={ll['L_observed']:.1f}, lambda*W={ll['lambda_times_W']:.1f}",
            "Lower the ready-brief buffer / WIP limits so work queues upstream instead of inside the system, as long as editors never starve.",
            "Shorter lead time at similar throughput; too-low limits cause starvation.",
            {"lean": {"ready_brief_buffer": max(3, cfg.lean.ready_brief_buffer - 2)}}, "Scenario: ready-brief buffer reduced by 2.")
    # 7. Heijunka
    dcv = float(mura["components"].set_index("component").cv["Daily accepted output (CV across days)"])
    if dcv > DAILY_CV_TRIGGER or cfg.demand.arrival_pattern != "uniform":
        add("heijunka", dcv,
            "Level demand release (Heijunka)",
            f"Daily accepted output varies with CV {dcv:.2f}; demand pattern is '{cfg.demand.arrival_pattern}'.",
            f"Daily demand CV {float(mura['components'].set_index('component').cv['Daily demand arrivals (CV across days)']):.2f}",
            "Release briefs at the daily pitch (36/day) instead of in batches; smooth the complexity mix per POD per day.",
            "Lower Mura; steadier daily output and queues.",
            {"demand": {"arrival_pattern": "uniform", "demand_cv": min(cfg.demand.demand_cv, 0.05)}}, "Scenario: uniform release, low demand variability.")
    # 8. skills
    skill = idle("skill_mismatch")
    if skill > 0.005 or (cfg.org.complex_capable_per_pod is not None and cfg.org.complex_capable_per_pod < cfg.org.editors_per_pod and cfg.complexity.mix["Complex"] >= 0.2):
        add("skills", max(skill * 10, 0.2),
            "Introduce skill-based resource sharing",
            f"Only {cfg.org.complex_capable_per_pod} editor(s) per POD can take Complex ads; {skill*100:.1f}% of editor capacity idles for skill mismatch.",
            f"Complex share {cfg.complexity.mix['Complex']*100:.0f}%",
            "Cross-train editors for Complex work and allow skill-matched cross-POD pulls.",
            "More flexible pull; fewer idle editors beside waiting complex work.",
            {"org": {"complex_capable_per_pod": None}, "lean": {"cross_pod_balancing": True}}, "Scenario: all editors certified for Complex + cross-POD pull.")
    # 9. assets
    a = idle("asset_wait")
    if a > IDLE_SHARE_TRIGGER:
        add("assets", a * 8,
            "Build a ready asset library",
            f"{a*100:.1f}% of editor capacity is idle waiting for assets.", f"Asset wait {coh['L_wait_asset'].mean():.2f} h/ad",
            "Pre-pack assets with every brief (checklist in the brief template).", "Less editor idle time and shorter lead time.",
            {"quality": {"asset_delay_probability": 0.0}}, "Scenario: asset delays eliminated (upper bound).")
    if not recs:
        add("none", 0.1, "Maintain the system and test a stretch scenario",
            "No KPI breaches the documented trigger thresholds.", f"Target attainment {kpis['target_attainment_pct']:.0f}%, "
            f"editor utilization {kpis['editor_utilization_pct']:.0f}%", "Keep the daily management routine; test demand +10-20% to find the next constraint.",
            "Early warning of the next bottleneck.", {"demand": {"weekly_target_ads": cfg.demand.weekly_target_ads * 1.15}}, "Scenario: demand +15%.")
    recs.sort(key=lambda r: -r["strength"])
    for i, r in enumerate(recs):
        r["priority"] = i + 1
    return recs
