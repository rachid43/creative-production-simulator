"""Dispatching rules (used by the DES) and the static Heijunka weekly planner.

DES rules
---------
* Default queue discipline is FIFO (time the ad entered the queue).
* With ``priority_scheduling`` urgent ads go first, then earliest due date, then FIFO.
* Editors *pull*: they take work only when idle, within the POD's editing WIP limit, and only
  from briefs that are already approved (ready).  Rework is pulled before new briefs
  ("finish before starting").
* Cross-POD balancing: an editor with no eligible work in their own POD may pull a *ready*
  brief from a donor POD holding at least ``cross_pod_min_queue`` ready briefs, if skills match.
  The ad stays owned by (and is credited to) its home POD.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config.schema import COMPLEXITIES, SimConfig
from .entities import Job, Worker

BIG = 10 ** 9


def queue_key(job: Job, priority_on: bool):
    if priority_on:
        return (0 if job.urgent else 1, job.due_time, job.wait_since, job.seq)
    return (job.wait_since, job.seq)


def _can_do(w: Worker, job: Job) -> bool:
    if job.complexity == "Complex" and not w.can_complex:
        return False
    return job.planned_editor is None or job.planned_editor == w.wid


# --------------------------------------------------------------------------- #
# Editing
# --------------------------------------------------------------------------- #
def pick_edit_job(sim, ed: Worker) -> Tuple[Optional[Job], bool]:
    """Return (job, is_rework) for editor ``ed`` or (None, False)."""
    cfg = sim.cfg
    lim = cfg.lean.editing_wip_limit if cfg.lean.enforce_wip_limits else BIG
    prio = cfg.lean.priority_scheduling
    pods = sim.pods

    own = pods[ed.pod]
    if own.editing >= lim:  # the working POD's editing station is full (limit counts work done by its editors)
        return None, False

    if cfg.lean.assignment_policy != "pull":
        best = None
        for pod in pods:
            for lst, rw in ((pod.rework, True), (pod.ready, False)):
                for j in lst:
                    if _can_do(ed, j):
                        k = (0 if rw else 1,) + queue_key(j, prio)
                        if best is None or k < best[0]:
                            best = (k, j, rw)
        return (best[1], best[2]) if best else (None, False)

    c = [j for j in own.rework if _can_do(ed, j)]
    if c:
        return min(c, key=lambda j: queue_key(j, prio)), True
    c = [j for j in own.ready if _can_do(ed, j)]
    if c:
        return min(c, key=lambda j: queue_key(j, prio)), False
    if cfg.lean.cross_pod_balancing:
        best_pod, best_c = None, []
        for pod in pods:
            if pod.idx == ed.pod:
                continue
            c = [j for j in pod.ready if _can_do(ed, j)]
            if len(c) >= cfg.lean.cross_pod_min_queue and len(c) > len(best_c):
                best_pod, best_c = pod, c
        if best_pod is not None:
            return min(best_c, key=lambda j: queue_key(j, prio)), False
    return None, False


def edit_idle_reason(sim, ed: Worker) -> str:
    """Why an available editor has nothing to do (used only for capacity-loss accounting)."""
    cfg = sim.cfg
    own = sim.pods[ed.pod]
    lim = cfg.lean.editing_wip_limit if cfg.lean.enforce_wip_limits else BIG
    if cfg.lean.assignment_policy != "pull":
        if own.editing >= lim:
            return "blocked_wip_limit"
        if any(j.planned_editor == ed.wid for p in sim.pods for j in (p.ready + p.rework)):
            return "skill_mismatch"
        if any(j.planned_editor == ed.wid for p in sim.pods for j in p.backlog):
            return "starved_brief_supply"
        if any(j.planned_editor == ed.wid and j.state == "briefing" for j in sim.jobs if j.status == "open"):
            return "brief_in_progress"
        return "no_demand"
    if own.ready or own.rework:
        if own.editing >= lim:
            return "blocked_wip_limit"
        return "skill_mismatch"
    if own.backlog:
        return "starved_brief_supply"       # ads are waiting for a strategist while editors have nothing ready
    if own.briefing > 0:
        return "brief_in_progress"          # unavoidable briefing process time; strategist is already working on it
    return "no_demand"


# --------------------------------------------------------------------------- #
# Briefing / review
# --------------------------------------------------------------------------- #
def pick_brief_job(sim, st: Worker) -> Optional[Job]:
    cfg = sim.cfg
    pod = sim.pods[st.pod]
    if not pod.backlog:
        return None
    if cfg.lean.enforce_wip_limits and (len(pod.ready) + pod.briefing) >= cfg.lean.ready_brief_buffer:
        return None
    return min(pod.backlog, key=lambda j: queue_key(j, cfg.lean.priority_scheduling))


def brief_idle_reason(sim, st: Worker) -> str:
    pod = sim.pods[st.pod]
    if pod.backlog:
        return "buffer_full_pull_signal"
    return "no_demand"


def _eligible_reviews(pod, now: float) -> List[Job]:
    return [j for j in pod.review_q if j.review_ready_at <= now + 1e-9]


def pick_review_job(sim, w: Worker, now: float) -> Optional[Job]:
    cfg = sim.cfg
    prio = cfg.lean.priority_scheduling
    if w.role == "strategist":
        own = _eligible_reviews(sim.pods[w.pod], now)
        if own:
            return min(own, key=lambda j: queue_key(j, prio))
        if cfg.lean.cross_pod_review_sharing:
            best = []
            for pod in sim.pods:
                if pod.idx == w.pod:
                    continue
                c = _eligible_reviews(pod, now)
                if len(c) >= cfg.lean.cross_pod_min_queue and len(c) > len(best):
                    best = c
            if best:
                return min(best, key=lambda j: queue_key(j, prio))
        return None
    # pooled reviewers (dedicated or Lead Creative) serve every POD, FIFO across PODs
    allc = [j for pod in sim.pods for j in _eligible_reviews(pod, now)]
    return min(allc, key=lambda j: queue_key(j, prio)) if allc else None


# --------------------------------------------------------------------------- #
# Static assignment used by the 'equal_count / equal_hours / capacity_aware' policies in the DES
# --------------------------------------------------------------------------- #
def choose_planned_editor(sim, job: Job) -> Optional[str]:
    pol = sim.cfg.lean.assignment_policy
    if pol == "pull":
        return None
    pool = sim.editors if sim.cfg.lean.cross_pod_balancing and pol == "capacity_aware" else sim.editors_by_pod[job.pod]
    pool = [e for e in pool if job.complexity != "Complex" or e.can_complex] or list(pool)
    if not pool:
        return None
    if pol == "equal_count":
        key = lambda e: (sim.assigned_count[e.wid], e.wid)
    elif pol == "equal_hours":
        key = lambda e: (sim.assigned_hours[e.wid], e.wid)
    else:  # capacity_aware: outstanding standard hours relative to speed-adjusted availability
        key = lambda e: ((sim.outstanding_hours[e.wid] + job.est_hours) / (e.speed * e.base_rate * (1 - sim.cfg.org.planned_absenteeism) + 1e-9), e.wid)
    e = min(pool, key=key)
    sim.assigned_count[e.wid] += 1
    sim.assigned_hours[e.wid] += job.est_hours
    sim.outstanding_hours[e.wid] += job.est_hours
    return e.wid


# =========================================================================== #
# Static Heijunka planner (weekly)
# =========================================================================== #
def _largest_remainder(total: float, shares: List[float]) -> List[int]:
    raw = [total * s for s in shares]
    base = [int(math.floor(x + 1e-9)) for x in raw]
    rem = int(round(total)) - sum(base)
    order = sorted(range(len(raw)), key=lambda i: -(raw[i] - base[i]))
    for i in order[: max(0, rem)]:
        base[i] += 1
    return base


def editor_speed(seed: int, index: int, cv: float) -> float:
    """Editor speed factor; identical formula to the DES so plan and simulation agree."""
    if cv <= 0:
        return 1.0
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2, index]))
    z = rng.standard_normal()
    s2 = math.log(1 + cv * cv)
    return math.exp(-s2 / 2 + math.sqrt(s2) * z)


def plan_week(cfg: SimConfig, method: str, seed: int = 0, allow_cross_pod: bool = False) -> dict:
    """Build a one-week production plan and score its balance.

    method: 'equal_count' | 'equal_hours' | 'capacity_aware'
    Capacity of an editor (in *standard hours*) = productive availability x weekly hours x
    (1 - planned absenteeism) x editor speed (+ overtime allowance is reported separately).
    """
    org = cfg.org
    H, D = org.hours_per_day, org.days_per_week
    shares = cfg.pod_shares
    per_pod = _largest_remainder(cfg.demand.weekly_target_ads, shares)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 31]))
    wk_w = cfg.weekday_weights()

    # ---- editors ------------------------------------------------------------
    editors = []
    idx = 0
    for p in range(org.n_pods):
        n_e = org.editors_in_pod(p)
        n_cx = n_e if org.complex_capable_per_pod is None else min(n_e, org.complex_capable_per_pod)
        for k in range(n_e):
            sp = editor_speed(cfg.seed if seed is None else seed, idx, org.editor_speed_cv) * org.editor_speed_multiplier
            cap = org.productive_availability * H * D * (1 - org.planned_absenteeism) * sp
            editors.append(dict(editor=f"P{p+1}-E{k+1}", pod=p, can_complex=k < n_cx, speed=sp, capacity_h=cap, load_h=0.0, jobs=[]))
            idx += 1

    # ---- jobs ---------------------------------------------------------------
    jobs = []
    for p in range(org.n_pods):
        n = per_pod[p]
        counts = _largest_remainder(n, [cfg.mix_for(p)[c] for c in COMPLEXITIES])
        seq = [c for c, k in zip(COMPLEXITIES, counts) for _ in range(k)]
        rng.shuffle(seq)
        days = []
        for d, k in enumerate(_largest_remainder(n, wk_w)):
            days += [d] * k
        for i, c in enumerate(seq):
            jobs.append(dict(pod=p, complexity=c, std_h=cfg.complexity.standard_hours[c], release_day=days[i] if i < len(days) else D - 1))
    # ---- allocation -----------------------------------------------------------
    by_pod = {p: [e for e in editors if e["pod"] == p] for p in range(org.n_pods)}
    rr = {p: 0 for p in range(org.n_pods)}
    unplaced = 0

    def pool_for(job):
        pool = editors if (allow_cross_pod and method == "capacity_aware") else by_pod[job["pod"]]
        sk = [e for e in pool if job["complexity"] != "Complex" or e["can_complex"]]
        return sk or pool

    order = jobs
    if method in ("equal_hours", "capacity_aware"):
        order = sorted(jobs, key=lambda j: (-j["std_h"], j["pod"]))  # LPT
    for j in order:
        pool = pool_for(j)
        if not pool:
            unplaced += 1
            j["editor"] = None
            continue
        if method == "equal_count":
            sk = [e for e in by_pod[j["pod"]] if j["complexity"] != "Complex" or e["can_complex"]] or by_pod[j["pod"]]
            e = sk[rr[j["pod"]] % len(sk)]
            rr[j["pod"]] += 1
        elif method == "equal_hours":
            e = min(pool, key=lambda x: x["load_h"])
        else:
            e = min(pool, key=lambda x: (x["load_h"] + j["std_h"]) / max(x["capacity_h"], 1e-9))
        j["editor"] = e["editor"]
        j["editor_pod"] = e["pod"]
        e["load_h"] += j["std_h"]
        e["jobs"].append(j)

    # ---- sequence within editor (earliest release first) and finish day ------------------------
    for e in editors:
        daily_cap = e["capacity_h"] / D if D else 0
        t = 0.0  # cumulative standard hours
        for j in sorted(e["jobs"], key=lambda x: (x["release_day"], -x["std_h"])):
            start_h = max(t, j["release_day"] * daily_cap)
            t = start_h + j["std_h"]
            j["start_day"] = int(min(D + 5, start_h // daily_cap)) if daily_cap > 0 else D + 5
            j["finish_day"] = int(math.ceil(t / daily_cap - 1e-9)) - 1 if daily_cap > 0 else D + 5
            j["fits_week"] = j["finish_day"] <= D - 1

    jobs_df = pd.DataFrame(jobs)
    ed_df = pd.DataFrame([{k: v for k, v in e.items() if k != "jobs"} | {
        "planned_ads": len(e["jobs"]),
        "simple": sum(1 for j in e["jobs"] if j["complexity"] == "Simple"),
        "medium": sum(1 for j in e["jobs"] if j["complexity"] == "Medium"),
        "complex": sum(1 for j in e["jobs"] if j["complexity"] == "Complex"),
        "ads_fit_in_week": sum(1 for j in e["jobs"] if j["fits_week"]),
    } for e in editors])
    ed_df["load_ratio"] = ed_df["load_h"] / ed_df["capacity_h"].replace(0, np.nan)
    ed_df["overflow_h"] = (ed_df["load_h"] - ed_df["capacity_h"]).clip(lower=0)
    ed_df["spare_h"] = (ed_df["capacity_h"] - ed_df["load_h"]).clip(lower=0)

    pod_df = ed_df.groupby("pod").agg(
        planned_ads=("planned_ads", "sum"), load_h=("load_h", "sum"), capacity_h=("capacity_h", "sum"),
        simple=("simple", "sum"), medium=("medium", "sum"), complex=("complex", "sum"),
        ads_fit_in_week=("ads_fit_in_week", "sum"),
    ).reset_index()
    pod_df["load_ratio"] = pod_df["load_h"] / pod_df["capacity_h"].replace(0, np.nan)
    pod_df["pod_name"] = pod_df["pod"].map(lambda p: f"POD {p+1}")

    lr = ed_df["load_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    cv = float(lr.std(ddof=0) / lr.mean()) if len(lr) and lr.mean() > 0 else 0.0
    summary = dict(
        method=method,
        balance_index=float(max(0.0, 1 - cv)),
        load_ratio_cv=cv,
        max_load_ratio=float(lr.max()) if len(lr) else 0.0,
        min_load_ratio=float(lr.min()) if len(lr) else 0.0,
        overloaded_editors=int((lr > 1.0).sum()),
        overflow_hours=float(ed_df["overflow_h"].sum()),
        planned_ads=int(len(jobs_df)),
        ads_fit_in_week=int(jobs_df["fits_week"].sum()) if "fits_week" in jobs_df else 0,
        unplaced=unplaced,
    )
    daily = (jobs_df.groupby("finish_day").size().reindex(range(D), fill_value=0)
             if "finish_day" in jobs_df else pd.Series(0, index=range(D)))
    # brief release: a brief must be ready the working day before its planned editing start
    jobs_df["brief_release_day"] = jobs_df["start_day"].clip(upper=D - 1).sub(1).clip(lower=0)
    release = jobs_df.pivot_table(index="brief_release_day", columns="pod", values="std_h", aggfunc="count", fill_value=0)
    release = release.reindex(index=range(D), columns=range(org.n_pods), fill_value=0)
    return dict(summary=summary, jobs=jobs_df, editors=ed_df, pods=pod_df, daily_output=daily, brief_release=release)
