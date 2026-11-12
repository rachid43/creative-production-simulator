"""Discrete-event simulation engine (SimPy) for the Lean Creative Production flow.

Flow:  Demand -> Briefing (strategist) -> Ready buffer -> Editing (pull) -> Review
       -> (Rework -> Editing -> Review)* -> Approved / Delivered

Design guarantees (see tests):
* every person is a finite resource processing one task at a time, only inside the shift calendar;
* editing starts only from an approved (ready) brief and review only after editing finishes;
* rework consumes real additional editor *and* reviewer capacity;
* an ad counts as delivered only when approved;
* WIP limits gate work release (brief buffer), editing start and the hand-off to review;
* nothing is dropped at the horizon - unfinished work stays in its queue and is reported;
* per-ad random inputs use common random numbers so scenarios are comparable.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import simpy

from config.schema import COMPLEXITIES, SimConfig
from .entities import (
    APPROVED, BACKLOG, BLOCKED, BRIEFING, CANCELLED, EDITING, OPEN_STATES, READY, REVIEW, REWORK,
    TOUCH_CATEGORIES, WAIT_CATEGORIES, Job, Worker,
)
from .resources import PodState, make_absence, productive_capacity
from .scheduling import (
    _largest_remainder, brief_idle_reason, choose_planned_editor, edit_idle_reason, editor_speed,
    pick_brief_job, pick_edit_job, pick_review_job,
)

_ND = NormalDist()
EPS = 1e-9


def _z(u: float) -> float:
    return _ND.inv_cdf(min(max(u, 1e-12), 1 - 1e-12))


def _lognormal(mean: float, cv: float, u: float) -> float:
    """Mean-preserving lognormal draw from a uniform ``u`` (CV=0 -> deterministic mean)."""
    if mean <= 0 or cv <= 0:
        return mean
    s2 = math.log(1 + cv * cv)
    return math.exp(math.log(mean) - s2 / 2 + math.sqrt(s2) * _z(u))


@dataclass
class SimulationResult:
    cfg: SimConfig
    seed: int
    t_end: float
    jobs: pd.DataFrame
    events: pd.DataFrame
    pod_snapshots: pd.DataFrame
    worker_snapshots: pd.DataFrame
    worker_day: pd.DataFrame
    workers: pd.DataFrame
    conservation: Dict[str, float]
    job_objects: List[Job]


class ProductionSimulation:
    def __init__(self, cfg: SimConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.seed = cfg.seed if seed is None else seed
        self.env = simpy.Environment()
        self.H = cfg.org.hours_per_day
        self.n_days = cfg.n_days
        self.T_end = cfg.horizon_hours
        self.pods: List[PodState] = [PodState(i) for i in range(cfg.org.n_pods)]
        self.jobs: List[Job] = []
        self.events: List[dict] = []
        self.snap_pod: List[dict] = []
        self.snap_worker: List[dict] = []
        self._wakes = {k: self.env.event() for k in ("edit", "staff", "slot")}
        self._ot_cache: Dict[tuple, bool] = {}
        self._marks: Dict[str, tuple] = {}
        self.rng_dyn = np.random.default_rng(np.random.SeedSequence([self.seed, 9]))
        self.assigned_count: Dict[str, int] = defaultdict(int)
        self.assigned_hours: Dict[str, float] = defaultdict(float)
        self.outstanding_hours: Dict[str, float] = defaultdict(float)
        self._build_workers()

    # ------------------------------------------------------------------ setup
    def _build_workers(self) -> None:
        org, q, cfg = self.cfg.org, self.cfg.quality, self.cfg
        a = org.productive_availability
        self.editors: List[Worker] = []
        self.strategists: List[Worker] = []
        self.reviewers: List[Worker] = []
        idx = 0
        for p in range(org.n_pods):
            n_e = org.editors_in_pod(p)
            n_cx = n_e if org.complex_capable_per_pod is None else min(n_e, org.complex_capable_per_pod)
            for k in range(n_e):
                self.editors.append(Worker(
                    wid=f"P{p+1}-E{k+1}", role="editor", pod=p, base_rate=a,
                    speed=editor_speed(self.seed, idx, org.editor_speed_cv) * org.editor_speed_multiplier, can_complex=k < n_cx,
                    absent=make_absence(self.seed, idx, self.n_days, org.planned_absenteeism)))
                idx += 1
            for k in range(org.strategists_per_pod):
                sid = f"S{p+1}" if org.strategists_per_pod == 1 else f"S{p+1}{chr(97 + k)}"
                self.strategists.append(Worker(
                    wid=sid, role="strategist", pod=p, base_rate=cfg.support_availability,
                    absent=make_absence(self.seed, 1000 + p * 10 + k, self.n_days, org.planned_absenteeism)))
        if q.review_mode == "dedicated":
            for k in range(q.dedicated_reviewers):
                self.reviewers.append(Worker(
                    wid=f"R{k+1}", role="reviewer", pod=-1, base_rate=cfg.support_availability,
                    absent=make_absence(self.seed, 2000 + k, self.n_days, org.planned_absenteeism)))
        elif q.review_mode == "lead_creative":
            self.reviewers.append(Worker(
                wid="LC", role="lead_creative", pod=-1, base_rate=cfg.support_availability * q.lead_creative_review_share,
                absent=make_absence(self.seed, 3000, self.n_days, org.planned_absenteeism)))
        self.editors_by_pod = {p: [e for e in self.editors if e.pod == p] for p in range(org.n_pods)}
        self.all_workers = self.editors + self.strategists + self.reviewers

    # --------------------------------------------------------------- calendar
    def _day(self, t: float) -> int:
        return int(t / self.H + EPS)

    def _ot_active(self, pod: int, d: int) -> bool:
        key = (pod, d)
        if key not in self._ot_cache:
            p = self.pods[pod]
            n_e = max(1, len(self.editors_by_pod[pod]))
            waiting = len(p.ready) + len(p.rework)
            self._ot_cache[key] = waiting / n_e >= self.cfg.org.overtime_trigger_queue_per_editor
        return self._ot_cache[key]

    def rate(self, w: Worker, d: int) -> float:
        """Productive hours per clock hour for worker ``w`` on day ``d`` (0 = absent)."""
        if w.absent is not None and w.absent[min(d, len(w.absent) - 1)]:
            return 0.0
        r = w.base_rate
        ot = self.cfg.org.overtime_hours_per_day
        if w.role == "editor" and ot > 0 and self._ot_active(w.pod, d):
            r = min(1.0, r * (self.H + ot) / self.H)
        return r

    def notify(self, *chans: str) -> None:
        for c in chans:
            old = self._wakes[c]
            self._wakes[c] = self.env.event()
            if not old.triggered:
                old.succeed()

    def _frozen(self) -> bool:
        return self.env.now >= self.T_end - EPS

    def _forever(self):
        yield self.env.event()

    def _wait_available(self, w: Worker):
        while True:
            if self._frozen():
                yield from self._forever()
            now = self.env.now
            d = self._day(now)
            if self.rate(w, d) > 0:
                return
            w.state = "absent"
            yield self.env.timeout(max((d + 1) * self.H - now, EPS))

    # -------------------------------------------------------------- accounting
    def _mark(self, w: Worker, cause: str, state: str = "idle") -> None:
        self._marks[w.wid] = (self.env.now, cause)
        w.state = state

    def _unmark(self, w: Worker) -> None:
        t0, cause = self._marks.pop(w.wid)
        self._account_idle(w, t0, self.env.now, cause)

    def _account_idle(self, w: Worker, t0: float, t1: float, cause: str) -> None:
        if t1 <= t0 + 1e-12:
            return
        for d in range(self._day(t0), self._day(t1 - EPS) + 1):
            lo, hi = max(t0, d * self.H), min(t1, (d + 1) * self.H)
            if hi > lo:
                w.idle[(d, cause)] += (hi - lo) * self.rate(w, d)

    def _log(self, etype: str, job: Optional[Job] = None, w: Optional[Worker] = None, **extra) -> None:
        if self.cfg.record_events:
            self.events.append(dict(t=self.env.now, event=etype, job_id=job.job_id if job else None,
                                    pod=job.pod if job else (w.pod if w else None),
                                    resource=w.wid if w else None, **extra))

    def _work(self, w: Worker, hours: float, job: Job, act: str):
        """Consume ``hours`` of *productive* time, honouring the shift calendar and absences."""
        rem = hours
        while rem > 1e-9:
            if self._frozen():
                yield from self._forever()
            now = self.env.now
            d = self._day(now)
            r = self.rate(w, d)
            seg = (d + 1) * self.H - now
            if r <= 0:
                yield self.env.timeout(max(seg, EPS))
                job.ledger["wait_unavailable"] += max(seg, EPS)
                continue
            dt = rem / r
            if dt >= seg - EPS:
                dt = seg
            prod = dt * r
            w.busy[(d, act)] += prod
            w.overhead[d] += dt - prod
            job.touch[act] += prod
            job.overhead += dt - prod
            yield self.env.timeout(dt)
            rem -= prod

    def _consume_clock(self, w: Worker, clock_hours: float, job: Job):
        """Unproductive clock time inside the shift (interruptions)."""
        rem = clock_hours
        while rem > 1e-9:
            if self._frozen():
                yield from self._forever()
            now = self.env.now
            d = self._day(now)
            r = self.rate(w, d)
            seg = (d + 1) * self.H - now
            if r <= 0:
                yield self.env.timeout(max(seg, EPS))
                job.ledger["wait_unavailable"] += max(seg, EPS)
                continue
            dt = min(rem, seg)
            w.idle[(d, "interruption")] += dt * r
            job.overhead += dt
            yield self.env.timeout(dt)
            rem -= dt

    # -------------------------------------------------------------------- jobs
    def _make_job(self, pod: int, arrival: float, ready_at_start: bool = False) -> Job:
        cfg = self.cfg
        seq = len(self.jobs)
        u = np.random.default_rng(np.random.SeedSequence([self.seed, 17, seq])).random(64)
        # complexity
        cum, cx = 0.0, COMPLEXITIES[-1]
        mix = cfg.mix_for(pod)
        for c in COMPLEXITIES:
            cum += mix[c]
            if u[0] < cum - 1e-12:
                cx = c
                break
        est = cfg.complexity.standard_hours[cx]
        urgent = bool(u[1] < cfg.demand.urgent_share)
        sla = cfg.demand.sla_days * self.H * (cfg.demand.urgent_sla_factor if urgent else 1.0)
        q = cfg.quality
        p_asset = cfg.effective_asset_delay_probability
        asset = -q.asset_delay_hours * math.log(1 - min(u[4], 1 - 1e-12)) if u[3] < p_asset else 0.0
        n_loops = q.max_rework_loops + 1
        rev_u, rev_h, rev_resp, rw_h = [], [], [], []
        for k in range(n_loops):
            b = 6 + 4 * k
            rev_u.append(float(u[b]))
            rev_h.append(_lognormal(q.review_hours, q.review_cv, u[b + 1]))
            rev_resp.append(_lognormal(q.review_response_delay_hours, 0.5, u[b + 2]) if q.review_response_delay_hours > 0 else 0.0)
            rw_h.append(_lognormal(q.rework_hours_fraction * est, q.rework_cv, u[b + 3]))
        job = Job(
            job_id=f"AD-{seq+1:05d}", seq=seq, pod=pod, complexity=cx, est_hours=est, arrival_time=arrival,
            due_time=arrival + sla, urgent=urgent,
            actual_hours=_lognormal(est, cfg.complexity.editing_time_cv, u[2]),
            brief_hours=_lognormal(cfg.effective_brief_hours, q.brief_cv, u[5]),
            asset_delay=asset, review_hours=rev_h, review_u=rev_u, review_resp=rev_resp, rework_hours=rw_h,
        )
        self.jobs.append(job)
        self.pods[pod].arrived += 1
        job.planned_editor = choose_planned_editor(self, job)
        return job

    def _arrive(self, pod: int, t: float, initial: bool = False, ready: bool = False) -> None:
        job = self._make_job(pod, t, ready_at_start=initial)
        p = self.pods[pod]
        if initial:
            job.initial = True
            p.arrived -= 1
            p.initial += 1
        if ready:
            job.brief_start = job.brief_end = 0.0
            job.state = READY
            job.enter_wait("wait_ready_queue", t)
            p.ready.append(job)
            self._log("initial_ready_brief", job)
        else:
            job.state = BACKLOG
            job.enter_wait("wait_brief_queue", t)
            p.backlog.append(job)
            self._log("initial_backlog" if initial else "arrival", job)
        self.notify("staff", "edit")

    def _arrival_schedule(self) -> List[tuple]:
        cfg, H, D = self.cfg, self.H, self.cfg.org.days_per_week
        w = cfg.weekday_weights()
        shares = cfg.pod_shares
        cv = cfg.demand.demand_cv
        s2 = math.log(1 + cv * cv) if cv > 0 else 0.0
        z = np.random.default_rng(np.random.SeedSequence([self.seed, 5])).standard_normal((self.n_days, len(self.pods)))
        carry = [0.0] * len(self.pods)
        out = []
        for d in range(self.n_days):
            for p in range(len(self.pods)):
                factor = math.exp(-s2 / 2 + math.sqrt(s2) * z[d, p]) if cv > 0 else 1.0
                carry[p] += cfg.demand.weekly_target_ads * w[d % D] * shares[p] * factor
                n = int(math.floor(carry[p] + 1e-9))
                carry[p] -= n
                if n:
                    jit = np.random.default_rng(np.random.SeedSequence([self.seed, 6, d, p])).random(n) if cv > 0 else np.full(n, 0.5)
                    out += [(d * H + (i + jit[i]) / n * H, p) for i in range(n)]
        out.sort()
        return out

    # --------------------------------------------------------------- processes
    def _arrivals(self):
        for t, p in self._arrival_schedule():
            if t > self.env.now:
                yield self.env.timeout(t - self.env.now)
            self._arrive(p, t)

    def _enqueue_review(self, job: Job) -> None:
        now = self.env.now
        job.leave_wait(now)
        job.review_ready_at = now
        job.enter_wait("wait_review_queue", now)
        self.pods[job.pod].review_q.append(job)
        self._log("review_queued", job)
        self.notify("staff")

    def _delayed_review(self, job: Job, delay: float):
        yield self.env.timeout(delay)
        self._enqueue_review(job)

    def _editor_loop(self, ed: Worker):
        while True:
            yield from self._wait_available(ed)
            job, is_rework = pick_edit_job(self, ed)
            if job is None:
                self._mark(ed, edit_idle_reason(self, ed))
                yield self._wakes["edit"]
                self._unmark(ed)
                continue
            yield from self._do_edit(ed, job, is_rework)

    def _do_edit(self, ed: Worker, job: Job, is_rework: bool):
        env, cfg = self.env, self.cfg
        now = env.now
        home = self.pods[job.pod]
        (home.rework if is_rework else home.ready).remove(job)
        job.leave_wait(now)
        work_pod = self.pods[ed.pod]          # the WIP limit applies to the POD whose editor does the work
        work_pod.editing += 1
        home.editing_owned += 1               # ownership/accounting stays with the home POD
        job.state = EDITING
        job.editor = ed.wid
        job.editor_history.append(ed.wid)
        if is_rework and len(job.editor_history) > 1 and job.editor_history[-1] != job.editor_history[-2]:
            job.handoffs += 1
            job.unnecessary_handoffs += 1
        if ed.pod != job.pod:
            job.transferred = True
            job.transfer_count += 1
            job.handoffs += 1
            job.unnecessary_handoffs += 1
            self._log("cross_pod_transfer", job, ed, editor_pod=ed.pod)
        if job.edit_start is None:
            job.edit_start = now
        act = "rework" if is_rework else "edit"
        hours = (job.rework_hours[job.rework_count - 1] if is_rework else job.actual_hours) / ed.speed
        ed.state, ed.current_job = act, job.job_id
        self._log("rework_start" if is_rework else "edit_start", job, ed)
        self.notify("staff")  # a ready slot opened -> strategist may brief again
        # interruptions (ad-hoc work), default rate 0
        q = cfg.quality
        if q.interruption_rate_per_hour > 0:
            n = int(self.rng_dyn.poisson(q.interruption_rate_per_hour * hours))
            if n:
                yield from self._consume_clock(ed, n * q.interruption_hours, job)
        # unavailable assets (first pass only)
        if not is_rework and job.asset_delay > 0:
            job.enter_wait("wait_asset", env.now)
            self._mark(ed, "asset_wait", "asset")
            yield env.timeout(job.asset_delay)
            job.leave_wait(env.now)
            self._unmark(ed)
        yield from self._work(ed, hours, job, act)
        now = env.now
        job.last_edit_end = now
        if not is_rework:
            job.edit_end = now
            job.actual_editing_hours = job.touch["edit"]
            job.edit_elapsed_first = now - job.edit_start
        self._log("edit_end" if not is_rework else "rework_end", job, ed)
        # hand-off to review is gated by the review WIP limit (kanban: finished work blocks the editor)
        lim = cfg.lean.review_wip_limit
        if cfg.lean.enforce_wip_limits and home.review_wip >= lim:
            job.state = BLOCKED
            job.enter_wait("wait_blocked_review", now)
            self._mark(ed, "blocked_review_wip", "blocked")
            while home.review_wip >= lim:
                yield self._wakes["slot"]
            job.leave_wait(env.now)
            self._unmark(ed)
        work_pod.editing -= 1
        home.editing_owned -= 1
        home.review_wip += 1
        job.handoffs += 1
        job.state = REVIEW
        ed.state, ed.current_job = "idle", None
        ed.jobs_completed += 0
        delay = job.review_resp[min(job.reviews_done, len(job.review_resp) - 1)]
        if delay > 0:
            job.enter_wait("wait_review_external", env.now)
            env.process(self._delayed_review(job, delay))
        else:
            self._enqueue_review(job)
        self.notify("edit")  # editing slot freed

    def _strategist_loop(self, st: Worker):
        mode = self.cfg.quality.review_mode
        review_first = self.cfg.quality.review_priority_over_briefing
        while True:
            yield from self._wait_available(st)
            order = (("review", "brief") if review_first else ("brief", "review")) if mode == "strategist" else ("brief",)
            job, kind = None, None
            for kind in order:
                job = pick_review_job(self, st, self.env.now) if kind == "review" else pick_brief_job(self, st)
                if job is not None:
                    break
            if job is None:
                self._mark(st, brief_idle_reason(self, st))
                yield self._wakes["staff"]
                self._unmark(st)
                continue
            if kind == "review":
                yield from self._do_review(st, job)
            else:
                yield from self._do_brief(st, job)

    def _do_brief(self, st: Worker, job: Job):
        env = self.env
        pod = self.pods[job.pod]
        pod.backlog.remove(job)
        job.leave_wait(env.now)
        pod.briefing += 1
        job.state = BRIEFING
        job.strategist = st.wid
        job.brief_start = env.now
        st.state, st.current_job = "brief", job.job_id
        self._log("brief_start", job, st)
        yield from self._work(st, job.brief_hours, job, "brief")
        job.brief_end = env.now
        pod.briefing -= 1
        pod.ready.append(job)
        job.state = READY
        job.handoffs += 1
        job.enter_wait("wait_ready_queue", env.now)
        st.state, st.current_job = "idle", None
        self._log("brief_ready", job, st)
        self.notify("edit", "staff")

    def _reviewer_loop(self, rv: Worker):
        while True:
            yield from self._wait_available(rv)
            job = pick_review_job(self, rv, self.env.now)
            if job is None:
                self._mark(rv, "no_reviews")
                yield self._wakes["staff"]
                self._unmark(rv)
                continue
            yield from self._do_review(rv, job)

    def _do_review(self, rv: Worker, job: Job):
        env, cfg = self.env, self.cfg
        pod = self.pods[job.pod]
        q = cfg.quality
        pod.review_q.remove(job)
        job.leave_wait(env.now)
        pod.in_review += 1
        job.reviewer = rv.wid
        if job.review_start is None:
            job.review_start = env.now
        k = job.reviews_done
        act = "review" if k == 0 else "rework_review"
        rv.state, rv.current_job = "reviewing", job.job_id
        self._log("review_start", job, rv)
        yield from self._work(rv, job.review_hours[k] / q.review_capacity_multiplier, job, act)
        now = env.now
        job.review_end = now
        job.reviews_done += 1
        pod.in_review -= 1
        rv.state, rv.current_job = "idle", None
        if k == 0:
            job.first_review_end = now
        p_rej = (1 - cfg.first_pass_for(job.pod)) if k == 0 else q.rework_repeat_probability
        reject = job.review_u[k] < p_rej - 1e-12
        pod.review_wip -= 1
        if not reject:
            if k == 0:
                job.first_pass_approved = True
            self._approve(job, now)
        else:
            if k == 0:
                job.first_pass_approved = False
            if job.rework_count < q.max_rework_loops:
                job.rework_count += 1
                job.state = REWORK
                job.handoffs += 1
                job.unnecessary_handoffs += 1
                job.enter_wait("wait_rework_queue", now)
                pod.rework.append(job)
                self._log("rework_requested", job, rv)
            elif q.on_max_loops == "accept":
                job.forced_accept = True
                self._approve(job, now)
            else:
                job.status, job.state = "cancelled", CANCELLED
                pod.cancelled += 1
                if job.planned_editor:
                    self.outstanding_hours[job.planned_editor] -= job.est_hours
                self._log("cancelled", job, rv)
        self.notify("slot", "edit", "staff")

    def _approve(self, job: Job, now: float) -> None:
        job.approval_time = now
        job.status, job.state = "delivered", APPROVED
        self.pods[job.pod].accepted += 1
        if job.planned_editor:
            self.outstanding_hours[job.planned_editor] -= job.est_hours
        self._log("approved", job)

    # -------------------------------------------------------------- snapshots
    def _snapshotter(self):
        iv = self.cfg.snapshot_interval_hours
        while True:
            self._snapshot()
            yield self.env.timeout(iv)

    def _snapshot(self) -> None:
        t = min(self.env.now, self.T_end)
        d = self._day(t)
        for p in self.pods:
            self.snap_pod.append(dict(
                t=t, day=d, pod=p.idx, backlog=len(p.backlog), briefing=p.briefing, ready=len(p.ready),
                editing=p.editing, editing_owned=p.editing_owned, rework_q=len(p.rework), review_wip=p.review_wip,
                review_q=len(p.review_q), in_review=p.in_review, accepted=p.accepted, arrived=p.arrived + p.initial,
                wip=p.wip_released, open_total=p.open_total))
        for w in self.all_workers:
            st = "absent" if self.rate(w, d) <= 0 else w.state
            self.snap_worker.append(dict(t=t, day=d, worker=w.wid, role=w.role, pod=w.pod, state=st, job_id=w.current_job))

    # -------------------------------------------------------------------- run
    def run(self) -> SimulationResult:
        cfg, env = self.cfg, self.env
        shares = cfg.pod_shares
        for p, n in enumerate(_largest_remainder(cfg.demand.initial_backlog_ads, shares)):
            for _ in range(n):
                self._arrive(p, 0.0, initial=True)
        for p in range(len(self.pods)):
            for _ in range(cfg.demand.initial_ready_briefs_per_pod):
                self._arrive(p, 0.0, initial=True, ready=True)
        for w in self.editors:
            env.process(self._editor_loop(w))
        for w in self.strategists:
            env.process(self._strategist_loop(w))
        for w in self.reviewers:
            env.process(self._reviewer_loop(w))
        env.process(self._arrivals())
        env.process(self._snapshotter())
        env.run(until=self.T_end + 1e-6)
        return self._finalize()

    def _finalize(self) -> SimulationResult:
        T = self.T_end
        for wid, (t0, cause) in list(self._marks.items()):
            w = next(x for x in self.all_workers if x.wid == wid)
            self._account_idle(w, t0, T, cause)
        self._marks.clear()
        for j in self.jobs:
            if j.status == "open" and j.wait_cat:
                j.ledger[j.wait_cat] += max(0.0, T - j.wait_since)
                j.wait_cat = None
        if not self.snap_pod or self.snap_pod[-1]["t"] < T - 1e-6:
            self._snapshot()
        return SimulationResult(
            cfg=self.cfg, seed=self.seed, t_end=T,
            jobs=self._jobs_frame(), events=pd.DataFrame(self.events),
            pod_snapshots=pd.DataFrame(self.snap_pod), worker_snapshots=pd.DataFrame(self.snap_worker),
            worker_day=self._worker_day_frame(), workers=self._workers_frame(),
            conservation=self._conservation(), job_objects=self.jobs)

    # ------------------------------------------------------------------ frames
    def _jobs_frame(self) -> pd.DataFrame:
        T = self.T_end
        rows = []
        for j in self.jobs:
            lead = j.lead_time
            rows.append(dict(
                job_id=j.job_id, seq=j.seq, pod=j.pod, strategist=j.strategist, editor=j.editor,
                first_editor=(j.editor_history[0] if j.editor_history else None),
                complexity=j.complexity, est_hours=j.est_hours, actual_editing_hours=j.actual_editing_hours,
                arrival=j.arrival_time, brief_start=j.brief_start, brief_end=j.brief_end,
                edit_start=j.edit_start, edit_end=j.edit_end, review_start=j.review_start, review_end=j.review_end,
                approval=j.approval_time, rework_count=j.rework_count, status=j.status, state=j.state,
                lead_time=lead, age_at_end=(T - j.arrival_time) if j.status == "open" else None,
                total_wait=j.total_wait, total_touch=j.total_touch, overhead=j.overhead,
                due=j.due_time, urgent=j.urgent,
                on_time=(None if j.approval_time is None else j.approval_time <= j.due_time + 1e-9),
                first_pass_approved=j.first_pass_approved, forced_accept=j.forced_accept,
                transferred=j.transferred, transfer_count=j.transfer_count, handoffs=j.handoffs,
                unnecessary_handoffs=j.unnecessary_handoffs, edit_elapsed_first=j.edit_elapsed_first,
                reviews_done=j.reviews_done, initial=j.initial,
                **{f"L_{c}": j.ledger.get(c, 0.0) for c in WAIT_CATEGORIES},
                **{f"T_{c}": j.touch.get(c, 0.0) for c in TOUCH_CATEGORIES}))
        return pd.DataFrame(rows)

    def _workers_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dict(worker=w.wid, role=w.role, pod=w.pod, speed=w.speed, can_complex=w.can_complex,
                                  base_rate=w.base_rate) for w in self.all_workers])

    def _worker_day_frame(self) -> pd.DataFrame:
        H, n = self.H, self.n_days
        acts = ("brief", "edit", "rework", "review", "rework_review")
        rows = []
        for w in self.all_workers:
            cap = productive_capacity(w, n, H)
            for d in range(n):
                busy = {a: w.busy.get((d, a), 0.0) for a in acts}
                b_tot = sum(busy.values())
                ot = max(0.0, b_tot - cap[d]) if w.role == "editor" else 0.0
                rows.append(dict(worker=w.wid, role=w.role, pod=w.pod, day=d, week=d // self.cfg.org.days_per_week,
                                 weekday=d % self.cfg.org.days_per_week, capacity_h=float(cap[d]),
                                 absent=bool(cap[d] == 0), busy_h=b_tot, ot_prod_h=ot,
                                 ot_clock_h=ot / w.base_rate if w.base_rate > 0 else 0.0,
                                 overhead_h=w.overhead.get(d, 0.0), **{f"busy_{a}": busy[a] for a in acts}))
        df = pd.DataFrame(rows)
        # idle causes as columns
        idle_rows = defaultdict(float)
        for w in self.all_workers:
            for (d, c), v in w.idle.items():
                if d < n:
                    idle_rows[(w.wid, d, c)] += v
        if idle_rows:
            idf = pd.DataFrame([dict(worker=k[0], day=k[1], cause=k[2], h=v) for k, v in idle_rows.items()])
            piv = idf.pivot_table(index=["worker", "day"], columns="cause", values="h", aggfunc="sum", fill_value=0.0)
            piv.columns = [f"idle_{c}" for c in piv.columns]
            df = df.merge(piv.reset_index(), on=["worker", "day"], how="left")
        idle_cols = [c for c in df.columns if c.startswith("idle_")]
        df[idle_cols] = df[idle_cols].fillna(0.0)
        df["idle_total_h"] = df[idle_cols].sum(axis=1) if idle_cols else 0.0
        return df

    def _conservation(self) -> Dict[str, float]:
        initial = sum(p.initial for p in self.pods)
        arrivals = sum(p.arrived for p in self.pods)
        completed = sum(p.accepted for p in self.pods)
        cancelled = sum(p.cancelled for p in self.pods)
        ending_queues = sum(p.open_total for p in self.pods)
        ending_jobs = sum(1 for j in self.jobs if j.status == "open")
        return dict(initial_wip=initial, arrivals=arrivals, completed=completed, cancelled=cancelled,
                    ending_wip=ending_queues, ending_wip_from_jobs=ending_jobs,
                    balance=initial + arrivals - completed - ending_queues - cancelled)


def run_simulation(cfg: SimConfig, seed: Optional[int] = None) -> SimulationResult:
    """Run one replication."""
    return ProductionSimulation(cfg, seed).run()
