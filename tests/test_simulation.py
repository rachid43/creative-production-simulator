import numpy as np
import pandas as pd
import pytest

from config.defaults import baseline_config
from config.schema import SimConfig
from simulation.engine import run_simulation
from tests.conftest import det_cfg


# ---------------------------------------------------------------- deterministic manual case
def test_deterministic_manual_timeline():
    r = run_simulation(det_cfg())
    j = r.jobs.sort_values("seq").reset_index(drop=True)
    assert list(j["status"]) == ["delivered"] * 3
    assert np.allclose(j["approval"], [3.75, 6.75, 9.75])
    assert np.allclose(j["brief_end"], [0.5, 1.0, 1.5])
    assert np.allclose(j["edit_start"], [0.5, 3.5, 6.5])
    assert np.allclose(j["edit_end"], [3.5, 6.5, 9.5])
    # lead time = approval - arrival (all arrive at t=0)
    assert np.allclose(j["lead_time"], [3.75, 6.75, 9.75])
    assert np.allclose(j["total_touch"], 3.75)
    # job 2: waited 0.5 h for the strategist and 2.5 h in the ready queue
    assert j.loc[1, "L_wait_brief_queue"] == pytest.approx(0.5)
    assert j.loc[1, "L_wait_ready_queue"] == pytest.approx(2.5)
    wd = r.worker_day
    assert wd[wd.worker == "P1-E1"]["busy_h"].sum() == pytest.approx(9.0)
    assert wd[wd.worker == "S1"]["busy_h"].sum() == pytest.approx(1.5 + 0.75)
    # editor utilisation = 9 / (5 days x 8 h) = 22.5 %
    e = wd[wd.role == "editor"]
    assert e.busy_h.sum() / e.capacity_h.sum() == pytest.approx(9 / 40)


def test_deterministic_rework_loop_consumes_capacity():
    cfg = det_cfg(demand=dict(initial_backlog_ads=1),
                  quality=dict(first_pass_approval_rate=0.0, rework_repeat_probability=0.0, rework_hours_fraction=0.5,
                               max_rework_loops=2))
    r = run_simulation(cfg)
    j = r.jobs.iloc[0]
    assert j["rework_count"] == 1 and not j["first_pass_approved"]
    # brief .5 + edit 3 + review .25 + rework 1.5 + review .25 = 5.5
    assert j["approval"] == pytest.approx(5.5)
    wd = r.worker_day
    assert wd[wd.worker == "P1-E1"]["busy_h"].sum() == pytest.approx(3.0 + 1.5)  # rework used real editor hours
    assert wd[wd.worker == "S1"]["busy_h"].sum() == pytest.approx(0.5 + 0.25 + 0.25)
    assert j["T_rework"] == pytest.approx(1.5) and j["T_rework_review"] == pytest.approx(0.25)


def test_max_rework_loops_forced_accept_and_cancel():
    q = dict(first_pass_approval_rate=0.0, rework_repeat_probability=1.0, max_rework_loops=2)
    r = run_simulation(det_cfg(demand=dict(initial_backlog_ads=1), quality=q))
    j = r.jobs.iloc[0]
    assert j["rework_count"] == 2 and j["forced_accept"] and j["status"] == "delivered"
    r2 = run_simulation(det_cfg(demand=dict(initial_backlog_ads=1), quality=dict(q, on_max_loops="cancel")))
    c = r2.conservation
    assert r2.jobs.iloc[0]["status"] == "cancelled" and c["cancelled"] == 1 and c["balance"] == 0


# ---------------------------------------------------------------- conservation
@pytest.mark.parametrize("over", [
    {},
    dict(demand=dict(initial_backlog_ads=40, initial_ready_briefs_per_pod=4)),
    dict(demand=dict(weekly_target_ads=260.0)),
    dict(org=dict(overtime_hours_per_day=1.5, planned_absenteeism=0.1)),
    dict(lean=dict(cross_pod_balancing=True, cross_pod_review_sharing=True), demand=dict(pod_share=[3, 1, 1, 1, 1, 1])),
    dict(quality=dict(first_pass_approval_rate=0.3, on_max_loops="cancel", max_rework_loops=1)),
    dict(quality=dict(review_mode="dedicated", dedicated_reviewers=1, review_response_delay_hours=3.0)),
])
def test_conservation_initial_plus_arrivals(over):
    cfg = baseline_config(horizon_weeks=2, warmup_weeks=0, **over)
    r = run_simulation(cfg, seed=7)
    c = r.conservation
    assert c["balance"] == 0
    assert c["initial_wip"] + c["arrivals"] == c["completed"] + c["ending_wip"] + c["cancelled"]
    # counters agree with the independent job list
    assert c["ending_wip"] == c["ending_wip_from_jobs"]
    assert len(r.jobs) == c["initial_wip"] + c["arrivals"]
    assert (r.jobs.status == "delivered").sum() == c["completed"]


def test_unfinished_work_is_preserved_in_queues(baseline_result):
    open_jobs = baseline_result.jobs[baseline_result.jobs.status == "open"]
    last = baseline_result.pod_snapshots[baseline_result.pod_snapshots.t == baseline_result.pod_snapshots.t.max()]
    assert len(open_jobs) == last.open_total.sum() > 0


# ---------------------------------------------------------------- precedence and counting rules
def test_precedence_and_approval_required(baseline_result):
    j = baseline_result.jobs
    d = j[j.status == "delivered"]
    for a, b in [("arrival", "brief_start"), ("brief_start", "brief_end"), ("brief_end", "edit_start"),
                 ("edit_start", "edit_end"), ("edit_end", "review_start"), ("review_start", "review_end"),
                 ("review_end", "approval")]:
        assert (d[a] <= d[b] + 1e-9).all(), (a, b)
    # open or cancelled ads never have an approval timestamp
    assert j[j.status != "delivered"].approval.isna().all()
    # an ad edited but not yet reviewed is WIP, not output
    pending = j[j.edit_end.notna() & j.approval.isna()]
    assert (pending.status == "open").all()


# ---------------------------------------------------------------- resource constraints
def test_no_resource_exceeds_calendar_capacity():
    cfg = baseline_config(horizon_weeks=3, warmup_weeks=0, org=dict(planned_absenteeism=0.1))
    r = run_simulation(cfg, seed=3)
    wd = r.worker_day
    # busy hours never exceed capacity (no overtime configured), absent days have zero work
    assert (wd.busy_h <= wd.capacity_h + 1e-6).all()
    assert (wd[wd.absent].busy_h == 0).all() and wd.absent.any()
    # busy + idle = capacity, i.e. time is fully accounted for
    assert np.allclose((wd.busy_h + wd.idle_total_h)[~wd.absent], wd.capacity_h[~wd.absent], atol=1e-6)


def test_overtime_adds_capacity_only_with_queue_and_is_bounded():
    cfg = baseline_config(horizon_weeks=2, warmup_weeks=0, demand=dict(weekly_target_ads=260.0),
                          org=dict(overtime_hours_per_day=2.0))
    r = run_simulation(cfg, seed=3)
    e = r.worker_day[r.worker_day.role == "editor"]
    assert e.ot_prod_h.sum() > 0
    a, H = cfg.org.productive_availability, cfg.org.hours_per_day
    assert (e.busy_h <= H + 1e-6).all()  # productive rate can never exceed 100 % of the shift
    r0 = run_simulation(baseline_config(horizon_weeks=2, warmup_weeks=0, demand=dict(weekly_target_ads=260.0)), seed=3)
    assert (r.jobs.status == "delivered").sum() > (r0.jobs.status == "delivered").sum()


def test_editor_never_works_two_ads_at_once():
    r = run_simulation(baseline_config(horizon_weeks=2, warmup_weeks=0, quality=dict(first_pass_approval_rate=0.6)), seed=5)
    ev = r.events
    for ed, g in ev[ev.event.isin(["edit_start", "edit_end", "rework_start", "rework_end"])].groupby("resource"):
        g = g.sort_values(["t"], kind="stable")
        open_ = 0
        for e in g.event:
            open_ += 1 if e.endswith("start") else -1
            assert 0 <= open_ <= 1, f"{ed} overlapped"


def test_pull_editing_requires_ready_brief():
    r = run_simulation(baseline_config(horizon_weeks=2, warmup_weeks=0), seed=5)
    j = r.jobs
    s = j[j.edit_start.notna()]
    assert (s.edit_start >= s.brief_end - 1e-9).all()


# ---------------------------------------------------------------- WIP limits
def test_wip_limits_respected_in_every_snapshot():
    cfg = baseline_config(horizon_weeks=2, warmup_weeks=0, demand=dict(weekly_target_ads=240.0, initial_backlog_ads=60),
                          snapshot_interval_hours=0.25)
    r = run_simulation(cfg, seed=11)
    ps = r.pod_snapshots
    assert (ps.editing <= cfg.lean.editing_wip_limit).all()
    assert (ps.review_wip <= cfg.lean.review_wip_limit).all()
    assert (ps.ready + ps.briefing <= cfg.lean.ready_brief_buffer).all()


def test_wip_limit_constrains_output_and_push_mode_inflates_wip():
    base = baseline_config(horizon_weeks=2, warmup_weeks=0, demand=dict(weekly_target_ads=240.0))
    tight = run_simulation(base.copy_with(lean=dict(editing_wip_limit=1)), seed=2)
    loose = run_simulation(base, seed=2)
    assert tight.jobs.status.eq("delivered").sum() < loose.jobs.status.eq("delivered").sum()  # 1 of 3 editors works per POD
    push = run_simulation(base.copy_with(lean=dict(enforce_wip_limits=False)), seed=2)
    assert push.pod_snapshots.groupby("t").wip.sum().mean() > loose.pod_snapshots.groupby("t").wip.sum().mean()


def test_review_wip_limit_blocks_editors():
    cfg = baseline_config(horizon_weeks=2, warmup_weeks=0, quality=dict(review_hours=1.5, review_mode="dedicated",
                                                                       dedicated_reviewers=1),
                          lean=dict(review_wip_limit=1))
    r = run_simulation(cfg, seed=4)
    assert r.jobs.L_wait_blocked_review.sum() > 0
    assert r.worker_day.get("idle_blocked_review_wip", pd.Series([0])).sum() > 0


# ---------------------------------------------------------------- starvation and review bottlenecks
def test_brief_starvation_creates_editor_idle_time():
    slow = baseline_config(horizon_weeks=2, warmup_weeks=0, quality=dict(brief_hours=3.0))
    fast = baseline_config(horizon_weeks=2, warmup_weeks=0)
    rs, rf = run_simulation(slow, seed=6), run_simulation(fast, seed=6)
    idle = lambda r: r.worker_day[r.worker_day.role == "editor"].get("idle_starved_brief_supply", pd.Series([0])).sum()
    assert idle(rs) > 3 * idle(rf)
    assert (rs.jobs.status == "delivered").sum() < (rf.jobs.status == "delivered").sum()


def test_review_bottleneck_grows_review_waiting_and_cuts_output():
    base = baseline_config(horizon_weeks=2, warmup_weeks=0)
    slow = base.copy_with(quality=dict(review_mode="dedicated", dedicated_reviewers=1, review_hours=0.5))
    ok = base.copy_with(quality=dict(review_mode="dedicated", dedicated_reviewers=6))
    rs, ro = run_simulation(slow, seed=8), run_simulation(ok, seed=8)
    assert rs.jobs.L_wait_review_queue.mean() > 5 * ro.jobs.L_wait_review_queue.mean()
    assert (rs.jobs.status == "delivered").sum() < (ro.jobs.status == "delivered").sum()


def test_review_capacity_not_double_counted_with_strategist():
    r = run_simulation(baseline_config(horizon_weeks=2, warmup_weeks=0), seed=2)
    wd = r.worker_day
    s = wd[wd.role == "strategist"]
    assert (s.busy_h <= s.capacity_h + 1e-6).all()  # briefing + reviewing share ONE capacity
    assert s.busy_brief.sum() > 0 and s.busy_review.sum() > 0


# ---------------------------------------------------------------- controlled-effect tests
def test_more_processing_time_means_longer_lead_and_lower_throughput():
    base = baseline_config(horizon_weeks=3, warmup_weeks=0)
    hard = base.copy_with(complexity=dict(standard_hours={"Simple": 2.5, "Medium": 5.0, "Complex": 8.0}))
    a, b = run_simulation(base, seed=9), run_simulation(hard, seed=9)
    da, db = a.jobs[a.jobs.status == "delivered"], b.jobs[b.jobs.status == "delivered"]
    assert db.lead_time.mean() > da.lead_time.mean() * 1.5
    assert len(db) < len(da)


def test_fewer_editors_reduces_throughput_and_raises_wip():
    base = baseline_config(horizon_weeks=3, warmup_weeks=0)
    short = base.copy_with(org=dict(editor_counts=[3, 3, 3, 2, 2, 2]))
    a, b = run_simulation(base, seed=9), run_simulation(short, seed=9)
    assert (b.jobs.status == "delivered").sum() < (a.jobs.status == "delivered").sum()
    assert b.conservation["ending_wip"] > a.conservation["ending_wip"]


def test_higher_demand_increases_backlog():
    base = baseline_config(horizon_weeks=3, warmup_weeks=0)
    a = run_simulation(base, seed=9)
    b = run_simulation(base.copy_with(demand=dict(weekly_target_ads=250.0)), seed=9)
    assert b.conservation["ending_wip"] > a.conservation["ending_wip"] + 30


def test_demand_cannot_create_capacity():
    """Throughput saturates at editing capacity however much demand is added."""
    base = baseline_config(horizon_weeks=3, warmup_weeks=0)
    outs = []
    for target in (260.0, 400.0, 800.0):
        r = run_simulation(base.copy_with(demand=dict(weekly_target_ads=target)), seed=9)
        outs.append((r.jobs.status == "delivered").sum())
    cap_weekly = 18 * 32 / 3.1 * 3
    assert max(outs) < cap_weekly * 1.02
    assert max(outs) - min(outs) < 0.10 * max(outs)


# ---------------------------------------------------------------- cross-POD ownership
def test_cross_pod_transfers_preserve_ownership_and_help():
    skew = dict(pod_share=[2.0, 2.0, 1.0, 1.0, 0.5, 0.5])
    base = baseline_config(horizon_weeks=3, warmup_weeks=0, demand=skew)
    off = run_simulation(base, seed=5)
    on = run_simulation(base.copy_with(lean=dict(cross_pod_balancing=True)), seed=5)
    t = on.jobs[on.jobs.transferred]
    assert len(t) > 0
    assert (t.editor.str.extract(r"P(\d+)-")[0].astype(int) - 1 != t.pod).any()
    # ownership never changes: per-POD arrivals identical with and without balancing
    assert (on.jobs.groupby("pod").size() == off.jobs.groupby("pod").size()).all()
    assert on.conservation["balance"] == 0
    assert (on.jobs.status == "delivered").sum() >= (off.jobs.status == "delivered").sum()
    assert off.jobs.transferred.sum() == 0


# ---------------------------------------------------------------- reproducibility
def test_same_seed_reproducible_and_different_seed_differs():
    cfg = baseline_config(horizon_weeks=2, warmup_weeks=0)
    a, b, c = run_simulation(cfg, 5), run_simulation(cfg, 5), run_simulation(cfg, 6)
    pd.testing.assert_frame_equal(a.jobs, b.jobs)
    assert not a.jobs.equals(c.jobs)


def test_common_random_numbers_same_job_inputs_across_scenarios():
    cfg = baseline_config(horizon_weeks=2, warmup_weeks=0)
    a = run_simulation(cfg, 5)
    b = run_simulation(cfg.copy_with(quality=dict(first_pass_approval_rate=0.6)), 5)
    assert (a.jobs.complexity == b.jobs.complexity).all()
    assert np.allclose(a.jobs.arrival, b.jobs.arrival)
    # lower first-pass approval can only add rework for the same ads
    assert b.jobs.rework_count.sum() >= a.jobs.rework_count.sum()


def test_ledger_closure_and_capacity_accounting(baseline_result):
    d = baseline_result.jobs[baseline_result.jobs.status == "delivered"]
    parts = d.total_wait + d.total_touch + d.overhead
    assert np.allclose(parts, d.lead_time, atol=1e-6)  # every hour of lead time is classified exactly once
