import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from analytics import kpis, lean_metrics as lm
from analytics.bottlenecks import detect_bottleneck
from config.defaults import PLAYBOOK, baseline_config
from config.schema import SimConfig
from simulation.engine import run_simulation
from simulation.scheduling import plan_week
from tests.conftest import det_cfg


# ----------------------------------------------------------------- baseline capacity and takt
def test_baseline_organisation_matches_playbook():
    cfg = baseline_config()
    assert cfg.org.n_pods == PLAYBOOK["n_pods"] == 6
    assert cfg.org.total_editors == 18 and cfg.org.strategists_per_pod == 1
    assert cfg.org.weekly_hours == 40 and cfg.demand.weekly_target_ads == 180
    assert cfg.complexity.standard_hours == {"Simple": 1.5, "Medium": 3.0, "Complex": 5.0}
    assert cfg.complexity.mean_standard_hours == pytest.approx(2.95)
    assert (cfg.lean.ready_brief_buffer, cfg.lean.editing_wip_limit, cfg.lean.review_wip_limit) == (6, 3, 3)


def test_takt_calculations_match_playbook():
    t = lm.takt_summary(baseline_config())
    assert t["daily_target"] == 36 and t["ads_per_editor_week"] == 10
    assert t["gross_takt_h_per_ad_per_editor"] == pytest.approx(4.0)
    assert t["effective_takt_h_per_ad_per_editor"] == pytest.approx(3.2)
    assert t["org_takt_minutes_between_ads"] == pytest.approx(40 * 60 / 180)
    # required hours per ad (standard + expected rework) vs available 32 h/editor/week
    assert t["required_hours_per_ad"] == pytest.approx(2.95 + t["expected_rework_hours"])
    assert t["available_productive_h_per_editor_week"] == pytest.approx(32.0)
    assert 0.95 < t["planned_load_ratio"] < 1.0           # feasible on paper but with < 5 % headroom
    # takt scales with demand
    t220 = lm.takt_summary(baseline_config(), weekly_target=220)
    assert t220["gross_takt_h_per_ad_per_editor"] == pytest.approx(720 / 220)


def test_expected_rework_formula():
    cfg = det_cfg(quality=dict(first_pass_approval_rate=0.5, rework_repeat_probability=0.0, rework_hours_fraction=0.5,
                               max_rework_loops=3))
    # 50 % need 0.5 x 3 h = 1.5 h of rework; no second loops
    assert lm.expected_rework_hours_per_ad(cfg) == pytest.approx(0.5 * 1.5)
    assert lm.expected_rework_hours_per_ad(det_cfg()) == 0.0


def test_workload_balance_index():
    assert lm.workload_balance_index([0.9, 0.9, 0.9]) == pytest.approx(1.0)
    assert lm.workload_balance_index([0.5, 1.0]) == pytest.approx(1 - 0.25 / 0.75)
    assert lm.workload_balance_index([0.0, 2.0, 0.0, 2.0]) == pytest.approx(0.0)


# ----------------------------------------------------------------- KPI correctness on the manual case
def test_kpis_match_manual_calculation():
    r = run_simulation(det_cfg())
    k = kpis.compute_kpis(r)
    assert k["accepted_total"] == 3
    assert k["lead_time_h"] == pytest.approx((3.75 + 6.75 + 9.75) / 3)
    assert k["cycle_time_edit_h"] == pytest.approx(3.0)
    assert k["first_pass_yield_pct"] == pytest.approx(100.0)
    assert k["editor_utilization_pct"] == pytest.approx(22.5)
    # flow efficiency: VA = briefing 0.5 + editing 3.0 per ad; review is necessary non-value-adding
    assert k["flow_efficiency_pct"] == pytest.approx(10.5 / 20.25 * 100)
    cfg2 = det_cfg()
    cfg2 = cfg2.copy_with(classification=dict(touch={**cfg2.classification.touch, "review": "VA"}))
    assert lm.flow_efficiency(r.jobs, cfg2) == pytest.approx(11.25 / 20.25 * 100)
    assert k["wip_end"] == 0 and k["backlog_end"] == 0


def test_lead_time_decomposition_sums_to_lead_time(baseline_result):
    dv = lm.delivered_in_window(baseline_result)
    dec = lm.lead_time_decomposition(baseline_result.jobs[baseline_result.jobs.status == "delivered"])
    assert dec.hours.sum() == pytest.approx(dv.lead_time.mean() if len(dv) == (baseline_result.jobs.status == "delivered").sum()
                                            else baseline_result.jobs[baseline_result.jobs.status == "delivered"].lead_time.mean())


def test_first_pass_yield_matches_job_table(baseline_result):
    j = baseline_result.jobs
    fr = j[j.first_pass_approved.notna() & j.review_start.notna()]
    k = kpis.compute_kpis(baseline_result)
    assert k["first_pass_yield_pct"] == pytest.approx(fr.first_pass_approved.astype(bool).mean() * 100)


def test_waste_pareto_sorted_and_complete(baseline_result):
    coh = lm.cohort(baseline_result)
    p = lm.waste_pareto(coh, baseline_result.cfg)
    assert p.hours.is_monotonic_decreasing and p.cum_pct.iloc[-1] == pytest.approx(100.0)
    # waste is not invented: with no rework, no assets, no feedback delay the corresponding categories are zero
    r = run_simulation(det_cfg())
    pr = lm.waste_pareto(lm.cohort(r), r.cfg).set_index("cause").hours
    assert pr["Rework processing"] == 0 and pr["Waiting for assets"] == 0 and pr["Waiting for external feedback"] == 0
    assert pr["Waiting for editor capacity"] == pytest.approx(2.5 + 5.0)  # jobs 2 and 3 queue for the single editor


def test_rework_hours_by_pod_reflect_quality():
    good = run_simulation(baseline_config(horizon_weeks=2, warmup_weeks=0, quality=dict(first_pass_approval_rate=0.95)), 3)
    bad = run_simulation(baseline_config(horizon_weeks=2, warmup_weeks=0, quality=dict(first_pass_approval_rate=0.5)), 3)
    assert bad.jobs.T_rework.sum() > 3 * good.jobs.T_rework.sum()


# ----------------------------------------------------------------- Mura / Muri / bottlenecks
def test_mura_higher_for_skewed_demand():
    even = run_simulation(baseline_config(horizon_weeks=3, warmup_weeks=0), 4)
    skew = run_simulation(baseline_config(horizon_weeks=3, warmup_weeks=0, demand=dict(pod_share=[3, 2, 1, 1, 0.5, 0.5])), 4)
    me, ms = lm.mura_analysis(even), lm.mura_analysis(skew)
    comp = lambda m, key: m["components"].set_index("component").cv[key]
    k = "POD workload / capacity (CV across PODs)"
    assert comp(ms, k) > 5 * comp(me, k)
    assert ms["index"] > me["index"]
    assert ms["workload_balance_index"] < me["workload_balance_index"]
    assert ms["imbalance"].concurrent.any()   # idle POD beside an overloaded POD is detected


def test_muri_increases_with_load():
    lo = run_simulation(baseline_config(horizon_weeks=3, warmup_weeks=0, demand=dict(weekly_target_ads=110.0)), 4)
    hi = run_simulation(baseline_config(horizon_weeks=3, warmup_weeks=0, demand=dict(weekly_target_ads=230.0)), 4)
    assert lm.muri_analysis(hi)["index"] > lm.muri_analysis(lo)["index"]
    assert lm.muri_analysis(hi)["capacity_shortfall_h"] > lm.muri_analysis(lo)["capacity_shortfall_h"]


def test_utilisation_band_thresholds_configurable():
    cfg = baseline_config()
    assert [lm.util_band(u, cfg) for u in (0.5, 0.75, 0.9, 0.97)] == ["Spare capacity", "Normal", "High load", "Critical load"]
    cfg2 = cfg.copy_with(thresholds=dict(util_spare_below=0.8, util_normal_upto=0.9, util_high_upto=0.98))
    assert lm.util_band(0.75, cfg2) == "Spare capacity"


def test_bottleneck_identifies_each_constraint():
    base = baseline_config(horizon_weeks=3, warmup_weeks=0)
    # editing capacity constraint
    r = run_simulation(base.copy_with(demand=dict(weekly_target_ads=240.0)), 4)
    assert detect_bottleneck(r)["primary_key"] == "editing"
    # brief supply constraint
    r = run_simulation(base.copy_with(quality=dict(brief_hours=3.0)), 4)
    assert detect_bottleneck(r)["primary_key"] == "brief"
    # review constraint
    r = run_simulation(base.copy_with(quality=dict(review_mode="dedicated", dedicated_reviewers=1, review_hours=0.5)), 4)
    assert detect_bottleneck(r)["primary_key"] == "review"
    # healthy system -> no binding constraint
    r = run_simulation(base.copy_with(demand=dict(weekly_target_ads=120.0)), 4)
    assert detect_bottleneck(r)["no_constraint"]


def test_most_utilised_resource_is_not_assumed_bottleneck():
    """Strategists near 100 % but editors starved by brief supply: the explanation must name briefing, not editing."""
    r = run_simulation(baseline_config(horizon_weeks=3, warmup_weeks=0, quality=dict(brief_hours=2.5)), 4)
    d = detect_bottleneck(r)
    assert d["primary_key"] == "brief" and d["editor_utilization"] < 0.9


# ----------------------------------------------------------------- Little's law
def test_littles_law_holds_in_stable_regime():
    cfg = baseline_config(horizon_weeks=10, warmup_weeks=2, demand=dict(weekly_target_ads=150.0))
    ll = lm.littles_law(run_simulation(cfg, 12))
    assert ll["stable"]
    assert ll["ratio"] == pytest.approx(1.0, abs=0.08)


def test_littles_law_flags_short_window_and_backlog():
    cfg = baseline_config(horizon_weeks=1, warmup_weeks=0, demand=dict(initial_backlog_ads=100, weekly_target_ads=200.0))
    ll = lm.littles_law(run_simulation(cfg, 12))
    assert not ll["stable"] and len(ll["caveats"]) >= 2


# ----------------------------------------------------------------- Heijunka planner
def test_heijunka_methods_place_all_work_and_capacity_aware_balances_best():
    cfg = baseline_config(org=dict(editor_counts=[3, 3, 3, 3, 2, 2]))
    plans = {m: plan_week(cfg, m, seed=3, allow_cross_pod=True) for m in ("equal_count", "equal_hours", "capacity_aware")}
    for m, p in plans.items():
        assert p["summary"]["planned_ads"] == 180 and p["jobs"].editor.notna().all()
        assert p["editors"].planned_ads.sum() == 180
        assert p["editors"].load_h.sum() == pytest.approx(p["jobs"].std_h.sum())
    wbi = {m: p["summary"]["balance_index"] for m, p in plans.items()}
    assert wbi["capacity_aware"] >= wbi["equal_hours"] >= wbi["equal_count"] - 0.02
    assert plans["capacity_aware"]["summary"]["max_load_ratio"] <= plans["equal_count"]["summary"]["max_load_ratio"]


def test_heijunka_respects_complex_skill():
    cfg = baseline_config(org=dict(complex_capable_per_pod=1))
    p = plan_week(cfg, "capacity_aware", seed=1)
    ed = p["editors"].set_index("editor")
    assert (ed.loc[~ed.can_complex, "complex"] == 0).all()


# ----------------------------------------------------------------- validation
def test_input_validation():
    with pytest.raises(ValidationError):
        baseline_config(complexity=dict(mix={"Simple": 0.5, "Medium": 0.5, "Complex": 0.5}))
    with pytest.raises(ValidationError):
        baseline_config(quality=dict(first_pass_approval_rate=0.8, rework_probability=0.5))   # inconsistent
    assert baseline_config(quality=dict(first_pass_approval_rate=0.8, rework_probability=0.2)).quality.rework_probability == 0.2
    with pytest.raises(ValidationError):
        baseline_config(horizon_weeks=2, warmup_weeks=2)
    with pytest.raises(ValidationError):
        baseline_config(org=dict(n_pods=3), demand=dict(pod_share=[1, 1]))
    with pytest.raises(ValidationError):
        baseline_config(thresholds=dict(util_spare_below=0.9, util_normal_upto=0.8))
    with pytest.raises(ValidationError):
        baseline_config(classification=dict(touch={"edit": "MAYBE"}))


def test_pod_status_thresholds():
    cfg = baseline_config()
    row = lambda att, util, q: pd.Series(dict(attainment=att, utilization=util, queue=q, editors=3))
    assert kpis.pod_status(row(1.0, 0.8, 0), cfg) == "green"
    assert kpis.pod_status(row(0.92, 0.8, 0), cfg) == "amber"
    assert kpis.pod_status(row(1.0, 0.9, 0), cfg) == "amber"
    assert kpis.pod_status(row(0.8, 0.8, 0), cfg) == "red"
    assert kpis.pod_status(row(1.0, 0.97, 0), cfg) == "red"
    assert kpis.pod_status(row(1.0, 0.8, 12), cfg) == "red"
