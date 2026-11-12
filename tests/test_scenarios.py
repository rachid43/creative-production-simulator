import numpy as np
import pandas as pd
import pytest

from config.defaults import baseline_config
from simulation.scenarios import (
    Scenario, comparison_table, mean_ci, preset_scenarios, run_replications, run_scenarios, t_quantile_95,
)


def test_six_presets_with_expected_changes():
    base = baseline_config()
    sc = {s.key: s.build(base) for s in preset_scenarios(base)}
    assert list(sc) == list("ABCDEF")
    assert sc["A"].model_dump() == base.model_dump()
    assert sc["B"].demand.weekly_target_ads == 220
    assert sc["C"].complexity.mix["Complex"] > base.complexity.mix["Complex"]
    assert sc["D"].quality.first_pass_approval_rate < base.quality.first_pass_approval_rate
    assert sc["D"].quality.review_response_delay_hours > 0
    assert sc["E"].org.total_editors == 16
    assert sc["F"].lean.cross_pod_balancing and sc["F"].lean.brief_standardization
    assert sc["F"].quality.first_pass_approval_rate > base.quality.first_pass_approval_rate


def test_custom_scenario_and_validation():
    s = Scenario("X", "Custom", "x", {"lean": {"editing_wip_limit": 2}})
    assert s.build(baseline_config()).lean.editing_wip_limit == 2
    with pytest.raises(Exception):
        Scenario("Y", "Bad", "", {"demand": {"weekly_target_ads": -5}}).build(baseline_config())


def test_scenarios_reproducible_with_same_seeds():
    base = baseline_config(horizon_weeks=2, warmup_weeks=0)
    scs = preset_scenarios(base)[:3]
    r1 = run_scenarios(scs, base, reps=2, seed0=11)
    r2 = run_scenarios(scs, base, reps=2, seed0=11)
    for k in r1:
        pd.testing.assert_frame_equal(r1[k], r2[k])


def test_baseline_in_comparison_has_zero_delta_and_directions_make_sense():
    base = baseline_config(horizon_weeks=3, warmup_weeks=1)
    scs = preset_scenarios(base)
    runs = run_scenarios([scs[0], scs[1], scs[3], scs[4], scs[5]], base, reps=4, seed0=21)
    tab = comparison_table(runs, "A")
    g = lambda s, m: tab[(tab.scenario == s) & (tab.metric == m)].iloc[0]
    assert g("A", "accepted_per_week").delta == 0
    assert g("D", "first_pass_yield_pct").delta < -10               # quality deterioration lowers FPY
    assert g("D", "lead_time_h").delta > 0
    assert g("E", "accepted_per_week").delta < 0                    # fewer editors -> less output
    assert g("E", "accepted_per_week").verdict == "worse"
    assert g("F", "lead_time_h").delta < 0 and g("F", "muda_waste_share_pct").delta < 0
    assert g("F", "first_pass_yield_pct").verdict == "better"
    assert g("B", "backlog_end").delta > 0


def test_confidence_interval_math_and_no_unsupported_significance():
    m, hw = mean_ci([10, 12, 14, 16])
    assert m == 13 and hw == pytest.approx(t_quantile_95(3) * np.std([10, 12, 14, 16], ddof=1) / 2)
    assert np.isnan(mean_ci([5])[1])
    # identical runs -> differences are noise, never labelled better/worse
    df = pd.DataFrame({c: [1.0, 1.1, 0.9] for c in
                       ["accepted_per_week", "target_attainment_pct", "lead_time_h", "wip_avg", "editor_utilization_pct",
                        "overtime_required_per_week", "on_time_delivery_pct", "first_pass_yield_pct", "flow_efficiency_pct",
                        "mura_index", "muri_index", "muda_waste_share_pct", "backlog_end"]})
    df["rep"] = range(3)
    tab = comparison_table({"A": df, "B": df.copy()}, "A")
    assert (tab[tab.scenario == "B"].verdict == "within replication noise").all()
    # a single replication can never be called distinguishable
    one = df.iloc[:1]
    t1 = comparison_table({"A": one, "B": one * 2}, "A")
    assert not t1[t1.scenario == "B"].distinguishable.any()


def test_replications_use_distinct_seeds_and_return_detailed_first_run():
    cfg = baseline_config(horizon_weeks=2, warmup_weeks=0)
    first, df = run_replications(cfg, n=3, seed0=5)
    assert first is not None and len(df) == 3 and list(df.seed) == [5, 6, 7]
    assert df.lead_time_h.nunique() > 1   # different seeds -> different stochastic outcomes
