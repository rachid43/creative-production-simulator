"""Role-level availability and per-POD overrides (mix, first-pass approval)."""
import pytest
from pydantic import ValidationError

from analytics import lean_metrics as lm
from config.defaults import baseline_config
from simulation.engine import run_simulation


def test_strategist_availability_differs_from_editors():
    cfg = baseline_config(horizon_weeks=2, warmup_weeks=0, org=dict(strategist_availability=0.5))
    wd = run_simulation(cfg, 3).worker_day
    assert wd[wd.role == "strategist"].capacity_h.max() == pytest.approx(0.5 * 8)
    assert wd[wd.role == "editor"].capacity_h.max() == pytest.approx(0.8 * 8)
    # default: same as editors
    wd0 = run_simulation(baseline_config(horizon_weeks=1, warmup_weeks=0), 3).worker_day
    assert wd0[wd0.role == "strategist"].capacity_h.max() == pytest.approx(0.8 * 8)


def test_lower_strategist_availability_cuts_output_when_briefing_binds():
    base = baseline_config(horizon_weeks=3, warmup_weeks=0, quality=dict(brief_hours=1.2))
    slow = base.copy_with(org=dict(strategist_availability=0.4))
    a, b = run_simulation(base, 5), run_simulation(slow, 5)
    assert (b.jobs.status == "delivered").sum() < (a.jobs.status == "delivered").sum()


def test_per_pod_mix_changes_workload_of_that_pod_only():
    heavy = {"Simple": 0.0, "Medium": 0.3, "Complex": 0.7}
    cfg = baseline_config(horizon_weeks=3, warmup_weeks=0, complexity=dict(pod_mix=[heavy] + [None] * 5))
    j = run_simulation(cfg, 4).jobs
    m = j.groupby("pod").est_hours.mean()
    assert m[0] > 4.0 and (m[1:] < 3.4).all()
    assert cfg.mean_std_hours_for(0) == pytest.approx(0.3 * 3 + 0.7 * 5)
    assert cfg.mean_std_hours_for(1) == pytest.approx(2.95)


def test_per_pod_first_pass_drives_rework_in_that_pod():
    cfg = baseline_config(horizon_weeks=5, warmup_weeks=0, quality=dict(pod_first_pass=[0.4] + [None] * 5))
    j = run_simulation(cfg, 4).jobs
    rw = j.groupby("pod").rework_count.mean()
    assert rw[0] > 2.5 * rw[1:].mean()   # expected ~4x (60 % vs 15 % first-review rejection)
    assert lm.expected_rework_hours_per_ad(cfg, 0) > 3 * lm.expected_rework_hours_per_ad(cfg, 1)
    # org-level planning numbers reflect the weak POD
    assert lm.takt_summary(cfg)["expected_rework_hours"] > lm.takt_summary(baseline_config())["expected_rework_hours"]


def test_pod_overrides_validated():
    with pytest.raises(ValidationError):
        baseline_config(quality=dict(pod_first_pass=[0.8, 0.8]))               # wrong length
    with pytest.raises(ValidationError):
        baseline_config(quality=dict(pod_first_pass=[1.4] + [None] * 5))
    with pytest.raises(ValidationError):
        baseline_config(complexity=dict(pod_mix=[{"Simple": 0.5, "Medium": 0.5, "Complex": 0.5}] + [None] * 5))
    with pytest.raises(ValidationError):
        baseline_config(org=dict(strategist_availability=0.01))


def test_conservation_holds_with_overrides():
    cfg = baseline_config(horizon_weeks=2, warmup_weeks=0, org=dict(strategist_availability=0.6),
                          quality=dict(pod_first_pass=[0.5] + [None] * 5),
                          complexity=dict(pod_mix=[None, None, {"Simple": 0.1, "Medium": 0.3, "Complex": 0.6}, None, None, None]))
    assert run_simulation(cfg, 2).conservation["balance"] == 0
