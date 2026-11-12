import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from config.defaults import baseline_config
from simulation.engine import run_simulation


def det_cfg(**over):
    """Fully deterministic 1-POD / 1-editor / 1-strategist configuration (all variability off, 100 % availability).

    Hand-calculated timeline (3 Medium ads at t=0, brief 0.5 h, edit 3 h, review 0.25 h, no rework):
      brief done at 0.5, 1.0, 1.5   editor busy 0.5-3.5, 3.5-6.5, 6.5-9.5
      reviews (strategist idle -> immediate) approve at 3.75, 6.75, 9.75
    """
    base = dict(
        horizon_weeks=1, warmup_weeks=0, seed=1, record_events=True, snapshot_interval_hours=0.25,
        org=dict(n_pods=1, editors_per_pod=1, strategists_per_pod=1, productive_availability=1.0, editor_speed_cv=0.0),
        demand=dict(weekly_target_ads=0.001, demand_cv=0.0, initial_backlog_ads=3, urgent_share=0.0),
        complexity=dict(mix={"Simple": 0.0, "Medium": 1.0, "Complex": 0.0}, editing_time_cv=0.0),
        quality=dict(first_pass_approval_rate=1.0, review_hours=0.25, review_cv=0.0, brief_hours=0.5, brief_cv=0.0,
                     asset_delay_probability=0.0, rework_cv=0.0),
    )
    return baseline_config(**base).copy_with(**over)


@pytest.fixture(scope="session")
def baseline_result():
    return run_simulation(baseline_config(horizon_weeks=4, warmup_weeks=0))
