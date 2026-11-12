"""Baseline configuration derived from Lean_Creative_Production_System.pdf.

Playbook-sourced values
-----------------------
* 6 PODs x (1 strategist + 3 editors) = 18 editors, 1 Head of Creative, 1 Lead Creative
* 5 days x 8 h; 180 ads/week (36/day; 30/POD/week; 10/editor/week)
* 80 % productive availability  -> effective takt 3.2 productive hours per ad
* Standard editing hours 1.5 / 3 / 5 for Simple / Medium / Complex
* Ready-brief buffer 6, editing WIP limit 3, review WIP limit 3 (per POD)

Everything else (mix, variability, first-pass rate, review time, brief time, asset delays,
SLA) is an **illustrative assumption**, not measured performance.  Replace with your own data.
"""
from __future__ import annotations

from .schema import SimConfig

# Playbook constants (used by tests / documentation, not by the engine)
PLAYBOOK = {
    "weekly_target": 180,
    "daily_target": 36,
    "pod_weekly_target": 30,
    "pod_daily_target": 6,
    "editor_weekly_target": 10,
    "editor_daily_target": 2,
    "gross_takt_hours_per_ad": 4.0,
    "effective_takt_hours_per_ad": 3.2,
    "n_pods": 6,
    "total_editors": 18,
}

ASSUMPTION_NOTES = [
    "Complexity mix 30/50/20 is illustrative (playbook gives only the standard hours).",
    "First-pass approval 85 %, repeat-rework 15 %, rework = 30 % of standard hours: illustrative.",
    "Brief time 0.6 h and review time 0.25 h per ad: illustrative (playbook does not specify).",
    "Review is a strategist responsibility by default (no dedicated reviewer role in the playbook).",
    "Asset-unavailability (5 % of edits, 1.5 h mean) is an illustrative event model; set to 0 to disable.",
    "SLA = 3 working days after demand arrival; 10 % of ads flagged urgent (half the SLA): illustrative.",
    "Editor speed variability (CV 5 %) and editing-time variability (CV 25 %) are illustrative.",
]


def baseline_config(**overrides) -> SimConfig:
    """The baseline (Scenario A) configuration."""
    cfg = SimConfig()
    return cfg.copy_with(**overrides) if overrides else cfg


def reset_values() -> dict:
    return baseline_config().model_dump()
