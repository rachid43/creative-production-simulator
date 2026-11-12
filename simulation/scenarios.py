"""Scenario presets, replication runner (common random numbers) and confidence intervals."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.defaults import baseline_config
from config.schema import SimConfig
from .engine import SimulationResult, run_simulation

# two-sided 95 % Student-t quantiles for df = 1..30 (df > 30 uses 2.01 up to 50, then 1.98)
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t_quantile_95(df: int) -> float:
    if df < 1:
        return float("nan")
    return _T95[df - 1] if df <= 30 else (2.01 if df <= 50 else 1.98)


def mean_ci(values) -> Tuple[float, float]:
    """Mean and 95 % CI half-width (Student t) over replications."""
    v = np.asarray([x for x in values if x == x], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float("nan")
    return float(v.mean()), float(t_quantile_95(len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


@dataclass
class Scenario:
    key: str
    name: str
    description: str
    overrides: Dict[str, dict] = field(default_factory=dict)

    def build(self, base: SimConfig) -> SimConfig:
        cfg = base.copy_with(**{k: dict(v) for k, v in self.overrides.items()}) if self.overrides else base
        return cfg


def preset_scenarios(base: Optional[SimConfig] = None) -> List[Scenario]:
    base = base or baseline_config()
    n = base.org.n_pods
    counts = [base.org.editors_in_pod(p) for p in range(n)]
    short = list(counts)
    removed = 0
    for p in range(n - 1, -1, -1):  # remove one editor from the last POD(s) until two are gone
        if removed < 2 and short[p] > 1:
            short[p] -= 1
            removed += 1
    fp = base.quality.first_pass_approval_rate
    return [
        Scenario("A", "A. Baseline", "6 PODs, 18 editors, 180 ads per week (current assumptions)."),
        Scenario("B", "B. Increased demand", "Weekly demand raised to 220 ads (+22 %).",
                 {"demand": {"weekly_target_ads": 220.0}}),
        Scenario("C", "C. Higher complexity", "Complex share doubled to 40 % (Simple 15 %, Medium 45 %).",
                 {"complexity": {"mix": {"Simple": 0.15, "Medium": 0.45, "Complex": 0.40}}}),
        Scenario("D", "D. Quality deterioration", "First-pass approval 65 %, repeat rework 30 %, review time x1.5, 2 h feedback delay.",
                 {"quality": {"first_pass_approval_rate": max(0.0, fp - 0.20), "rework_repeat_probability": 0.30,
                              "review_hours": base.quality.review_hours * 1.5, "review_response_delay_hours": 2.0}}),
        Scenario("E", "E. Resource shortage", "Two editors removed (one each from the last two PODs).",
                 {"org": {"editor_counts": short}}),
        Scenario("F", "F. Lean improvement", "Standardised briefs, first-pass approval 92 %, cross-POD balancing and review sharing.",
                 {"lean": {"brief_standardization": True, "cross_pod_balancing": True, "cross_pod_review_sharing": True},
                  "quality": {"first_pass_approval_rate": 0.92}}),
    ]


def fast_cfg(cfg: SimConfig) -> SimConfig:
    """Replication config without the heavy event log and with coarse snapshots (analysis needs only summaries)."""
    return cfg.copy_with(record_events=False, snapshot_interval_hours=max(cfg.snapshot_interval_hours, 2.0))


def run_replications(cfg: SimConfig, n: Optional[int] = None, seed0: Optional[int] = None,
                     progress: Optional[Callable[[float], None]] = None, keep_first: bool = True
                     ) -> Tuple[Optional[SimulationResult], pd.DataFrame]:
    """Run ``n`` replications with seeds seed0, seed0+1, ... Returns (first detailed result, summary table)."""
    from analytics.kpis import summarize

    n = n or cfg.replications
    seed0 = cfg.seed if seed0 is None else seed0
    rows, first = [], None
    light = fast_cfg(cfg)
    for r in range(n):
        if r == 0 and keep_first:
            res = run_simulation(cfg, seed0)
            first = res
        else:
            res = run_simulation(light, seed0 + r)
        row = summarize(res)
        row.update(rep=r, seed=seed0 + r)
        rows.append(row)
        if progress:
            progress((r + 1) / n)
    return first, pd.DataFrame(rows)


def run_scenarios(scenarios: List[Scenario], base: SimConfig, reps: int, seed0: int,
                  progress: Optional[Callable[[float], None]] = None) -> Dict[str, pd.DataFrame]:
    """Same seeds for every scenario (common random numbers) -> paired, fair comparison."""
    out = {}
    for i, sc in enumerate(scenarios):
        cfg = sc.build(base)
        _, df = run_replications(cfg, reps, seed0, keep_first=False)
        out[sc.key] = df
        if progress:
            progress((i + 1) / len(scenarios))
    return out


COMPARE_METRICS: List[Tuple[str, str, str]] = [
    # (column, label, better) better in {"up","down","none"}
    ("accepted_per_week", "Throughput (accepted ads/week)", "up"),
    ("target_attainment_pct", "Target attainment (%)", "up"),
    ("lead_time_h", "Lead time (working h)", "down"),
    ("wip_avg", "Average WIP (ads)", "down"),
    ("editor_utilization_pct", "Editor utilization (%)", "none"),
    ("overtime_required_per_week", "Overtime required (h/week)", "down"),
    ("on_time_delivery_pct", "On-time delivery (%)", "up"),
    ("first_pass_yield_pct", "First-pass yield (%)", "up"),
    ("flow_efficiency_pct", "Flow efficiency (%)", "up"),
    ("mura_index", "Mura index (0-100)", "down"),
    ("muri_index", "Muri index (0-100)", "down"),
    ("muda_waste_share_pct", "Muda: waste share of job-hours (%)", "down"),
    ("backlog_end", "Backlog at horizon (ads)", "down"),
]


def comparison_table(runs: Dict[str, pd.DataFrame], baseline_key: str = "A", names: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Long table: scenario x metric with mean, CI, paired delta vs baseline and its CI."""
    base = runs[baseline_key].sort_values("rep").reset_index(drop=True)
    rows = []
    for key, df in runs.items():
        df = df.sort_values("rep").reset_index(drop=True)
        for col, label, better in COMPARE_METRICS:
            m, hw = mean_ci(df[col])
            bm, _ = mean_ci(base[col])
            n = min(len(df), len(base))
            diff = (df[col].values[:n] - base[col].values[:n]) if key != baseline_key else np.zeros(n)
            dm, dhw = mean_ci(diff)
            distinct = bool(key != baseline_key and n >= 3 and dhw == dhw and abs(dm) > dhw)
            if key == baseline_key or not distinct:
                verdict = "-" if key == baseline_key else "within replication noise"
            else:
                improved = (dm > 0) == (better == "up") if better != "none" else None
                verdict = "better" if improved else ("worse" if improved is False else "changed")
            rows.append(dict(scenario=key, name=(names or {}).get(key, key), metric=col, label=label, better=better,
                             mean=m, ci95=hw, delta=dm, delta_ci95=dhw,
                             delta_pct=(dm / bm * 100) if bm and bm == bm and bm != 0 else np.nan,
                             distinguishable=distinct, verdict=verdict, n=len(df)))
    return pd.DataFrame(rows)
