"""Generate docs/example_results.md from the baseline and the six preset scenarios.

Usage:  python examples.py
"""
from analytics.report import build_analysis
from config.defaults import baseline_config
from simulation.scenarios import comparison_table, preset_scenarios, run_replications, run_scenarios


def main(reps: int = 10):
    cfg = baseline_config()
    first, df = run_replications(cfg, reps, cfg.seed)
    a = build_analysis(first, df)
    k, ts = a["kpis"], a["takt"]
    scs = preset_scenarios(cfg)
    names = {s.key: s.name for s in scs}
    tab = comparison_table(run_scenarios(scs, cfg, reps, cfg.seed), "A", names)
    piv = tab.pivot_table(index="label", columns="name", values="mean", sort=False)[[s.name for s in scs]].round(1)
    out = ["# Example simulation results", "",
           "**Simulated** with the default (illustrative) assumptions; not actual operational data.",
           f"Horizon {cfg.horizon_weeks} weeks (warm-up {cfg.warmup_weeks}), seed {cfg.seed}, {reps} replications. Durations are working hours.", "",
           "## Baseline (Scenario A)", "",
           f"- Planned load: {ts['required_hours_per_editor_week']:.1f} h required vs {ts['available_productive_h_per_editor_week']:.1f} h available per editor-week = {ts['planned_load_ratio']*100:.0f}%",
           f"- Simulated accepted ads/week: {k['accepted_per_week']:.1f} (target 180); attainment {k['target_attainment_pct']:.1f}%",
           f"- Lead time {k['lead_time_h']:.1f} h; flow efficiency {k['flow_efficiency_pct']:.0f}%; editor utilization {k['editor_utilization_pct']:.0f}%",
           f"- Likely constraint: {a['bott']['primary']}", "", "Evidence:", *[f"- {e}" for e in a["bott"]["explanation"]], "",
           "## Scenario comparison (means over replications)", "", piv.to_markdown(), ""]
    import os
    os.makedirs("docs", exist_ok=True)
    open("docs/example_results.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
