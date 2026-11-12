# Example simulation results

**Simulated** with the default (illustrative) assumptions; not actual operational data.
Horizon 5 weeks (warm-up 1), seed 42, 10 replications. Durations are working hours.

## Baseline (Scenario A)

- Planned load: 31.0 h required vs 32.0 h available per editor-week = 97%
- Simulated accepted ads/week: 175.0 (target 180); attainment 97.2%
- Lead time 9.6 h; flow efficiency 37%; editor utilization 94%
- Likely constraint: Editing capacity (overloaded editors)

Evidence:
- Top candidate: Editing capacity (overloaded editors) (evidence score 0.55).
- editor utilization 94%
- 32.6% of job-hours waiting for an editor
- backlog trend +5.0 ads/week with ready buffer 39% full
- +15% editing speed adds +2.2 ads/week
- Editor utilisation, waiting for editors and throughput sensitivity agree: extra editing capacity limits output.

## Scenario comparison (means over replications)

| label                              |   A. Baseline |   B. Increased demand |   C. Higher complexity |   D. Quality deterioration |   E. Resource shortage |   F. Lean improvement |
|:-----------------------------------|--------------:|----------------------:|-----------------------:|---------------------------:|-----------------------:|----------------------:|
| Throughput (accepted ads/week)     |         175.8 |                 181.6 |                  150.8 |                      146.3 |                  158   |                 179.4 |
| Target attainment (%)              |          97.7 |                  82.5 |                   83.8 |                       81.3 |                   87.8 |                  99.6 |
| Lead time (working h)              |           9.5 |                  24.7 |                   23.8 |                       29.9 |                   17.8 |                   5.6 |
| Average WIP (ads)                  |          38.4 |                  50.5 |                   52.6 |                       65   |                   41.5 |                  24.8 |
| Editor utilization (%)             |          94.6 |                  98.1 |                   98.6 |                       85.1 |                   95.4 |                  93.1 |
| Overtime required (h/week)         |          17   |                 135.7 |                  127.1 |                       47.4 |                   86.1 |                   8.7 |
| On-time delivery (%)               |          97.4 |                  41   |                   44.6 |                       25.1 |                   68.6 |                  99.9 |
| First-pass yield (%)               |          85.7 |                  85.6 |                   85.5 |                       65.9 |                   85.9 |                  97   |
| Flow efficiency (%)                |          37.6 |                  14.5 |                   17.7 |                       11.9 |                   19.9 |                  60.5 |
| Mura index (0-100)                 |          25.6 |                  19.2 |                   18.9 |                       20.4 |                   27.7 |                  32.5 |
| Muri index (0-100)                 |          67.7 |                  80.8 |                   74   |                       46.3 |                   66   |                  54.8 |
| Muda: waste share of job-hours (%) |          50.8 |                  84.6 |                   81   |                       86.8 |                   79.7 |                  19   |
| Backlog at horizon (ads)           |           8.3 |                 164.5 |                  115.2 |                      137.9 |                   90   |                   0.5 |
