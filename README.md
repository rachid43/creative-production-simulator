# Lean Creative Production Simulator

A discrete-event simulation (SimPy) of a performance-marketing creative organisation, wrapped in a Streamlit
command center. It models the flow **Demand → Briefing → Ready buffer → Editing (pull) → Review → Rework → Approved**
with finite people, shift calendars, WIP limits and variability, and diagnoses **Mura** (unevenness),
**Muri** (overburden) and **Muda** (waste). Baseline: `Lean_Creative_Production_System.pdf`
(6 PODs × 1 strategist + 3 editors, 180 ads/week, 80 % productive availability, effective takt 3.2 h).

> Everything shown is **simulated** from stated assumptions, not operational data. Values not in the playbook
> (complexity mix, first-pass rate, brief/review time, asset delays, SLA, variability) are *illustrative* — see
> `config/defaults.py` and the "Baseline assumptions" panel; replace them with your own measurements.

## Install, test, run

```bash
cd lean_creative_simulator
python3 -m venv .venv && source .venv/bin/activate     # Python 3.11+
pip install -r requirements-dev.txt   # requirements.txt = runtime only (used by Streamlit Cloud)
pytest -q                                               # 58 tests incl. manual-calculation and conservation checks
streamlit run app.py                                    # simplified 4-page view (opens http://localhost:8501)
streamlit run app_advanced.py                           # full 9-page analyst version
python examples.py                                      # regenerates docs/example_results.md
```

## Simple app (`app.py`) - for the Head of Creative and Lead Creative
1. **Overview** - "can we deliver the target?" verdict, 6 KPIs, plan vs simulated chart, POD status, what holds output back, top 3 actions.
2. **PODs** - daily view: network, POD scorecard, rebalancing hint, editor detail.
3. **Waste & Flow** - Mura/Muri/Muda in one line each, lead-time breakdown, biggest delay causes, editor load heatmap.
4. **What-if** - baseline vs presets, with noise flagged.
Twelve sidebar inputs; everything else uses the playbook baseline.

## Pages of the full app (`app_advanced.py`)
1. **Executive Command Center** – KPI cards, "is 180/week achievable?" verdict, plan-vs-simulated cumulative chart, POD side-by-side, delay Pareto, CSV-zip and PDF export.
2. **POD Production Network** – six PODs with strategist, editors, kanban flow (Backlog→Briefing→Ready→Editing→Review→Approved), status colours, playback slider/▶ over time-stamped snapshots, editor-level drill-down.
3. **Mura / Muri / Muda** – heatmaps, box plots, Pareto charts, indices with documented formulas.
4. **Takt & Flow** – takt vs cycle vs lead time vs throughput, Little's Law with stationarity caveats.
5. **Heijunka** – equal-count vs equal-hours vs capacity-aware weekly plans, brief release schedule, and a simulation test of the assignment policies.
6. **Scenario Laboratory** – presets A–F + custom, common random numbers, replications, 95 % CIs, paired differences.
7. **Improvement Recommendations** – each tied to observed KPIs with a one-click paired simulation test.

## Model essentials (all enforced and tested)
* Clock = **working hours** (weekends excluded); lead/wait times are working hours. A person converts shift time to productive time at *productive availability*; work pauses overnight and on absence days. Overtime only lifts the productive rate (cap 100 %) when the POD has waiting work.
* Editors **pull** from approved briefs; editing WIP limit counts work done by the POD's own editors; finished ads block the editor if the **review WIP limit** is full; strategists brief only while ready+briefing < buffer. `Enforce WIP limits` off = push.
* Review is performed by the **strategist** (shares one capacity with briefing – never double counted), the **Lead Creative** (configurable share) or **dedicated reviewers**.
* Rework consumes real editor *and* reviewer hours; after max loops an ad is force-accepted or cancelled (config).
* Only approved ads count as throughput. Nothing is dropped at the horizon: `initial WIP + arrivals = completed + ending WIP + cancelled`.
* Cross-POD pulls keep the ad **owned by (and credited to)** its home POD.
* Per-ad random inputs are pre-drawn from `(seed, ad index)` → scenarios share random numbers.
* Every hour of an ad's lead time lands in exactly one category (touch, overhead, or a named wait), which feeds flow efficiency and waste.

## Key definitions
| Metric | Definition |
|---|---|
| Takt | available production time / demand (4.0 h gross, 3.2 h productive per ad per editor at baseline) |
| Cycle time | productive (touch) hours per ad |
| Lead time | approval − arrival |
| Throughput / attainment | accepted ads per week; accepted / target |
| First-pass yield | approved at first review / first reviewed |
| WIP | brief started, not yet accepted (backlog awaiting brief is separate) |
| Flow efficiency | value-adding hours / lead time (classification configurable) |
| Editor utilization | busy editing hours / available productive hours |
| Workload Balance Index | 1 − CV(planned standard hours ÷ capacity) across editors |
| Mura index | mean of min(1, CV/ref)×100 over 6 documented CV components |
| Muri index | 100×(0.5·share of resource-days 85–95 % + 1.0·share >95 %) |
| Muda share | waste-classified job-hours / all job-hours |
| Little's Law | L vs λ·W, flagged when the window isn't stationary |

## Interpreting the baseline
Mean standard hours (2.95 h) fit inside the 3.2 h effective takt, but expected rework leaves only ~3 % headroom
(97 % planned load). The simulation therefore lands slightly under 180/week with queues driven by variability — the
constraint evidence (utilisation + waiting + sensitivity re-runs) points to editing capacity. See `docs/example_results.md`.

## Limitations
* Finite horizons penalise attainment by the work still in flight; use ≥1 warm-up week for steady-state, 0 for backlog/first-week analysis.
* Standardisation effects, asset delays, urgency, SLA and the Kingman illustration are assumptions/approximations.
* Scenario "better/worse" labels appear only when the paired 95 % CI excludes zero; no other significance claims are made.
* PDF export is a text/table summary (no chart images).

## Layout
`config/` schema (Pydantic) + baseline · `simulation/` engine, entities, resources, scheduling (dispatch rules + Heijunka planner), scenarios · `analytics/` KPIs, Lean metrics, bottlenecks, recommendations, export · `visualization/` charts, POD network, pages · `tests/`.
