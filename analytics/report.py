"""Bundle all analytics for one run, and export (CSV zip, executive-summary PDF)."""
from __future__ import annotations

import io
import json
import zipfile
from typing import Dict, Optional

import numpy as np
import pandas as pd

from simulation.engine import SimulationResult
from simulation.scenarios import mean_ci

from . import kpis as K
from . import lean_metrics as lm
from .bottlenecks import detect_bottleneck, throughput_sensitivity
from .recommendations import build_recommendations


def build_analysis(res: SimulationResult, reps: Optional[pd.DataFrame] = None, with_sensitivity: bool = True) -> Dict:
    cfg = res.cfg
    k = K.compute_kpis(res)
    sens = throughput_sensitivity(cfg, res.seed, n_seeds=3) if with_sensitivity else None
    bott = detect_bottleneck(res, sens)
    mura = lm.mura_analysis(res)
    muri = lm.muri_analysis(res)
    coh = lm.cohort(res)
    k["muda_waste_share_pct"] = lm.waste_share(coh, cfg)
    k["mura_index"], k["muri_index"] = mura["index"], muri["index"]
    pods = K.pod_table(res)
    status = {int(r["pod"]): K.pod_status(r, cfg) for _, r in pods.iterrows()}
    # flow efficiency by POD
    fe = []
    for p in range(cfg.org.n_pods):
        d = lm.delivered_in_window(res)
        fe.append(dict(name=f"POD {p+1}", flow_eff=lm.flow_efficiency(d[d["pod"] == p], cfg)))
    return dict(
        res=res, cfg=cfg, reps=reps, kpis=k, sens=sens, bott=bott, mura=mura, muri=muri, pods=pods, pod_status=status,
        editors=K.editor_table(res), daily=K.daily_table(res), takt=lm.takt_summary(cfg), little=lm.littles_law(res),
        decomposition=lm.lead_time_decomposition(res.jobs[res.jobs["status"] == "delivered"]),
        waste=lm.waste_pareto(coh, cfg), flow_eff_pod=pd.DataFrame(fe),
        recs=build_recommendations(res, k, bott, mura, muri), cohort=coh,
    )


def kpi_ci_table(reps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in reps.columns:
        if c in ("rep", "seed") or not np.issubdtype(reps[c].dtype, np.number):
            continue
        m, hw = mean_ci(reps[c])
        rows.append(dict(kpi=c, mean=m, ci95_halfwidth=hw, min=reps[c].min(), max=reps[c].max(), n=len(reps)))
    return pd.DataFrame(rows)


def results_zip(a: Dict) -> bytes:
    res: SimulationResult = a["res"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", "SIMULATED output of the Lean Creative Production Simulator - not actual operational data.\n"
                                 "All durations are working hours on the simulation clock (weekends excluded).\n")
        z.writestr("config.json", res.cfg.model_dump_json(indent=2))
        z.writestr("jobs.csv", res.jobs.to_csv(index=False))
        z.writestr("worker_day.csv", res.worker_day.to_csv(index=False))
        z.writestr("pod_snapshots.csv", res.pod_snapshots.to_csv(index=False))
        z.writestr("pod_summary.csv", a["pods"].to_csv(index=False))
        z.writestr("editor_summary.csv", a["editors"].to_csv(index=False))
        z.writestr("daily.csv", a["daily"].to_csv(index=False))
        z.writestr("kpis.json", json.dumps({k: (None if (isinstance(v, float) and v != v) else v) for k, v in a["kpis"].items()}, indent=2, default=float))
        if len(res.events):
            z.writestr("events.csv", res.events.to_csv(index=False))
        if a.get("reps") is not None:
            z.writestr("replications.csv", a["reps"].to_csv(index=False))
            z.writestr("kpi_confidence_intervals.csv", kpi_ci_table(a["reps"]).to_csv(index=False))
    return buf.getvalue()


def executive_pdf(a: Dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    k, cfg, b = a["kpis"], a["cfg"], a["bott"]
    f = lambda v, d=1: "n/a" if v != v else f"{v:.{d}f}"
    st = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    S = []
    S.append(Paragraph("Lean Creative Production Simulator - Executive Summary", st["Title"]))
    S.append(Paragraph("<b>All figures are SIMULATED from the stated assumptions, not actual operational data.</b> "
                       f"Horizon {cfg.horizon_weeks} weeks (warm-up {cfg.warmup_weeks}), seed {a['res'].seed}, "
                       f"{len(a['reps']) if a.get('reps') is not None else 1} replication(s). Durations are working hours.", st["Normal"]))
    S.append(Spacer(1, 10))
    rows = [["KPI", "Value"],
            ["Weekly target (ads)", f(k["weekly_target"], 0)], ["Simulated accepted ads / week", f(k["accepted_per_week"])],
            ["Target attainment %", f(k["target_attainment_pct"])], ["Estimated editing capacity (ads/week)", f(k["estimated_capacity_ads_per_week"])],
            ["Avg editing cycle time (h)", f(k["cycle_time_edit_h"], 2)], ["Avg lead time (working h)", f(k["lead_time_h"], 2)],
            ["First-pass yield %", f(k["first_pass_yield_pct"])], ["Avg editor utilization %", f(k["editor_utilization_pct"])],
            ["Average WIP (ads)", f(k["wip_avg"])], ["On-time delivery %", f(k["on_time_delivery_pct"])],
            ["Flow efficiency %", f(k["flow_efficiency_pct"])], ["Mura index (0-100)", f(k["mura_index"])],
            ["Muri index (0-100)", f(k["muri_index"])], ["Muda waste share %", f(k["muda_waste_share_pct"])],
            ["Likely constraint", b["primary"]]]
    t = Table(rows, colWidths=[260, 240])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a78d6")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    S += [t, Spacer(1, 10), Paragraph("Bottleneck evidence", st["Heading3"])]
    S += [Paragraph("• " + e, st["Normal"]) for e in b["explanation"]]
    S += [Spacer(1, 8), Paragraph("Top recommendations (hypotheses to test, not guarantees)", st["Heading3"])]
    for r in a["recs"][:5]:
        S.append(Paragraph(f"<b>{r['priority']}. {r['title']}</b> - {r['observed']} Proposed: {r['intervention']}", st["Normal"]))
        S.append(Spacer(1, 3))
    doc.build(S)
    return buf.getvalue()
