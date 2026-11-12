"""Core KPIs computed from simulation output (nothing is estimated unless labelled so)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from simulation.engine import SimulationResult
from . import lean_metrics as lm


def _safe_mean(s) -> float:
    s = pd.Series(s).dropna()
    return float(s.mean()) if len(s) else float("nan")


def compute_kpis(res: SimulationResult) -> Dict[str, float]:
    cfg, w = res.cfg, lm.window(res)
    j = res.jobs
    dv = lm.delivered_in_window(res)
    weeks = w["weeks"]
    target_total = cfg.demand.weekly_target_ads * weeks
    accepted = len(dv)

    # editing cycle time: productive editing hours per ad whose first editing pass finished in window
    edited = j[(j["edit_end"].notna()) & (j["edit_end"] >= w["t_w"] - 1e-9)]
    cyc_first = _safe_mean(edited["T_edit"])
    cyc_total = _safe_mean(edited["T_edit"] + edited["T_rework"])

    # first-pass yield
    fr = j[j["first_pass_approved"].notna() & j["review_start"].notna()]
    fpy = float(fr["first_pass_approved"].astype(bool).mean() * 100) if len(fr) else float("nan")

    ed = lm.editor_day_table(res)
    util = float(ed["busy_h"].sum() / ed["capacity_h"].sum() * 100) if ed["capacity_h"].sum() > 0 else float("nan")
    ps = res.pod_snapshots
    ps_w = ps[ps["t"] >= w["t_w"] - 1e-9]
    wip_t = ps_w.groupby("t")["wip"].sum()
    sys_t = ps_w.groupby("t")["open_total"].sum()
    last = ps[ps["t"] == ps["t"].max()]
    # on-time delivery: on-time accepted / (accepted + open ads already past due at horizon)
    open_j = j[(j["status"] == "open") & (j["arrival"] >= w["t_w"] - 1e-9)]
    overdue_open = int((open_j["due"] < res.t_end).sum())
    on_time = int(dv["on_time"].astype(bool).sum()) if len(dv) else 0
    otd = on_time / (accepted + overdue_open) * 100 if (accepted + overdue_open) > 0 else float("nan")

    ts = lm.takt_summary(cfg)
    fe = lm.flow_efficiency(dv, cfg)
    coh = lm.cohort(res)
    k = dict(
        weekly_target=cfg.demand.weekly_target_ads, weeks_measured=weeks,
        accepted_total=accepted, accepted_per_week=accepted / weeks if weeks else float("nan"),
        throughput_per_day=accepted / (weeks * w["D"]) if weeks else float("nan"),
        target_attainment_pct=accepted / target_total * 100 if target_total else float("nan"),
        estimated_capacity_ads_per_week=ts["capacity_ads_per_week"],
        capacity_vs_target_pct=ts["capacity_vs_target"] * 100,
        cycle_time_edit_h=cyc_first, cycle_time_edit_incl_rework_h=cyc_total,
        edit_elapsed_h=_safe_mean(edited["edit_elapsed_first"]),
        lead_time_h=_safe_mean(dv["lead_time"]),
        lead_time_median_h=float(dv["lead_time"].median()) if len(dv) else float("nan"),
        lead_time_p90_h=float(dv["lead_time"].quantile(0.9)) if len(dv) else float("nan"),
        touch_time_h=_safe_mean(dv["total_touch"]),
        wait_time_h=_safe_mean(dv["total_wait"]),
        first_pass_yield_pct=fpy,
        editor_utilization_pct=util,
        wip_avg=float(wip_t.mean()) if len(wip_t) else float("nan"),
        wip_end=int(last["wip"].sum()),
        system_open_avg=float(sys_t.mean()) if len(sys_t) else float("nan"),
        backlog_end=int(last["backlog"].sum()),
        open_total_end=int(last["open_total"].sum()),
        on_time_delivery_pct=otd, overdue_open_ads=overdue_open,
        flow_efficiency_pct=fe,
        overtime_clock_h_per_week=float(ed["ot_clock_h"].sum() / weeks) if weeks else 0.0,
        rework_rate_pct=float((dv["rework_count"] > 0).mean() * 100) if len(dv) else float("nan"),
        weighted_units_per_week=float(dv["est_hours"].sum() / weeks) if weeks else float("nan"),
        transfers=int(coh["transferred"].sum()),
    )
    return k


def pod_table(res: SimulationResult) -> pd.DataFrame:
    cfg, w = res.cfg, lm.window(res)
    j = res.jobs
    dv = lm.delivered_in_window(res)
    ed = lm.editor_day_table(res)
    wk = w["weeks"]
    shares = cfg.pod_shares
    rows = []
    last = res.pod_snapshots[res.pod_snapshots["t"] == res.pod_snapshots["t"].max()].set_index("pod")
    for p in range(cfg.org.n_pods):
        e = ed[ed["pod"] == p]
        d = dv[dv["pod"] == p]
        target = cfg.demand.weekly_target_ads * shares[p] * wk
        cap, busy = e["capacity_h"].sum(), e["busy_h"].sum()
        edited = j[(j["pod"] == p) & j["first_pass_approved"].notna() & j["review_start"].notna()]
        rows.append(dict(
            pod=p, name=f"POD {p+1}", editors=cfg.org.editors_in_pod(p), target=target, accepted=len(d),
            attainment=len(d) / target if target else np.nan, utilization=busy / cap if cap > 0 else np.nan,
            capacity_h=cap, busy_h=busy, wip=int(last.loc[p, "wip"]), backlog=int(last.loc[p, "backlog"]),
            ready=int(last.loc[p, "ready"]), editing=int(last.loc[p, "editing_owned"]), review_wip=int(last.loc[p, "review_wip"]),
            rework_q=int(last.loc[p, "rework_q"]), queue=int(last.loc[p, "backlog"] + last.loc[p, "ready"] + last.loc[p, "rework_q"]),
            lead_time_h=_safe_mean(d["lead_time"]), fpy=float(edited["first_pass_approved"].astype(bool).mean() * 100) if len(edited) else np.nan,
            rework_h=float((j[j["pod"] == p]["T_rework"] + j[j["pod"] == p]["T_rework_review"]).sum()),
            on_time=float(d["on_time"].astype(bool).mean() * 100) if len(d) else np.nan,
            transfers_out=int(j[(j["pod"] == p)]["transferred"].sum()),
        ))
    return pd.DataFrame(rows)


def pod_status(row: pd.Series, cfg, queue_per_editor: float = None) -> str:
    """Green / Amber / Red using the configurable, documented thresholds in ``ThresholdConfig``.

    Red   : attainment < attain_amber, OR utilization >= pod_util_red, OR waiting editing work per editor >= pod_queue_red_per_editor
    Amber : attainment < attain_green, OR utilization >= pod_util_amber
    Green : otherwise
    """
    t = cfg.thresholds
    att, util = row["attainment"], row["utilization"]
    qpe = (row["queue"] / max(row["editors"], 1)) if queue_per_editor is None else queue_per_editor
    if (att == att and att < t.attain_amber) or (util == util and util >= t.pod_util_red) or qpe >= t.pod_queue_red_per_editor:
        return "red"
    if (att == att and att < t.attain_green) or (util == util and util >= t.pod_util_amber):
        return "amber"
    return "green"


def editor_table(res: SimulationResult) -> pd.DataFrame:
    w = lm.window(res)
    j = res.jobs
    dv = lm.delivered_in_window(res)
    ed = lm.editor_day_table(res)
    g = ed.groupby(["worker", "pod"]).agg(capacity_h=("capacity_h", "sum"), busy_h=("busy_h", "sum"),
                                          edit_h=("busy_edit", "sum"), rework_h=("busy_rework", "sum"),
                                          ot_clock_h=("ot_clock_h", "sum"),
                                          absent_days=("absent", "sum")).reset_index()
    idle_cols = [c for c in ed.columns if c.startswith("idle_")]
    for c in idle_cols:
        g[c] = ed.groupby("worker")[c].sum().reindex(g["worker"]).values
    g["utilization"] = g["busy_h"] / g["capacity_h"].replace(0, np.nan)
    g["ads_accepted"] = g["worker"].map(dv.groupby("editor").size()).fillna(0).astype(int)
    first = j[j["edit_end"].notna()]
    g["ads_edited"] = g["worker"].map(first.groupby("first_editor").size()).fillna(0).astype(int)
    g["avg_cycle_h"] = g["worker"].map(first.groupby("first_editor")["T_edit"].mean())
    sp = res.workers.set_index("worker")
    g["speed"] = g["worker"].map(sp["speed"])
    g["can_complex"] = g["worker"].map(sp["can_complex"])
    return g


def daily_table(res: SimulationResult) -> pd.DataFrame:
    cfg, w = res.cfg, lm.window(res)
    j = res.jobs
    n, H, D = w["n_days"], w["H"], w["D"]
    arr = j.assign(day=(j["arrival"] // H).astype(int)).groupby("day").size().reindex(range(n), fill_value=0)
    done = j[j["status"] == "delivered"].assign(day=lambda x: (x["approval"] // H).clip(upper=n - 1).astype(int))
    out = done.groupby("day").size().reindex(range(n), fill_value=0)
    df = pd.DataFrame(dict(day=range(n), week=[d // D for d in range(n)], weekday=[d % D for d in range(n)],
                           arrivals=arr.values, accepted=out.values))
    df["cum_accepted"] = df["accepted"].cumsum()
    df["planned_daily"] = cfg.demand.weekly_target_ads / D
    df["cum_planned"] = df["planned_daily"].cumsum()
    return df


def summarize(res: SimulationResult) -> Dict[str, float]:
    """Flat scalar summary of one replication (used by scenario comparison and CSV export)."""
    k = compute_kpis(res)
    mura = lm.mura_analysis(res)
    muri = lm.muri_analysis(res)
    coh = lm.cohort(res)
    k.update(
        mura_index=mura["index"], muri_index=muri["index"], muda_waste_share_pct=lm.waste_share(coh, res.cfg),
        workload_balance_index=mura["workload_balance_index"],
        overtime_required_per_week=muri["overtime_required_per_week"],
        capacity_shortfall_h=muri["capacity_shortfall_h"],
    )
    return k
