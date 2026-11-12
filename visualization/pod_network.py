"""Interactive POD production network (Plotly shapes) driven by time-stamped simulation snapshots."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from analytics.kpis import pod_status
from simulation.engine import SimulationResult

from .charts import STATUS

EDITOR_COLORS = {"edit": "#2a78d6", "rework": "#eb6834", "idle": "#c9c9c4", "blocked": "#d64545", "asset": "#eda100",
                 "absent": "#ffffff", "brief": "#2a78d6", "reviewing": "#6250d6", "reviewing ": "#6250d6"}
STATE_LABEL = {"edit": "editing", "rework": "rework", "idle": "idle / waiting for work", "blocked": "blocked (review WIP limit)",
               "asset": "waiting for assets", "absent": "absent", "brief": "briefing", "reviewing": "reviewing"}
STAGES = ["Backlog", "Briefing", "Ready", "Editing", "Review", "Approved"]
BUSY = ("edit", "rework")


def snapshot_times(res: SimulationResult) -> np.ndarray:
    return np.sort(res.pod_snapshots["t"].unique())


def pod_state_at(res: SimulationResult, t: float) -> pd.DataFrame:
    """State of every POD at (nearest snapshot to) time ``t`` including status colour per the configurable thresholds."""
    cfg = res.cfg
    times = snapshot_times(res)
    ts = times[np.abs(times - t).argmin()]
    ps = res.pod_snapshots[res.pod_snapshots["t"] == ts].set_index("pod")
    ws = res.worker_snapshots
    at = ws[ws["t"] == ts]
    ed_hist = ws[(ws["role"] == "editor") & (ws["t"] <= ts) & (ws["state"] != "absent")]
    util = ed_hist.assign(b=ed_hist["state"].isin(BUSY)).groupby("pod")["b"].mean()
    H, D = cfg.org.hours_per_day, cfg.org.days_per_week
    shares = cfg.pod_shares
    rows: List[dict] = []
    for p in range(cfg.org.n_pods):
        r = ps.loc[p]
        planned = cfg.demand.weekly_target_ads * shares[p] * ts / (H * D)
        att = r["accepted"] / planned if planned >= 2 else np.nan
        row = dict(pod=p, name=f"POD {p+1}", editors=cfg.org.editors_in_pod(p), backlog=int(r["backlog"]), briefing=int(r["briefing"]),
                   ready=int(r["ready"]), editing=int(r["editing_owned"]), review=int(r["review_wip"]), rework_q=int(r["rework_q"]),
                   accepted=int(r["accepted"]), planned=planned, attainment=att, utilization=float(util.get(p, np.nan)),
                   queue=int(r["backlog"] + r["ready"] + r["rework_q"]), wip=int(r["wip"]))
        row["status"] = pod_status(pd.Series(row), cfg)
        row["editor_states"] = [(x.worker, x.state, x.job_id) for x in at[(at.role == "editor") & (at.pod == p)].itertuples()]
        row["strategist_states"] = [(x.worker, x.state, x.job_id) for x in at[(at.role == "strategist") & (at.pod == p)].itertuples()]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["t"] = float(ts)
    return out


def pod_network_figure(res: SimulationResult, t: float, selected: Optional[int] = None, height: int = 760) -> go.Figure:
    cfg = res.cfg
    st = pod_state_at(res, t)
    lean = cfg.lean
    n = len(st)
    cols = 3
    rows_n = int(np.ceil(n / cols))
    W, Hc = 10.0, 10.0
    shapes, ann = [], []
    hx, hy, ht = [], [], []

    def rect(x0, y0, x1, y1, fill, line, lw=1, dash=None, opacity=1.0):
        shapes.append(dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, fillcolor=fill, line=dict(color=line, width=lw, dash=dash or "solid"),
                           opacity=opacity, layer="below"))

    def text(x, y, s, size=11, color="#555", anchor="center", bold=False):
        ann.append(dict(x=x, y=y, text=f"<b>{s}</b>" if bold else s, showarrow=False, font=dict(size=size, color=color), xanchor=anchor))

    for i, r in st.iterrows():
        c, rr = i % cols, i // cols
        ox, oy = c * W + 0.3, (rows_n - 1 - rr) * Hc + 0.3
        sc = STATUS[r["status"]]
        sel = selected == r["pod"]
        rect(ox, oy, ox + 9.4, oy + 9.4, "rgba(128,128,128,0.07)", sc, lw=5 if sel else 3)
        icon = {"green": "● On track", "amber": "▲ At risk", "red": "■ Bottleneck / overburden"}[r["status"]]
        text(ox + 0.25, oy + 9.0, f"POD {r['pod']+1}", 15, "#222", "left", True)
        text(ox + 9.15, oy + 9.0, icon, 11, sc, "right", True)
        att = "n/a yet" if r["attainment"] != r["attainment"] else f"{r['attainment']*100:.0f}%"
        ut = "n/a" if r["utilization"] != r["utilization"] else f"{r['utilization']*100:.0f}%"
        text(ox + 0.25, oy + 8.35, f"Accepted {r['accepted']}/{r['planned']:.0f} plan  ·  attain. {att}  ·  util {ut}", 10, "#666", "left")
        hx.append(ox + 4.7); hy.append(oy + 8.6)
        ht.append(f"<b>POD {r['pod']+1}</b> — {r['status'].upper()}<br>Accepted: {r['accepted']}<br>Plan-to-date: {r['planned']:.1f}<br>"
                  f"Attainment: {att}<br>Editor utilization (sampled): {ut}<br>Queue (backlog+ready+rework): {r['queue']}<br>WIP: {r['wip']}")
        # strategist + editors
        for (wid, state, job) in r["strategist_states"][:1]:
            col = EDITOR_COLORS.get(state, "#c9c9c4")
            shapes.append(dict(type="circle", x0=ox + 0.35, y0=oy + 6.75, x1=ox + 1.45, y1=oy + 7.85, fillcolor=col, line=dict(color="#6250d6", width=2.5)))
            text(ox + 0.9, oy + 7.3, "S", 12, "#fff" if state != "idle" else "#444", "center", True)
            text(ox + 0.9, oy + 6.45, f"Strategist<br>{STATE_LABEL.get(state, state)}", 8, "#666")
            hx.append(ox + 0.9); hy.append(oy + 7.3)
            ht.append(f"<b>Strategist {wid}</b><br>{STATE_LABEL.get(state, state)}<br>Current: {job or '-'}")
        ne = len(r["editor_states"])
        step = min(1.9, 7.2 / max(ne, 1))
        for k, (wid, state, job) in enumerate(r["editor_states"]):
            x = ox + 2.4 + k * step
            col = EDITOR_COLORS.get(state, "#c9c9c4")
            shapes.append(dict(type="circle", x0=x, y0=oy + 6.75, x1=x + 1.1, y1=oy + 7.85, fillcolor=col,
                               line=dict(color="#888" if state == "absent" else col, width=2, dash="dot" if state == "absent" else "solid")))
            text(x + 0.55, oy + 7.3, f"E{k+1}", 11, "#fff" if state in ("edit", "rework", "blocked", "asset") else "#444", "center", True)
            text(x + 0.55, oy + 6.45, (job or STATE_LABEL.get(state, state))[-5:] if job else STATE_LABEL.get(state, state).split(" ")[0], 8, "#666")
            hx.append(x + 0.55); hy.append(oy + 7.3)
            ht.append(f"<b>{wid}</b><br>{STATE_LABEL.get(state, state)}<br>Current ad: {job or '-'}")
        # kanban flow: Backlog -> Briefing -> Ready -> Editing -> Review -> Approved
        vals = [r["backlog"], r["briefing"], r["ready"], r["editing"], r["review"] + r["rework_q"], r["accepted"]]
        refs = [12, max(1, cfg.org.strategists_per_pod), lean.ready_brief_buffer, max(lean.editing_wip_limit, 1), lean.review_wip_limit + 2,
                max(r["planned"], 1)]
        bw, gap = 1.3, 0.22
        x0 = ox + 0.2
        for s_i, (lab, v, ref) in enumerate(zip(STAGES, vals, refs)):
            bx = x0 + s_i * (bw + gap)
            frac = min(1.0, v / ref) if ref else 0
            over = v > ref and s_i < 5
            rect(bx, oy + 1.5, bx + bw, oy + 5.3, "rgba(128,128,128,0.10)", "#999", lw=1)
            if frac > 0:
                fillc = "#d64545" if (over and s_i in (0, 2, 4)) else ("#2e9e5b" if s_i == 5 else "#2a78d6")
                rect(bx, oy + 1.5, bx + bw, oy + 1.5 + 3.8 * frac, fillc, fillc, lw=0, opacity=0.55)
            text(bx + bw / 2, oy + 3.4, f"{v}", 18, "#222", "center", True)
            text(bx + bw / 2, oy + 1.05, lab if lab != "Review" else "Review<br>+rework", 9, "#555")
            hx.append(bx + bw / 2); hy.append(oy + 3.4)
            ht.append(f"<b>{lab}</b> (POD {r['pod']+1}): {v}" + (f"<br>limit/reference: {ref:.0f}" if s_i < 5 else ""))
            if s_i < len(STAGES) - 1:
                ann.append(dict(x=bx + bw + gap, y=oy + 3.4, ax=bx + bw, ay=oy + 3.4, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                                arrowhead=2, arrowsize=1.2, arrowwidth=1.8, arrowcolor="#777"))
        text(ox + 0.25, oy + 0.35, f"Queue waiting for editors: {r['queue']}  ·  WIP: {r['wip']}  ·  rework queue: {r['rework_q']}", 10, "#666", "left")
    # legend
    lx = 0.4
    for k, lab in (("edit", "editing"), ("rework", "rework"), ("idle", "idle"), ("blocked", "blocked"), ("asset", "asset wait"), ("absent", "absent")):
        shapes.append(dict(type="circle", x0=lx, y0=-0.85, x1=lx + 0.4, y1=-0.45, fillcolor=EDITOR_COLORS[k], line=dict(color="#888", width=1)))
        text(lx + 0.55, -0.65, lab, 10, "#666", "left")
        lx += 2.3
    text(lx + 0.4, -0.65, "Bar fill = stage load vs its limit (red = over limit)", 10, "#666", "left")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers", marker=dict(size=26, opacity=0), hovertext=ht, hoverinfo="text", showlegend=False))
    day = int(st.attrs["t"] // cfg.org.hours_per_day)
    clock = st.attrs["t"] - day * cfg.org.hours_per_day
    fig.update_layout(shapes=shapes, annotations=ann, height=height, margin=dict(l=5, r=5, t=40, b=5), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", title=dict(text=("Production network — end of week " + str(cfg.horizon_weeks)) if st.attrs["t"] >= res.t_end - 1e-6 else
                                                      f"Production network — week {day // cfg.org.days_per_week + 1}, day {day % cfg.org.days_per_week + 1}, hour {clock:.0f}",
                                                 x=0, font=dict(size=15)))
    fig.update_xaxes(visible=False, range=[0, cols * W], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-1.2, rows_n * Hc], scaleanchor=None, fixedrange=True)
    return fig
