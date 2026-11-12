"""Plotly figure builders. Pure functions: data in, figure out (no Streamlit here).

Colour conventions (validated reference palette, categorical slots used in fixed order):
  blue = actual/simulated, orange = rework/overload, aqua = secondary series, grey = target/neutral.
  Status colours are reserved for green/amber/red state and utilisation bands.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GREY = "#8a8a85"
STATUS = {"green": "#2e9e5b", "amber": "#e0a100", "red": "#d64545"}
BAND_COLORS = {"Spare capacity": "#9ecae1", "Normal": "#2e9e5b", "High load": "#e0a100", "Critical load": "#d64545"}
SEQ = [[0, "#eaf2fc"], [0.5, "#6fa6e6"], [1, "#12417d"]]
HEAT = [[0, "#9ecae1"], [0.55, "#2e9e5b"], [0.78, "#e0a100"], [1.0, "#d64545"]]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _style(fig: go.Figure, title: str = "", h: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15)), height=h, margin=dict(l=10, r=10, t=48 if title else 16, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
        legend=dict(orientation="h", y=-0.18, x=0), hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(128,128,128,0.4)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


# --------------------------------------------------------------------------- executive
def cumulative_plan_vs_actual(daily: pd.DataFrame, target_weekly: float, d_w: int, D: int) -> go.Figure:
    """Cumulative accepted ads Mon..Fri: planned daily pitch vs simulated (mean of measured weeks, each week faint)."""
    d = daily[daily["day"] >= d_w].copy()
    d["wk_cum"] = d.groupby("week")["accepted"].cumsum()
    fig = go.Figure()
    labels = [WEEKDAYS[i] for i in range(D)]
    for wk, g in d.groupby("week"):
        fig.add_trace(go.Scatter(x=labels, y=g["wk_cum"], mode="lines", line=dict(color=BLUE, width=1), opacity=0.3,
                                 name=f"Week {wk+1}", showlegend=False, hovertemplate=f"Week {wk+1}: %{{y}}<extra></extra>"))
    mean = d.groupby("weekday")["wk_cum"].mean().reindex(range(D))
    plan = [target_weekly / D * (i + 1) for i in range(D)]
    fig.add_trace(go.Scatter(x=labels, y=plan, mode="lines+markers", name="Planned (daily pitch)",
                             line=dict(color=GREY, width=2, dash="dash"), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=labels, y=mean, mode="lines+markers", name="Simulated accepted (mean of weeks)",
                             line=dict(color=BLUE, width=3), marker=dict(size=9)))
    fig.update_yaxes(title="Cumulative accepted ads")
    return _style(fig, "Cumulative accepted ads: plan vs simulation")


def pod_target_vs_actual(pt: pd.DataFrame, status: Dict[int, str]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pt["name"], y=pt["target"], name="Target", marker_color="#c9c9c4"))
    fig.add_trace(go.Bar(x=pt["name"], y=pt["accepted"], name="Accepted (simulated)",
                         marker_color=[STATUS[status[p]] for p in pt["pod"]], text=[f"{a:.0f}%" for a in pt["attainment"] * 100],
                         textposition="outside"))
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(title="Ads in measured window")
    return _style(fig, "POD output vs target (bar colour = status)")


def pareto(df: pd.DataFrame, cat: str, val: str, title: str, unit: str = "hours") -> go.Figure:
    df = df.sort_values(val, ascending=False).reset_index(drop=True)
    tot = df[val].sum()
    cum = df[val].cumsum() / tot * 100 if tot > 0 else df[val] * 0
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[cat], y=df[val], name=unit, marker_color=BLUE, hovertemplate="%{x}<br>%{y:,.1f} " + unit + "<extra></extra>"))
    fig.add_trace(go.Scatter(x=df[cat], y=cum, name="Cumulative %", yaxis="y2", mode="lines+markers", line=dict(color=ORANGE, width=2)))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105], title="Cumulative %", showgrid=False))
    fig.update_yaxes(title=unit)
    fig.update_xaxes(tickangle=-30)
    return _style(fig, title, h=400)


# --------------------------------------------------------------------------- Mura
def pod_workload_heatmap(pw: pd.DataFrame) -> go.Figure:
    z = pw.pivot(index="pod", columns="week", values="load_ratio").sort_index()
    fig = go.Figure(go.Heatmap(z=z.values * 100, x=[f"Week {w+1}" for w in z.columns], y=[f"POD {p+1}" for p in z.index],
                               colorscale=HEAT, zmin=50, zmax=110, colorbar=dict(title="Load %"),
                               text=np.round(z.values * 100), texttemplate="%{text}", hovertemplate="%{y} %{x}<br>planned load / capacity: %{z:.0f}%<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "POD workload heatmap (planned standard hours incl. expected rework / capacity)", legend=False)


def editor_util_heatmap(ed_day: pd.DataFrame, title="Editor utilization heatmap (editor x day)") -> go.Figure:
    z = ed_day.pivot(index="worker", columns="day", values="util")
    z = z.reindex(sorted(z.index, key=lambda s: (int(s[1:s.index('-')]), s)))
    fig = go.Figure(go.Heatmap(z=z.values * 100, x=[f"D{d+1}" for d in z.columns], y=z.index, colorscale=HEAT, zmin=0, zmax=110,
                               colorbar=dict(title="Util %"), hovertemplate="%{y} %{x}: %{z:.0f}%<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, title, h=max(360, 18 * len(z) + 120), legend=False)


def daily_variability(daily: pd.DataFrame, d_w: int, planned: float) -> go.Figure:
    d = daily[daily["day"] >= d_w]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["day"] + 1, y=d["accepted"], name="Accepted", marker_color=BLUE))
    fig.add_trace(go.Scatter(x=d["day"] + 1, y=d["arrivals"], name="Arrivals", mode="lines+markers", line=dict(color=AQUA, width=2)))
    fig.add_hline(y=planned, line_dash="dash", line_color=GREY, annotation_text=f"pitch {planned:.0f}/day")
    fig.update_xaxes(title="Working day")
    fig.update_yaxes(title="Ads per day")
    return _style(fig, "Daily production variability")


def workload_box(ed_day: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for p, g in ed_day.groupby("pod"):
        fig.add_trace(go.Box(y=g["busy_h"], name=f"POD {p+1}", boxpoints="outliers", marker_color=BLUE, line=dict(width=1.5)))
    fig.update_yaxes(title="Editor busy hours per day")
    return _style(fig, "Workload distribution (editor-days by POD)", legend=False)


def demand_vs_capacity(res_daily: pd.DataFrame, demand_h: pd.Series, cap_h: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=demand_h.index + 1, y=demand_h.values, name="Arriving workload (std h + expected rework)", marker_color=ORANGE, opacity=0.8))
    fig.add_trace(go.Scatter(x=cap_h.index + 1, y=cap_h.values, name="Available editing capacity (std h)", mode="lines", line=dict(color=BLUE, width=3)))
    fig.update_xaxes(title="Working day")
    fig.update_yaxes(title="Hours")
    return _style(fig, "Demand vs available capacity (daily)")


def mura_components(table: pd.DataFrame) -> go.Figure:
    t = table.sort_values("score")
    fig = go.Figure(go.Bar(x=t["score"], y=t["component"], orientation="h", marker_color=BLUE,
                           text=[f"CV {c:.2f}" for c in t["cv"]], textposition="outside",
                           hovertemplate="%{y}<br>score %{x:.0f}/100<extra></extra>"))
    fig.update_xaxes(range=[0, 115], title="Component score (CV / reference CV x 100, capped)")
    return _style(fig, "Mura index components", legend=False, h=330)


# --------------------------------------------------------------------------- Muri
def worker_util_heatmap(wd: pd.DataFrame, role: str) -> go.Figure:
    g = wd[wd["role"] == role]
    z = g.pivot(index="worker", columns="day", values="util")
    fig = go.Figure(go.Heatmap(z=z.values * 100, x=[f"D{d+1}" for d in z.columns], y=z.index, colorscale=HEAT, zmin=0, zmax=110,
                               colorbar=dict(title="Util %"), hovertemplate="%{y} %{x}: %{z:.0f}%<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, f"{role.replace('_', ' ').title()} utilization by day", h=max(220, 22 * len(z) + 110), legend=False)


def capacity_vs_assigned(per_w: pd.DataFrame) -> go.Figure:
    t = per_w.sort_values("worker", key=lambda s: s.map(lambda x: (int(x[1:x.index('-')]) if '-' in x else 99, x)))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t["worker"], y=t["cap"], name="Available productive hours", marker_color="#c9c9c4"))
    fig.add_trace(go.Bar(x=t["worker"], y=t["busy"], name="Busy (worked) hours", marker_color=[BAND_COLORS[b] for b in t["band"]]))
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(title="Hours in measured window")
    return _style(fig, "Capacity vs assigned workload (bar colour = load band)")


def overload_distribution(wd: pd.DataFrame) -> go.Figure:
    order = list(BAND_COLORS)
    c = wd.groupby("band").size().reindex(order, fill_value=0)
    fig = go.Figure(go.Bar(x=c.index, y=c.values / max(c.sum(), 1) * 100, marker_color=[BAND_COLORS[b] for b in c.index],
                           text=[f"{v}" for v in c.values], textposition="outside"))
    fig.update_yaxes(title="% of resource-days")
    return _style(fig, "Resource overload distribution (all resources, resource-days per load band)", legend=False, h=330)


def util_histogram(wd: pd.DataFrame, cfg) -> go.Figure:
    fig = px.histogram(wd[wd["role"] == "editor"], x=wd[wd["role"] == "editor"]["util"] * 100, nbins=30, color_discrete_sequence=[BLUE])
    t = cfg.thresholds
    for v, lab in ((t.util_spare_below, "70%"), (t.util_normal_upto, "85%"), (t.util_high_upto, "95%")):
        fig.add_vline(x=v * 100, line_dash="dash", line_color=GREY, annotation_text=lab)
    fig.update_xaxes(title="Editor daily utilization %")
    return _style(fig, "Editor-day utilization distribution", legend=False, h=300)


def overtime_forecast(pw: pd.DataFrame) -> go.Figure:
    t = pw.groupby("pod")["ot_required_clock_h"].sum().reset_index()
    fig = go.Figure(go.Bar(x=[f"POD {p+1}" for p in t["pod"]], y=t["ot_required_clock_h"], marker_color=ORANGE,
                           hovertemplate="%{x}: %{y:.0f} overtime clock-hours needed<extra></extra>"))
    fig.update_yaxes(title="Overtime clock-hours needed (whole window)")
    return _style(fig, "Overtime forecast: hours needed beyond regular capacity", legend=False, h=320)


def bottleneck_ranking(table: pd.DataFrame) -> go.Figure:
    t = table.sort_values("score")
    fig = go.Figure(go.Bar(x=t["score"], y=t["candidate"], orientation="h", marker_color=[ORANGE if s == t["score"].max() else BLUE for s in t["score"]],
                           hovertext=t["evidence"], hovertemplate="%{y}<br>score %{x:.2f}<br>%{hovertext}<extra></extra>"))
    fig.update_xaxes(range=[0, 1], title="Evidence score (0-1)")
    return _style(fig, "Bottleneck ranking", legend=False, h=340)


# --------------------------------------------------------------------------- Muda
def lead_time_decomposition(dec: pd.DataFrame, by: Optional[pd.DataFrame] = None) -> go.Figure:
    pal = [BLUE, AQUA, VIOLET, ORANGE, YELLOW, "#9aa0a6", RED, MAGENTA, "#7b8794", "#c58b00", "#b5651d", "#5f6b7a", "#d48a8a", "#999"]
    fig = go.Figure()
    d = dec[dec["hours"] > 1e-9]
    for i, r in d.reset_index(drop=True).iterrows():
        fig.add_trace(go.Bar(y=["Mean lead time"], x=[r["hours"]], name=r["component"], orientation="h", marker_color=pal[i % len(pal)],
                             hovertemplate=f"{r['component']}: %{{x:.2f}} h<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Working hours per accepted ad")
    return _style(fig, "Lead-time decomposition (touch vs waiting)", h=250)


def rework_by_pod(pt: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=pt["name"], y=pt["rework_h"], marker_color=ORANGE, hovertemplate="%{x}: %{y:.1f} h<extra></extra>"))
    fig.update_yaxes(title="Rework editing + review hours")
    return _style(fig, "Rework hours by POD", legend=False, h=320)


def wip_trend(ps: pd.DataFrame) -> go.Figure:
    tot = ps.groupby("t")[["wip", "backlog"]].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tot["t"] / 8, y=tot["wip"], name="WIP (briefing started, not accepted)", line=dict(color=BLUE, width=2.5)))
    fig.add_trace(go.Scatter(x=tot["t"] / 8, y=tot["backlog"], name="Backlog (awaiting brief)", line=dict(color=ORANGE, width=2)))
    fig.update_xaxes(title="Working days")
    fig.update_yaxes(title="Ads")
    return _style(fig, "WIP trend over time")


def flow_eff_by_pod(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=df["name"], y=df["flow_eff"], marker_color=AQUA, text=[f"{v:.0f}%" for v in df["flow_eff"]], textposition="outside"))
    fig.update_yaxes(title="Flow efficiency %", rangemode="tozero")
    return _style(fig, "Flow efficiency by POD (value-adding time / lead time)", legend=False, h=320)


def capacity_loss_bar(df: pd.DataFrame) -> go.Figure:
    colors = [ORANGE if k == "loss" else "#b8b8b2" for k in df["kind"]]
    fig = go.Figure(go.Bar(x=df["cause"], y=df["hours"], marker_color=colors, hovertemplate="%{x}: %{y:.1f} editor-h<extra></extra>"))
    fig.update_xaxes(tickangle=-25)
    fig.update_yaxes(title="Editor productive hours")
    return _style(fig, "Editor capacity not used for editing (orange = avoidable loss, grey = demand-limited / necessary)", legend=False, h=380)


# --------------------------------------------------------------------------- flow
def littles_law_chart(series: pd.Series, L: float, lamW: float) -> go.Figure:
    fig = go.Figure(go.Scatter(x=series.index / 8, y=series.values, name="Ads in system", line=dict(color=BLUE, width=2)))
    fig.add_hline(y=L, line_color=BLUE, line_dash="dot", annotation_text=f"L observed {L:.1f}")
    fig.add_hline(y=lamW, line_color=ORANGE, line_dash="dash", annotation_text=f"lambda x W {lamW:.1f}", annotation_position="bottom right")
    fig.update_xaxes(title="Working days")
    fig.update_yaxes(title="Ads in system (arrived, not accepted)")
    return _style(fig, "Little's Law check", legend=False, h=330)


def lead_time_hist(dv: pd.DataFrame, sla_h: float) -> go.Figure:
    fig = px.histogram(dv, x="lead_time", nbins=40, color_discrete_sequence=[BLUE])
    fig.add_vline(x=sla_h, line_dash="dash", line_color=RED, annotation_text="SLA")
    fig.update_xaxes(title="Lead time (working hours)")
    return _style(fig, "Lead-time distribution", legend=False, h=320)


def editor_timeline(jobs: pd.DataFrame, pod: int, t0: float, t1: float) -> go.Figure:
    j = jobs[(jobs["edit_start"].notna()) & (jobs["first_editor"].fillna("").str.startswith(f"P{pod+1}-"))]
    j = j[(j["edit_start"] < t1) & (j["edit_start"] >= t0)]
    fig = go.Figure()
    cmap = {"Simple": AQUA, "Medium": BLUE, "Complex": VIOLET}
    for cx, g in j.groupby("complexity"):
        fig.add_trace(go.Bar(base=g["edit_start"], x=(g["edit_end"].fillna(t1) - g["edit_start"]), y=g["first_editor"], orientation="h",
                             name=cx, marker_color=cmap.get(cx, BLUE), hovertext=g["job_id"],
                             hovertemplate="%{hovertext}<br>start %{base:.1f}h<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Clock (working hours)")
    return _style(fig, f"POD {pod+1} editor timeline (first editing pass)", h=300)


# --------------------------------------------------------------------------- scenario / heijunka
def scenario_metric_chart(tab: pd.DataFrame, metric: str, names: Dict[str, str]) -> go.Figure:
    t = tab[tab["metric"] == metric]
    label = t["label"].iloc[0]
    fig = go.Figure(go.Bar(x=[names[s] for s in t["scenario"]], y=t["mean"], error_y=dict(type="data", array=t["ci95"].fillna(0), visible=True),
                           marker_color=[GREY if s == "A" else BLUE for s in t["scenario"]],
                           hovertemplate="%{x}<br>%{y:.2f} (95% CI half-width %{error_y.array:.2f})<extra></extra>"))
    fig.update_xaxes(tickangle=-20)
    return _style(fig, f"{label} (mean, 95% CI)", legend=False, h=330)


def scenario_delta_chart(tab: pd.DataFrame, names: Dict[str, str], metrics: List[str]) -> go.Figure:
    t = tab[(tab["metric"].isin(metrics)) & (tab["scenario"] != "A")]
    fig = go.Figure()
    for i, (s, g) in enumerate(t.groupby("scenario")):
        fig.add_trace(go.Bar(x=g["label"], y=g["delta_pct"], name=names[s], marker_color=[BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET][i % 6]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Change vs baseline (%)")
    fig.update_xaxes(tickangle=-20)
    return _style(fig, "Change versus baseline (%)", h=420)


def heijunka_load_chart(plans: Dict[str, pd.DataFrame], labels: Dict[str, str]) -> go.Figure:
    fig = go.Figure()
    cols = {"equal_count": ORANGE, "equal_hours": YELLOW, "capacity_aware": BLUE}
    for m, ed in plans.items():
        fig.add_trace(go.Bar(x=ed["editor"], y=ed["load_ratio"] * 100, name=labels[m], marker_color=cols[m]))
    fig.add_hline(y=100, line_dash="dash", line_color=RED, annotation_text="100% of capacity")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Load / capacity %")
    return _style(fig, "Load-to-capacity ratio per editor by allocation method", h=400)


def heijunka_plan_heatmap(ed: pd.DataFrame) -> go.Figure:
    cols = ["simple", "medium", "complex"]
    z = ed.set_index("editor")[cols]
    fig = go.Figure(go.Heatmap(z=z.values, x=["Simple", "Medium", "Complex"], y=z.index, colorscale=SEQ, text=z.values, texttemplate="%{text}",
                               colorbar=dict(title="Ads")))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Planned ads by editor and complexity", h=max(380, 18 * len(z) + 100), legend=False)


# --------------------------------------------------------------------------- simplified-app charts
CLASS_COLOR = {"VA": BLUE, "NNVA": "#9aa0a6", "WASTE": ORANGE}
CLASS_LABEL = {"VA": "Value-adding work", "NNVA": "Necessary, not value-adding (review, overhead)", "WASTE": "Waste (waiting, blocking, rework)"}


def lead_time_breakdown(dec: pd.DataFrame, cfg) -> go.Figure:
    """One row per component of the mean lead time, coloured by Lean class, labelled with hours and share."""
    cl = cfg.classification
    d = dec[dec["hours"] > 1e-6].copy()

    def cls(key):
        if key == "overhead":
            return cl.touch.get("overhead", "NNVA")
        return (cl.touch if key.startswith("T_") else cl.waits).get(key[2:], "WASTE")

    d["cls"] = d["key"].map(cls)
    total = d["hours"].sum()
    d = d.sort_values("hours")
    fig = go.Figure()
    for c in ("VA", "NNVA", "WASTE"):
        g = d[d["cls"] == c]
        if len(g):
            fig.add_trace(go.Bar(y=g["component"], x=g["hours"], orientation="h", name=CLASS_LABEL[c], marker_color=CLASS_COLOR[c],
                                 text=[f"{h:.2f} h · {h/total*100:.0f}%" for h in g["hours"]], textposition="outside", cliponaxis=False,
                                 hovertemplate="%{y}: %{x:.2f} h<extra></extra>"))
    fig.update_xaxes(title="Working hours per accepted ad", range=[0, d["hours"].max() * 1.35])
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=0))
    return _style(fig, f"Where an ad's {total:.1f} working hours go", h=max(380, 34 * len(d) + 140))


def pareto_horizontal(df: pd.DataFrame, title: str) -> go.Figure:
    """Largest cause at the top; bar label = hours and cumulative share (no overlapping legend or axis names)."""
    d = df.sort_values("hours", ascending=False).head(8).reset_index(drop=True)
    d["cum"] = d["hours"].cumsum() / max(df["hours"].sum(), 1e-9) * 100
    d = d.iloc[::-1]
    fig = go.Figure(go.Bar(y=d["cause"], x=d["hours"], orientation="h", marker_color=[ORANGE if i == len(d) - 1 else "#f0a98a" for i in range(len(d))],
                           text=[f"{h:,.0f} h · {c:.0f}% cum." for h, c in zip(d["hours"], d["cum"])], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:,.0f} job-hours<extra></extra>"))
    fig.update_xaxes(title="Job-hours of waste in the measured window", range=[0, d["hours"].max() * 1.7])
    return _style(fig, title + " (label: hours · cumulative share)", legend=False, h=max(360, 42 * len(d) + 110))


def load_heatmap(wd: pd.DataFrame, cfg, d_w: int) -> go.Figure:
    """Utilization by person and day with DISCRETE bands that match the configured thresholds."""
    t = cfg.thresholds
    g = wd[wd["role"].isin(["editor", "strategist"]) & (wd["day"] >= d_w) & (wd["capacity_h"] > 0)].copy()
    g["util"] = g["busy_h"] / g["capacity_h"]
    D = cfg.org.days_per_week
    order = sorted(g["worker"].unique(), key=lambda w: (g[g["worker"] == w]["pod"].iloc[0], 0 if w.startswith("S") else 1, w))
    z = g.pivot(index="worker", columns="day", values="util").reindex(order)
    cuts = [0, t.util_spare_below, t.util_normal_upto, t.util_high_upto, 1.2]
    cols = [BAND_COLORS["Spare capacity"], BAND_COLORS["Normal"], BAND_COLORS["High load"], BAND_COLORS["Critical load"]]
    scale, hi = [], cuts[-1]
    for i, c in enumerate(cols):
        scale += [[cuts[i] / hi, c], [cuts[i + 1] / hi, c]]
    labels = [f"W{(d // D) + 1} {WEEKDAYS[d % D]}" for d in z.columns]
    fig = go.Figure(go.Heatmap(z=z.values, x=labels, y=order, colorscale=scale, zmin=0, zmax=hi, showscale=False, xgap=1, ygap=1,
                               hovertemplate="%{y} · %{x}: %{z:.0%}<extra></extra>"))
    for lab, c in zip(("Spare < %d%%" % (t.util_spare_below * 100), "Normal %d-%d%%" % (t.util_spare_below * 100, t.util_normal_upto * 100),
                       "High %d-%d%%" % (t.util_normal_upto * 100, t.util_high_upto * 100), "Critical > %d%%" % (t.util_high_upto * 100)), cols):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=c, symbol="square"), name=lab))
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(tickangle=-60, type="category")
    fig.update_layout(legend=dict(orientation="h", y=1.06, x=0))
    return _style(fig, "Utilization by person and day (S = strategist, E = editor; grouped by POD)", h=max(520, 17 * len(order) + 160))
