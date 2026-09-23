"""Plotly-Abbildungen der Korrelierte-Gleichgewichte-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import kor_constants as C

LINE_COLOR = "#4c78a8"
REF_COLOR = "#7f7f7f"
GOOD = "#54a24b"
BAD = "#e45756"
WARN = "#f58518"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_range(ratios):
    """Für reine Nash-Gleichgewichte, korrelierte und grobe korrelierte Gleichgewichte: von der besten bis zur schlechtesten Summe der Wartezeiten (geteilt durch das Optimum)."""
    rows = [("Reines Nash-Gleichgewicht", ratios["ne_min"], ratios["ne_max"], LINE_COLOR), ("Korreliertes Gleichgewicht", ratios["ce_min"], ratios["ce_max"], GOOD),
            ("Grobes korreliertes Gleichgewicht", ratios["cce_min"], ratios["cce_max"], WARN)]
    fig = go.Figure()
    for name, lo, hi, color in rows:
        fig.add_trace(go.Scatter(x=[lo, hi], y=[name, name], mode="lines+markers+text", line=dict(color=color, width=6), marker=dict(size=12, color=color), text=[f"{lo:.3f}", f"{hi:.3f}"],
                                 textposition=["top left", "top right"], showlegend=False, hovertemplate="%{x:.4f}<extra></extra>"))
    fig.add_vline(x=1.0, line=dict(color=REF_COLOR, dash="dot"), annotation_text="Optimum", annotation_position="bottom right")
    hi_all = max(r[2] for r in rows)
    fig.update_xaxes(title_text="Summe der Wartezeiten / Optimum", range=[0.99, hi_all * 1.03])
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 260)


def build_recommendation(values, recommended, truck):
    """Erwartete Wartezeit von Lkw `truck`, dem Tor `recommended` empfohlen wurde, bei Gehorsam (grün) und bei festem Abweichen zu jedem anderen Tor."""
    m = len(values)
    colors = [GOOD if h == recommended else "#c9c9c9" for h in range(m)]
    fig = go.Figure(go.Bar(x=[f"Tor {h + 1}" + (" (Empfehlung)" if h == recommended else "") for h in range(m)], y=values, marker_color=colors, text=[f"{v:.2f}" for v in values], textposition="outside"))
    fig.update_yaxes(title_text="Erwartete Wartezeit (min)", range=[min(values) * 0.9, max(values) * 1.06])
    return _base(fig, 300)


def build_mediator_experiment(res_mixed, res_uniform):
    """Links: bestes reines Nash, bestes korreliertes und bestes grobes korreliertes Gleichgewicht; rechts: die jeweils schlechtesten. Mittel über feste Instanzen, geteilt durch das Optimum."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Beste Gleichgewichte", "Schlechteste Gleichgewichte"), horizontal_spacing=0.1)
    names_l = ("Nash rein", "Korreliert", "Grob korreliert")
    for name, res, color in (("gemischte Größen", res_mixed, LINE_COLOR), ("einheitliche Größen", res_uniform, GOOD)):
        fig.add_trace(go.Bar(x=names_l, y=[res["ne_min_mean"], res["ce_min_mean"], res["cce_min_mean"]], name=name, marker_color=color,
                             text=[f"{v:.3f}" for v in (res["ne_min_mean"], res["ce_min_mean"], res["cce_min_mean"])], textposition="outside"), row=1, col=1)
        fig.add_trace(go.Bar(x=names_l, y=[res["ne_max_mean"], res["ce_max_mean"], res["cce_max_mean"]], name=name, marker_color=color, showlegend=False,
                             text=[f"{v:.3f}" for v in (res["ne_max_mean"], res["ce_max_mean"], res["cce_max_mean"])], textposition="outside"), row=1, col=2)
    fig.update_yaxes(range=[1.0, 1.25], title_text="Summe der Wartezeiten / Optimum", row=1, col=1)
    fig.update_yaxes(range=[1.0, 1.25], row=1, col=2)
    fig.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_scaling(rows):
    """Wie weit über dem Optimum liegen das beste reine Nash-, das beste korrelierte und das beste grobe korrelierte Gleichgewicht bei wachsender Lkw-Zahl (Mittel, in Prozent)?"""
    xs = [r["n"] for r in rows]
    fig = go.Figure()
    for key, name, color in (("ne_min", "Bestes Nash (rein)", LINE_COLOR), ("ce_min", "Bestes korreliertes", GOOD), ("cce_min", "Bestes grobes korreliertes", WARN)):
        fig.add_trace(go.Scatter(x=xs, y=[100 * (r[key] - 1) for r in rows], mode="lines+markers", name=name, line=dict(color=color, width=2.5)))
    fig.update_xaxes(title_text="Lkw", dtick=1)
    fig.update_yaxes(title_text="über dem Optimum (%)")
    return _base(fig, 320)


def build_learning_gap(res):
    """Relativer CE-Gap der empirischen Verteilung über die Tage (Mittel über Instanzen, beide Achsen logarithmisch): Regret-Matching gegen Hedge."""
    fig = go.Figure()
    for method, color in (("rm", GOOD), ("hedge", LINE_COLOR)):
        fig.add_trace(go.Scatter(x=list(res["days"]), y=np.maximum(res[method]["ce_gap"], 1e-6), mode="lines+markers", name=C.LEARN_LABELS[method], line=dict(color=color, width=2.5)))
    fig.update_xaxes(title_text="Tage", type="log")
    fig.update_yaxes(title_text="CE-Lücke / bezahlte Wartezeit", type="log")
    return _base(fig, 320)


def build_learning_outcome(res):
    """Ausgang der letzten 100 Tage (Summe der Wartezeiten / Optimum) im Vergleich zu bestem/schlechtestem reinem Nash und bestem/schlechtestem korrelierten Gleichgewicht derselben Instanzen."""
    labels = ["Regret-Matching", "Hedge", "Bestes Nash", "Bestes CE"]
    values = [res["rm"]["window"], res["hedge"]["window"], res["ref_ne_min"], res["ref_ce_min"]]
    colors = [GOOD, LINE_COLOR, REF_COLOR, "#2ca02c"]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=[f"{v:.3f}" for v in values], textposition="outside"))
    fig.update_yaxes(title_text="Summe der Wartezeiten / Optimum", range=[1.0, max(values) * 1.03])
    return _base(fig, 320)


def build_bimatrix(costs):
    """Erwartete Wartezeiten der beiden Lkw im Mini-Spiel: reines Gleichgewicht (der Schlechtergestellte), gemischtes Gleichgewicht, gerechter Vermittler. `costs` = {Name: (Lkw 1, Lkw 2)}."""
    names = list(costs)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[costs[k][0] for k in names], name="Lkw 1", marker_color=LINE_COLOR, text=[f"{costs[k][0]:.2f}" for k in names], textposition="outside"))
    fig.add_trace(go.Bar(x=names, y=[costs[k][1] for k in names], name="Lkw 2", marker_color=WARN, text=[f"{costs[k][1]:.2f}" for k in names], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Erwartete Wartezeit (min)", range=[0, max(max(v) for v in costs.values()) * 1.15])
    return _base(fig, 300)
