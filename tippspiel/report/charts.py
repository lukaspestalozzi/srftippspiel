"""Plotly figure builders (spec §6.7.2). Each returns an HTML <div> fragment carrying the
figure as an **inert JSON payload**; the report template's lazy-render runtime calls
Plotly.newPlot on it the first time its section is revealed. Rendering eagerly instead
(plotly's to_html script tags) blocks page load for ~10s+ — the report embeds ~200 figures,
all inside closed <details>. plotly.js is inlined once by the html_writer. Every chart has
hover tooltips showing exact values.
"""

from __future__ import annotations

import plotly.graph_objects as go

from ..model.scoreline import ScorelineDistribution


def _fig_to_div(fig: go.Figure) -> str:
    """Wrap a figure as ``<div class="lazy-plot">`` + JSON for the template's lazy renderer.

    ``min-height`` reserves the figure's final height so revealing a section doesn't shift
    the layout when the charts pop in. ``</`` is escaped so a payload can never terminate
    its own <script> element."""
    height = int(fig.layout.height or 450)
    payload = fig.to_json().replace("</", "<\\/")
    return (
        f'<div class="lazy-plot" style="min-height:{height}px">'
        f'<script type="application/json">{payload}</script></div>'
    )


def ldw_bar(dist: ScorelineDistribution, home: str, away: str) -> str:
    """Single horizontal segmented bar: home-win / draw / away-win."""
    h, d, a = dist.p_home_win(), dist.p_draw(), dist.p_away_win()
    fig = go.Figure()
    segments = [
        (f"{home} win", h, "#2c7fb8"),
        ("Draw", d, "#999999"),
        (f"{away} win", a, "#de2d26"),
    ]
    for label, val, color in segments:
        fig.add_bar(
            y=["L/D/W"],
            x=[val],
            name=label,
            orientation="h",
            marker_color=color,
            hovertemplate=f"{label}: %{{x:.1%}}<extra></extra>",
            text=[f"{val:.0%}"],
            textposition="inside",
        )
    fig.update_layout(
        barmode="stack",
        height=110,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(orientation="h", y=-0.4),
        xaxis=dict(range=[0, 1], tickformat=".0%"),
    )
    return _fig_to_div(fig)


def scoreline_heatmap(
    dist: ScorelineDistribution, rec_home: int | None = None, rec_away: int | None = None
) -> str:
    """Interactive heatmap of the full (home goals × away goals) probability matrix.

    rows = home goals (y), cols = away goals (x). The recommended cell is marked.
    """
    m = dist.matrix
    gmax = dist.gmax
    axis = list(range(gmax + 1))
    fig = go.Figure(
        go.Heatmap(
            z=m,
            x=axis,
            y=axis,
            colorscale="YlGnBu",
            colorbar=dict(title="P", tickformat=".1%"),
            hovertemplate="Home %{y} : %{x} Away<br>P = %{z:.2%}<extra></extra>",
        )
    )
    if rec_home is not None and rec_away is not None:
        fig.add_shape(
            type="rect",
            x0=rec_away - 0.5,
            x1=rec_away + 0.5,
            y0=rec_home - 0.5,
            y1=rec_home + 0.5,
            line=dict(color="#e6550d", width=3),
        )
        fig.add_annotation(
            x=rec_away,
            y=rec_home,
            text="TIP",
            showarrow=False,
            font=dict(color="#e6550d", size=11),
            yshift=-16,
        )
    fig.update_layout(
        height=340,
        margin=dict(l=50, r=10, t=30, b=45),
        xaxis_title="Away goals",
        yaxis_title="Home goals",
        xaxis=dict(dtick=1),
        yaxis=dict(dtick=1, autorange="reversed"),
    )
    return _fig_to_div(fig)


def advancement_stacked_bar(group: str, rows: list[dict]) -> str:
    """Per-group stacked bar: P(win group)/P(2nd)/P(3rd)/P(eliminated) per team.

    rows: [{team, win, second, third, eliminated, se}], se = MC standard error.
    """
    teams = [r["team"] for r in rows]
    fig = go.Figure()
    layers = [
        ("Win group", "win", "#238b45"),
        ("2nd", "second", "#74c476"),
        ("3rd", "third", "#fdae6b"),
        ("Eliminated", "eliminated", "#cccccc"),
    ]
    for label, key, color in layers:
        vals = [r[key] for r in rows]
        ses = [r.get("se", 0.0) for r in rows]
        fig.add_bar(
            x=teams,
            y=vals,
            name=label,
            marker_color=color,
            customdata=ses,
            hovertemplate=f"%{{x}} — {label}: %{{y:.1%}} (±%{{customdata:.2%}})<extra></extra>",
        )
    fig.update_layout(
        barmode="stack",
        height=320,
        margin=dict(l=40, r=10, t=30, b=40),
        title=f"Group {group} advancement",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        legend=dict(orientation="h", y=-0.2),
    )
    return _fig_to_div(fig)


def title_odds_bar(rows: list[tuple[str, float]]) -> str:
    """Horizontal bar chart of the top teams by P(win title). rows = [(name, prob)]."""
    rows = sorted(rows, key=lambda r: r[1])
    names = [r[0] for r in rows]
    probs = [r[1] for r in rows]
    fig = go.Figure(
        go.Bar(
            x=probs,
            y=names,
            orientation="h",
            marker_color="#2c7fb8",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(300, 22 * len(rows)),
        margin=dict(l=80, r=10, t=30, b=40),
        title="Title odds (top teams)",
        xaxis=dict(tickformat=".0%"),
    )
    return _fig_to_div(fig)


def bracket_progression(rows: list[dict], rounds: list[str] | None = None) -> str:
    """Round-by-round reach probabilities for the leading teams (spec §6.7.2 item 6).

    rows: [{team, probs: [...]}], one line per team; ``rounds`` are the matching x-axis
    labels (defaults to the WC2026 R32->Final sequence). A clean static diagram of how far
    each contender is likely to advance.
    """
    if rounds is None:
        rounds = ["Qualify R32", "Reach R16", "Reach QF", "Reach SF", "Reach Final", "Champion"]
    fig = go.Figure()
    for r in rows:
        fig.add_trace(
            go.Scatter(
                x=rounds,
                y=r["probs"],
                mode="lines+markers",
                name=r["team"],
                hovertemplate=r["team"] + " — %{x}: %{y:.1%}<extra></extra>",
            )
        )
    fig.update_layout(
        height=460,
        margin=dict(l=50, r=10, t=30, b=40),
        title="Advancement by round (leading teams)",
        yaxis=dict(range=[0, 1], tickformat=".0%", title="probability"),
        legend=dict(orientation="v"),
    )
    return _fig_to_div(fig)


# Fixed-order categorical palette for the rating-history lines (CVD-validated ordering:
# worst adjacent-pair ΔE 24.2 under protanopia on the light surface). Slots are assigned to
# the highlighted teams in rank order and stay with the team across all three history charts,
# so the same side wears the same colour in the Elo, attack and defence plots.
_SERIES_COLORS = [
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
]
_CONTEXT_COLOR = "#898781"  # muted gray for the non-highlighted (legend-toggle) teams


def rating_history_lines(
    series: list[tuple[str, list[tuple[str, float]]]],
    *,
    title: str,
    ytitle: str,
    highlight: list[str],
    yfmt: str = ".0f",
) -> str:
    """Per-team rating trajectories over time (one line per team).

    ``series`` = ``[(team_name, [(iso_date, value), ...]), ...]``. Teams in ``highlight``
    (max 8 — the categorical-palette ceiling) are drawn in colour and visible by default;
    every other team starts as a gray legend-only trace the reader can toggle on, so the
    chart stays readable with a 48-team field."""
    slot = {name: i for i, name in enumerate(highlight)}
    fig = go.Figure()
    for name, points in series:
        if not points:
            continue
        i = slot.get(name)
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                mode="lines",
                name=name,
                line=dict(
                    width=2,
                    color=_SERIES_COLORS[i % len(_SERIES_COLORS)]
                    if i is not None
                    else _CONTEXT_COLOR,
                ),
                visible=True if i is not None else "legendonly",
                hovertemplate=name + " — %{x}: %{y:" + yfmt + "}<extra></extra>",
            )
        )
    fig.update_layout(
        height=440,
        margin=dict(l=50, r=10, t=40, b=40),
        title=title,
        yaxis_title=ytitle,
        legend=dict(orientation="v"),
    )
    return _fig_to_div(fig)


def bonus_candidates_bar(question: str, rows: list[tuple[str, float]]) -> str:
    rows = sorted(rows, key=lambda r: r[1])[-8:]
    fig = go.Figure(
        go.Bar(
            x=[r[1] for r in rows],
            y=[r[0] for r in rows],
            orientation="h",
            marker_color="#756bb1",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(200, 26 * len(rows)),
        margin=dict(l=80, r=10, t=30, b=30),
        title=question,
        xaxis=dict(tickformat=".0%"),
    )
    return _fig_to_div(fig)
