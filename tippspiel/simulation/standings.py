"""Group standings with FIFA tiebreakers (spec §3.2), vectorised across MC iterations.

Tiebreakers applied in exact order:
  1. points  2. goal difference  3. goals scored
  4. head-to-head among the tied teams only: (a) points (b) GD (c) goals
  5. remaining FIFA criteria (disciplinary / drawing of lots) — approximated by
     ``break_remaining_ties``, a deterministic seeded random tiebreak.

Criteria 1-3 are vectorised via a composite sort key. Criterion 4 (head-to-head) and 5
are applied per-iteration only to the rows that still have an exact (pts, gd, gf) tie,
keeping the common path fully vectorised.
"""

from __future__ import annotations

import numpy as np

# Layout = ordered list of (home_local_idx, away_local_idx) for the group's 6 matches.
Layout = list[tuple[int, int]]

_GD_OFFSET = 1000  # gd is within +-(3*gmax); offset keeps the composite key positive.


def compute_stats(
    home_goals: np.ndarray, away_goals: np.ndarray, layout: Layout, nteams: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (points, goal_difference, goals_for), each shape [N, nteams]."""
    n = home_goals.shape[0]
    pts = np.zeros((n, nteams))
    gf = np.zeros((n, nteams))
    ga = np.zeros((n, nteams))
    for m, (h, a) in enumerate(layout):
        hg = home_goals[:, m]
        ag = away_goals[:, m]
        home_win = hg > ag
        draw = hg == ag
        pts[:, h] += np.where(home_win, 3, np.where(draw, 1, 0))
        pts[:, a] += np.where(hg < ag, 3, np.where(draw, 1, 0))
        gf[:, h] += hg
        ga[:, h] += ag
        gf[:, a] += ag
        ga[:, a] += hg
    return pts, gf - ga, gf


def _composite_key(pts: np.ndarray, gd: np.ndarray, gf: np.ndarray) -> np.ndarray:
    """Single sortable key encoding (points, gd, gf) lexicographically (descending)."""
    return pts * 1e7 + (gd + _GD_OFFSET) * 1e3 + gf


def break_remaining_ties(tied: list[int], rand_keys: np.ndarray) -> list[int]:
    """Criterion 5 approximation (spec §3.2): deterministic seeded random tiebreak.

    APPROXIMATION — FIFA's real criterion 5 is disciplinary/fair-play points then a
    drawing of lots, which we cannot reproduce from goal data. We substitute a stable
    ordering by a per-team random key supplied by the seeded simulator RNG. Isolated and
    named so it can be replaced with the real fair-play rule later.
    """
    return sorted(tied, key=lambda t: rand_keys[t])


def h2h_table(
    subset: list[int], home_row: np.ndarray, away_row: np.ndarray, layout: Layout
) -> dict[int, tuple[int, int, int]]:
    """Head-to-head (pts, gd, gf) for one iteration, over matches among `subset` only.

    Public helper: the vectorised standings path (``_order_block``) and the deterministic report
    resolver (``known_participants``) both rank tied teams through this same head-to-head table."""
    s = set(subset)
    pts = {t: 0 for t in subset}
    gf = {t: 0 for t in subset}
    ga = {t: 0 for t in subset}
    for m, (h, a) in enumerate(layout):
        if h in s and a in s:
            hg, ag = int(home_row[m]), int(away_row[m])
            gf[h] += hg
            ga[h] += ag
            gf[a] += ag
            ga[a] += hg
            if hg > ag:
                pts[h] += 3
            elif hg < ag:
                pts[a] += 3
            else:
                pts[h] += 1
                pts[a] += 1
    return {t: (pts[t], gf[t] - ga[t], gf[t]) for t in subset}


def _resolve_tied_row(
    order_local: list[int],
    pts_row: np.ndarray,
    gd_row: np.ndarray,
    gf_row: np.ndarray,
    home_row: np.ndarray,
    away_row: np.ndarray,
    layout: Layout,
    rand_row: np.ndarray,
) -> list[int]:
    """Re-order one iteration's teams, applying H2H (crit 4) then random (crit 5) to any
    teams sharing the same (pts, gd, gf)."""

    def primary(t: int) -> tuple[int, int, int]:
        return (int(pts_row[t]), int(gd_row[t]), int(gf_row[t]))

    # Group teams by identical primary key, preserving descending primary order.
    by_primary = sorted(order_local, key=primary, reverse=True)
    final: list[int] = []
    i = 0
    while i < len(by_primary):
        j = i
        while j < len(by_primary) and primary(by_primary[j]) == primary(by_primary[i]):
            j += 1
        block = by_primary[i:j]
        if len(block) == 1:
            final.extend(block)
        else:
            final.extend(_order_block(block, home_row, away_row, layout, rand_row))
        i = j
    return final


def _order_block(block, home_row, away_row, layout, rand_row) -> list[int]:
    """Order a set of teams tied on (pts, gd, gf): head-to-head, then random."""
    h2h = h2h_table(block, home_row, away_row, layout)
    ranked = sorted(block, key=lambda t: h2h[t], reverse=True)
    # Resolve any sub-groups still tied on H2H with the random tiebreak.
    out: list[int] = []
    i = 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and h2h[ranked[j]] == h2h[ranked[i]]:
            j += 1
        sub = ranked[i:j]
        out.extend(sub if len(sub) == 1 else break_remaining_ties(sub, rand_row))
        i = j
    return out


def rank_group(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    layout: Layout,
    rand: np.ndarray,
    nteams: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (order, points, gd, gf).

    ``order`` has shape [N, nteams]: local team indices ordered best (col 0) to worst.
    """
    pts, gd, gf = compute_stats(home_goals, away_goals, layout, nteams)
    key = _composite_key(pts, gd, gf)
    order = np.argsort(-key, axis=1, kind="stable")

    # Rows with an exact (pts, gd, gf) tie need head-to-head / random resolution.
    sorted_key = np.sort(key, axis=1)
    tie_rows = np.where(np.any(np.diff(sorted_key, axis=1) == 0, axis=1))[0]
    for r in tie_rows:
        order[r] = _resolve_tied_row(
            list(order[r]), pts[r], gd[r], gf[r],
            home_goals[r], away_goals[r], layout, rand[r],
        )
    return order, pts, gd, gf
