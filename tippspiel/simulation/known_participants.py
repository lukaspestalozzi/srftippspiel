"""Resolve knockout participants that are already *determined* by the played results.

``fixtures.csv`` keeps knockout slots as structured references (``W:A`` / ``R:A`` /
``3RD:74:ABCDF`` / ``WIN:M101`` / ``LOSE:M101``) — the format's single source of truth. The
Monte-Carlo simulator resolves those references itself every iteration. This module does the
*deterministic* counterpart for the **report / tip** path: given the results played so far, it
rewrites a knockout slot to a concrete team **only when the outcome is already certain**, leaving
every still-open slot as its reference.

Why a separate, deterministic pass (rather than reading it off the simulation): the predict / tip /
report path must work without a simulation (``diagnose --no-sim``), and a slot is filled here only
when it is *unambiguous* — so a filled team always equals the team the simulator yields with
probability 1.0. Genuine ties that FIFA would settle by fair-play points or drawing of lots (which
cannot be derived from goals, and which the simulator randomises) are deliberately left open.

The simulator is untouched: it keeps operating on the raw reference fixtures.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..model.types import Match, Result, TeamRef
from .bracket import Bracket
from .standings import _h2h_table, compute_stats
from .thirds import select_best_thirds


def _determined_placings(
    pts_row: np.ndarray,
    gd_row: np.ndarray,
    gf_row: np.ndarray,
    home_row: np.ndarray,
    away_row: np.ndarray,
    layout: list[tuple[int, int]],
    nteams: int,
) -> list[int | None]:
    """Rank a completed group by FIFA criteria points -> GD -> GF -> head-to-head.

    Returns a list of length ``nteams``: the local team index at each rank (0 = winner), or
    ``None`` where a tie survives all derivable criteria (so the placing is *not* determined).
    """

    def primary(t: int) -> tuple[int, int, int]:
        return (int(pts_row[t]), int(gd_row[t]), int(gf_row[t]))

    by_primary = sorted(range(nteams), key=primary, reverse=True)
    result: list[int | None] = [None] * nteams
    i = 0
    while i < nteams:
        j = i
        while j < nteams and primary(by_primary[j]) == primary(by_primary[i]):
            j += 1
        block = by_primary[i:j]
        if len(block) == 1:
            result[i] = block[0]
        else:
            # Criterion 4: head-to-head among the tied teams only.
            h2h = _h2h_table(block, home_row, away_row, layout)
            ranked = sorted(block, key=lambda t: h2h[t], reverse=True)
            p = 0
            while p < len(ranked):
                q = p
                while q < len(ranked) and h2h[ranked[q]] == h2h[ranked[p]]:
                    q += 1
                # A singleton sub-block is determined; a surviving tie stays None.
                if q - p == 1:
                    result[i + p] = ranked[p]
                p = q
        i = j
    return result


def _group_layouts(fixtures: list[Match]) -> dict[str, dict]:
    """Per-group ordered matches, local<->global team maps and the match layout."""
    groups: dict[str, list[Match]] = {}
    for m in fixtures:
        if m.group:
            groups.setdefault(m.group, []).append(m)
    layouts: dict[str, dict] = {}
    for letter, ms in groups.items():
        ms = sorted(ms, key=lambda m: m.match_id)
        local_ids = sorted({m.home.team_id for m in ms} | {m.away.team_id for m in ms})
        local = {tid: i for i, tid in enumerate(local_ids)}
        layout = [(local[m.home.team_id], local[m.away.team_id]) for m in ms]
        layouts[letter] = {"matches": ms, "layout": layout, "global": local_ids}
    return layouts


def _resolve_groups(
    fixtures: list[Match], results: dict[str, Result]
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, tuple[int, int, int]]]:
    """Determined group placings from the played results.

    Returns (winner, runner_up, third) mapping group letter -> team_id (only where determined),
    plus ``third_stats`` mapping group letter -> the third-placed team's (pts, gd, gf) — present
    only for a group whose third place is itself determined.
    """
    winner: dict[str, str] = {}
    runner: dict[str, str] = {}
    third: dict[str, str] = {}
    third_stats: dict[str, tuple[int, int, int]] = {}
    for letter, info in _group_layouts(fixtures).items():
        ms, layout, glob = info["matches"], info["layout"], info["global"]
        if not all(m.match_id in results for m in ms):
            continue  # group still in progress
        hg = np.array([[results[m.match_id].home_goals for m in ms]], dtype=np.int64)
        ag = np.array([[results[m.match_id].away_goals for m in ms]], dtype=np.int64)
        nteams = len(glob)
        pts, gd, gf = compute_stats(hg, ag, layout, nteams)
        order = _determined_placings(pts[0], gd[0], gf[0], hg[0], ag[0], layout, nteams)
        if order[0] is not None:
            winner[letter] = glob[order[0]]
        if order[1] is not None:
            runner[letter] = glob[order[1]]
        if len(order) > 2 and order[2] is not None:
            t = order[2]
            third[letter] = glob[t]
            third_stats[letter] = (int(pts[0, t]), int(gd[0, t]), int(gf[0, t]))
    return winner, runner, third, third_stats


def _resolve_third_slots(
    ko_matches: list[Match],
    group_letters: list[str],
    thirds_allocation: dict | None,
    third: dict[str, str],
    third_stats: dict[str, tuple[int, int, int]],
) -> dict[int, str]:
    """Map each ``3RD`` slot to a concrete team, or {} if the best-thirds picture is not yet
    certain (any group's third undetermined, or the 8th vs 9th third tied on pts/GD/GF)."""
    bracket = Bracket(ko_matches, group_letters, thirds_allocation)
    if not bracket.third_slots:
        return {}
    # Need every group's third determined to rank them.
    if any(g not in third_stats for g in bracket.group_letters):
        return {}
    k = len(bracket.third_slots)
    pts = np.array([[third_stats[g][0] for g in bracket.group_letters]], dtype=float)
    gd = np.array([[third_stats[g][1] for g in bracket.group_letters]], dtype=float)
    gf = np.array([[third_stats[g][2] for g in bracket.group_letters]], dtype=float)
    rand = np.zeros_like(pts)
    qualified, order = select_best_thirds(pts, gd, gf, rand, k=k)
    # Ambiguous if the qualifying boundary (k-th vs (k+1)-th best) is an exact pts/GD/GF tie.
    if order.shape[1] > k:
        a, b = order[0, k - 1], order[0, k]
        if (pts[0, a], gd[0, a], gf[0, a]) == (pts[0, b], gd[0, b], gf[0, b]):
            return {}
    slot_group_idx, _ = bracket.assign_thirds(qualified)
    out: dict[int, str] = {}
    for pos, slot in enumerate(bracket.third_slots):
        letter = bracket.group_letters[int(slot_group_idx[0, pos])]
        out[slot] = third[letter]
    return out


def _match_num(match_id: str) -> int:
    digits = "".join(c for c in match_id if c.isdigit())
    return int(digits) if digits else 0


def resolve_known_participants(
    fixtures: list[Match],
    results: dict[str, Result],
    thirds_allocation: dict | None = None,
) -> list[Match]:
    """Return a copy of ``fixtures`` with knockout references replaced by concrete teams wherever
    the played results already determine them. Group fixtures and still-open slots are untouched."""
    winner, runner, third, third_stats = _resolve_groups(fixtures, results)
    ko_matches = [m for m in fixtures if m.group is None]
    group_letters = sorted({m.group for m in fixtures if m.group})

    third_slot_team: dict[int, str] = {}
    if ko_matches and group_letters:
        third_slot_team = _resolve_third_slots(
            ko_matches, group_letters, thirds_allocation, third, third_stats
        )

    # Forward pass over knockout matches (ascending id) so a WIN/LOSE reference can read the
    # already-resolved winner/loser of an earlier match.
    resolved_side: dict[str, tuple[str | None, str | None]] = {}
    match_winner: dict[str, str] = {}
    match_loser: dict[str, str] = {}

    def resolve_ref(ref: TeamRef) -> str | None:
        if ref.is_concrete:
            return ref.team_id
        r = ref.ko_ref
        if r.kind == "winner":
            return winner.get(r.group)
        if r.kind == "runner_up":
            return runner.get(r.group)
        if r.kind == "third_pooled":
            return third_slot_team.get(r.slot)
        if r.kind == "winner_of":
            return match_winner.get(r.match_id)
        if r.kind == "loser_of":
            return match_loser.get(r.match_id)
        return None

    for m in sorted(ko_matches, key=lambda m: _match_num(m.match_id)):
        home_id = resolve_ref(m.home)
        away_id = resolve_ref(m.away)
        resolved_side[m.match_id] = (home_id, away_id)
        r = results.get(m.match_id)
        if r is None or home_id is None or away_id is None:
            continue
        if r.winner_team_id:
            win_id = r.winner_team_id
            lose_id = away_id if win_id == home_id else home_id
        elif r.home_goals > r.away_goals:
            win_id, lose_id = home_id, away_id
        elif r.home_goals < r.away_goals:
            win_id, lose_id = away_id, home_id
        else:
            continue  # 120-minute draw with no recorded shootout winner
        match_winner[m.match_id], match_loser[m.match_id] = win_id, lose_id

    resolved: list[Match] = []
    for m in fixtures:
        if m.group is not None:
            resolved.append(m)
            continue
        home_id, away_id = resolved_side[m.match_id]
        home = TeamRef(team_id=home_id) if home_id is not None else m.home
        away = TeamRef(team_id=away_id) if away_id is not None else m.away
        resolved.append(replace(m, home=home, away=away))
    return resolved
