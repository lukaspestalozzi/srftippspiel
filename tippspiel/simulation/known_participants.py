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

from dataclasses import dataclass, replace

import numpy as np

from ..model.types import Match, Result, TeamRef
from .bracket import Bracket
from .standings import h2h_table
from .thirds import select_best_thirds


@dataclass(frozen=True)
class TeamStanding:
    """One team's row in a group table, computed from the matches played so far."""

    team_id: str
    rank: int  # 1-based position within the group (display order)
    played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    goal_diff: int
    points: int
    # True only when the group is complete *and* this rank is strictly settled by the derivable
    # FIFA criteria (points -> GD -> GF -> head-to-head) — i.e. not in a tie that would need
    # fair-play/lots. The knockout resolver fills a slot only from a ``placing_certain`` row.
    placing_certain: bool


@dataclass(frozen=True)
class GroupStanding:
    letter: str
    complete: bool  # every group match has a recorded result
    rows: tuple[TeamStanding, ...]  # ordered best (rank 1) to worst


def _rank_and_flag(
    points: list[int],
    gd: list[int],
    gf: list[int],
    sub_layout: list[tuple[int, int]],
    home_row: list[int],
    away_row: list[int],
    glob: list[str],
    complete: bool,
) -> list[tuple[int, bool]]:
    """Order a group's teams by FIFA criteria points -> GD -> GF -> head-to-head, breaking any
    residual tie by ``team_id`` for a stable total order.

    Returns ``(local_idx, placing_certain)`` per rank. ``placing_certain`` is True only when the
    group is complete and the position is strictly separated (singleton on primary, then on
    head-to-head); a surviving tie is ordered by ``team_id`` and flagged not-certain.
    """

    def primary(t: int) -> tuple[int, int, int]:
        return (points[t], gd[t], gf[t])

    nteams = len(points)
    by_primary = sorted(range(nteams), key=primary, reverse=True)
    out: list[tuple[int, bool]] = []
    i = 0
    while i < nteams:
        j = i
        while j < nteams and primary(by_primary[j]) == primary(by_primary[i]):
            j += 1
        block = by_primary[i:j]
        if len(block) == 1:
            out.append((block[0], complete))
        else:
            # Criterion 4: head-to-head among the tied teams only.
            h2h = h2h_table(block, home_row, away_row, sub_layout)
            ranked = sorted(block, key=lambda t: h2h[t], reverse=True)
            p = 0
            while p < len(ranked):
                q = p
                while q < len(ranked) and h2h[ranked[q]] == h2h[ranked[p]]:
                    q += 1
                sub = ranked[p:q]
                if len(sub) == 1:
                    out.append((sub[0], complete))
                else:  # genuine tie (needs fair-play/lots): order by team_id, not certain
                    out.extend((t, False) for t in sorted(sub, key=lambda t: glob[t]))
                p = q
        i = j
    return out


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


def compute_group_standings(
    fixtures: list[Match], results: dict[str, Result]
) -> list[GroupStanding]:
    """Current group tables from the matches played so far — one ``GroupStanding`` per group,
    sorted by group letter, each carrying its teams ordered by the FIFA tiebreakers.

    This is the shared calculation step: the report renders these tables and the knockout resolver
    reads the certain placings off them (see ``_placings_from_standings``).
    """
    layouts = _group_layouts(fixtures)
    standings: list[GroupStanding] = []
    for letter in sorted(layouts):
        info = layouts[letter]
        ms, layout, glob = info["matches"], info["layout"], info["global"]
        nteams = len(glob)
        played = [0] * nteams
        wins = [0] * nteams
        draws = [0] * nteams
        losses = [0] * nteams
        gf = [0] * nteams
        ga = [0] * nteams
        points = [0] * nteams
        sub_layout: list[tuple[int, int]] = []
        home_row: list[int] = []
        away_row: list[int] = []
        for j, m in enumerate(ms):
            if m.match_id not in results:
                continue
            r = results[m.match_id]
            h, a = layout[j]
            hg, ag = r.home_goals, r.away_goals
            sub_layout.append((h, a))
            home_row.append(hg)
            away_row.append(ag)
            played[h] += 1
            played[a] += 1
            gf[h] += hg
            ga[h] += ag
            gf[a] += ag
            ga[a] += hg
            if hg > ag:
                wins[h] += 1
                losses[a] += 1
                points[h] += 3
            elif hg < ag:
                wins[a] += 1
                losses[h] += 1
                points[a] += 3
            else:
                draws[h] += 1
                draws[a] += 1
                points[h] += 1
                points[a] += 1
        gd = [gf[t] - ga[t] for t in range(nteams)]
        complete = all(m.match_id in results for m in ms)
        order = _rank_and_flag(points, gd, gf, sub_layout, home_row, away_row, glob, complete)
        rows = tuple(
            TeamStanding(
                team_id=glob[t], rank=rank, played=played[t],
                wins=wins[t], draws=draws[t], losses=losses[t],
                goals_for=gf[t], goals_against=ga[t], goal_diff=gd[t],
                points=points[t], placing_certain=certain,
            )
            for rank, (t, certain) in enumerate(order, start=1)
        )
        standings.append(GroupStanding(letter=letter, complete=complete, rows=rows))
    return standings


def _placings_from_standings(
    standings: list[GroupStanding],
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, tuple[int, int, int]]]:
    """Certain group placings for the knockout resolver, read off the standings tables.

    Returns (winner, runner_up, third) mapping group letter -> team_id (only where the placing is
    certain), plus ``third_stats`` (pts, gd, gf) for each group whose third place is itself certain.
    """
    winner: dict[str, str] = {}
    runner: dict[str, str] = {}
    third: dict[str, str] = {}
    third_stats: dict[str, tuple[int, int, int]] = {}
    for g in standings:
        rows = g.rows
        if len(rows) > 0 and rows[0].placing_certain:
            winner[g.letter] = rows[0].team_id
        if len(rows) > 1 and rows[1].placing_certain:
            runner[g.letter] = rows[1].team_id
        if len(rows) > 2 and rows[2].placing_certain:
            third[g.letter] = rows[2].team_id
            third_stats[g.letter] = (rows[2].points, rows[2].goal_diff, rows[2].goals_for)
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
    standings: list[GroupStanding] | None = None,
) -> list[Match]:
    """Return a copy of ``fixtures`` with knockout references replaced by concrete teams wherever
    the played results already determine them. Group fixtures and still-open slots are untouched.

    ``standings`` may be supplied (from ``compute_group_standings``) to avoid recomputing the group
    tables; when omitted they are computed here. The certain placings drive which slots are filled.
    """
    ko_matches = [m for m in fixtures if m.group is None]
    # A completed tournament lists concrete knockout participants (no references), so there is
    # nothing to resolve — and a Bracket cannot be built from such a fixed bracket. Short-circuit.
    if not any(side.ko_ref for m in ko_matches for side in (m.home, m.away)):
        return list(fixtures)

    if standings is None:
        standings = compute_group_standings(fixtures, results)
    winner, runner, third, third_stats = _placings_from_standings(standings)
    group_letters = sorted({m.group for m in fixtures if m.group})

    third_slot_team: dict[int, str] = {}
    has_third_ref = any(
        side.ko_ref is not None and side.ko_ref.kind == "third_pooled"
        for m in ko_matches for side in (m.home, m.away)
    )
    if group_letters and has_third_ref:
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
