"""Resolve thin ``results.csv`` rows against the committed international-match corpus.

A tournament's results may carry the scoreline **inline** (``match_id,home_goals,away_goals,
winner_team_id`` — used by synthetic test fixtures and legacy data) or **reference** the corpus:
a thin row ``match_id,date,winner_team_id`` names which played match it is, and the scoreline is
read from ``international_results.csv`` by joining on the match date + the fixture's two teams.

This removes the duplication between each tournament's ``results.csv`` and the same matches sitting
in the corpus that feeds the Elo fit. The join is on the **unordered** team pair (the corpus may
assign home/away differently at a neutral venue); the corpus scoreline is then re-oriented to the
fixture's home/away. Resolution failures (no fixture, non-concrete participants, zero or multiple
corpus rows) raise loudly — for a row the maintainer asserted is played, a miss is a data error.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..model.types import Match, Result
from .historical_results_adapter import DEFAULT_CORPUS, corpus_name_for


class ResultResolutionError(ValueError):
    """A thin results row could not be resolved to exactly one corpus match."""


def build_corpus_index(
    corpus_path: str | Path = DEFAULT_CORPUS,
) -> dict[tuple[str, frozenset[str]], list[tuple[str, int, int]]]:
    """Index played corpus matches by ``(date, {home_name, away_name})`` for O(1) lookup.

    Each value is the list of ``(home_team_name, home_goals, away_goals)`` rows with that key
    (normally exactly one; >1 signals an ambiguous join the resolver rejects)."""
    index: dict[tuple[str, frozenset[str]], list[tuple[str, int, int]]] = {}
    with Path(corpus_path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            date = (row.get("date") or "").strip()
            hg, ag = (row.get("home_score") or "").strip(), (row.get("away_score") or "").strip()
            if not date or not hg or not ag or hg == "NA" or ag == "NA":
                continue
            home = (row.get("home_team") or "").strip()
            away = (row.get("away_team") or "").strip()
            index.setdefault((date, frozenset((home, away))), []).append((home, int(hg), int(ag)))
    return index


def resolve_corpus_result(
    match_id: str,
    date: str,
    winner_team_id: str | None,
    fixtures_by_id: dict[str, Match],
    name_by_id: dict[str, str],
    index: dict[tuple[str, frozenset[str]], list[tuple[str, int, int]]],
) -> Result:
    """Resolve one thin results row to a ``Result`` by joining against the corpus index."""
    fx = fixtures_by_id.get(match_id)
    if fx is None:
        raise ResultResolutionError(f"results row {match_id!r} has no matching fixture")
    if not (fx.home.is_concrete and fx.away.is_concrete):
        raise ResultResolutionError(
            f"results row {match_id!r}: fixture participants are not concrete team ids"
        )
    home_name = name_by_id.get(fx.home.team_id)
    away_name = name_by_id.get(fx.away.team_id)
    if home_name is None or away_name is None:
        raise ResultResolutionError(
            f"results row {match_id!r}: unknown team_id {fx.home.team_id}/{fx.away.team_id}"
        )
    home_corpus, away_corpus = corpus_name_for(home_name), corpus_name_for(away_name)
    rows = index.get((date, frozenset((home_corpus, away_corpus))))
    if not rows:
        raise ResultResolutionError(
            f"results row {match_id!r}: no corpus match on {date} for "
            f"{home_corpus} vs {away_corpus}"
        )
    if len(rows) > 1:
        raise ResultResolutionError(
            f"results row {match_id!r}: {len(rows)} corpus matches on {date} for "
            f"{home_corpus} vs {away_corpus} (ambiguous)"
        )
    corpus_home, hg, ag = rows[0]
    # Re-orient the corpus scoreline to the fixture's home/away.
    home_goals, away_goals = (hg, ag) if corpus_home == home_corpus else (ag, hg)
    return Result(
        match_id=match_id,
        home_goals=home_goals,
        away_goals=away_goals,
        winner_team_id=winner_team_id,
    )
