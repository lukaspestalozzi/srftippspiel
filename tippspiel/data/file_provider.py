"""File-based DataProvider (spec §6.1.2–6.1.5).

Reads teams.csv, fixtures.csv, results.csv. The knockout bracket is derived from the
fixtures themselves (knockout fixtures reference group placings / earlier matches); the
only optional sidecar is a third-place combination->slot allocation table. Matches present
in results.csv are treated as played and fixed; matches absent are predicted/simulated.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from ..model.stages import Stage
from ..model.types import Match, Result, Team, TeamRef
from .base import DataProvider, Odds1X2


def _opt_float(raw: str | None) -> float:
    """Parse an optional float cell; blank/missing -> 0.0."""
    s = (raw or "").strip()
    return float(s) if s else 0.0


def _parse_kickoff(raw: str) -> datetime:
    s = raw.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _devig_proportional(odds_home: float, odds_draw: float, odds_away: float) -> Odds1X2:
    """De-vig decimal 1X2 odds into a normalised probability triple.

    Implied probabilities are ``1/odds``; their sum (the "booksum") exceeds 1 by the
    bookmaker's margin. The proportional method scales them back to sum to 1 — the standard,
    transparent default. ``method=`` is intentionally not exposed yet; Shin's method can be
    added here later without an ``odds.csv`` schema change.
    """
    imp_h, imp_d, imp_a = 1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away
    booksum = imp_h + imp_d + imp_a
    return Odds1X2(imp_h / booksum, imp_d / booksum, imp_a / booksum)


class FileDataProvider(DataProvider):
    def __init__(
        self,
        teams_file: str | Path,
        fixtures_file: str | Path,
        results_file: str | Path,
        thirds_allocation_file: str | Path | None = None,
        odds_file: str | Path | None = None,
    ) -> None:
        self.teams_file = Path(teams_file)
        self.fixtures_file = Path(fixtures_file)
        self.results_file = Path(results_file)
        self.thirds_allocation_file = (
            Path(thirds_allocation_file) if thirds_allocation_file else None
        )
        self.odds_file = Path(odds_file) if odds_file else None

    def get_teams(self) -> list[Team]:
        teams: list[Team] = []
        with self.teams_file.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if not row.get("team_id"):
                    continue
                # att_elo/def_elo are optional (added by `tippspiel fit-offdef`); absent or
                # blank -> 0.0, which leaves the predictor at its pure-Elo behaviour.
                teams.append(
                    Team(
                        team_id=row["team_id"].strip(),
                        name=row["name"].strip(),
                        elo=float(row["elo"]),
                        att_elo=_opt_float(row.get("att_elo")),
                        def_elo=_opt_float(row.get("def_elo")),
                    )
                )
        return teams

    def get_fixtures(self) -> list[Match]:
        fixtures: list[Match] = []
        with self.fixtures_file.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if not row.get("match_id"):
                    continue
                group = (row.get("group") or "").strip() or None
                venue = (row.get("venue_country") or "").strip() or None
                fixtures.append(
                    Match(
                        match_id=row["match_id"].strip(),
                        stage=Stage(row["stage"].strip()),
                        home=TeamRef.parse(row["home_ref"]),
                        away=TeamRef.parse(row["away_ref"]),
                        kickoff=_parse_kickoff(row["kickoff_utc"]),
                        group=group,
                        venue_country=venue,
                    )
                )
        return fixtures

    def get_results(self) -> list[Result]:
        if not self.results_file.exists():
            return []
        results: list[Result] = []
        with self.results_file.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if not row.get("match_id"):
                    continue
                winner = (row.get("winner_team_id") or "").strip() or None
                results.append(
                    Result(
                        match_id=row["match_id"].strip(),
                        home_goals=int(row["home_goals"]),
                        away_goals=int(row["away_goals"]),
                        winner_team_id=winner,
                    )
                )
        return results

    def get_thirds_allocation(self) -> dict:
        """Optional explicit third-place combination->slot table; {} if not supplied."""
        if not self.thirds_allocation_file or not self.thirds_allocation_file.exists():
            return {}
        return json.loads(self.thirds_allocation_file.read_text(encoding="utf-8"))

    def get_odds(self) -> dict[str, Odds1X2]:
        """Optional per-match de-vigged 1X2 odds keyed by match_id; {} if not supplied.

        Reads ``odds.csv`` (``match_id,odds_home,odds_draw,odds_away``, raw decimal odds —
        auditable, de-vigged at load). Rows are optional per match; a match absent here falls
        back to the Elo predictor in ``MarketOddsPredictor``.
        """
        if not self.odds_file or not self.odds_file.exists():
            return {}
        odds: dict[str, Odds1X2] = {}
        with self.odds_file.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                mid = (row.get("match_id") or "").strip()
                if not mid:
                    continue
                odds[mid] = _devig_proportional(
                    float(row["odds_home"]), float(row["odds_draw"]), float(row["odds_away"])
                )
        return odds
