"""File-based DataProvider (spec §6.1.2–6.1.5) plus the R32 bracket map loader.

Reads teams.csv, fixtures.csv, results.csv and r32_bracket_map.json. Matches present in
results.csv are treated as played and fixed; matches absent are predicted/simulated.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from ..model.stages import Stage
from ..model.types import Match, Result, Team, TeamRef
from .base import DataProvider


def _parse_kickoff(raw: str) -> datetime:
    s = raw.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class FileDataProvider(DataProvider):
    def __init__(
        self,
        teams_file: str | Path,
        fixtures_file: str | Path,
        results_file: str | Path,
        bracket_map_file: str | Path | None = None,
    ) -> None:
        self.teams_file = Path(teams_file)
        self.fixtures_file = Path(fixtures_file)
        self.results_file = Path(results_file)
        self.bracket_map_file = Path(bracket_map_file) if bracket_map_file else None

    def get_teams(self) -> list[Team]:
        teams: list[Team] = []
        with self.teams_file.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if not row.get("team_id"):
                    continue
                trend = row.get("elo_trend", "").strip()
                teams.append(
                    Team(
                        team_id=row["team_id"].strip(),
                        name=row["name"].strip(),
                        elo=float(row["elo"]),
                        elo_trend=float(trend) if trend else None,
                    )
                )
        return teams

    def get_fixtures(self) -> list[Match]:
        fixtures: list[Match] = []
        with self.fixtures_file.open(newline="") as fh:
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
        with self.results_file.open(newline="") as fh:
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

    def get_bracket_map(self) -> dict:
        if not self.bracket_map_file or not self.bracket_map_file.exists():
            raise FileNotFoundError("bracket_map_file is required for simulation")
        return json.loads(self.bracket_map_file.read_text())
