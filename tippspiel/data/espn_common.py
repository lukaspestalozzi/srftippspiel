"""Shared ESPN scoreboard helpers for the maintainer fetch tools.

Used by ``espn_odds_fetch.py`` and ``espn_results_fetch.py``: loading ``teams.csv``,
``fixtures.csv`` and ``results.csv``, fetching the public scoreboard JSON per match date, and
matching a repo fixture to its ESPN event by team identity. Offline, **not on the runtime path**.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.request
from pathlib import Path

import tippspiel

REPO = Path(tippspiel.__file__).parent.parent
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# ESPN display names that differ from the repo's ``teams.csv`` names. Extend as needed.
ALIASES = {
    "ir iran": "iran",
    "korea republic": "south korea",
    "usa": "united states",
    "côte d'ivoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "cabo verde": "cape verde",
    "congo dr": "dr congo",
    "bosnia & herzegovina": "bosnia and herzegovina",
    "bosnia-herzegovina": "bosnia and herzegovina",
    "turkey": "türkiye",
    "czech republic": "czechia",
}


def norm(name: str) -> str:
    return ALIASES.get(name.strip().lower(), name.strip().lower())


def get_json(url: str, *, retries: int = 4) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=25) as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001 — maintainer tool, best-effort with backoff
            if i == retries - 1:
                raise
            time.sleep(2**i)
    return {}


def load_teams(tdir: Path) -> dict[str, str]:
    """Normalised team name -> repo ``team_id``, from ``teams.csv``."""
    teams: dict[str, str] = {}
    with (tdir / "teams.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            teams[norm(row["name"])] = row["team_id"]
    return teams


def load_team_names(tdir: Path) -> dict[str, str]:
    """Repo ``team_id`` -> display ``name`` (the inverse of ``load_teams``), from ``teams.csv``."""
    with (tdir / "teams.csv").open(newline="", encoding="utf-8") as fh:
        return {row["team_id"]: row["name"] for row in csv.DictReader(fh)}


def load_concrete_fixtures(tdir: Path) -> list[dict]:
    """Fixtures with a real, dated two-team matchup -- skips structural KO refs (``W:A``, ...)."""
    fixtures = []
    with (tdir / "fixtures.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ko = row.get("kickoff_utc", "")
            home, away = row.get("home_ref", ""), row.get("away_ref", "")
            if ":" in home or ":" in away or "T" not in ko:
                continue
            fixtures.append({
                "match_id": row["match_id"],
                "stage": row.get("stage", ""),
                "date": ko[:10].replace("-", ""),
                "home_id": home,
                "away_id": away,
                "kickoff_utc": ko,
                "venue_country": (row.get("venue_country") or "").strip(),
            })
    return fixtures


def load_played_match_ids(tdir: Path) -> set[str]:
    """``match_id``s already recorded in ``results.csv`` (empty if the file doesn't exist)."""
    path = tdir / "results.csv"
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["match_id"] for row in csv.DictReader(fh)}


def fetch_scoreboard(slug: str, dates: list[str]) -> dict[str, list]:
    """ESPN scoreboard events, one fetch per ``YYYYMMDD`` date in ``dates``."""
    scoreboard: dict[str, list] = {}
    for d in dates:
        url = (f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}"
               f"/scoreboard?dates={d}")
        try:
            scoreboard[d] = get_json(url).get("events", [])
        except Exception:  # noqa: BLE001
            scoreboard[d] = []
    return scoreboard


def find_event(
    events: list[dict], teams: dict[str, str], home_id: str, away_id: str
) -> tuple[dict, dict[str, str]] | None:
    """The scoreboard event matching ``home_id``/``away_id`` by team identity, or ``None``.

    Returns ``(event, ids)`` where ``ids`` maps ESPN's ``"home"``/``"away"`` to repo team ids.
    Tolerates malformed/partial event entries -- one bad event is skipped, not fatal.
    """
    for e in events:
        try:
            comps = e.get("competitions") or []
            if not comps:
                continue
            ids: dict[str, str] = {}
            for c in comps[0].get("competitors", []):
                name = (c.get("team") or {}).get("displayName")
                if not name:
                    continue
                rid = teams.get(norm(name))
                if rid:
                    ids[c.get("homeAway")] = rid
            if {ids.get("home"), ids.get("away")} == {home_id, away_id}:
                return e, ids
        except Exception:  # noqa: BLE001 — one malformed event shouldn't abort the run
            continue
    return None
