"""Maintainer tool: diff ``teams.csv`` base elo against eloratings.net's current ratings.

Offline, **not on the runtime path** (mirrors ``espn_odds_fetch`` / ``eloratings_adapter``).
eloratings.net publishes its current rating table and a code->name lookup as plain
tab-separated files (``World.tsv``, ``en.teams.tsv``) fetchable with a browser ``User-Agent``
(the same trick as the ESPN feeds). This tool fetches both, maps eloratings' 2-letter codes to
this repo's ``team_id``s via ``teams.csv``, restricts to the teams that **played** (per
``results.csv``/``fixtures.csv``), and reports only the ones whose rating has moved.

Usage (run from the repo root, network required)::

    python -m tippspiel.data.eloratings_diff wc2026
"""

from __future__ import annotations

import csv
import sys
import time
import urllib.request
from pathlib import Path

from tippspiel.data.espn_common import (
    REPO,
    UA,
    load_concrete_fixtures,
    load_played_match_ids,
    load_teams,
    norm,
)


def _fetch_text(url: str, *, retries: int = 4) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=25) as fh:
                return fh.read().decode("utf-8")
        except Exception:  # noqa: BLE001 — maintainer tool, best-effort with backoff
            if i == retries - 1:
                raise
            time.sleep(2**i)
    return ""


def parse_world_tsv(text: str) -> dict[str, float]:
    """eloratings team code -> current rating, from ``World.tsv`` (no header)."""
    ratings: dict[str, float] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 4:
            continue
        ratings[parts[2]] = float(parts[3])
    return ratings


def parse_teams_tsv(text: str) -> dict[str, list[str]]:
    """eloratings team code -> ``[name, alias, ...]``, from ``en.teams.tsv`` (no header)."""
    names: dict[str, list[str]] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 2 or not parts[0]:
            continue
        names[parts[0]] = [p for p in parts[1:] if p]
    return names


def build_code_to_team_id(
    teams_tsv: dict[str, list[str]], name_to_team_id: dict[str, str]
) -> dict[str, str]:
    """eloratings code -> repo ``team_id``, by matching names.

    ``name_to_team_id`` is keyed by ``espn_common.norm()`` (as returned by ``load_teams``); the
    eloratings names are normalised with the same function so both sides of the lookup use the
    one alias table.
    """
    code_to_id = {}
    for code, names in teams_tsv.items():
        for name in names:
            team_id = name_to_team_id.get(norm(name))
            if team_id:
                code_to_id[code] = team_id
                break
    return code_to_id


def diff_ratings(
    tdir: Path, world_tsv_text: str, teams_tsv_text: str
) -> tuple[list[tuple[str, float, float]], list[str]]:
    """Elo movements for teams that **played**, per ``results.csv``.

    Returns ``(moved, unresolved)``: ``moved`` is ``[(team_id, old_elo, new_elo), ...]`` for
    played teams whose rating differs from the committed ``teams.csv`` value; ``unresolved`` is
    played ``team_id``s with no eloratings code mapping or no rating in ``World.tsv`` yet.
    """
    name_to_id = load_teams(tdir)

    fixtures = {f["match_id"]: f for f in load_concrete_fixtures(tdir)}
    played_teams: set[str] = set()
    for match_id in load_played_match_ids(tdir):
        fixture = fixtures.get(match_id)
        if fixture:
            played_teams.add(fixture["home_id"])
            played_teams.add(fixture["away_id"])

    current_elo: dict[str, float] = {}
    with (tdir / "teams.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            current_elo[row["team_id"]] = float(row["elo"])

    world = parse_world_tsv(world_tsv_text)
    code_to_id = build_code_to_team_id(parse_teams_tsv(teams_tsv_text), name_to_id)
    id_to_code = {team_id: code for code, team_id in code_to_id.items()}

    moved: list[tuple[str, float, float]] = []
    unresolved: list[str] = []
    for team_id in sorted(played_teams):
        code = id_to_code.get(team_id)
        new = world.get(code) if code else None
        old = current_elo.get(team_id)
        if new is None or old is None:
            unresolved.append(team_id)
        elif new != old:
            moved.append((team_id, old, new))
    return moved, unresolved


def main(tournament: str) -> None:
    tdir = REPO / "tippspiel" / "data" / "tournaments" / tournament
    if not tdir.is_dir():
        sys.exit(f"no such tournament: {tournament!r} ({tdir} does not exist)")
    world_text = _fetch_text("https://www.eloratings.net/World.tsv")
    teams_text = _fetch_text("https://www.eloratings.net/en.teams.tsv")
    moved, unresolved = diff_ratings(tdir, world_text, teams_text)
    for team_id, old, new in moved:
        print(f"{team_id} {old:.0f} -> {new:.0f}")
    if unresolved:
        print(f"unresolved (unchanged, unmapped, or not yet processed): {', '.join(unresolved)}",
              file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m tippspiel.data.eloratings_diff <tournament>")
    main(sys.argv[1])
