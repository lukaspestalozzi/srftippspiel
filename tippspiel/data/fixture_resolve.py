"""Shared helpers for the odds fetchers: tippable fixtures + the frozen-odds rule.

A fixture is tippable when it has two known teams and no recorded result yet. For the group
stage that is just the un-played group matches; for the knockout stage the participants are
stored in ``fixtures.csv`` as structured references (``W:A`` / ``R:B`` / ``3RD:74:…``), so they
are resolved to concrete teams with :func:`resolve_known_participants` — the same deterministic
pass the report/tip path uses — but **only where the played results already make them certain**.

This lets both odds fetchers (ESPN, Polymarket) price knockout matches as soon as the bracket is
settled, not just group matches. Offline maintainer helper, **not on the runtime path**.

:func:`frozen_match_ids` + :func:`write_odds_preserving_frozen` implement the **frozen-odds
rule**: once a match has kicked off (or is recorded in ``results.csv``), its committed odds row
is a historical pre-match snapshot — a refresh must neither re-price it (an in-play or post-match
price is not a pre-match odd) nor drop it from the file.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from .espn_common import REPO, load_played_match_ids
from .file_provider import FileDataProvider
from ..simulation.known_participants import (
    compute_group_standings,
    resolve_known_participants,
)

_ODDS_FIELDS = ["match_id", "odds_home", "odds_draw", "odds_away"]


def tournament_dir(tournament: str) -> Path:
    """The repo data directory for a tournament name (e.g. ``wc2026``)."""
    return REPO / "tippspiel" / "data" / "tournaments" / tournament


def load_tippable_fixtures(tdir: str | Path) -> list[dict]:
    """Concrete, un-played fixtures (group + resolved knockout) as plain dicts.

    Each dict carries the same keys as ``espn_common.load_concrete_fixtures`` (``match_id``,
    ``stage``, ``date`` as ``YYYYMMDD``, ``home_id``, ``away_id``, ``kickoff_utc``,
    ``venue_country``) so it is a drop-in fixture source for the fetchers. Knockout slots that the
    played results have not yet settled are left unresolved and therefore omitted.
    """
    tdir = Path(tdir)
    thirds = tdir / "thirds_allocation.json"
    provider = FileDataProvider(
        tdir / "teams.csv",
        tdir / "fixtures.csv",
        tdir / "results.csv",
        thirds if thirds.exists() else None,
    )
    fixtures = provider.get_fixtures()
    results = {r.match_id: r for r in provider.get_results()}
    standings = compute_group_standings(fixtures, results)
    resolved = resolve_known_participants(
        fixtures, results, provider.get_thirds_allocation(), standings
    )

    out: list[dict] = []
    for m in resolved:
        if m.match_id in results:
            continue  # already played
        if not (m.home.team_id and m.away.team_id) or m.home.ko_ref or m.away.ko_ref:
            continue  # an open knockout slot — participants not yet certain
        ko = m.kickoff
        out.append({
            "match_id": m.match_id,
            "stage": m.stage.value,
            "date": ko.strftime("%Y%m%d"),
            "home_id": m.home.team_id,
            "away_id": m.away.team_id,
            "kickoff_utc": ko.isoformat(),
            "venue_country": m.venue_country or "",
        })
    return out


def frozen_match_ids(tdir: str | Path, *, now: datetime | None = None) -> set[str]:
    """Matches whose odds are locked: recorded in ``results.csv`` **or already kicked off**.

    A committed odds row is a *pre-match* snapshot. Once the match starts there is no pre-match
    price to fetch any more — an in-play or post-match quote is a different thing — so a refresh
    must leave these matches' rows exactly as committed. ``now`` (default: current UTC time) is
    the kickoff cutoff; tests pass a fixed value.
    """
    tdir = Path(tdir)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:  # normalise a naive cutoff to UTC, same as the kickoffs below
        now = now.replace(tzinfo=timezone.utc)
    frozen = set(load_played_match_ids(tdir))
    with (tdir / "fixtures.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mid, raw = (row.get("match_id") or "").strip(), (row.get("kickoff_utc") or "").strip()
            if not mid or not raw:
                continue
            kick = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if kick.tzinfo is None:
                kick = kick.replace(tzinfo=timezone.utc)
            if kick <= now:
                frozen.add(mid)
    return frozen


def write_odds_preserving_frozen(
    out_path: str | Path, fresh_rows: list[dict], frozen: set[str]
) -> tuple[int, int]:
    """Write ``fresh_rows`` (odds.csv schema) to ``out_path``, keeping frozen rows untouched.

    Rows already in the file whose ``match_id`` is frozen are copied through **verbatim** (same
    strings, original order, first) — the pre-match snapshot of a played/kicked-off match never
    changes. ``fresh_rows`` follow; any fresh row for a frozen match is dropped defensively (it
    would be an in-play price). Returns ``(rows_written, rows_preserved)``.
    """
    out = Path(out_path)
    preserved: list[dict] = []
    if out.exists():
        with out.open(newline="", encoding="utf-8") as fh:
            preserved = [
                {k: row.get(k, "") for k in _ODDS_FIELDS}
                for row in csv.DictReader(fh)
                if (row.get("match_id") or "").strip() in frozen
            ]
    kept_ids = {r["match_id"] for r in preserved}
    rows = preserved + [
        r for r in fresh_rows if r["match_id"] not in frozen and r["match_id"] not in kept_ids
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        # LF explicitly: the committed data files are LF, and byte-stable output is what makes
        # "frozen rows preserved" visible as an empty git diff.
        writer = csv.DictWriter(fh, fieldnames=_ODDS_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), len(preserved)
