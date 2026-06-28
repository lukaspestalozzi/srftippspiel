"""Shared helper for the odds fetchers: the list of *tippable* concrete fixtures.

A fixture is tippable when it has two known teams and no recorded result yet. For the group
stage that is just the un-played group matches; for the knockout stage the participants are
stored in ``fixtures.csv`` as structured references (``W:A`` / ``R:B`` / ``3RD:74:…``), so they
are resolved to concrete teams with :func:`resolve_known_participants` — the same deterministic
pass the report/tip path uses — but **only where the played results already make them certain**.

This lets both odds fetchers (ESPN, Polymarket) price knockout matches as soon as the bracket is
settled, not just group matches. Offline maintainer helper, **not on the runtime path**.
"""

from __future__ import annotations

from pathlib import Path

from .espn_common import REPO
from .file_provider import FileDataProvider
from ..simulation.known_participants import (
    compute_group_standings,
    resolve_known_participants,
)


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
