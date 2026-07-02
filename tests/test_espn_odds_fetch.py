"""Tests for the odds fetch tool (tippspiel/data/espn_odds_fetch.py)."""

import csv

import tippspiel.data.espn_odds_fetch as eof
from tippspiel.data.espn_odds_fetch import fetch_odds


def _event(event_id, home_name, away_name):
    return {
        "id": event_id,
        "competitions": [{
            "competitors": [
                {"team": {"displayName": home_name}, "homeAway": "home"},
                {"team": {"displayName": away_name}, "homeAway": "away"},
            ]
        }],
    }


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_fetch_odds_finds_event_listed_under_previous_local_date(tmp_path, monkeypatch):
    """A western-venue late-UTC kickoff (e.g. ``…T03:00:00Z``) is filed by ESPN under the prior
    *local* date; the scoreboard lookup must search the same ±1-day window as the results fetcher
    or the fixture is never priced (WC2026 M85/M87/M92/M94 were all missing for this reason).
    """
    # Fixture dated 2026-07-03 (UTC), but ESPN lists the event under 2026-07-02 (local).
    monkeypatch.setattr(eof, "load_teams",
                        lambda tdir: {"switzerland": "SUI", "algeria": "ALG"})
    monkeypatch.setattr(eof, "load_tippable_fixtures", lambda tdir: [{
        "match_id": "M85", "stage": "R32", "date": "20260703",
        "home_id": "SUI", "away_id": "ALG", "kickoff_utc": "2026-07-03T03:00:00Z",
        "venue_country": "USA",
    }])
    requested: list[str] = []

    def scoreboard(slug, dates):
        requested.extend(dates)
        return {"20260702": [_event("760498", "Switzerland", "Algeria")]}

    monkeypatch.setattr(eof, "fetch_scoreboard", scoreboard)
    monkeypatch.setattr(eof, "_odds_for_event", lambda slug, event_id: (1.95, 3.30, 4.40))

    out = tmp_path / "odds_espn.csv"
    assert fetch_odds("wc2026", "fifa.world", out) == 1
    assert "20260702" in requested  # the prior local day was fetched at all
    rows = _read_rows(out)
    assert rows == [{"match_id": "M85", "odds_home": "1.95",
                     "odds_draw": "3.30", "odds_away": "4.40"}]


def test_fetch_odds_orients_prices_by_team_identity(tmp_path, monkeypatch):
    """When ESPN's home side is the repo's away side, the home/away prices must swap."""
    monkeypatch.setattr(eof, "load_teams",
                        lambda tdir: {"switzerland": "SUI", "algeria": "ALG"})
    monkeypatch.setattr(eof, "load_tippable_fixtures", lambda tdir: [{
        "match_id": "M85", "stage": "R32", "date": "20260702",
        "home_id": "SUI", "away_id": "ALG", "kickoff_utc": "2026-07-02T20:00:00Z",
        "venue_country": "USA",
    }])
    # ESPN lists Algeria as its home side; its home price belongs to ALG = repo away.
    monkeypatch.setattr(eof, "fetch_scoreboard", lambda slug, dates: {
        "20260702": [_event("760498", "Algeria", "Switzerland")],
    })
    monkeypatch.setattr(eof, "_odds_for_event", lambda slug, event_id: (4.40, 3.30, 1.95))

    out = tmp_path / "odds_espn.csv"
    assert fetch_odds("wc2026", "fifa.world", out) == 1
    rows = _read_rows(out)
    assert rows == [{"match_id": "M85", "odds_home": "1.95",
                     "odds_draw": "3.30", "odds_away": "4.40"}]
