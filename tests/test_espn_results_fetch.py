"""Tests for the results-recording fetch tool (tippspiel/data/espn_results_fetch.py)."""

from datetime import datetime, timezone

import tippspiel.data.espn_results_fetch as erf
from tippspiel.data.espn_common import date_window
from tippspiel.data.espn_results_fetch import fetch_results

# Fixed "now" so the kickoff-has-passed filter is deterministic, not wall-clock dependent (after
# every fixture date used below).
_NOW = datetime(2026, 7, 15, tzinfo=timezone.utc)


def test_date_window_spans_neighbouring_days():
    assert date_window("20260617") == ["20260616", "20260617", "20260618"]
    # Crosses a month boundary correctly.
    assert date_window("20260601") == ["20260531", "20260601", "20260602"]


def _event(home_name, away_name, home_score, away_score):
    return {
        "status": {"type": {"state": "post"}},
        "competitions": [{
            "competitors": [
                {"team": {"displayName": home_name}, "homeAway": "home", "score": str(home_score)},
                {"team": {"displayName": away_name}, "homeAway": "away", "score": str(away_score)},
            ]
        }],
    }


def test_fetch_results_finds_match_listed_under_previous_local_date(monkeypatch):
    """A western-venue late-UTC kickoff (e.g. ``…T02:00:00Z``) is filed by ESPN under the prior
    *local* date; the scoreboard lookup must search the ±1-day window or it's permanently skipped.
    """
    # Fixture dated 2026-06-18 (UTC), but ESPN lists the event under 2026-06-17 (local).
    monkeypatch.setattr(erf, "load_teams", lambda tdir: {"uzbekistan": "UZB", "colombia": "COL"})
    monkeypatch.setattr(erf, "load_played_match_ids", lambda tdir: set())
    monkeypatch.setattr(erf, "load_tippable_fixtures", lambda tdir: [{
        "match_id": "G_K_2", "stage": "GROUP", "date": "20260618",
        "home_id": "UZB", "away_id": "COL", "kickoff_utc": "2026-06-18T02:00:00Z",
        "venue_country": "MEX",
    }])
    monkeypatch.setattr(
        erf, "fetch_scoreboard",
        lambda slug, dates: {"20260617": [_event("Uzbekistan", "Colombia", 1, 3)]},
    )

    rows = fetch_results("wc2026", "fifa.world", now=_NOW)
    assert len(rows) == 1
    row = rows[0]
    assert row["match_id"] == "G_K_2"
    assert (row["home_goals"], row["away_goals"]) == (1, 3)
    assert row["date"] == "2026-06-18"  # corpus join (±1 day) reconciles this to the local date


def test_fetch_results_records_resolved_knockout_match(monkeypatch):
    """Knockout fixtures store participants as structural refs (``W:A``/``R:B``/``3RD:…``), so the
    candidate list must come from :func:`load_tippable_fixtures` (which resolves them from the
    played results), not the raw ``fixtures.csv`` rows — otherwise no KO result is ever found.
    """
    monkeypatch.setattr(erf, "load_teams", lambda tdir: {"brazil": "BRA", "france": "FRA"})
    monkeypatch.setattr(erf, "load_played_match_ids", lambda tdir: set())
    # A resolved KO match: load_tippable_fixtures has already turned W:E / R:C into concrete teams.
    monkeypatch.setattr(erf, "load_tippable_fixtures", lambda tdir: [{
        "match_id": "M75", "stage": "R32", "date": "20260630",
        "home_id": "BRA", "away_id": "FRA", "kickoff_utc": "2026-06-30T01:00:00+00:00",
        "venue_country": "MEX",
    }])
    monkeypatch.setattr(
        erf, "fetch_scoreboard",
        lambda slug, dates: {"20260630": [_event("Brazil", "France", 2, 1)]},
    )

    rows = fetch_results("wc2026", "fifa.world", now=_NOW)
    assert len(rows) == 1
    assert rows[0]["match_id"] == "M75"
    assert rows[0]["stage"] == "R32"
    assert (rows[0]["home_goals"], rows[0]["away_goals"]) == (2, 1)


def test_fetch_results_auto_fills_shootout_winner(monkeypatch):
    """A knockout match level after 90' should take its winner_team_id from the feed's
    shootoutScore, not leave it blank for the maintainer."""
    monkeypatch.setattr(erf, "load_teams", lambda tdir: {"germany": "GER", "paraguay": "PAR"})
    monkeypatch.setattr(erf, "load_played_match_ids", lambda tdir: set())
    monkeypatch.setattr(erf, "load_tippable_fixtures", lambda tdir: [{
        "match_id": "M74", "stage": "R32", "date": "20260629",
        "home_id": "GER", "away_id": "PAR", "kickoff_utc": "2026-06-29T20:00:00+00:00",
        "venue_country": "USA",
    }])
    event = _event("Germany", "Paraguay", 1, 1)
    competitors = event["competitions"][0]["competitors"]
    competitors[0]["shootoutScore"] = 3  # Germany
    competitors[1]["shootoutScore"] = 4  # Paraguay wins the shootout
    monkeypatch.setattr(erf, "fetch_scoreboard", lambda slug, dates: {"20260629": [event]})

    rows = fetch_results("wc2026", "fifa.world", now=_NOW)
    assert len(rows) == 1
    assert rows[0]["shootout"] is True
    assert rows[0]["winner_team_id"] == "PAR"
