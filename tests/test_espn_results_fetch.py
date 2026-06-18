"""Tests for the results-recording fetch tool (tippspiel/data/espn_results_fetch.py)."""

import tippspiel.data.espn_results_fetch as erf
from tippspiel.data.espn_results_fetch import _date_window, fetch_results


def test_date_window_spans_neighbouring_days():
    assert _date_window("20260617") == ["20260616", "20260617", "20260618"]
    # Crosses a month boundary correctly.
    assert _date_window("20260601") == ["20260531", "20260601", "20260602"]


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
    monkeypatch.setattr(erf, "load_concrete_fixtures", lambda tdir: [{
        "match_id": "G_K_2", "stage": "GROUP", "date": "20260618",
        "home_id": "UZB", "away_id": "COL", "kickoff_utc": "2026-06-18T02:00:00Z",
        "venue_country": "MEX",
    }])
    monkeypatch.setattr(
        erf, "fetch_scoreboard",
        lambda slug, dates: {"20260617": [_event("Uzbekistan", "Colombia", 1, 3)]},
    )

    rows = fetch_results("wc2026", "fifa.world")
    assert len(rows) == 1
    row = rows[0]
    assert row["match_id"] == "G_K_2"
    assert (row["home_goals"], row["away_goals"]) == (1, 3)
    assert row["date"] == "2026-06-18"  # corpus join (±1 day) reconciles this to the local date
