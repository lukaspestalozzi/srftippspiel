"""Tests for the ESPN fetch tools' shared helpers (tippspiel/data/espn_common.py)."""

import csv
from pathlib import Path

import tippspiel
from tippspiel.data.espn_common import (
    find_event,
    load_concrete_fixtures,
    load_played_match_ids,
    load_teams,
    norm,
)

REPO = Path(tippspiel.__file__).parent.parent
WC2026 = REPO / "tippspiel" / "data" / "tournaments" / "wc2026"


def test_norm_applies_aliases():
    assert norm("USA") == "united states"
    assert norm("Korea Republic") == "south korea"
    assert norm("  Czech Republic  ") == "czechia"
    assert norm("Spain") == "spain"


def _event(home_name="Mexico", away_name="South Africa", home_side="home", event_id="123"):
    return {
        "id": event_id,
        "competitions": [{
            "competitors": [
                {"team": {"displayName": home_name}, "homeAway": home_side},
                {"team": {"displayName": away_name},
                 "homeAway": "away" if home_side == "home" else "home"},
            ]
        }],
    }


def test_find_event_matches_by_team_identity():
    teams = {"mexico": "MEX", "south africa": "RSA"}
    events = [_event()]
    found = find_event(events, teams, "MEX", "RSA")
    assert found is not None
    event, ids = found
    assert event["id"] == "123"
    assert ids == {"home": "MEX", "away": "RSA"}


def test_find_event_returns_none_when_no_match():
    teams = {"mexico": "MEX", "south africa": "RSA"}
    events = [_event(home_name="Brazil", away_name="Morocco")]
    assert find_event(events, teams, "MEX", "RSA") is None


def test_find_event_skips_malformed_events_without_crashing():
    teams = {"mexico": "MEX", "south africa": "RSA"}
    malformed = [
        {},  # no "competitions" at all
        {"competitions": []},  # empty competitions
        {"competitions": [{}]},  # no "competitors"
        {"competitions": [{"competitors": [{"homeAway": "home"}]}]},  # no "team"
        {"competitions": [{"competitors": [{"team": {}, "homeAway": "home"}]}]},  # no displayName
        {"competitions": [{"competitors": [{"team": {"displayName": "Mexico"}}]}]},  # no homeAway
    ]
    # None of the malformed events should raise, and none should match.
    assert find_event(malformed, teams, "MEX", "RSA") is None

    # A good event after a run of malformed ones is still found.
    events = [*malformed, _event()]
    found = find_event(events, teams, "MEX", "RSA")
    assert found is not None
    assert found[0]["id"] == "123"


def test_load_teams_wc2026():
    teams = load_teams(WC2026)
    assert teams["mexico"] == "MEX"
    assert teams["spain"] == "ESP"


def test_load_concrete_fixtures_skips_structural_refs():
    fixtures = load_concrete_fixtures(WC2026)
    assert fixtures  # non-empty
    for f in fixtures:
        assert ":" not in f["home_id"]
        assert ":" not in f["away_id"]
        assert f["kickoff_utc"].endswith("Z")
        assert len(f["date"]) == 8

    by_id = {f["match_id"]: f for f in fixtures}
    assert by_id["G_A_1"]["home_id"] == "MEX"
    assert by_id["G_A_1"]["away_id"] == "RSA"
    assert by_id["G_A_1"]["stage"] == "GROUP"


def test_load_played_match_ids_wc2026():
    # Structural invariant, independent of how far the tournament has progressed:
    # the played set is exactly the match_ids in results.csv, and every one of them
    # is a real fixture. (Don't hardcode which matches are played — that assumption
    # goes stale as matchdays are recorded, culminating in the final.)
    with (WC2026 / "results.csv").open(newline="", encoding="utf-8") as fh:
        expected = {row["match_id"] for row in csv.DictReader(fh)}

    played = load_played_match_ids(WC2026)
    assert played == expected
    # A match id that isn't in results.csv isn't reported as played.
    assert "NOPE_999" not in played
    # Every played id is a declared fixture (results.csv never references a phantom match).
    fixture_ids = {row.split(",", 1)[0] for row in
                   (WC2026 / "fixtures.csv").read_text(encoding="utf-8").splitlines()[1:]}
    assert played <= fixture_ids
