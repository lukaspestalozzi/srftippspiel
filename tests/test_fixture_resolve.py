"""Frozen-odds rule tests: frozen_match_ids + write_odds_preserving_frozen + consensus wiring.

Once a match is played (in results.csv) or has kicked off, its committed odds row is a historical
pre-match snapshot: a refresh must neither re-price it nor drop it from the file.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone

from tippspiel.data.fixture_resolve import frozen_match_ids, write_odds_preserving_frozen
from tippspiel.data.odds_consensus import build_consensus

_NOW = datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc)


def _tournament_dir(tmp_path):
    (tmp_path / "fixtures.csv").write_text(
        "match_id,stage,group,home_ref,away_ref,kickoff_utc,venue_country\n"
        "G_A_1,GROUP,A,AAA,BBB,2026-06-18T16:00:00Z,USA\n"   # played
        "G_A_2,GROUP,A,CCC,DDD,2026-06-20T10:00:00Z,USA\n"   # kicked off, result not yet recorded
        "M1,R16,,W:A,R:A,2026-06-28T16:00:00Z,USA\n",        # future (refs are irrelevant here)
        encoding="utf-8",
    )
    (tmp_path / "results.csv").write_text(
        "match_id,date,winner_team_id\nG_A_1,2026-06-18,\n", encoding="utf-8"
    )
    return tmp_path


def test_frozen_match_ids_played_and_kicked_off(tmp_path):
    tdir = _tournament_dir(tmp_path)
    assert frozen_match_ids(tdir, now=_NOW) == {"G_A_1", "G_A_2"}
    # After the knockout kickoff too, everything is frozen.
    later = datetime(2026, 6, 28, 17, 0, tzinfo=timezone.utc)
    assert frozen_match_ids(tdir, now=later) == {"G_A_1", "G_A_2", "M1"}


def test_frozen_match_ids_accepts_naive_now(tmp_path):
    # A naive cutoff is normalised to UTC instead of raising on aware-vs-naive comparison.
    tdir = _tournament_dir(tmp_path)
    assert frozen_match_ids(tdir, now=_NOW.replace(tzinfo=None)) == {"G_A_1", "G_A_2"}


def test_frozen_match_ids_without_results_file(tmp_path):
    tdir = _tournament_dir(tmp_path)
    (tdir / "results.csv").unlink()
    assert frozen_match_ids(tdir, now=_NOW) == {"G_A_1", "G_A_2"}  # kickoff-based only


def _rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_write_preserves_frozen_rows_verbatim(tmp_path):
    out = tmp_path / "odds.csv"
    out.write_text(
        "match_id,odds_home,odds_draw,odds_away\n"
        "G_A_1,1.50,4.30,7.00\n"    # frozen -> must survive unchanged
        "M1,2.00,3.00,4.00\n",      # not frozen -> replaced by this run's fresh coverage
        encoding="utf-8",
    )
    fresh = [
        {"match_id": "M1", "odds_home": "2.10", "odds_draw": "3.10", "odds_away": "3.90"},
        {"match_id": "M2", "odds_home": "1.80", "odds_draw": "3.50", "odds_away": "4.50"},
        # Defensive: a fresh row for a frozen match is an in-play price -> dropped.
        {"match_id": "G_A_1", "odds_home": "1.01", "odds_draw": "9.99", "odds_away": "9.99"},
    ]
    total, kept = write_odds_preserving_frozen(out, fresh, {"G_A_1"})
    assert (total, kept) == (3, 1)
    by_id = {r["match_id"]: r for r in _rows(out)}
    assert by_id["G_A_1"] == {"match_id": "G_A_1", "odds_home": "1.50",
                              "odds_draw": "4.30", "odds_away": "7.00"}
    assert by_id["M1"]["odds_home"] == "2.10"  # future fixture re-priced
    assert by_id["M2"]["odds_home"] == "1.80"  # newly priced fixture added


def test_write_drops_stale_unfrozen_rows(tmp_path):
    # An unfrozen row absent from this run's fresh coverage drops out (feed no longer prices it),
    # exactly as before the frozen-odds rule.
    out = tmp_path / "odds.csv"
    out.write_text(
        "match_id,odds_home,odds_draw,odds_away\nM1,2.00,3.00,4.00\n", encoding="utf-8"
    )
    total, kept = write_odds_preserving_frozen(out, [], set())
    assert (total, kept) == (0, 0)
    assert _rows(out) == []


def test_write_without_existing_file(tmp_path):
    out = tmp_path / "odds.csv"
    fresh = [{"match_id": "M1", "odds_home": "2.00", "odds_draw": "3.00", "odds_away": "4.00"}]
    total, kept = write_odds_preserving_frozen(out, fresh, {"G_A_1"})
    assert (total, kept) == (1, 0)
    assert _rows(out)[0]["match_id"] == "M1"


def test_consensus_preserves_frozen_rows(tmp_path, capsys):
    # The committed consensus row of a started match stays verbatim even when the sidecars would
    # blend to a different value (their rows may date from different fetch times).
    src = tmp_path / "odds_espn.csv"
    src.write_text(
        "match_id,odds_home,odds_draw,odds_away\n"
        "G_A_1,1.10,8.00,15.00\n"   # would re-blend the frozen match
        "M1,2.00,4.00,4.00\n",      # vig-free trio -> de-vig is an exact identity
        encoding="utf-8",
    )
    out = tmp_path / "odds.csv"
    out.write_text(
        "match_id,odds_home,odds_draw,odds_away\nG_A_1,1.50,4.30,7.00\n", encoding="utf-8"
    )
    total = build_consensus([src], out, frozen={"G_A_1"})
    assert total == 2
    by_id = {r["match_id"]: r for r in _rows(out)}
    assert by_id["G_A_1"]["odds_home"] == "1.50"  # untouched, not re-blended from the sidecar
    assert by_id["M1"]["odds_home"] == "2.00"     # de-vig of a single fair source ~ identity
