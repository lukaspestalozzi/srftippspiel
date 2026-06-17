"""Write-side corpus helpers + the espn_results_fetch record path (offline)."""

from __future__ import annotations

from pathlib import Path

from tippspiel.config import write_offdef_snapshot_date
from tippspiel.data.corpus_update import (
    latest_results_date,
    set_corpus_score,
    snapshot_after,
)
from tippspiel.data.espn_results_fetch import record_results

_CORPUS_HEADER = "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"


def _corpus(*rows: str) -> list[str]:
    return [_CORPUS_HEADER, *(r if r.endswith("\n") else r + "\n" for r in rows)]


def test_fill_same_orientation():
    lines = _corpus("2026-06-11,Mexico,South Africa,NA,NA,FIFA World Cup,Zapopan,Mexico,FALSE")
    date, action = set_corpus_score(
        lines, date_hint="2026-06-11", home_corpus="Mexico", away_corpus="South Africa",
        home_goals=2, away_goals=0)
    assert (date, action) == ("2026-06-11", "filled")
    assert lines[1] == "2026-06-11,Mexico,South Africa,2,0,FIFA World Cup,Zapopan,Mexico,FALSE\n"


def test_fill_swapped_orientation():
    # Corpus stores South Africa as home; fixture has Mexico home -> goals re-oriented to the row.
    lines = _corpus("2026-06-11,South Africa,Mexico,NA,NA,FIFA World Cup,Zapopan,Mexico,TRUE")
    set_corpus_score(lines, date_hint="2026-06-11", home_corpus="Mexico",
                     away_corpus="South Africa", home_goals=2, away_goals=0)
    assert lines[1].split(",")[3:5] == ["0", "2"]  # South Africa(home) 0 - 2 Mexico(away)


def test_date_within_one_day():
    lines = _corpus("2026-06-11,Mexico,South Africa,NA,NA,FIFA World Cup,Zapopan,Mexico,FALSE")
    # kickoff UTC date is the next day; corpus date is local -> still matched (±1 day).
    date, action = set_corpus_score(
        lines, date_hint="2026-06-12", home_corpus="Mexico", away_corpus="South Africa",
        home_goals=1, away_goals=1)
    assert (date, action) == ("2026-06-11", "filled")


def test_idempotent_exists():
    lines = _corpus("2026-06-11,Mexico,South Africa,2,0,FIFA World Cup,Zapopan,Mexico,FALSE")
    before = list(lines)
    date, action = set_corpus_score(
        lines, date_hint="2026-06-11", home_corpus="Mexico", away_corpus="South Africa",
        home_goals=2, away_goals=0)
    assert action == "exists" and lines == before


def test_append_when_absent():
    lines = _corpus("2026-06-11,Mexico,South Africa,2,0,FIFA World Cup,Zapopan,Mexico,FALSE")
    date, action = set_corpus_score(
        lines, date_hint="2026-07-19", home_corpus="Spain", away_corpus="Brazil",
        home_goals=1, away_goals=0, neutral=True)
    assert action == "appended" and date == "2026-07-19"
    assert lines[-1] == "2026-07-19,Spain,Brazil,1,0,FIFA World Cup,,,TRUE\n"


def test_latest_results_date_and_snapshot(tmp_path):
    rc = tmp_path / "results.csv"
    rc.write_text("match_id,date,winner_team_id\nG_A_1,2026-06-11,\nG_B_1,2026-06-16,\n",
                  encoding="utf-8")
    assert latest_results_date(rc) == "2026-06-16"
    assert snapshot_after("2026-06-16") == "2026-06-17"


def test_write_offdef_snapshot_date_preserves_comments(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "offdef:\n  epochs: 3\n  snapshot_date: \"2026-06-16\"  # day after matchday\n", encoding="utf-8")
    changed = write_offdef_snapshot_date(cfg, "2026-06-17")
    assert changed
    text = cfg.read_text()
    assert 'snapshot_date: "2026-06-17"  # day after matchday' in text
    assert "epochs: 3" in text
    assert not write_offdef_snapshot_date(cfg, "2026-06-17")  # idempotent no-op


def _seed_tournament(tmp_path: Path) -> tuple[Path, Path, Path]:
    tdir = tmp_path / "wc"
    tdir.mkdir()
    (tdir / "teams.csv").write_text(
        "team_id,name,elo\nMEX,Mexico,1700\nRSA,South Africa,1600\n", encoding="utf-8")
    (tdir / "results.csv").write_text("match_id,date,winner_team_id\n", encoding="utf-8")
    corpus = tmp_path / "corpus.csv"
    corpus.write_text(
        _CORPUS_HEADER
        + "2026-06-11,Mexico,South Africa,NA,NA,FIFA World Cup,Zapopan,Mexico,FALSE\n",
        encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("offdef:\n  snapshot_date: \"2026-06-01\"\n", encoding="utf-8")
    return tdir, corpus, config


def test_record_results_end_to_end(tmp_path):
    tdir, corpus, config = _seed_tournament(tmp_path)
    rows = [{"match_id": "G_A_1", "home_id": "MEX", "away_id": "RSA", "home_goals": 2,
             "away_goals": 0, "stage": "GROUP", "date": "2026-06-11", "venue_country": "MEX",
             "shootout": False}]
    # dry-run writes nothing
    dry = record_results(tdir, rows, corpus_path=corpus, config_path=config, write=False)
    assert dry["snapshot"] == "2026-06-12" and dry["plan"][0]["action"] == "filled"
    assert "NA,NA" in corpus.read_text()  # untouched
    # write commits to all three places
    out = record_results(tdir, rows, corpus_path=corpus, config_path=config, write=True)
    assert out["written"]
    assert "2026-06-11,Mexico,South Africa,2,0," in corpus.read_text()
    assert "G_A_1,2026-06-11," in (tdir / "results.csv").read_text()
    assert 'snapshot_date: "2026-06-12"' in config.read_text()
    # idempotent: re-running records nothing (match already in results.csv)
    again = record_results(tdir, rows, corpus_path=corpus, config_path=config, write=True)
    assert again["plan"] == []


def test_record_results_config_without_snapshot_does_not_abort(tmp_path):
    tdir, corpus, _config = _seed_tournament(tmp_path)
    bad_config = tmp_path / "no_snapshot.yaml"
    bad_config.write_text("offdef:\n  epochs: 3\n", encoding="utf-8")  # no snapshot_date line
    rows = [{"match_id": "G_A_1", "home_id": "MEX", "away_id": "RSA", "home_goals": 2,
             "away_goals": 0, "stage": "GROUP", "date": "2026-06-11", "venue_country": "MEX",
             "shootout": False}]
    # corpus + results still get written; the missing config line warns, doesn't raise.
    out = record_results(tdir, rows, corpus_path=corpus, config_path=bad_config, write=True)
    assert out["written"]
    assert "G_A_1,2026-06-11," in (tdir / "results.csv").read_text()
