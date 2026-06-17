"""Corpus-results resolver + dual-mode FileDataProvider tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tippspiel.data.corpus_results import (
    ResultResolutionError,
    build_corpus_index,
    resolve_corpus_result,
)
from tippspiel.data.file_provider import FileDataProvider
from tippspiel.model.stages import Stage
from tippspiel.model.types import KnockoutRef, Match, TeamRef

_KICK = datetime(2022, 11, 20, 16, 0, tzinfo=timezone.utc)


def _match(mid, home_id, away_id):
    return Match(mid, Stage.GROUP, TeamRef(team_id=home_id), TeamRef(team_id=away_id), _KICK)


def _ctx():
    fixtures = {"M": _match("M", "QAT", "ECU")}
    names = {"QAT": "Qatar", "ECU": "Ecuador"}
    index = {("2022-11-20", frozenset(("Qatar", "Ecuador"))): [("Qatar", 0, 2)]}
    return fixtures, names, index


def test_resolve_same_orientation():
    fixtures, names, index = _ctx()
    r = resolve_corpus_result("M", "2022-11-20", None, fixtures, names, index)
    assert (r.home_goals, r.away_goals) == (0, 2)


def test_resolve_swapped_orientation():
    # Corpus stores the match with Ecuador as home; fixture has Qatar home -> goals re-oriented.
    fixtures, names, _ = _ctx()
    index = {("2022-11-20", frozenset(("Qatar", "Ecuador"))): [("Ecuador", 2, 0)]}
    r = resolve_corpus_result("M", "2022-11-20", None, fixtures, names, index)
    assert (r.home_goals, r.away_goals) == (0, 2)


def test_penalty_winner_passthrough():
    fixtures, names, index = _ctx()
    r = resolve_corpus_result("M", "2022-11-20", "ECU", fixtures, names, index)
    assert r.winner_team_id == "ECU"


def test_not_found_raises():
    fixtures, names, index = _ctx()
    with pytest.raises(ResultResolutionError):
        resolve_corpus_result("M", "2022-11-21", None, fixtures, names, index)


def test_ambiguous_raises():
    fixtures, names, _ = _ctx()
    index = {("2022-11-20", frozenset(("Qatar", "Ecuador"))): [("Qatar", 0, 2), ("Qatar", 1, 1)]}
    with pytest.raises(ResultResolutionError):
        resolve_corpus_result("M", "2022-11-20", None, fixtures, names, index)


def test_non_concrete_fixture_raises():
    fixtures = {"M": Match("M", Stage.R16, TeamRef(ko_ref=KnockoutRef(kind="winner", group="A")),
                           TeamRef(team_id="ECU"), _KICK)}
    names = {"ECU": "Ecuador"}
    with pytest.raises(ResultResolutionError):
        resolve_corpus_result("M", "2022-11-20", None, fixtures, names, {})


def test_build_corpus_index_drops_unplayed(tmp_path):
    corpus = tmp_path / "corpus.csv"
    corpus.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2022-11-20,Qatar,Ecuador,0,2,FIFA World Cup,Al Khor,Qatar,FALSE\n"
        "2026-06-27,Spain,Brazil,NA,NA,FIFA World Cup,x,USA,TRUE\n",
        encoding="utf-8",
    )
    index = build_corpus_index(corpus)
    assert ("2022-11-20", frozenset(("Qatar", "Ecuador"))) in index
    assert all("Spain" not in key[1] for key in index)  # unplayed NA row dropped


def test_provider_dual_mode(tmp_path):
    # One inline-scoreline row and one corpus-reference row in the same results.csv.
    (tmp_path / "teams.csv").write_text(
        "team_id,name,elo\nQAT,Qatar,1500\nECU,Ecuador,1600\nSEN,Senegal,1700\nNED,Netherlands,1900\n",
        encoding="utf-8",
    )
    (tmp_path / "fixtures.csv").write_text(
        "match_id,stage,group,home_ref,away_ref,kickoff_utc,venue_country\n"
        "G_A_1,GROUP,A,QAT,ECU,2022-11-20T16:00:00Z,QAT\n"
        "G_A_2,GROUP,A,SEN,NED,2022-11-21T16:00:00Z,QAT\n",
        encoding="utf-8",
    )
    (tmp_path / "results.csv").write_text(
        "match_id,date,home_goals,away_goals,winner_team_id\n"
        "G_A_1,2022-11-20,,,\n"            # corpus reference (no inline score)
        "G_A_2,,3,1,\n",                    # inline scoreline
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.csv"
    corpus.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2022-11-20,Qatar,Ecuador,0,2,FIFA World Cup,Al Khor,Qatar,FALSE\n",
        encoding="utf-8",
    )
    provider = FileDataProvider(
        tmp_path / "teams.csv", tmp_path / "fixtures.csv", tmp_path / "results.csv",
        corpus_file=corpus,
    )
    by_id = {r.match_id: r for r in provider.get_results()}
    assert (by_id["G_A_1"].home_goals, by_id["G_A_1"].away_goals) == (0, 2)  # resolved
    assert (by_id["G_A_2"].home_goals, by_id["G_A_2"].away_goals) == (3, 1)  # inline
