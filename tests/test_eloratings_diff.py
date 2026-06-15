"""Tests for the eloratings.net diff tool (tippspiel/data/eloratings_diff.py)."""

from pathlib import Path
from unittest.mock import patch

from tippspiel.data.eloratings_diff import (
    build_code_to_team_id,
    diff_ratings,
    main,
    parse_teams_tsv,
    parse_world_tsv,
)

WORLD_TSV = "16\t16\tMX\t1885\n26\t26\tKR\t1786\n42\t42\tCZ\t1712\n"
TEAMS_TSV = "MX\tMexico\nKR\tSouth Korea\nCZ\tCzechia\n"


def test_parse_world_tsv():
    ratings = parse_world_tsv(WORLD_TSV)
    assert ratings == {"MX": 1885.0, "KR": 1786.0, "CZ": 1712.0}


def test_parse_world_tsv_skips_short_lines():
    assert parse_world_tsv("16\t16\tMX\n\n") == {}


def test_parse_teams_tsv():
    names = parse_teams_tsv(TEAMS_TSV)
    assert names == {"MX": ["Mexico"], "KR": ["South Korea"], "CZ": ["Czechia"]}


def test_parse_teams_tsv_keeps_aliases():
    names = parse_teams_tsv("AG\tAntigua and Barbuda\tAntigua & Barbuda\tAntigua/Barbuda\n")
    assert names["AG"] == ["Antigua and Barbuda", "Antigua & Barbuda", "Antigua/Barbuda"]


def test_build_code_to_team_id_basic():
    code_to_id = build_code_to_team_id(parse_teams_tsv(TEAMS_TSV), {"mexico": "MEX", "south korea": "KOR"})
    assert code_to_id == {"MX": "MEX", "KR": "KOR"}


def test_build_code_to_team_id_turkey_alias():
    # eloratings calls it "Turkey"; teams.csv uses "Türkiye" -- resolved via espn_common's
    # shared alias table (the same one load_teams() uses to build name_to_team_id).
    teams_tsv = {"TR": ["Turkey"]}
    code_to_id = build_code_to_team_id(teams_tsv, {"türkiye": "TUR"})
    assert code_to_id == {"TR": "TUR"}


def _write_tournament(tdir: Path) -> None:
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "teams.csv").write_text(
        "team_id,name,elo\n"
        "MEX,Mexico,1881\n"
        "RSA,South Africa,1511\n"
        "KOR,South Korea,1786\n"
        "CZE,Czechia,1712\n"
        "ESP,Spain,2157\n",
        encoding="utf-8",
    )
    (tdir / "fixtures.csv").write_text(
        "match_id,stage,group,home_ref,away_ref,kickoff_utc,venue_country\n"
        "G_A_1,GROUP,A,MEX,RSA,2026-06-11T19:00:00Z,MEX\n"
        "G_A_2,GROUP,A,KOR,CZE,2026-06-12T02:00:00Z,MEX\n"
        "G_A_3,GROUP,A,ESP,RSA,2026-06-20T19:00:00Z,MEX\n",
        encoding="utf-8",
    )
    (tdir / "results.csv").write_text(
        "match_id,home_goals,away_goals,winner_team_id\n"
        "G_A_1,2,0,\n"
        "G_A_2,2,1,\n",
        encoding="utf-8",
    )


def test_diff_ratings(tmp_path: Path):
    _write_tournament(tmp_path)

    # MEX moved 1881 -> 1885; KOR/CZE unchanged; RSA's code ("ZA") is absent from the teams_tsv
    # (unmapped); ESP didn't play so it's neither moved, unchanged, nor unresolved.
    world_tsv = "16\t16\tMX\t1885\n26\t26\tKR\t1786\n42\t42\tCZ\t1712\n"
    teams_tsv = "MX\tMexico\nKR\tSouth Korea\nCZ\tCzechia\n"

    moved, unchanged, unresolved = diff_ratings(tmp_path, world_tsv, teams_tsv)

    assert moved == [("MEX", 1881.0, 1885.0)]
    assert unchanged == ["CZE", "KOR"]
    assert unresolved == ["RSA"]


def test_main_prints_summary_even_with_zero_movers(tmp_path: Path, capsys):
    tdir = tmp_path / "tippspiel" / "data" / "tournaments" / "demo"
    _write_tournament(tdir)

    # Nothing moved: MEX/KOR/CZE all match teams.csv; RSA stays unresolved (no "ZA" code).
    world_tsv = "16\t16\tMX\t1881\n26\t26\tKR\t1786\n42\t42\tCZ\t1712\n"
    teams_tsv = "MX\tMexico\nKR\tSouth Korea\nCZ\tCzechia\n"

    with (
        patch("tippspiel.data.eloratings_diff.REPO", tmp_path),
        patch("tippspiel.data.eloratings_diff._fetch_text", side_effect=[world_tsv, teams_tsv]),
    ):
        main("demo")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "0 movers; 3 played teams already up-to-date" in captured.err
    assert "RSA" in captured.err
