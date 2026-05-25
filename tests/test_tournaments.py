"""Tournaments layer: bundle resolution + format-general validation (spec §10)."""

import json

import pytest

from tippspiel.cli import validate_data
from tippspiel.config import available_tournaments, resolve_tournament


def test_available_includes_both_bundles():
    avail = available_tournaments()
    assert "wc2026" in avail
    assert "womenseuro2025" in avail


def test_resolve_unknown_tournament_raises():
    with pytest.raises(ValueError):
        resolve_tournament("does_not_exist")


def test_validate_data_passes_for_both_bundles():
    for name in ("wc2026", "womenseuro2025"):
        assert validate_data(resolve_tournament(name)) == []


def test_womenseuro_format_is_16_teams_no_thirds_qf_first():
    b = resolve_tournament("womenseuro2025")
    assert b.completed is True
    teams = b.teams_file.read_text().strip().splitlines()[1:]  # minus header
    assert len(teams) == 16
    bm = json.loads(b.bracket_map_file.read_text())
    assert bm["first_round"]["stage"] == "QF"
    assert bm["_meta"]["third_place_slots"] == []          # no third-place qualifiers
    assert len(bm["first_round"]["slots"]) == 4            # 4 quarter-finals


def test_wc2026_format_is_48_teams_with_thirds_r32_first():
    b = resolve_tournament("wc2026")
    assert b.completed is False
    bm = json.loads(b.bracket_map_file.read_text())
    assert bm["first_round"]["stage"] == "R32"
    assert len(bm["_meta"]["third_place_slots"]) == 8      # best-8 thirds advance
    assert len(bm["first_round"]["slots"]) == 16


def test_bonus_questions_are_tournament_scoped():
    wc = resolve_tournament("wc2026")
    we = resolve_tournament("womenseuro2025")
    assert {q.id for q in wc.bonus_questions} >= {"champion", "swiss_progress"}
    assert {q.id for q in we.bonus_questions} == {"champion"}
