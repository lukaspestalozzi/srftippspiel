"""Knockout bracket tests — derived from the fixtures (spec §3.3 / §6.1.4 / §10)."""

from pathlib import Path

import numpy as np
import pytest

import tippspiel
from tippspiel.config import load_tournament
from tippspiel.data.file_provider import FileDataProvider
from tippspiel.simulation.bracket import Bracket, _match_slots

REPO = Path(tippspiel.__file__).parent.parent


@pytest.fixture
def bracket() -> Bracket:
    b = load_tournament(REPO / "config.yaml")  # FIFA World Cup 2026
    prov = FileDataProvider(b.teams_file, b.fixtures_file, b.results_file,
                            b.thirds_allocation_file)
    fixtures = prov.get_fixtures()
    groups = sorted({m.group for m in fixtures if m.group})
    ko = [m for m in fixtures if m.group is None]
    return Bracket(ko, groups, prov.get_thirds_allocation())


def test_structure(bracket: Bracket):
    assert len(bracket.first_round) == 16
    assert bracket.first_round_stage == "R32"
    assert bracket.stage_chain == ["R32", "R16", "QF", "SF", "FINAL"]
    assert bracket.third_slots == [74, 77, 79, 80, 81, 82, 85, 87]
    assert len(bracket.progression) == 16  # matches 89..104


def test_forced_assignment_respects_unique_allowed_slots(bracket: Bracket):
    # Group K is allowed only in slot 80, group L only in slot 87 — so whenever both
    # qualify they must take exactly those slots.
    g = {c: i for i, c in enumerate("ABCDEFGHIJKL")}
    qualified = np.zeros((1, 12), dtype=bool)
    for letter in "CDEGHIKL":
        qualified[0, g[letter]] = True
    slot_group_idx, _ = bracket.assign_thirds(qualified)
    pos = {s: p for p, s in enumerate(bracket.third_slots)}
    assert slot_group_idx[0, pos[80]] == g["K"]
    assert slot_group_idx[0, pos[87]] == g["L"]


def test_assignment_is_valid_for_all_combinations(bracket: Bracket):
    # Every assignment must be a perfect matching into allowed slots (distinct groups,
    # each in its slot's allowed set).
    for _mask, assign in bracket._table.items():
        assert len(set(assign.tolist())) == 8
        for p, slot in enumerate(bracket.third_slots):
            letter = "ABCDEFGHIJKL"[assign[p]]
            assert letter in bracket.allowed_by_slot[slot]


def test_match_slots_matching_is_perfect():
    allowed = {0: {"A", "B"}, 1: {"B", "C"}, 2: {"C", "D"}}
    result = _match_slots(("A", "B", "C"), allowed, [0, 1, 2])
    assert sorted(result.values()) == ["A", "B", "C"]
    assert set(result.keys()) == {0, 1, 2}
