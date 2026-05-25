"""R32 bracket assembly tests (spec §3.3 / §6.1.4 / §10)."""

import json
from pathlib import Path

import numpy as np
import pytest

import tippspiel
from tippspiel.simulation.bracket import Bracket, _match_slots

BRACKET_FILE = Path(tippspiel.__file__).parent / "data" / "r32_bracket_map.json"


@pytest.fixture
def bracket() -> Bracket:
    return Bracket(json.loads(BRACKET_FILE.read_text()))


def test_structure(bracket: Bracket):
    assert len(bracket.r32_specs) == 16
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
    for mask, assign in bracket._table.items():
        assert len(set(assign.tolist())) == 8
        for p, slot in enumerate(bracket.third_slots):
            letter = "ABCDEFGHIJKL"[assign[p]]
            assert letter in bracket.allowed_by_slot[slot]


def test_match_slots_matching_is_perfect():
    allowed = {0: {"A", "B"}, 1: {"B", "C"}, 2: {"C", "D"}}
    result = _match_slots(("A", "B", "C"), allowed, [0, 1, 2])
    assert sorted(result.values()) == ["A", "B", "C"]
    assert set(result.keys()) == {0, 1, 2}
