"""Round-of-32 assembly and knockout progression (spec §3.3, §6.1.4).

The third-placed slot assignment depends on which combination of 8 groups supplies the
qualifying third-placed teams. The official FIFA 'Annex C' table is a known data risk
(see r32_bracket_map.json). When that table is not supplied for a combination, we fall
back to ``_match_slots`` — a deterministic bipartite matching that respects each slot's
allowed source groups. This is documented as an approximation.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _g2i(letter: str) -> int:
    return ord(letter) - ord("A")


def _match_slots(group_letters: tuple[str, ...], allowed_by_slot: dict[int, set[str]],
                 slot_order: list[int]) -> dict[int, str]:
    """Deterministic bipartite matching: assign each qualifying group to one allowed slot.

    Uses augmenting paths over groups in sorted order for reproducibility. Any group that
    cannot be matched (should not occur for valid combinations) is placed in a remaining
    slot deterministically so assembly never fails.
    """
    groups = sorted(group_letters)
    adj = {g: [s for s in slot_order if g in allowed_by_slot[s]] for g in groups}
    slot_to_group: dict[int, str] = {}

    def augment(g: str, visited: set[int]) -> bool:
        for s in adj[g]:
            if s in visited:
                continue
            visited.add(s)
            if s not in slot_to_group or augment(slot_to_group[s], visited):
                slot_to_group[s] = g
                return True
        return False

    for g in groups:
        augment(g, set())

    matched_groups = set(slot_to_group.values())
    leftover_groups = [g for g in groups if g not in matched_groups]
    leftover_slots = [s for s in slot_order if s not in slot_to_group]
    for g, s in zip(leftover_groups, leftover_slots):
        slot_to_group[s] = g
    return slot_to_group


class Bracket:
    def __init__(self, bracket_map: dict) -> None:
        self.map = bracket_map
        self.third_slots: list[int] = list(bracket_map["_meta"].get("third_place_slots", []))
        self.allowed_by_slot: dict[int, set[str]] = {}
        # First knockout round (R32 for WC2026, QF for a 16-team tournament, etc.).
        first = bracket_map["first_round"]
        self.first_round_stage: str = first["stage"]
        self.first_round_ids: list[str] = sorted(first["slots"], key=int)
        self.first_round_specs: list[tuple] = []  # (home_spec, away_spec) per first-round match
        for mid in self.first_round_ids:
            slot = first["slots"][mid]
            self.first_round_specs.append((self._spec(slot["home"]), self._spec(slot["away"])))
            for side in ("home", "away"):
                ref = slot[side]
                if ref["type"] == "third":
                    self.allowed_by_slot[ref["slot"]] = set(ref["allowed_groups"])
        # Progression matches (later knockout rounds), in order.
        self.progression: list[tuple] = []
        for mid in sorted(bracket_map["progression"], key=int):
            p = bracket_map["progression"][mid]
            self.progression.append((mid, self._spec(p["home"]), self._spec(p["away"]), p["stage"]))
        # Ordered advancement chain: first-round stage, then each later stage in order of
        # appearance, excluding the third-place consolation. e.g. [R32, R16, QF, SF, FINAL]
        # for WC2026; [QF, SF, FINAL] for Women's Euro 2025.
        chain = [self.first_round_stage]
        for _mid, _h, _a, stage in self.progression:
            if stage != "THIRD_PLACE" and stage not in chain:
                chain.append(stage)
        self.stage_chain: list[str] = chain
        self._explicit = bracket_map.get("third_place_allocation", {})
        # The third-place combination table is only needed when some groups send a third-placed
        # team to the knockouts (WC2026); skipped entirely otherwise.
        self._table = self._build_table() if self.third_slots else {}

    @staticmethod
    def _spec(ref: dict):
        t = ref["type"]
        if t == "winner":
            return ("W", _g2i(ref["group"]))
        if t == "runner_up":
            return ("RU", _g2i(ref["group"]))
        if t == "third":
            return ("3RD", ref["slot"])
        if t == "winner_of":
            return ("WIN", ref["match"])
        if t == "loser_of":
            return ("LOSE", ref["match"])
        raise ValueError(f"unknown ref type {t}")

    def _build_table(self) -> dict[int, np.ndarray]:
        """Precompute, for every 8-of-12 combination, the group index filling each third
        slot. Keyed by the 12-bit mask of qualifying groups; value is an int array of
        length len(third_slots) giving the group index per slot (in self.third_slots order).
        """
        table: dict[int, np.ndarray] = {}
        for combo in combinations("ABCDEFGHIJKL", 8):
            mask = sum(1 << _g2i(g) for g in combo)
            key = "".join(combo)
            if key in self._explicit:
                assign = {int(s): g for s, g in self._explicit[key].items()}
            else:
                assign = _match_slots(combo, self.allowed_by_slot, self.third_slots)
            table[mask] = np.array([_g2i(assign[s]) for s in self.third_slots], dtype=np.int64)
        return table

    def assign_thirds(self, qualified: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map qualifying-thirds combinations to slot group indices.

        qualified: bool [N, 12]. Returns (slot_group_idx [N, n_slots], mask [N]).
        slot_group_idx[i, p] = group index whose third-placed team fills self.third_slots[p].
        """
        weights = (1 << np.arange(12)).astype(np.int64)
        mask = (qualified.astype(np.int64) * weights).sum(axis=1)
        unique, inverse = np.unique(mask, return_inverse=True)
        compact = np.stack([self._table[int(m)] for m in unique])  # [U, n_slots]
        return compact[inverse], mask
