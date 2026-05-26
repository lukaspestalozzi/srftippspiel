"""Knockout-bracket assembly, derived from the fixtures themselves (spec §3.3, §6.1.4).

The tournament format is self-describing from ``fixtures.csv``: knockout fixtures reference
group placings (``W:A`` / ``R:A`` / ``3RD:<slot>:<groups>``) or earlier-match outcomes
(``WIN:M101`` / ``LOSE:M101``). The first knockout round is exactly those fixtures whose
participants come from group standings; every later round references prior matches.

For a "best thirds" format (WC 2026), each receiving slot lists its allowed source groups
inline in the fixture ref. Which group's third lands in which slot, for a given combination
of qualifying thirds, is the official FIFA "Annex C" table — a known data risk. When an
explicit combination->slot table is supplied (``thirds_allocation``) it is used; otherwise
we fall back to ``_match_slots`` — a deterministic bipartite matching that respects each
slot's allowed groups. This is documented as an approximation.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..model.types import Match


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


def _match_sort_key(match_id: str) -> int:
    """Order matches by the numeric part of their id (M73 < M104, M1 < M7)."""
    digits = "".join(c for c in match_id if c.isdigit())
    return int(digits) if digits else 0


class Bracket:
    """Knockout structure derived from the knockout fixtures of a tournament.

    ``ko_matches`` are the non-group fixtures (each side a :class:`KnockoutRef`).
    ``group_letters`` is the full set of group letters in standings order (needed only to
    enumerate third-place combinations). ``thirds_allocation`` is the optional explicit
    combination->slot table.
    """

    def __init__(self, ko_matches: list[Match], group_letters: list[str],
                 thirds_allocation: dict | None = None) -> None:
        self.group_letters = list(group_letters)
        self._gidx = {g: i for i, g in enumerate(self.group_letters)}

        first_matches: list[Match] = []
        prog_matches: list[Match] = []
        for m in ko_matches:
            if m.home.ko_ref.is_group_ref and m.away.ko_ref.is_group_ref:
                first_matches.append(m)
            else:
                prog_matches.append(m)
        first_matches.sort(key=lambda m: _match_sort_key(m.match_id))
        prog_matches.sort(key=lambda m: _match_sort_key(m.match_id))

        if not first_matches:
            raise ValueError("no first knockout round found (no fixtures filled from groups)")
        self.first_round_stage: str = first_matches[0].stage.value
        # (match_id, home_spec, away_spec) for each first-round match.
        self.first_round: list[tuple] = [
            (m.match_id, self._spec(m.home), self._spec(m.away)) for m in first_matches
        ]
        # (match_id, home_spec, away_spec, stage) for each later-round match.
        self.progression: list[tuple] = [
            (m.match_id, self._spec(m.home), self._spec(m.away), m.stage.value)
            for m in prog_matches
        ]

        # Third-place receiving slots + their allowed source groups (empty if no thirds qualify).
        self.allowed_by_slot: dict[int, set[str]] = {}
        for m in first_matches:
            for side in (m.home, m.away):
                if side.ko_ref.kind == "third_pooled":
                    self.allowed_by_slot[side.ko_ref.slot] = set(side.ko_ref.allowed_groups)
        self.third_slots: list[int] = sorted(self.allowed_by_slot)

        # Advancement chain: first-round stage, then each later stage in order of appearance,
        # excluding the third-place consolation. e.g. [R32, R16, QF, SF, FINAL] for WC2026.
        chain = [self.first_round_stage]
        for _mid, _h, _a, stage in self.progression:
            if stage != "THIRD_PLACE" and stage not in chain:
                chain.append(stage)
        self.stage_chain: list[str] = chain

        self._explicit = thirds_allocation or {}
        # The combination table is only needed when some groups send a third to the knockouts.
        self._table = self._build_table() if self.third_slots else {}

    def _spec(self, ref):
        r = ref.ko_ref
        if r.kind == "winner":
            return ("W", self._gidx[r.group])
        if r.kind == "runner_up":
            return ("RU", self._gidx[r.group])
        if r.kind == "third_pooled":
            return ("3RD", r.slot)
        if r.kind == "winner_of":
            return ("WIN", r.match_id)
        if r.kind == "loser_of":
            return ("LOSE", r.match_id)
        raise ValueError(f"unknown ref kind {r.kind}")

    def _build_table(self) -> dict[int, np.ndarray]:
        """Precompute, for every combination of qualifying-thirds groups, the group index
        filling each third slot. Keyed by the bitmask of qualifying groups; value is an int
        array of length len(third_slots) giving the group index per slot (in self.third_slots
        order).
        """
        k = len(self.third_slots)
        table: dict[int, np.ndarray] = {}
        for combo in combinations(self.group_letters, k):
            mask = sum(1 << self._gidx[g] for g in combo)
            key = "".join(combo)
            if key in self._explicit:
                assign = {int(s): g for s, g in self._explicit[key].items()}
            else:
                assign = _match_slots(combo, self.allowed_by_slot, self.third_slots)
            table[mask] = np.array([self._gidx[assign[s]] for s in self.third_slots],
                                   dtype=np.int64)
        return table

    def assign_thirds(self, qualified: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map qualifying-thirds combinations to slot group indices.

        qualified: bool [N, n_groups]. Returns (slot_group_idx [N, n_slots], mask [N]).
        slot_group_idx[i, p] = group index whose third-placed team fills self.third_slots[p].
        """
        ng = len(self.group_letters)
        weights = (1 << np.arange(ng)).astype(np.int64)
        mask = (qualified.astype(np.int64) * weights).sum(axis=1)
        unique, inverse = np.unique(mask, return_inverse=True)
        compact = np.stack([self._table[int(m)] for m in unique])  # [U, n_slots]
        return compact[inverse], mask
