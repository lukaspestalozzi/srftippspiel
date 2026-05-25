"""TipStrategy interface (spec §6.3.1).

A TipStrategy converts the COMPLETE set of MatchPredictions plus the TournamentOutcome
into a complete TipSet. It operates on the whole slate at once, not match-by-match,
because rank-optimising strategies (Phase 3) do not decompose per match. This
whole-slate signature is a deliberate design constraint — do not narrow it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..model.types import Match, MatchPrediction, TipSet, TournamentOutcome


class TipStrategy(ABC):
    name: str

    @abstractmethod
    def generate_tips(
        self,
        predictions: dict[str, MatchPrediction],
        outcome: TournamentOutcome | None,
        fixtures: list[Match],
    ) -> TipSet:
        """Produce one tip per tippable fixture plus bonus answers. ``outcome`` may be
        None in Phase 1 (group-only runs)."""
        ...
