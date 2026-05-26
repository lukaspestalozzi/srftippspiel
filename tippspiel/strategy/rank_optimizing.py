"""RankOptimizingStrategy + FieldModel (spec §6.3.3) — Phase 3 STUB ONLY.

Abstract intent: for a very large pool (~200,000 participants), maximising your own
expected points scores well but rarely wins — winning requires deliberate contrarian
variance on high-uncertainty matches. This strategy would maximise P(rank = 1) (or
P(top-N)) instead of E[points], which needs a FieldModel estimating how the field tips.

Provide interfaces and NotImplementedError stubs only. The whole-slate TipStrategy
interface (§6.3.1) already accommodates this; no refactor is needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..model.types import Match, MatchPrediction, TipSet, TournamentOutcome
from .base import TipStrategy


class FieldModel(ABC):
    @abstractmethod
    def opponent_tip_distribution(
        self, prediction: MatchPrediction
    ) -> dict[tuple[int, int], float]:
        """Estimate the distribution of opponents' tips for a match."""
        ...


class RankOptimizingStrategy(TipStrategy):
    name = "rank_optimizing"

    def __init__(self, field_model: FieldModel | None = None) -> None:
        self.field_model = field_model

    def generate_tips(
        self,
        predictions: dict[str, MatchPrediction],
        outcome: TournamentOutcome | None,
        fixtures: list[Match],
    ) -> TipSet:
        raise NotImplementedError("RankOptimizingStrategy is a Phase 3 stub (spec §6.3.3).")
