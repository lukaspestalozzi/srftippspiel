"""Predictor interface (spec §6.2.1).

A Predictor converts a single Match (two concrete teams + context) into a
MatchPrediction carrying a full ScorelineDistribution. Swapping predictors is a local
change. Phase 1 ships EloPoissonPredictor; Phase 3 adds MarketOddsPredictor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..model.types import Match, MatchPrediction, Team


class Predictor(ABC):
    name: str
    # Which ratings file a tournament should feed this predictor (see TournamentBundle.teams_files).
    ratings_kind: str = "elo"

    @abstractmethod
    def predict(self, match: Match, teams: dict[str, Team]) -> MatchPrediction:
        """Return a MatchPrediction for a match with concrete participants."""
        ...
