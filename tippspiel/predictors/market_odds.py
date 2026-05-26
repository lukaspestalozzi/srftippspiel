"""MarketOddsPredictor (spec §6.2.6) — Phase 3 STUB ONLY.

Abstract intent: take bookmaker 1X2 odds, remove the vig (normalise implied
probabilities), expand to a scoreline via ``expand_1x2_to_scoreline`` (§6.2.5),
optionally blending with Elo as a fallback when odds are missing.

Do not implement until Phase 3 is greenlit. The Phase 1 Predictor interface already
accommodates this with no refactor.
"""

from __future__ import annotations

from ..model.types import Match, MatchPrediction, Team
from .base import Predictor


class MarketOddsPredictor(Predictor):
    name = "market_odds"

    def predict(self, match: Match, teams: dict[str, Team]) -> MatchPrediction:
        raise NotImplementedError("MarketOddsPredictor is a Phase 3 stub (spec §6.2.6).")
