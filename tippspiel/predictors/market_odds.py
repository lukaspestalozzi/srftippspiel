"""MarketOddsPredictor (spec §6.2.6) — Phase 3, blended market-odds predictor.

Bookmaker 1X2 odds are the most predictive freely-available football signal (the market
beats Elo for match forecasting; Hvattum & Arntzen 2010, Wunderlich & Memmert 2018). This
predictor uses de-vigged 1X2 odds for the scoreline where odds are supplied, and falls back
to a wrapped ``EloPoissonPredictor`` everywhere else — future knockout rounds, and every
synthetic matchup the simulator generates (its ``_pair_*`` match-ids always miss the odds
map). The 1X2 triple is expanded to a full scoreline via ``expand_1x2_to_scoreline`` (§6.2.5).

``market_weight`` blends the two sources where odds exist (log-linear pooling of the two
scoreline matrices, ``pool_log_linear``): 1.0 is the pure market expansion (the historical
behaviour), 0.0 the pure Elo model, in between an ensemble — the Bächinger-style
model x market mix, with the weight tunable against the backtests (``tippspiel tune
--market``). ``match_draw`` switches the expansion to also match the de-vigged draw price
(solving the total-goals level per match instead of assuming a fixed one).

Odds are injected at construction as a ``dict[match_id, Odds1X2]`` rather than added to the
frozen ``Match`` dataclass, so the ``Predictor.predict(match, teams)`` interface is unchanged.
"""

from __future__ import annotations

from ..data.base import Odds1X2
from ..model.scoreline import ScorelineDistribution
from ..model.types import Match, MatchPrediction, Team
from .base import Predictor
from .elo_poisson import EloPoissonPredictor
from .expansion import expand_1x2_to_scoreline, pool_log_linear


class MarketOddsPredictor(Predictor):
    name = "market_odds"

    def __init__(
        self,
        odds: dict[str, Odds1X2] | None = None,
        fallback: Predictor | None = None,
        total_goals: float = 2.6,
        gmax: int = 7,
        ko_goal_scale: float = 1.0,
        market_weight: float = 1.0,
        match_draw: bool = False,
    ) -> None:
        self.odds = dict(odds or {})
        self.fallback = fallback if fallback is not None else EloPoissonPredictor(gmax=gmax)
        self.total_goals = total_goals
        self.gmax = gmax
        self.ko_goal_scale = ko_goal_scale
        if not 0.0 <= market_weight <= 1.0:
            raise ValueError(f"market_weight must be in [0, 1], got {market_weight}")
        self.market_weight = market_weight
        self.match_draw = match_draw
        # The simulator reads predictor.gmax to size one flat CDF shared across both the
        # market and fallback paths; a mismatch would silently corrupt sampling.
        fb_gmax = getattr(self.fallback, "gmax", None)
        if fb_gmax is not None and fb_gmax != self.gmax:
            raise ValueError(
                f"MarketOddsPredictor.gmax ({self.gmax}) must equal the fallback's gmax "
                f"({fb_gmax}); the simulator sizes one shared scoreline grid from gmax."
            )

    @property
    def params(self) -> dict:
        return {
            "total_goals": self.total_goals,
            "gmax": self.gmax,
            "ko_goal_scale": self.ko_goal_scale,
            "market_weight": self.market_weight,
            "match_draw": self.match_draw,
            "n_odds": len(self.odds),
            "fallback": getattr(self.fallback, "params", {}),
        }

    def predict(self, match: Match, teams: dict[str, Team]) -> MatchPrediction:
        o = self.odds.get(match.match_id)
        if o is None or self.market_weight <= 0.0:
            return self.fallback.predict(match, teams)
        # Odds settle on 90 minutes; lift the goal total for knockout (120-minute) scorelines.
        # This shifts only exact-scoreline mass, not the L/D/W tendency the expander matches.
        goal_scale = self.ko_goal_scale if match.stage.is_knockout else 1.0
        scoreline = expand_1x2_to_scoreline(
            o.p_home, o.p_draw, o.p_away,
            total_goals=self.total_goals, gmax=self.gmax,
            match_draw=self.match_draw, goal_scale=goal_scale,
        )
        if self.market_weight < 1.0:
            model = self.fallback.predict(match, teams).scoreline
            scoreline = ScorelineDistribution(
                pool_log_linear(scoreline.matrix, model.matrix, self.market_weight)
            )
        return MatchPrediction(
            match_id=match.match_id,
            scoreline=scoreline,
            predictor_name=self.name,
            predictor_params=self.params,
        )
