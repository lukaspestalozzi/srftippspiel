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

``divergence_threshold`` makes the blend *targeted*: a fixture keeps the pure model
prediction unless the model's 1X2 diverges from the de-vigged market by more than the
threshold on some outcome (the diagnostic value-check's ``max |delta|``), in which case the
blend (or pure market, at ``market_weight=1``) takes over. The rationale: the market mostly
agrees with a calibrated model, and where it disagrees *sharply* it usually knows something
(lineups, injuries, squad quality a slow rating misses) — so only those fixtures defer to
it. ``0.0`` (default) disables the gate and blends every odds-backed fixture as before.

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
        divergence_threshold: float = 0.0,
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
        if not 0.0 <= divergence_threshold <= 1.0:
            raise ValueError(
                f"divergence_threshold must be in [0, 1], got {divergence_threshold}"
            )
        self.divergence_threshold = divergence_threshold
        # The simulator reads predictor.gmax to size one flat CDF shared across both the
        # market and fallback paths; a mismatch would silently corrupt sampling.
        fb_gmax = getattr(self.fallback, "gmax", None)
        if fb_gmax is not None and fb_gmax != self.gmax:
            raise ValueError(
                f"MarketOddsPredictor.gmax ({self.gmax}) must equal the fallback's gmax "
                f"({fb_gmax}); the simulator sizes one shared scoreline grid from gmax."
            )

    @property
    def alpha(self) -> float:
        """The off/def volume weight actually in play — the fallback's (the market path has no
        Elo inputs). Surfaced so report/diagnostic off/def displays gate correctly when this
        wrapper is the active predictor."""
        return float(getattr(self.fallback, "alpha", 0.0))

    @property
    def params(self) -> dict:
        return {
            "total_goals": self.total_goals,
            "gmax": self.gmax,
            "ko_goal_scale": self.ko_goal_scale,
            "market_weight": self.market_weight,
            "match_draw": self.match_draw,
            "divergence_threshold": self.divergence_threshold,
            "n_odds": len(self.odds),
            "fallback": getattr(self.fallback, "params", {}),
        }

    def predict(self, match: Match, teams: dict[str, Team]) -> MatchPrediction:
        o = self.odds.get(match.match_id)
        if o is None or self.market_weight <= 0.0:
            return self.fallback.predict(match, teams)
        model = None
        if self.market_weight < 1.0 or self.divergence_threshold > 0.0:
            model = self.fallback.predict(match, teams)
        if self.divergence_threshold > 0.0:
            dist = model.scoreline
            divergence = max(
                abs(dist.p_home_win() - o.p_home),
                abs(dist.p_draw() - o.p_draw),
                abs(dist.p_away_win() - o.p_away),
            )
            if divergence < self.divergence_threshold:
                return model
        # Odds settle on 90 minutes; lift the goal total for knockout (120-minute) scorelines.
        # The home-vs-away win balance the expander matches is preserved, but the lift does
        # thin the draw mass (more expected goals -> fewer draws), as 120-minute results show.
        goal_scale = self.ko_goal_scale if match.stage.is_knockout else 1.0
        scoreline = expand_1x2_to_scoreline(
            o.p_home, o.p_draw, o.p_away,
            total_goals=self.total_goals, gmax=self.gmax,
            match_draw=self.match_draw, goal_scale=goal_scale,
        )
        if self.market_weight < 1.0:
            scoreline = ScorelineDistribution(
                pool_log_linear(scoreline.matrix, model.scoreline.matrix, self.market_weight)
            )
        return MatchPrediction(
            match_id=match.match_id,
            scoreline=scoreline,
            predictor_name=self.name,
            predictor_params=self.params,
        )
