"""AttackDefencePoissonPredictor: goal rates from per-team attack/defence ratings.

    lambda_home = exp(base_log_rate + attack_home - defence_away + home_advantage*is_host)
    lambda_away = exp(base_log_rate + attack_away - defence_home)

The ratings come from ``Team.attack``/``Team.defence`` (produced by ``build-elo`` with
``model: attack_defence``); a team without them falls back to 0 (i.e. the base rate). The
scoreline matrix (incl. the optional Dixon-Coles correction) is the same one the Elo predictor
uses, so only the goal-rate mapping differs.
"""

from __future__ import annotations

import math

from ..model.scoreline import ScorelineDistribution
from ..model.types import Match, MatchPrediction, Team
from .base import Predictor
from .elo_poisson import scoreline_from_rates


class AttackDefencePoissonPredictor(Predictor):
    name = "attack_defence_poisson"
    ratings_kind = "attack_defence"  # reads the computed Team.attack/Team.defence ratings

    def __init__(
        self,
        base_log_rate: float = 0.3,
        home_advantage: float = 0.15,
        rho: float = 0.0,
        gmax: int = 7,
        ko_goal_scale: float = 1.0,
    ) -> None:
        self.base_log_rate = base_log_rate
        self.home_advantage = home_advantage
        self.rho = rho
        self.gmax = gmax
        self.ko_goal_scale = ko_goal_scale

    @property
    def params(self) -> dict:
        return {
            "base_log_rate": self.base_log_rate,
            "home_advantage": self.home_advantage,
            "rho": self.rho,
            "gmax": self.gmax,
            "ko_goal_scale": self.ko_goal_scale,
        }

    def goal_rates(
        self, atk_h: float, def_h: float, atk_a: float, def_a: float, home_is_host: bool = False
    ) -> tuple[float, float]:
        ha = self.home_advantage if home_is_host else 0.0
        lambda_h = math.exp(self.base_log_rate + atk_h - def_a + ha)
        lambda_a = math.exp(self.base_log_rate + atk_a - def_h)
        return lambda_h, lambda_a

    def predict(self, match: Match, teams: dict[str, Team]) -> MatchPrediction:
        if not match.participants_known:
            raise ValueError(
                f"AttackDefencePoissonPredictor needs concrete participants for {match.match_id}"
            )
        home = teams[match.home.team_id]
        away = teams[match.away.team_id]
        home_is_host = (
            match.venue_country is not None and match.venue_country == home.team_id
        )
        lambda_h, lambda_a = self.goal_rates(
            home.attack or 0.0, home.defence or 0.0,
            away.attack or 0.0, away.defence or 0.0,
            home_is_host,
        )
        if match.stage.is_knockout:
            lambda_h *= self.ko_goal_scale
            lambda_a *= self.ko_goal_scale
        matrix = scoreline_from_rates(lambda_h, lambda_a, self.gmax, self.rho)
        return MatchPrediction(
            match_id=match.match_id,
            scoreline=ScorelineDistribution(matrix),
            predictor_name=self.name,
            predictor_params=self.params,
        )
