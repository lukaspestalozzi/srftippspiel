"""EloPoissonPredictor (spec §6.2.2) with optional Dixon-Coles correction (§6.2.3).

Goal rates use the MULTIPLICATIVE form. An additive Elo term produces negative goal
rates for lopsided matches, which is invalid for a Poisson model — so the rates here
are strictly positive for any Elo gap:

    d        = elo_home - elo_away + home_advantage
    lambda_h = (mu / 2) * exp( +k * d )
    lambda_a = (mu / 2) * exp( -k * d )

Scoreline matrix P(h, a) = Poisson(h; lambda_h) * Poisson(a; lambda_a), truncated to
[0, gmax] and renormalised.
"""

from __future__ import annotations

import math

import numpy as np

from ..model.scoreline import ScorelineDistribution
from ..model.types import Match, MatchPrediction, Team
from .base import Predictor


def _poisson_pmf_vector(lam: float, gmax: int) -> np.ndarray:
    """Poisson PMF for k = 0..gmax (un-normalised over the truncation)."""
    ks = np.arange(gmax + 1)
    log_pmf = -lam + ks * math.log(lam) - np.array([math.lgamma(k + 1) for k in ks])
    return np.exp(log_pmf)


def scoreline_from_rates(lambda_h: float, lambda_a: float, gmax: int, rho: float) -> np.ndarray:
    """Independent-Poisson scoreline matrix P(h, a) over [0, gmax]^2, optionally Dixon-Coles
    corrected and renormalised. Shared by the Elo and attack/defence predictors."""
    ph = _poisson_pmf_vector(lambda_h, gmax)
    pa = _poisson_pmf_vector(lambda_a, gmax)
    matrix = np.outer(ph, pa)
    if rho != 0.0:
        matrix = _apply_dixon_coles(matrix, lambda_h, lambda_a, rho)
    matrix = np.clip(matrix, 0.0, None)
    return matrix / matrix.sum()


class EloPoissonPredictor(Predictor):
    name = "elo_poisson"

    def __init__(
        self,
        mu: float = 2.6,
        k: float = 0.0015,
        gmax: int = 7,
        rho: float = 0.0,
        host_elo_bonus: float = 0.0,
        ko_goal_scale: float = 1.0,
    ) -> None:
        self.mu = mu
        self.k = k
        self.gmax = gmax
        self.rho = rho
        self.host_elo_bonus = host_elo_bonus
        self.ko_goal_scale = ko_goal_scale

    @property
    def params(self) -> dict:
        return {
            "mu": self.mu,
            "k": self.k,
            "gmax": self.gmax,
            "rho": self.rho,
            "host_elo_bonus": self.host_elo_bonus,
            "ko_goal_scale": self.ko_goal_scale,
        }

    def goal_rates(
        self, elo_home: float, elo_away: float, home_is_host: bool = False
    ) -> tuple[float, float]:
        bonus = self.host_elo_bonus if home_is_host else 0.0
        d = elo_home - elo_away + bonus
        lambda_h = (self.mu / 2.0) * math.exp(+self.k * d)
        lambda_a = (self.mu / 2.0) * math.exp(-self.k * d)
        return lambda_h, lambda_a

    def scoreline_matrix(self, lambda_h: float, lambda_a: float) -> np.ndarray:
        return scoreline_from_rates(lambda_h, lambda_a, self.gmax, self.rho)

    def predict(self, match: Match, teams: dict[str, Team]) -> MatchPrediction:
        if not match.participants_known:
            raise ValueError(
                f"EloPoissonPredictor needs concrete participants for {match.match_id}"
            )
        home = teams[match.home.team_id]
        away = teams[match.away.team_id]
        # A team is "host" when it plays in its own country (general across tournaments).
        home_is_host = (
            match.venue_country is not None and match.venue_country == home.team_id
        )
        lambda_h, lambda_a = self.goal_rates(home.elo, away.elo, home_is_host)
        if match.stage.is_knockout:
            # Knockout results are recorded as the 120-minute scoreline (extra time included),
            # but the goal rates model ~90 minutes; scale up to match.
            lambda_h *= self.ko_goal_scale
            lambda_a *= self.ko_goal_scale
        matrix = self.scoreline_matrix(lambda_h, lambda_a)
        return MatchPrediction(
            match_id=match.match_id,
            scoreline=ScorelineDistribution(matrix),
            predictor_name=self.name,
            predictor_params=self.params,
        )


def _apply_dixon_coles(
    matrix: np.ndarray, lambda_h: float, lambda_a: float, rho: float
) -> np.ndarray:
    """Dixon-Coles tau adjustment to the four low-score cells (0,0),(0,1),(1,0),(1,1)."""
    m = matrix.copy()
    if m.shape[0] < 2:
        return m
    m[0, 0] *= 1.0 - lambda_h * lambda_a * rho
    m[0, 1] *= 1.0 + lambda_h * rho
    m[1, 0] *= 1.0 + lambda_a * rho
    m[1, 1] *= 1.0 - rho
    return m
