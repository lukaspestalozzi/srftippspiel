"""EloPoissonPredictor (spec §6.2.2) with optional Dixon-Coles correction (§6.2.3).

Goal rates use the MULTIPLICATIVE form. An additive Elo term produces negative goal
rates for lopsided matches, which is invalid for a Poisson model — so the rates here
are strictly positive for any Elo gap:

    d        = elo_home - elo_away + home_advantage
    lambda_h = (mu / 2) * exp( +k * d )
    lambda_a = (mu / 2) * exp( -k * d )

Offensive/defensive volume term (``alpha``). A single scalar Elo fixes the goal *ratio*
(who is likelier to win, via ``k*d``) but pins every match's *total* goals to ``mu`` — it
can't tell a Spain-vs-Norway shoot-out from an Italy-vs-Greece stalemate. Per-team
offensive/defensive log-rate ratings (``Team.att_elo`` / ``Team.def_elo``, fitted from
historical match goals by ``tippspiel/training/offdef_elo.py``) supply exactly that missing
*volume* dimension. They enter as a symmetric term added to **both** sides:

    vol          = ((att_home + att_away) - (def_home + def_away)) / 2
    log lambda_h = log(mu/2) + k*d + alpha*vol
    log lambda_a = log(mu/2) - k*d + alpha*vol

so two strong attacks vs. two leaky defences (``vol`` high) lift both rates → a high-scoring
match, while two stingy defences (``vol`` low/negative) damp them → a tight one. The Elo
tendency term ``k*d`` is left at **full** strength, so the well-calibrated win/draw/loss split
is untouched; ``alpha`` only adds goal-volume information. ``alpha=0`` reproduces the pure-Elo
model exactly. Crucially the off/def *difference* (which side is stronger) is deliberately
**not** added on top of Elo — backtesting showed the scalar Elo already captures tendency
better than the fitted ratings do, so only their symmetric (volume) part earns its place.
Because att/def are zero-centred over the fitting field, an average matchup still expects
``mu`` total goals. ``tippspiel tune`` selects ``alpha``. Rates stay strictly positive for any
inputs (exponential form).

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
        alpha: float = 0.0,
    ) -> None:
        self.mu = mu
        self.k = k
        self.gmax = gmax
        self.rho = rho
        self.host_elo_bonus = host_elo_bonus
        self.ko_goal_scale = ko_goal_scale
        # Off/def volume weight: 0 = pure Elo (back-compatible default); higher feeds more of
        # the fitted attack-vs-defence goal-volume signal into both sides' rates.
        self.alpha = alpha

    @property
    def params(self) -> dict:
        return {
            "mu": self.mu,
            "k": self.k,
            "gmax": self.gmax,
            "rho": self.rho,
            "host_elo_bonus": self.host_elo_bonus,
            "ko_goal_scale": self.ko_goal_scale,
            "alpha": self.alpha,
        }

    def goal_rates(
        self,
        elo_home: float,
        elo_away: float,
        home_is_host: bool = False,
        att_home: float = 0.0,
        def_home: float = 0.0,
        att_away: float = 0.0,
        def_away: float = 0.0,
    ) -> tuple[float, float]:
        bonus = self.host_elo_bonus if home_is_host else 0.0
        d = elo_home - elo_away + bonus
        log_base = math.log(self.mu / 2.0)
        # Symmetric off/def "volume" term: combined attack minus combined defence. Added to
        # both sides so it shifts total goals without touching the Elo win/draw/loss tendency.
        vol = self.alpha * ((att_home + att_away) - (def_home + def_away)) / 2.0
        log_h = log_base + self.k * d + vol
        log_a = log_base - self.k * d + vol
        return math.exp(log_h), math.exp(log_a)

    def scoreline_matrix(self, lambda_h: float, lambda_a: float) -> np.ndarray:
        ph = _poisson_pmf_vector(lambda_h, self.gmax)
        pa = _poisson_pmf_vector(lambda_a, self.gmax)
        matrix = np.outer(ph, pa)
        if self.rho != 0.0:
            matrix = _apply_dixon_coles(matrix, lambda_h, lambda_a, self.rho)
        matrix = np.clip(matrix, 0.0, None)
        return matrix / matrix.sum()

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
        lambda_h, lambda_a = self.goal_rates(
            home.elo, away.elo, home_is_host,
            att_home=home.att_elo, def_home=home.def_elo,
            att_away=away.att_elo, def_away=away.def_elo,
        )
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
