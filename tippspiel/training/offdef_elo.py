"""Offensive / defensive Elo: per-team attack & defence ratings fitted from match goals.

A single scalar Elo applies symmetrically to both sides' goal rates, so it cannot separate
an attack-minded side from a defensive one at the same overall strength. This module learns
**two** ratings per team from the goals teams actually score and concede, using an online,
Elo-style update — the same ``rating += K * (observed - expected)`` shape as classical Elo,
but applied to **goals** rather than win/loss.

Model (log-Poisson). For a match with the home team's venue advantage ``gamma`` (0 at a
neutral venue), the expected goals are

    lambda_home = (mu / 2) * exp(att_home - def_away + gamma)
    lambda_away = (mu / 2) * exp(att_away - def_home)

and after each match we nudge the four ratings toward the residual ``g - lambda_hat``:

    att_home += k_att * w * (g_home - lambda_home)      # scored more than expected -> attack up
    def_away -= k_def * w * (g_home - lambda_home)      # conceded more than expected -> defence down
    att_away += k_att * w * (g_away - lambda_away)
    def_home -= k_def * w * (g_away - lambda_away)

``w`` is the match-importance weight (friendlies count less than World Cup games). This update
is exactly stochastic gradient descent on the Poisson negative-log-likelihood w.r.t. the
log-rates, i.e. the natural "Elo for goals". Matches are processed in chronological order so
ratings track form; a few epochs let early ratings benefit from later context.

Sign convention (matches ``EloPoissonPredictor``): higher ``att`` = scores more than the
field; higher ``def`` = concedes fewer (stingier). Ratings are finally **zero-centred** over
the field separately for att and def, so an average matchup expects ``mu`` total goals and the
absolute level carries no meaning — only ``att_home - def_away`` does.

This is an offline one-shot fit (run by ``tippspiel fit-offdef``), not the simulator, so it is
a plain sequential loop rather than a vectorised kernel — the updates are inherently online.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class OffDefParams:
    """Fit hyperparameters. ``mu`` is the corpus baseline mean goals/match (the rate an
    average matchup expects); the rest control the online update."""

    mu: float = 2.7
    k_att: float = 0.02
    k_def: float = 0.02
    gamma_home: float = 0.20
    residual_cap: float = 5.0
    epochs: int = 3


@dataclass(frozen=True)
class HistMatch:
    """One historical match in the fitting corpus (already name-resolved + weighted).

    Carries two independent importance signals derived from the corpus competition label:
    ``weight`` (the off/def goal-fit weight, see ``WeightTiers``) and ``k_importance`` (the
    World-Football-Elo K base for the scalar-Elo fit, see ``scalar_elo``). ``k_importance``
    defaults to 0 so records built without it leave the scalar fitter inert.
    """

    date: str  # ISO yyyy-mm-dd; used only for chronological ordering
    home: str
    away: str
    home_goals: int
    away_goals: int
    weight: float
    neutral: bool
    k_importance: float = 0.0


@dataclass(frozen=True)
class OffDefRating:
    att: float
    def_: float


def fit_off_def(
    matches: list[HistMatch], params: OffDefParams | None = None
) -> dict[str, OffDefRating]:
    """Fit attack/defence ratings for every team appearing in ``matches``.

    Deterministic: matches are sorted by date (stable), so the same corpus + params always
    yield the same ratings. Returns ``team_name -> OffDefRating`` (zero-centred).
    """
    params = params or OffDefParams()
    ordered = sorted(matches, key=lambda m: m.date)
    att: dict[str, float] = defaultdict(float)
    deff: dict[str, float] = defaultdict(float)
    half = params.mu / 2.0
    cap = params.residual_cap

    for _epoch in range(params.epochs):
        for m in ordered:
            home_adv = 0.0 if m.neutral else params.gamma_home
            lam_h = half * math.exp(att[m.home] - deff[m.away] + home_adv)
            lam_a = half * math.exp(att[m.away] - deff[m.home])
            res_h = _clip(m.home_goals - lam_h, cap)
            res_a = _clip(m.away_goals - lam_a, cap)
            wkh = params.k_att * m.weight
            wkd = params.k_def * m.weight
            att[m.home] += wkh * res_h
            deff[m.away] -= wkd * res_h
            att[m.away] += wkh * res_a
            deff[m.home] -= wkd * res_a

    return _centre(att, deff)


def _clip(x: float, cap: float) -> float:
    """Clamp a goal residual to +/- cap so blowouts don't dominate the fit."""
    return cap if x > cap else (-cap if x < -cap else x)


def _centre(att: dict[str, float], deff: dict[str, float]) -> dict[str, OffDefRating]:
    """Zero-centre att and def separately over the field, so the average matchup expects mu
    total goals (only ``att_home - def_away`` carries signal; a common shift cancels)."""
    teams = set(att) | set(deff)
    if not teams:
        return {}
    mean_att = sum(att.values()) / len(teams)
    mean_def = sum(deff.values()) / len(teams)
    return {
        t: OffDefRating(att=att[t] - mean_att, def_=deff[t] - mean_def)
        for t in teams
    }
