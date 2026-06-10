"""Shared utility: 1X2 -> scoreline expansion (spec §6.2.5).

Given an L/D/W triple and an assumed total-goals level, return a ScorelineDistribution
consistent with that triple. Used by any outcome-only predictor (the Phase 3 market
predictor). Implementation: search for the Poisson (lambda_h, lambda_a) pair whose
matrix reproduces the target L/D/W, holding lambda_h + lambda_a fixed at the assumed
total and adjusting the split. With ``match_draw=True`` the total is solved too (an
outer bisection matches the de-vigged draw price instead of discarding it).

Also home to ``pool_log_linear``, the model x market ensemble used by the blended
``MarketOddsPredictor``.
"""

from __future__ import annotations

import numpy as np

from ..model.scoreline import ScorelineDistribution
from .elo_poisson import _poisson_pmf_vector


def _ldw_of(matrix: np.ndarray) -> tuple[float, float, float]:
    total = matrix.sum()
    home = np.tril(matrix, -1).sum() / total
    draw = np.trace(matrix) / total
    away = np.triu(matrix, 1).sum() / total
    return float(home), float(draw), float(away)


def pool_log_linear(
    market: np.ndarray, model: np.ndarray, weight: float, eps: float = 1e-12
) -> np.ndarray:
    """Log-linear (geometric) pooling of two scoreline matrices, renormalised.

    ``weight`` is the market share: 1.0 returns ``market`` (renormalised), 0.0 returns
    ``model``. Pooling two independent-Poisson matrices cell-wise keeps the result an
    independent-Poisson matrix (rates lambda_mkt^w * lambda_elo^(1-w)), so the blend stays a
    well-formed scoreline distribution and preserves the model side's shape information
    (off/def volume, Dixon-Coles) that a 1X2-level blend would discard.
    """
    pooled = np.exp(
        weight * np.log(np.clip(market, eps, None))
        + (1.0 - weight) * np.log(np.clip(model, eps, None))
    )
    return pooled / pooled.sum()


def expand_1x2_to_scoreline(
    p_home: float,
    p_draw: float,
    p_away: float,
    total_goals: float = 2.6,
    gmax: int = 7,
    iterations: int = 60,
    match_draw: bool = False,
    goal_scale: float = 1.0,
) -> ScorelineDistribution:
    """Find a Poisson scoreline matrix whose L/D/W best matches the target triple.

    Default (``match_draw=False``): ``total_goals`` is held fixed; the home/away split (and
    hence the L/D/W balance) is bisected on the log goal-rate ratio to match the target
    home-vs-away win balance. The draw probability is then *implied* by the assumed total —
    the de-vigged draw price is discarded.

    ``match_draw=True`` solves the total too: an outer bisection on ``total_goals`` matches
    the target draw probability (draw mass is monotonically decreasing in total goals at a
    fixed balance), with the inner balance bisection nested per step — so all three de-vigged
    prices shape the matrix, and a market pricing a likely draw yields a correspondingly
    tight scoreline. Unattainable draw targets clamp at the total-goals bounds.

    ``goal_scale`` multiplies the goal rates (e.g. the knockout 120-minute lift). With
    ``match_draw=False`` it is equivalent to pre-scaling ``total_goals`` (kept that way for
    back-compat); with ``match_draw=True`` it is applied *after* the solve, so the matched
    L/D/W triple refers to the market's 90-minute prices while the totals get the lift.
    """
    s = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s
    target = p_home - p_away  # win-balance to match

    def matrix_for(log_ratio: float, total: float) -> np.ndarray:
        # lambda_h * lambda_a fixed via geometric split around total/2.
        half = total / 2.0
        lambda_h = half * np.exp(log_ratio)
        lambda_a = half * np.exp(-log_ratio)
        m = np.outer(_poisson_pmf_vector(lambda_h, gmax), _poisson_pmf_vector(lambda_a, gmax))
        return m / m.sum()

    def solve_ratio(total: float, n_iter: int) -> float:
        lo, hi = -3.0, 3.0
        for _ in range(n_iter):
            mid = (lo + hi) / 2.0
            h, _d, a = _ldw_of(matrix_for(mid, total))
            if (h - a) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    if not match_draw:
        total = total_goals * goal_scale
        return ScorelineDistribution(matrix_for(solve_ratio(total, iterations), total))

    # Nested bisection; ~30 steps each is ample precision and keeps the cost of the
    # 30x30 inner solves negligible per call.
    n = min(iterations, 30)
    lo_t, hi_t = 0.8, 6.0
    for _ in range(n):
        mid_t = (lo_t + hi_t) / 2.0
        _h, d, _a = _ldw_of(matrix_for(solve_ratio(mid_t, n), mid_t))
        if d > p_draw:
            lo_t = mid_t  # too much draw mass -> needs more goals
        else:
            hi_t = mid_t
    total = (lo_t + hi_t) / 2.0
    log_ratio = solve_ratio(total, iterations)
    half = (total / 2.0) * goal_scale
    m = np.outer(
        _poisson_pmf_vector(half * np.exp(log_ratio), gmax),
        _poisson_pmf_vector(half * np.exp(-log_ratio), gmax),
    )
    return ScorelineDistribution(m / m.sum())
