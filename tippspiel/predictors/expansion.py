"""Shared utility: 1X2 -> scoreline expansion (spec §6.2.5).

Given an L/D/W triple and an assumed total-goals level, return a ScorelineDistribution
consistent with that triple. Used by any outcome-only predictor (the Phase 3 market
predictor). Implementation: search for the Poisson (lambda_h, lambda_a) pair whose
matrix reproduces the target L/D/W, holding lambda_h + lambda_a fixed at the assumed
total and adjusting the split.
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


def expand_1x2_to_scoreline(
    p_home: float,
    p_draw: float,
    p_away: float,
    total_goals: float = 2.6,
    gmax: int = 7,
    iterations: int = 60,
) -> ScorelineDistribution:
    """Find a Poisson scoreline matrix whose L/D/W best matches the target triple.

    ``total_goals`` is held fixed; the home/away split (and hence the L/D/W balance) is
    bisected on the log goal-rate ratio to match the target home-vs-away win balance.
    """
    s = p_home + p_draw + p_away
    p_home, p_away = p_home / s, p_away / s

    def matrix_for(log_ratio: float) -> np.ndarray:
        # lambda_h * lambda_a fixed via geometric split around total_goals/2.
        half = total_goals / 2.0
        lambda_h = half * np.exp(log_ratio)
        lambda_a = half * np.exp(-log_ratio)
        m = np.outer(_poisson_pmf_vector(lambda_h, gmax), _poisson_pmf_vector(lambda_a, gmax))
        return m / m.sum()

    target = p_home - p_away  # win-balance to match
    lo, hi = -3.0, 3.0
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        h, _d, a = _ldw_of(matrix_for(mid))
        if (h - a) < target:
            lo = mid
        else:
            hi = mid
    return ScorelineDistribution(matrix_for((lo + hi) / 2.0))
