"""ScorelineDistribution: a 2-D probability matrix over (home_goals, away_goals).

The central prediction object (spec §5.5). Rows index home goals, columns away goals,
for goals in ``[0, gmax]``. The matrix is renormalised to sum to 1.0 after truncation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScorelineDistribution:
    matrix: np.ndarray  # shape (gmax+1, gmax+1), sums to 1.0

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=float)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError(f"scoreline matrix must be square 2-D, got {m.shape}")
        total = m.sum()
        if total <= 0:
            raise ValueError("scoreline matrix must have positive mass")
        # Store a normalised copy; frozen dataclass requires object.__setattr__.
        object.__setattr__(self, "matrix", m / total)

    @property
    def gmax(self) -> int:
        return self.matrix.shape[0] - 1

    # --- L/D/W triple -----------------------------------------------------------
    def p_home_win(self) -> float:
        return float(np.tril(self.matrix, -1).sum())  # home > away

    def p_draw(self) -> float:
        return float(np.trace(self.matrix))  # home == away

    def p_away_win(self) -> float:
        return float(np.triu(self.matrix, 1).sum())  # home < away

    # --- marginals --------------------------------------------------------------
    def p_home_goals(self, h: int) -> float:
        if not 0 <= h <= self.gmax:
            return 0.0
        return float(self.matrix[h, :].sum())

    def p_away_goals(self, a: int) -> float:
        if not 0 <= a <= self.gmax:
            return 0.0
        return float(self.matrix[:, a].sum())

    def p_goal_difference(self, d: int) -> float:
        """P(home_goals - away_goals == d)."""
        return float(np.trace(self.matrix, offset=-d))

    def cell(self, h: int, a: int) -> float:
        if not (0 <= h <= self.gmax and 0 <= a <= self.gmax):
            return 0.0
        return float(self.matrix[h, a])

    def most_likely_scorelines(self, n: int) -> list[tuple[int, int, float]]:
        """Top-n cells (home_goals, away_goals, prob), highest first.

        Ties broken deterministically by lower total goals, then lower home goals.
        """
        cells = [
            (h, a, float(self.matrix[h, a]))
            for h in range(self.gmax + 1)
            for a in range(self.gmax + 1)
        ]
        cells.sort(key=lambda c: (-c[2], c[0] + c[1], c[0]))
        return cells[:n]
