"""World Football Elo (the eloratings.net algorithm) as a ``RatingModel``.

    We      = 1 / (1 + 10^(-dr/400))          expected home result
    dr      = R_home + home_advantage*(not neutral) - R_away
    W       = 1.0 / 0.5 / 0.0                   home win / draw / loss
    G       = goal-difference multiplier        (1, 1.5, or (11+|gd|)/8)
    delta   = K * G * (W - We)                  K from the importance tier * recency weight
    R_home += delta ;  R_away -= delta          zero-sum
"""

from __future__ import annotations

from collections.abc import Iterable

from .config import EloConfig
from .matches import HistoricalMatch
from .ratings import RatingModel


def goal_difference_multiplier(goal_diff: int) -> float:
    agd = abs(goal_diff)
    if agd <= 1:
        return 1.0
    if agd == 2:
        return 1.5
    return (11 + agd) / 8.0


class WorldFootballElo(RatingModel):
    def __init__(self, cfg: EloConfig) -> None:
        self.cfg = cfg
        self._r: dict[str, float] = {}
        self._tier_k = [(sub.lower(), float(k)) for sub, k in cfg.tier_k]

    def seed(self, team: str) -> None:
        self._r.setdefault(team, self.cfg.seed_rating)

    def teams(self) -> Iterable[str]:
        return self._r.keys()

    def rating(self, team: str) -> float:
        return self._r[team]

    def k_for(self, tournament: str) -> float:
        text = (tournament or "").lower()
        for sub, k in self._tier_k:
            if sub in text:
                return k
        return self.cfg.tier_k_fallback

    def _dr(self, m: HistoricalMatch) -> float:
        ha = 0.0 if m.neutral else self.cfg.home_advantage
        return (self._r[m.home] + ha) - self._r[m.away]

    def expected(self, m: HistoricalMatch) -> float:
        return 1.0 / (1.0 + 10.0 ** (-self._dr(m) / 400.0))

    def update(self, m: HistoricalMatch) -> None:
        we = self.expected(m)
        goal_diff = m.home_score - m.away_score
        w = 1.0 if goal_diff > 0 else (0.5 if goal_diff == 0 else 0.0)
        g = goal_difference_multiplier(goal_diff)
        k = self.k_for(m.tournament) * m.weight
        delta = k * g * (w - we)
        self._r[m.home] += delta
        self._r[m.away] -= delta
