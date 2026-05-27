"""Attack/defence rating model: two ratings per team, fit online by SGD on the Poisson
log-likelihood of match goals.

For a match (neutral-aware home advantage on the log-goal scale)::

    lam_home = exp(c + atk[home] - def[away] + ha * (not neutral))
    lam_away = exp(c + atk[away] - def[home])

The gradient of the Poisson log-likelihood with respect to the log-rate is ``observed - expected``,
which gives the online update (a learning-rate-scaled step, weighted by the match recency weight)::

    atk[home] += lr*w*(gh - lam_home);  def[away] -= lr*w*(gh - lam_home)
    atk[away] += lr*w*(ga - lam_away);  def[home] -= lr*w*(ga - lam_away)

Unlike the zero-sum Elo update this is plain gradient ascent, so an optional ``shrinkage`` pulls
each touched rating toward 0 to regularise teams with little/sparse history. The model exposes the
attack/defence pair (consumed by ``AttackDefencePoissonPredictor``) and a combined scalar ``rating``
(``seed + 400*(atk - def)``) so the single-Elo report/emission path still works.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from .config import EloConfig
from .matches import HistoricalMatch
from .ratings import RatingModel

_SCALE = 400.0  # log-rate -> Elo-ish points, for the combined scalar rating only


class AttackDefenceElo(RatingModel):
    def __init__(self, cfg: EloConfig) -> None:
        self.cfg = cfg
        self._atk: dict[str, float] = {}
        self._def: dict[str, float] = {}

    def seed(self, team: str) -> None:
        self._atk.setdefault(team, 0.0)
        self._def.setdefault(team, 0.0)

    def teams(self) -> Iterable[str]:
        return self._atk.keys()

    def _rates(self, m: HistoricalMatch) -> tuple[float, float]:
        ha = 0.0 if m.neutral else self.cfg.ad_home_advantage
        c = self.cfg.base_log_rate
        lam_h = math.exp(c + self._atk[m.home] - self._def[m.away] + ha)
        lam_a = math.exp(c + self._atk[m.away] - self._def[m.home])
        return lam_h, lam_a

    def expected(self, m: HistoricalMatch) -> float:
        """Tendency We in [0, 1] from the rate difference (for the ranking report only)."""
        lam_h, lam_a = self._rates(m)
        return lam_h / (lam_h + lam_a)

    def update(self, m: HistoricalMatch) -> None:
        lam_h, lam_a = self._rates(m)
        lr = self.cfg.learning_rate * m.weight
        eh = m.home_score - lam_h
        ea = m.away_score - lam_a
        self._atk[m.home] += lr * eh
        self._def[m.away] -= lr * eh
        self._atk[m.away] += lr * ea
        self._def[m.home] -= lr * ea
        s = self.cfg.ad_shrinkage
        if s:
            for t in (m.home, m.away):
                self._atk[t] *= 1.0 - s
                self._def[t] *= 1.0 - s

    def attack_defence(self, team: str) -> tuple[float, float]:
        return self._atk[team], self._def[team]

    def attack_defence_ratings(self) -> dict[str, tuple[float, float]]:
        return {t: (self._atk[t], self._def[t]) for t in self._atk}

    def rating(self, team: str) -> float:
        return self.cfg.seed_rating + _SCALE * (self._atk[team] - self._def[team])
