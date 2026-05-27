"""Rating-model abstraction + the chronological forward-pass driver.

A ``RatingModel`` owns its per-team state shape; the driver only seeds teams, folds matches in
date order, and reads back one scalar Elo per team. This is the seam for a future attack/defence
model (two ratings per team): it implements the same interface and the driver is unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .matches import HistoricalMatch


class RatingModel(ABC):
    @abstractmethod
    def seed(self, team: str) -> None:
        """Initialize state for a newly-seen team (idempotent)."""

    @abstractmethod
    def expected(self, match: HistoricalMatch) -> float:
        """Expected home result We in [0, 1] given current state + venue."""

    @abstractmethod
    def update(self, match: HistoricalMatch) -> None:
        """Fold ONE chronological match into team state (honouring ``match.weight``)."""

    @abstractmethod
    def rating(self, team: str) -> float:
        """Single scalar Elo for ``team`` — the value emitted to teams.csv."""

    @abstractmethod
    def teams(self) -> Iterable[str]:
        """All teams seen so far."""

    def ratings(self) -> dict[str, float]:
        return {t: self.rating(t) for t in self.teams()}


def build_ratings(matches: Iterable[HistoricalMatch], model: RatingModel) -> dict[str, float]:
    """Run the single chronological forward pass and return team -> scalar Elo.

    Matches are sorted by ``(date, home, away)`` so the result is deterministic regardless of
    input order.
    """
    for m in sorted(matches, key=lambda x: (x.date, x.home, x.away)):
        model.seed(m.home)
        model.seed(m.away)
        model.update(m)
    return model.ratings()
