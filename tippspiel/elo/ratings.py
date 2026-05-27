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


def run_forward_pass(matches: Iterable[HistoricalMatch], model: RatingModel) -> RatingModel:
    """Fold the matches into ``model`` in ``(date, home, away)`` order (deterministic regardless of
    input order) and return the model, so callers can read scalar ratings *or* a richer state
    (e.g. attack/defence pairs)."""
    for m in sorted(matches, key=lambda x: (x.date, x.home, x.away)):
        model.seed(m.home)
        model.seed(m.away)
        model.update(m)
    return model


def build_ratings(matches: Iterable[HistoricalMatch], model: RatingModel) -> dict[str, float]:
    """Run the chronological forward pass and return team -> scalar Elo."""
    return run_forward_pass(matches, model).ratings()
