"""Bonus question handling (spec §6.6).

Bonus questions are modelled generically. Only ChampionBonus (50 pts) is implemented;
the full list of further 20-point questions is an open question for Q and is wired up
via this same mechanism once supplied.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..model.types import TournamentOutcome


class BonusQuestion(ABC):
    question_id: str
    points: int

    @abstractmethod
    def resolve(self, outcome: TournamentOutcome) -> dict[str, float]:
        """Return a probability distribution over candidate answers."""
        ...


class ChampionBonus(BonusQuestion):
    question_id = "champion"

    def __init__(self, points: int = 50) -> None:
        self.points = points

    def resolve(self, outcome: TournamentOutcome) -> dict[str, float]:
        dist = {
            team_id: metrics.get("wins_title", 0.0)
            for team_id, metrics in outcome.advancement.items()
        }
        return {k: v for k, v in dist.items() if v > 0.0}


# Registry so config-listed questions resolve to implementations.
_BONUS_REGISTRY: dict[str, type[BonusQuestion]] = {
    "champion": ChampionBonus,
}


def build_bonus_questions(configs) -> list[BonusQuestion]:
    """Instantiate bonus questions from config entries (id + points)."""
    questions: list[BonusQuestion] = []
    for cfg in configs:
        cls = _BONUS_REGISTRY.get(cfg.id)
        if cls is None:
            # Unknown question id: no resolver yet (awaiting Q's full list). Skip
            # rather than fail so the rest of the pipeline still runs.
            continue
        questions.append(cls(points=cfg.points))
    return questions
