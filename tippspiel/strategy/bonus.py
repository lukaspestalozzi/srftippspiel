"""Bonus question handling (spec §6.6).

Bonus questions are modelled generically: each resolves the Monte Carlo ``TournamentOutcome``
to a probability distribution over candidate answers, and the strategy recommends the mode
(the answer most likely to be exactly right, matching the pool's exact-match scoring).

Implemented questions:
  - champion        — World Champion (50 pts), from title probabilities.
  - swiss_progress  — how far Switzerland advances, from its advancement probabilities.
  - swiss_goals     — total goals Switzerland scores, from the simulated goal tally.
  - zero_zero_count — number of 0:0 matches (120-min results) across the tournament.
  - top_scorer_goals— Golden Boot tally, from a fixed historical prior (no squad data exists,
    so this is not derived from the simulation; see data/historical_stats.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..data.historical_stats import top_scorer_prior
from ..model.types import TournamentOutcome


class BonusQuestion(ABC):
    question_id: str
    points: int
    label: str = ""

    @abstractmethod
    def resolve(self, outcome: TournamentOutcome) -> dict[str, float]:
        """Return a probability distribution over candidate answers."""
        ...


class ChampionBonus(BonusQuestion):
    question_id = "champion"
    label = "Weltmeister"

    def __init__(self, points: int = 50) -> None:
        self.points = points

    def resolve(self, outcome: TournamentOutcome) -> dict[str, float]:
        dist = {
            team_id: metrics.get("wins_title", 0.0)
            for team_id, metrics in outcome.advancement.items()
        }
        return {k: v for k, v in dist.items() if v > 0.0}


class SwissProgressBonus(BonusQuestion):
    """How far does Switzerland advance? Telescopes the cumulative advancement
    probabilities into a distribution over the (mutually exclusive) exit stages."""

    question_id = "swiss_progress"
    label = "Wie weit kommt die Schweiz?"

    def __init__(self, points: int = 20, team_id: str = "SUI") -> None:
        self.points = points
        self.team_id = team_id

    def resolve(self, outcome: TournamentOutcome) -> dict[str, float]:
        a = outcome.advancement.get(self.team_id)
        if not a:
            return {}
        q = a.get("qualifies_r32", 0.0)
        r16 = a.get("reach_r16", 0.0)
        qf = a.get("reach_qf", 0.0)
        sf = a.get("reach_sf", 0.0)
        final = a.get("reach_final", 0.0)
        champ = a.get("wins_title", 0.0)
        dist = {
            "Gruppenphase": 1.0 - q,
            "Sechzehntelfinal": q - r16,
            "Achtelfinal": r16 - qf,
            "Viertelfinal": qf - sf,
            "Halbfinal": sf - final,
            "Final": final - champ,
            "Weltmeister": champ,
        }
        return {k: max(0.0, v) for k, v in dist.items() if v > 0.0}


class TeamTotalGoalsBonus(BonusQuestion):
    """Total goals a team scores over the tournament (120-min play; no shootout goals)."""

    question_id = "swiss_goals"
    label = "Tore der Schweiz im Turnier"

    def __init__(self, points: int = 20, team_id: str = "SUI") -> None:
        self.points = points
        self.team_id = team_id

    def resolve(self, outcome: TournamentOutcome) -> dict[str, float]:
        return dict(outcome.team_goal_distribution.get(self.team_id, {}))


class ZeroZeroCountBonus(BonusQuestion):
    """Number of matches that end 0:0 (result after 120 minutes) across all 104 fixtures."""

    question_id = "zero_zero_count"
    label = "Anzahl 0:0-Spiele (nach 120 Min)"

    def __init__(self, points: int = 20) -> None:
        self.points = points

    def resolve(self, outcome: TournamentOutcome) -> dict[str, float]:
        return dict(outcome.zero_zero_distribution)


class TopScorerGoalsBonus(BonusQuestion):
    """Golden Boot goal tally. No player/squad data exists, so this is a fixed historical
    prior (built from recent World Cup top-scorer tallies) rather than a simulation output;
    it ignores ``outcome``. See data/historical_stats.py for the construction."""

    question_id = "top_scorer_goals"
    label = "Tore des Torschützenkönigs"

    def __init__(self, points: int = 20) -> None:
        self.points = points

    def resolve(self, outcome: TournamentOutcome) -> dict[str, float]:
        return top_scorer_prior()


# Registry so config-listed questions resolve to implementations.
_BONUS_REGISTRY: dict[str, type[BonusQuestion]] = {
    "champion": ChampionBonus,
    "swiss_progress": SwissProgressBonus,
    "swiss_goals": TeamTotalGoalsBonus,
    "zero_zero_count": ZeroZeroCountBonus,
    "top_scorer_goals": TopScorerGoalsBonus,
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
