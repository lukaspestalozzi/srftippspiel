"""Core typed, immutable data model (spec §5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from .stages import Stage

if TYPE_CHECKING:
    from .scoreline import ScorelineDistribution


@dataclass(frozen=True)
class Team:
    team_id: str
    name: str
    elo: float
    elo_trend: float | None = None


@dataclass(frozen=True)
class TeamRef:
    """A match participant that may not yet be known (spec §5.3).

    Either a concrete ``team_id`` or a ``placeholder`` describing how the slot is
    filled (e.g. ``"winner of Group A"``, ``"3rd place from Group B/E/F/I"``,
    ``"winner of match 73"``). The simulator resolves placeholders; the predictor and
    tip strategy operate only on concrete participants.
    """

    team_id: str | None = None
    placeholder: str | None = None

    def __post_init__(self) -> None:
        if (self.team_id is None) == (self.placeholder is None):
            raise ValueError("TeamRef must have exactly one of team_id or placeholder")

    @property
    def is_concrete(self) -> bool:
        return self.team_id is not None

    @classmethod
    def parse(cls, raw: str) -> "TeamRef":
        """A bare 3-letter uppercase code is a concrete team; anything else is a placeholder."""
        s = raw.strip()
        if len(s) == 3 and s.isalpha() and s.isupper():
            return cls(team_id=s)
        return cls(placeholder=s)


@dataclass(frozen=True)
class Match:
    match_id: str
    stage: Stage
    home: TeamRef
    away: TeamRef
    kickoff: datetime  # timezone-aware, stored UTC
    group: str | None = None
    venue_country: str | None = None

    @property
    def participants_known(self) -> bool:
        return self.home.is_concrete and self.away.is_concrete


@dataclass(frozen=True)
class Result:
    match_id: str
    home_goals: int
    away_goals: int
    # For knockout matches decided by penalties: who actually advanced. The scoreline
    # is the result after 120 minutes (a shootout counts as a draw for tip purposes).
    winner_team_id: str | None = None


@dataclass(frozen=True)
class MatchPrediction:
    match_id: str
    scoreline: "ScorelineDistribution"
    predictor_name: str
    predictor_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Tip:
    match_id: str
    tip_home: int
    tip_away: int
    expected_points: float
    rationale: str = ""


@dataclass(frozen=True)
class TipSet:
    tips: dict[str, Tip]  # match_id -> Tip (only for fixtures we tip)
    bonus_answers: dict[str, str] = field(default_factory=dict)  # question_id -> answer


@dataclass(frozen=True)
class TournamentOutcome:
    """Aggregated Monte Carlo output (spec §5.8)."""

    advancement: dict[str, dict[str, float]]  # team_id -> {metric: probability}
    opponent_distribution: dict[str, dict[str, dict[str, float]]]
    bonus_probabilities: dict[str, dict[str, float]]  # question_id -> {answer: prob}
    mc_iterations: int
    mc_seed: int
    mc_standard_error: float
