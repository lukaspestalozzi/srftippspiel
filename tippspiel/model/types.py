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
    # Offensive / defensive log-goal-rate deviations from the field average, fitted from
    # historical match goals (``tippspiel/training/offdef_elo.py``; snapshot persisted by
    # ``tippspiel fit-ratings``). Both default to 0.0 — an absent/un-fitted snapshot leaves the
    # predictor at its pure-Elo behaviour. Convention: higher ``att_elo`` = scores more than
    # the average team; higher ``def_elo`` = concedes fewer (stingier defence). A value of
    # +0.3 ≈ e^0.3 ≈ 1.35× the average rate. See EloPoissonPredictor for how they blend in.
    att_elo: float = 0.0
    def_elo: float = 0.0


@dataclass(frozen=True)
class KnockoutRef:
    """A structured reference describing how a knockout slot is filled (spec §5.3).

    Knockout participants are not known a-priori, so a fixture's home/away may instead
    *reference* a group placing or the outcome of an earlier match. This makes the
    tournament format self-describing from ``fixtures.csv`` alone (no separate bracket map).

    Wire syntax used in ``fixtures.csv`` (anything that is not a bare 3-letter team code):

    * ``W:A``            — winner of group A
    * ``R:A``            — runner-up of group A
    * ``3RD:74:ABCDF``   — a best-placed third fills slot 74, drawn from groups A/B/C/D/F
                           (the allowed source groups are inlined; see CLAUDE.md)
    * ``WIN:M101``       — winner of match M101
    * ``LOSE:M101``      — loser of match M101
    """

    kind: str  # winner | runner_up | third_pooled | winner_of | loser_of
    group: str | None = None  # winner / runner_up
    slot: int | None = None  # third_pooled
    allowed_groups: tuple[str, ...] = ()  # third_pooled
    match_id: str | None = None  # winner_of / loser_of

    _GROUP_KINDS = ("winner", "runner_up", "third_pooled")

    @property
    def is_group_ref(self) -> bool:
        """True if filled from group standings (first knockout round); False if from a match."""
        return self.kind in self._GROUP_KINDS

    @classmethod
    def parse(cls, raw: str) -> "KnockoutRef":
        s = raw.strip()
        tag, _, rest = s.partition(":")
        if tag == "W":
            return cls(kind="winner", group=rest)
        if tag == "R":
            return cls(kind="runner_up", group=rest)
        if tag == "3RD":
            slot_str, _, groups = rest.partition(":")
            return cls(kind="third_pooled", slot=int(slot_str),
                       allowed_groups=tuple(groups))
        if tag == "WIN":
            return cls(kind="winner_of", match_id=rest)
        if tag == "LOSE":
            return cls(kind="loser_of", match_id=rest)
        raise ValueError(f"unrecognised knockout reference {raw!r}")

    def describe(self) -> str:
        if self.kind == "winner":
            return f"Winner Group {self.group}"
        if self.kind == "runner_up":
            return f"Runner-up Group {self.group}"
        if self.kind == "third_pooled":
            return f"3rd place (slot {self.slot})"
        if self.kind == "winner_of":
            return f"Winner of {self.match_id}"
        if self.kind == "loser_of":
            return f"Loser of {self.match_id}"
        return self.kind


@dataclass(frozen=True)
class TeamRef:
    """A match participant that may not yet be known (spec §5.3).

    Either a concrete ``team_id`` or a structured ``ko_ref`` describing how the slot is
    filled. The simulator resolves knockout references from the group standings / match
    outcomes; the predictor and tip strategy operate only on concrete participants.
    """

    team_id: str | None = None
    ko_ref: KnockoutRef | None = None

    def __post_init__(self) -> None:
        if (self.team_id is None) == (self.ko_ref is None):
            raise ValueError("TeamRef must have exactly one of team_id or ko_ref")

    @property
    def is_concrete(self) -> bool:
        return self.team_id is not None

    @property
    def placeholder(self) -> str | None:
        """Human-readable description of an unresolved knockout slot (None if concrete)."""
        return self.ko_ref.describe() if self.ko_ref else None

    @classmethod
    def parse(cls, raw: str) -> "TeamRef":
        """A bare 3-letter uppercase code is a concrete team; anything else is a knockout ref."""
        s = raw.strip()
        if len(s) == 3 and s.isalpha() and s.isupper():
            return cls(team_id=s)
        return cls(ko_ref=KnockoutRef.parse(s))


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
    # team_id -> {goal_count (as str): probability} of total goals scored over the tournament
    # (120-minute scorelines only; penalty-shootout goals excluded).
    team_goal_distribution: dict[str, dict[str, float]] = field(default_factory=dict)
    # goal-count (as str) -> probability of that many 0:0 matches across all 104 fixtures.
    zero_zero_distribution: dict[str, float] = field(default_factory=dict)
