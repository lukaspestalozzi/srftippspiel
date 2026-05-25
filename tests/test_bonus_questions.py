"""Unit tests for the bonus-question resolvers (distribution math + mode recommendation)."""

import pytest

from tippspiel.config import BonusQuestionConfig
from tippspiel.model.types import TournamentOutcome
from tippspiel.strategy.bonus import (
    SwissProgressBonus,
    TeamTotalGoalsBonus,
    TopScorerGoalsBonus,
    ZeroZeroCountBonus,
    build_bonus_questions,
)


def _outcome(**kw) -> TournamentOutcome:
    base = dict(
        advancement={}, opponent_distribution={}, bonus_probabilities={},
        mc_iterations=1000, mc_seed=1, mc_standard_error=0.0,
    )
    base.update(kw)
    return TournamentOutcome(**base)


def _mode(dist):
    return max(dist, key=dist.get)


def test_swiss_progress_telescopes_and_sums_to_one():
    adv = {"SUI": {
        "reach_r32": 0.8, "reach_r16": 0.5, "reach_qf": 0.3,
        "reach_sf": 0.15, "reach_final": 0.06, "wins_title": 0.02,
    }}
    dist = SwissProgressBonus().resolve(_outcome(advancement=adv))
    assert dist["Gruppenphase"] == pytest.approx(0.2)
    assert dist["Sechzehntelfinal"] == pytest.approx(0.3)
    assert dist["Achtelfinal"] == pytest.approx(0.2)
    assert dist["Viertelfinal"] == pytest.approx(0.15)
    assert dist["Halbfinal"] == pytest.approx(0.09)
    assert dist["Final"] == pytest.approx(0.04)
    assert dist["Weltmeister"] == pytest.approx(0.02)
    assert sum(dist.values()) == pytest.approx(1.0)
    assert _mode(dist) == "Sechzehntelfinal"


def test_swiss_progress_empty_without_data():
    assert SwissProgressBonus().resolve(_outcome()) == {}


def test_team_goals_passthrough_and_mode():
    o = _outcome(team_goal_distribution={"SUI": {"4": 0.3, "6": 0.7}})
    dist = TeamTotalGoalsBonus().resolve(o)
    assert dist == {"4": 0.3, "6": 0.7}
    assert _mode(dist) == "6"


def test_zero_zero_passthrough():
    o = _outcome(zero_zero_distribution={"5": 0.4, "6": 0.6})
    assert ZeroZeroCountBonus().resolve(o) == {"5": 0.4, "6": 0.6}


def test_top_scorer_prior_valid_and_mode_in_observed_range():
    dist = TopScorerGoalsBonus().resolve(_outcome())
    assert sum(dist.values()) == pytest.approx(1.0)
    assert all(p > 0.0 for p in dist.values())
    assert 5 <= int(_mode(dist)) <= 9


def test_build_registers_all_configured_questions():
    ids = ("champion", "swiss_progress", "swiss_goals", "zero_zero_count", "top_scorer_goals")
    qs = build_bonus_questions([BonusQuestionConfig(id=i, points=20) for i in ids])
    assert {q.question_id for q in qs} == set(ids)
    assert all(q.label for q in qs)  # every question has a human-readable label


def test_unknown_question_is_skipped():
    assert build_bonus_questions([BonusQuestionConfig(id="nope", points=5)]) == []
