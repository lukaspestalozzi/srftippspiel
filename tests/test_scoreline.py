"""ScorelineDistribution unit + property tests (spec §10)."""

import numpy as np
import pytest

from tippspiel.model.scoreline import ScorelineDistribution
from tippspiel.predictors.elo_poisson import EloPoissonPredictor


def test_renormalises_on_construction():
    sd = ScorelineDistribution(np.ones((3, 3)))
    assert sd.matrix.sum() == pytest.approx(1.0)
    assert sd.cell(0, 0) == pytest.approx(1 / 9)


def test_rejects_zero_mass():
    with pytest.raises(ValueError):
        ScorelineDistribution(np.zeros((3, 3)))


def test_marginals_and_ldw_consistency():
    p = EloPoissonPredictor()
    sd = ScorelineDistribution(p.scoreline_matrix(1.8, 1.0))
    # L/D/W partitions all mass.
    assert sd.p_home_win() + sd.p_draw() + sd.p_away_win() == pytest.approx(1.0)
    # Marginals each sum to 1.
    assert sum(sd.p_home_goals(h) for h in range(sd.gmax + 1)) == pytest.approx(1.0)
    assert sum(sd.p_away_goals(a) for a in range(sd.gmax + 1)) == pytest.approx(1.0)
    # Goal-difference distribution sums to 1 across its full support.
    gd = sum(sd.p_goal_difference(d) for d in range(-sd.gmax, sd.gmax + 1))
    assert gd == pytest.approx(1.0)


def test_most_likely_is_sorted_and_deterministic():
    sd = ScorelineDistribution(
        np.array([[0.4, 0.1, 0.0], [0.1, 0.3, 0.0], [0.0, 0.1, 0.0]])
    )
    top = sd.most_likely_scorelines(2)
    assert top[0][:2] == (0, 0)
    assert top[1][:2] == (1, 1)


@pytest.mark.parametrize("elo_h,elo_a", [(1900, 1200), (1500, 1500), (1300, 1850)])
def test_all_probabilities_in_unit_interval(elo_h, elo_a):
    p = EloPoissonPredictor()
    lh, la = p.goal_rates(elo_h, elo_a)
    sd = ScorelineDistribution(p.scoreline_matrix(lh, la))
    for fn in (sd.p_home_win, sd.p_draw, sd.p_away_win):
        assert 0.0 <= fn() <= 1.0
    assert np.all(sd.matrix >= 0.0)
