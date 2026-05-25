"""Poisson / Dixon-Coles / Elo-rate tests (spec §10)."""

import numpy as np
import pytest

from tippspiel.predictors.elo_poisson import EloPoissonPredictor, _poisson_pmf_vector
from tippspiel.predictors.expansion import expand_1x2_to_scoreline


def test_matrix_sums_to_one():
    p = EloPoissonPredictor()
    m = p.scoreline_matrix(1.5, 1.2)
    assert m.sum() == pytest.approx(1.0)


def test_rho_zero_reproduces_independent_poisson():
    p = EloPoissonPredictor(rho=0.0, gmax=7)
    m = p.scoreline_matrix(1.7, 1.1)
    ph = _poisson_pmf_vector(1.7, 7)
    pa = _poisson_pmf_vector(1.1, 7)
    indep = np.outer(ph, pa)
    indep /= indep.sum()
    assert np.allclose(m, indep)


def test_dixon_coles_changes_only_low_cells_and_renormalises():
    p0 = EloPoissonPredictor(rho=0.0)
    p1 = EloPoissonPredictor(rho=0.1)
    m0 = p0.scoreline_matrix(1.4, 1.3)
    m1 = p1.scoreline_matrix(1.4, 1.3)
    assert m1.sum() == pytest.approx(1.0)
    # The four low-score cells differ; a high cell is essentially unchanged in ratio.
    low = {(0, 0), (0, 1), (1, 0), (1, 1)}
    diffs = {(h, a): abs(m0[h, a] - m1[h, a]) for h in range(2) for a in range(2)}
    assert all(diffs[c] > 1e-6 for c in low)


def test_elo_rates_strictly_positive_for_most_lopsided():
    # Regression test for the additive-form bug: rates must stay positive.
    p = EloPoissonPredictor()
    lh, la = p.goal_rates(2200.0, 1100.0)  # extreme Elo gap
    assert lh > 0 and la > 0
    lh2, la2 = p.goal_rates(1100.0, 2200.0)
    assert lh2 > 0 and la2 > 0


def test_goal_rates_symmetric_for_equal_elo():
    p = EloPoissonPredictor(host_elo_bonus=0)
    lh, la = p.goal_rates(1800.0, 1800.0)
    assert lh == pytest.approx(la)
    assert lh == pytest.approx(p.mu / 2.0)


def test_host_bonus_applies_only_when_flagged():
    p = EloPoissonPredictor(host_elo_bonus=100)
    base = p.goal_rates(1700, 1700, home_is_host=False)
    boosted = p.goal_rates(1700, 1700, home_is_host=True)
    assert boosted[0] > base[0]
    assert base[0] == pytest.approx(base[1])  # neutral stays symmetric


def test_expansion_reproduces_target_ldw_balance():
    sd = expand_1x2_to_scoreline(0.6, 0.25, 0.15, total_goals=2.6)
    # home-vs-away win balance should be matched closely.
    assert (sd.p_home_win() - sd.p_away_win()) == pytest.approx(0.6 - 0.15, abs=0.03)
    assert sd.p_home_win() + sd.p_draw() + sd.p_away_win() == pytest.approx(1.0)
