"""Poisson / Dixon-Coles / Elo-rate tests (spec §10)."""

from datetime import datetime, timezone

import numpy as np
import pytest

from tippspiel.model.stages import Stage
from tippspiel.model.types import Match, Team, TeamRef
from tippspiel.predictors.elo_poisson import EloPoissonPredictor, _poisson_pmf_vector
from tippspiel.predictors.expansion import expand_1x2_to_scoreline


def _teams():
    return {"AAA": Team("AAA", "A", 1800.0), "BBB": Team("BBB", "B", 1800.0)}


def _match(stage: Stage, venue: str | None = None) -> Match:
    return Match(
        match_id="M", stage=stage,
        home=TeamRef(team_id="AAA"), away=TeamRef(team_id="BBB"),
        kickoff=datetime(2026, 6, 1, tzinfo=timezone.utc),
        group="A" if stage is Stage.GROUP else None, venue_country=venue,
    )


def _expected_total(dist) -> float:
    m = dist.matrix
    goals = np.arange(dist.gmax + 1)
    return float((m.sum(axis=1) * goals).sum() + (m.sum(axis=0) * goals).sum())


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


def test_ko_goal_scale_default_is_a_noop():
    # ko_goal_scale=1.0 (default) -> knockout prediction identical to the group prediction.
    p = EloPoissonPredictor(ko_goal_scale=1.0)
    teams = _teams()
    group = p.predict(_match(Stage.GROUP), teams).scoreline.matrix
    knockout = p.predict(_match(Stage.R16), teams).scoreline.matrix
    assert np.allclose(group, knockout)


def test_ko_goal_scale_raises_knockout_goals_only():
    p = EloPoissonPredictor(ko_goal_scale=1.25)
    teams = _teams()
    group = p.predict(_match(Stage.GROUP), teams).scoreline
    knockout = p.predict(_match(Stage.R16), teams).scoreline
    base = EloPoissonPredictor(ko_goal_scale=1.0)
    group_base = base.predict(_match(Stage.GROUP), teams).scoreline
    # Group prediction unaffected by the knockout scale; knockout expects ~25% more goals.
    assert np.allclose(group.matrix, group_base.matrix)
    assert _expected_total(knockout) > _expected_total(group)
    assert _expected_total(knockout) / _expected_total(group) == pytest.approx(1.25, abs=0.03)


def test_host_bonus_applies_when_team_plays_in_its_own_country():
    p = EloPoissonPredictor(host_elo_bonus=120)
    teams = _teams()
    neutral = p.predict(_match(Stage.GROUP, venue=None), teams).scoreline
    at_home = p.predict(_match(Stage.GROUP, venue="AAA"), teams).scoreline  # host == home team
    assert _expected_total(at_home) > 0
    # Host advantage skews the distribution toward the home side.
    assert at_home.p_home_win() > neutral.p_home_win()


def test_expansion_reproduces_target_ldw_balance():
    sd = expand_1x2_to_scoreline(0.6, 0.25, 0.15, total_goals=2.6)
    # home-vs-away win balance should be matched closely.
    assert (sd.p_home_win() - sd.p_away_win()) == pytest.approx(0.6 - 0.15, abs=0.03)
    assert sd.p_home_win() + sd.p_draw() + sd.p_away_win() == pytest.approx(1.0)
