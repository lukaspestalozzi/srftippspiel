"""EV optimiser tests with hand-computed cases (spec §10)."""

import math

import numpy as np
import pytest

from tippspiel.model.scoreline import ScorelineDistribution
from tippspiel.strategy.expected_points import best_tip, expected_points


def _dist(cells: dict[tuple[int, int], float], gmax: int) -> ScorelineDistribution:
    m = np.zeros((gmax + 1, gmax + 1))
    for (h, a), p in cells.items():
        m[h, a] = p
    return ScorelineDistribution(m)


def _poisson_dist(lh: float, la: float, gmax: int = 7) -> ScorelineDistribution:
    ph = np.array([math.exp(-lh) * lh**k / math.factorial(k) for k in range(gmax + 1)])
    pa = np.array([math.exp(-la) * la**k / math.factorial(k) for k in range(gmax + 1)])
    m = np.outer(ph, pa)
    return ScorelineDistribution(m / m.sum())


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def test_clear_favorite_hand_computed():
    # Home heavily favoured. Hand calc: EV(2-0) = 5*0.8 + 0.5 + 0.9 + 3*0.5 = 6.9.
    dist = _dist({(2, 0): 0.5, (1, 0): 0.3, (0, 0): 0.1, (1, 1): 0.1}, gmax=2)
    assert expected_points(dist, 2, 0, 1) == pytest.approx(6.9)
    assert expected_points(dist, 1, 0, 1) == pytest.approx(6.2)
    th, ta, ev = best_tip(dist, weight=1)
    assert (th, ta) == (2, 0)
    assert ev == pytest.approx(6.9)


def test_near_even_draw_hand_computed():
    # Symmetric distribution with draw mass. Hand calc: EV(1-1) = 5*0.4 + 0.45 + 0.45 + 3*0.4 = 4.1.
    dist = _dist(
        {(0, 0): 0.2, (1, 1): 0.2, (1, 0): 0.15, (0, 1): 0.15,
         (2, 1): 0.1, (1, 2): 0.1, (2, 0): 0.05, (0, 2): 0.05},
        gmax=2,
    )
    assert expected_points(dist, 1, 1, 1) == pytest.approx(4.1)
    assert expected_points(dist, 1, 0, 1) == pytest.approx(3.1)
    th, ta, ev = best_tip(dist, weight=1)
    assert (th, ta) == (1, 1)
    assert ev == pytest.approx(4.1)


def test_knockout_doubling():
    # Same favourite distribution, knockout weight W=2 -> EV doubles to 13.8.
    dist = _dist({(2, 0): 0.5, (1, 0): 0.3, (0, 0): 0.1, (1, 1): 0.1}, gmax=2)
    th, ta, ev = best_tip(dist, weight=2)
    assert (th, ta) == (2, 0)
    assert ev == pytest.approx(13.8)


def test_exact_score_is_sum_of_components():
    # If a single cell has all the mass, EV of tipping it == 10 (5+1+1+3) at W=1.
    dist = _dist({(3, 1): 1.0}, gmax=4)
    assert expected_points(dist, 3, 1, 1) == pytest.approx(10.0)
    assert expected_points(dist, 3, 1, 2) == pytest.approx(20.0)


def test_tiebreak_is_deterministic():
    # Two tips with identical EV: prefer higher exact-cell prob, then fewer goals.
    dist = _dist({(1, 0): 0.5, (0, 1): 0.5}, gmax=3)
    th, ta, _ = best_tip(dist, weight=1)
    # Both 1-0 and 0-1 have equal EV and exact prob; lower total goals tie, lower home goals -> 0-1.
    assert (th, ta) == (0, 1)


def test_realism_tolerance_zero_is_legacy():
    # Default tolerance reproduces the strict EV-maximiser byte-for-byte (back-compat).
    dist = _dist({(2, 0): 0.5, (1, 0): 0.3, (0, 0): 0.1, (1, 1): 0.1}, gmax=2)
    assert best_tip(dist, 1, 0.0) == best_tip(dist, 1) == (2, 0, pytest.approx(6.9))
    tie = _dist({(1, 0): 0.5, (0, 1): 0.5}, gmax=3)
    assert best_tip(tie, 1, 0.0)[:2] == (0, 1)  # legacy tie-break preserved


def test_realism_flips_shutout_to_same_margin_both_score():
    # Goal-rich favourite: strict EV tips the shutout 1:0; a small tolerance moves it to the
    # nearest expected scoreline 2:1 — both teams score, with the SAME tendency and margin.
    dist = _poisson_dist(2.0, 0.7)
    assert best_tip(dist, 1, 0.0)[:2] == (1, 0)
    rh, ra, _ = best_tip(dist, 1, 0.15)
    assert (rh, ra) == (2, 1)
    assert _sgn(rh - ra) == _sgn(1 - 0)  # tendency preserved (home win)
    assert (rh - ra) == (1 - 0)          # margin preserved (+1)


def test_realism_pick_is_never_farther_from_expected_than_strict():
    # The tolerance pick minimises L1 distance to the expected scoreline over a superset of the
    # strict candidates, so it is never farther from the expected score.
    for lh, la in [(2.0, 0.7), (1.6, 1.1), (2.4, 1.5)]:
        dist = _poisson_dist(lh, la)
        eh, ea = dist.expected_goals()
        sh, sa, _ = best_tip(dist, 1, 0.0)
        rh, ra, _ = best_tip(dist, 1, 0.3)
        assert abs(rh - eh) + abs(ra - ea) <= abs(sh - eh) + abs(sa - ea) + 1e-9


def test_realism_keeps_a_tight_game_low_total():
    # Few expected goals -> the tip stays low-total even with tolerance (realism is proportional
    # to the prediction; it does not invent scoring).
    dist = _poisson_dist(0.9, 0.6)
    rh, ra, _ = best_tip(dist, 1, 0.15)
    assert rh + ra <= 2


def test_realism_never_flips_tendency():
    # Even a large tolerance keeps a clear favourite a home-win tip (never a draw / away win).
    dist = _poisson_dist(2.2, 0.8)
    rh, ra, _ = best_tip(dist, 1, 0.5)
    assert rh > ra
