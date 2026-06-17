"""Scalar World-Football-Elo fitter + K-importance mapping tests."""

from __future__ import annotations

from tippspiel.data.historical_results_adapter import KTiers, elo_k_importance
from tippspiel.training.offdef_elo import HistMatch
from tippspiel.training.scalar_elo import (
    ScalarEloParams,
    _goal_diff_multiplier,
    fit_scalar_elo,
)


def _m(home, away, hg, ag, *, date="2020-01-01", neutral=True, k=40.0):
    return HistMatch(date, home, away, hg, ag, weight=1.0, neutral=neutral, k_importance=k)


def test_dominant_team_rises_pool_sinks():
    # A beats everyone; the field it beats ends below the start rating, A well above.
    matches = []
    for i, opp in enumerate(["B", "C", "D", "B", "C", "D"]):
        matches.append(_m("A", opp, 2, 0, date=f"2020-01-0{i+1}"))
    out = fit_scalar_elo(matches, ScalarEloParams(start_rating=1500.0))
    assert out["A"] > 1500.0
    for opp in ("B", "C", "D"):
        assert out[opp] < 1500.0
    # Zero-sum-ish: total stays at field*start (every delta is +x/-x).
    assert abs(sum(out.values()) - 1500.0 * len(out)) < 1e-6


def test_determinism_and_order_independence():
    a = [_m("A", "B", 1, 0, date="2020-01-01"), _m("B", "C", 3, 1, date="2020-02-01")]
    b = list(reversed(a))  # fitter sorts by date, so input order must not matter
    assert fit_scalar_elo(a) == fit_scalar_elo(b)


def test_goal_difference_amplifies():
    narrow = fit_scalar_elo([_m("A", "B", 1, 0)])
    blowout = fit_scalar_elo([_m("A", "B", 5, 0)])
    assert blowout["A"] > narrow["A"] > 1500.0


def test_goal_diff_multiplier_curve():
    assert _goal_diff_multiplier(1) == 1.0
    assert _goal_diff_multiplier(2) == 1.5
    assert _goal_diff_multiplier(3) == (11 + 3) / 8.0
    assert _goal_diff_multiplier(4) > _goal_diff_multiplier(3)


def test_home_advantage_dampens_expected_home_win():
    # The same home win earns the home side fewer rating points when home advantage is modelled
    # (the win was more expected), so a neutral fit moves A further than a home-advantaged one.
    home = fit_scalar_elo([_m("A", "B", 1, 0, neutral=False)], ScalarEloParams(home_advantage=200))
    neutral = fit_scalar_elo([_m("A", "B", 1, 0, neutral=True)], ScalarEloParams(home_advantage=200))
    assert neutral["A"] > home["A"]


def test_k_importance_ordering():
    t = KTiers()
    assert elo_k_importance("Friendly", t) == t.friendly
    assert elo_k_importance("FIFA World Cup qualification", t) == t.qualifier
    assert elo_k_importance("UEFA Nations League", t) == t.qualifier
    assert elo_k_importance("UEFA Euro", t) == t.continental
    assert elo_k_importance("FIFA World Cup", t) == t.world_cup
    assert elo_k_importance("Some Obscure Cup", t) == t.minor
    assert (t.friendly < t.minor < t.qualifier < t.continental < t.world_cup)


def test_higher_k_moves_more():
    big = fit_scalar_elo([_m("A", "B", 1, 0, k=60.0)])
    small = fit_scalar_elo([_m("A", "B", 1, 0, k=20.0)])
    assert big["A"] - 1500.0 > small["A"] - 1500.0
