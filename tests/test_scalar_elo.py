"""Scalar World-Football-Elo fitter + K-importance mapping tests."""

from __future__ import annotations

from tippspiel.data.historical_results_adapter import KTiers, elo_k_importance
from tippspiel.training.offdef_elo import HistMatch
from tippspiel.training.scalar_elo import (
    ScalarEloParams,
    _goal_diff_multiplier,
    fit_scalar_elo,
    fit_scalar_elo_history,
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


def test_same_date_order_independence():
    # Matches sharing a date must still be order-independent (the content secondary sort key);
    # Elo is path-dependent, so a date-only sort would let input order change the result.
    a = [_m("A", "B", 2, 0, date="2020-01-01"), _m("C", "D", 1, 0, date="2020-01-01"),
         _m("A", "C", 0, 1, date="2020-01-01")]
    assert fit_scalar_elo(a) == fit_scalar_elo(list(reversed(a)))


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


# --------------------------------------------------------------------------- history
def _history_corpus():
    return [
        _m("A", "B", 2, 0, date="2020-01-01"),
        _m("A", "C", 1, 1, date="2020-06-01"),
        _m("B", "C", 0, 3, date="2021-01-01"),
        _m("A", "B", 0, 1, date="2021-06-01"),
    ]


def test_history_endpoint_matches_fit():
    # Same single pass: a tracked team's last point is exactly its fitted rating.
    matches = _history_corpus()
    fitted = fit_scalar_elo(matches)
    hist = fit_scalar_elo_history(matches, track={"A", "C"})
    assert hist["A"][-1][1] == fitted["A"]
    assert hist["C"][-1][1] == fitted["C"]


def test_history_is_chronological_one_point_per_match():
    hist = fit_scalar_elo_history(_history_corpus(), track={"A"})
    dates = [d for d, _ in hist["A"]]
    assert dates == sorted(dates) == ["2020-01-01", "2020-06-01", "2021-06-01"]  # A played 3


def test_history_tracks_only_requested_teams_and_window():
    hist = fit_scalar_elo_history(_history_corpus(), track={"B"}, start_date="2021-01-01")
    assert set(hist) == {"B"}
    # Pre-window matches still move the rating but are not recorded.
    assert [d for d, _ in hist["B"]] == ["2021-01-01", "2021-06-01"]
    assert hist["B"][-1][1] == fit_scalar_elo(_history_corpus())["B"]
