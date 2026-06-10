"""MarketOddsPredictor + odds loading / de-vig tests (spec §6.2.6)."""

from datetime import datetime, timezone

import numpy as np
import pytest

from tippspiel.data.base import Odds1X2
from tippspiel.data.file_provider import FileDataProvider, _devig_proportional
from tippspiel.model.stages import Stage
from tippspiel.model.types import Match, Team, TeamRef
from tippspiel.predictors.elo_poisson import EloPoissonPredictor
from tippspiel.predictors.expansion import expand_1x2_to_scoreline, pool_log_linear
from tippspiel.predictors.market_odds import MarketOddsPredictor


def _teams():
    return {"AAA": Team("AAA", "A", 1800.0), "BBB": Team("BBB", "B", 1700.0)}


def _match(match_id="M", stage=Stage.GROUP):
    return Match(
        match_id=match_id, stage=stage,
        home=TeamRef(team_id="AAA"), away=TeamRef(team_id="BBB"),
        kickoff=datetime(2026, 6, 1, tzinfo=timezone.utc),
        group="A" if stage is Stage.GROUP else None, venue_country=None,
    )


def test_devig_proportional_normalises_and_is_deterministic():
    o = _devig_proportional(2.0, 3.5, 4.0)
    assert o.p_home + o.p_draw + o.p_away == pytest.approx(1.0)
    # Favourite (lowest odds) keeps the highest probability.
    assert o.p_home > o.p_draw and o.p_home > o.p_away
    assert _devig_proportional(2.0, 3.5, 4.0) == o  # pure arithmetic, repeatable


def test_devig_removes_margin():
    # Fair (margin-free) odds: implied probs already sum to 1 -> unchanged.
    fair = _devig_proportional(2.0, 4.0, 4.0)  # 0.5 + 0.25 + 0.25 = 1.0
    assert (fair.p_home, fair.p_draw, fair.p_away) == pytest.approx((0.5, 0.25, 0.25))


def test_odds_path_reproduces_win_balance():
    odds = {"M": _devig_proportional(1.5, 4.0, 6.0)}  # strong home favourite
    p = MarketOddsPredictor(odds=odds, total_goals=2.6, gmax=7)
    dist = p.predict(_match(), _teams()).scoreline
    o = odds["M"]
    # Same balance the shared expander targets (matches the expansion test tolerance).
    assert (dist.p_home_win() - dist.p_away_win()) == pytest.approx(o.p_home - o.p_away, abs=0.03)
    assert dist.p_home_win() + dist.p_draw() + dist.p_away_win() == pytest.approx(1.0)
    # Equivalent to calling the expander directly with the same args.
    direct = expand_1x2_to_scoreline(o.p_home, o.p_draw, o.p_away, total_goals=2.6, gmax=7)
    assert np.allclose(dist.matrix, direct.matrix)


def test_falls_back_to_elo_when_no_odds_for_match():
    fb = EloPoissonPredictor(gmax=7)
    p = MarketOddsPredictor(odds={}, fallback=fb, gmax=7)
    got = p.predict(_match(), _teams()).scoreline
    want = fb.predict(_match(), _teams()).scoreline
    assert np.allclose(got.matrix, want.matrix)


def test_falls_back_on_unknown_match_id():
    # The simulator generates synthetic _pair_* match-ids that never appear in the odds map.
    fb = EloPoissonPredictor(gmax=7)
    p = MarketOddsPredictor(odds={"M": _devig_proportional(2.0, 3.3, 3.5)}, fallback=fb, gmax=7)
    synthetic = _match(match_id="_pair_AAA_BBB")
    assert np.allclose(
        p.predict(synthetic, _teams()).scoreline.matrix,
        fb.predict(synthetic, _teams()).scoreline.matrix,
    )


def test_gmax_attribute_present_and_must_match_fallback():
    p = MarketOddsPredictor(gmax=7)
    assert p.gmax == 7
    with pytest.raises(ValueError):
        MarketOddsPredictor(fallback=EloPoissonPredictor(gmax=5), gmax=7)


def test_ko_goal_scale_lifts_knockout_total_only():
    odds = {"M": _devig_proportional(2.2, 3.3, 3.3)}
    p = MarketOddsPredictor(odds=odds, total_goals=2.6, gmax=7, ko_goal_scale=1.3)

    def total(dist):
        m = dist.matrix
        goals = np.arange(dist.gmax + 1)
        return float((m.sum(axis=1) * goals).sum() + (m.sum(axis=0) * goals).sum())

    group = p.predict(_match("M", Stage.GROUP), _teams()).scoreline
    ko = p.predict(_match("M", Stage.R16), _teams()).scoreline
    assert total(ko) > total(group)
    # Tendency is unaffected by total-goals scaling (only exact-scoreline mass shifts).
    assert ko.p_home_win() == pytest.approx(group.p_home_win(), abs=0.02)


def test_market_weight_one_is_byte_identical_to_pure_market_path():
    odds = {"M": _devig_proportional(1.5, 4.0, 6.0)}
    o = odds["M"]
    p = MarketOddsPredictor(odds=odds, total_goals=2.6, gmax=7)  # default market_weight=1.0
    direct = expand_1x2_to_scoreline(o.p_home, o.p_draw, o.p_away, total_goals=2.6, gmax=7)
    assert np.array_equal(p.predict(_match(), _teams()).scoreline.matrix, direct.matrix)


def test_market_weight_zero_is_the_fallback():
    odds = {"M": _devig_proportional(1.5, 4.0, 6.0)}
    fb = EloPoissonPredictor(gmax=7)
    p = MarketOddsPredictor(odds=odds, fallback=fb, gmax=7, market_weight=0.0)
    assert np.array_equal(
        p.predict(_match(), _teams()).scoreline.matrix,
        fb.predict(_match(), _teams()).scoreline.matrix,
    )


def test_blend_is_the_normalised_geometric_mean():
    odds = {"M": _devig_proportional(1.5, 4.0, 6.0)}
    o = odds["M"]
    fb = EloPoissonPredictor(gmax=7)
    p = MarketOddsPredictor(odds=odds, fallback=fb, gmax=7, market_weight=0.5)
    got = p.predict(_match(), _teams()).scoreline.matrix
    market = expand_1x2_to_scoreline(o.p_home, o.p_draw, o.p_away, total_goals=2.6, gmax=7)
    model = fb.predict(_match(), _teams()).scoreline.matrix
    want = np.sqrt(market.matrix * model)
    want /= want.sum()
    assert np.allclose(got, want)
    # The blend's win probability sits between the two components'.
    lo = min(market.p_home_win(), float(np.tril(model, -1).sum()))
    hi = max(market.p_home_win(), float(np.tril(model, -1).sum()))
    blend_home = float(np.tril(got, -1).sum())
    assert lo - 1e-9 <= blend_home <= hi + 1e-9


def test_blend_falls_back_when_no_odds_regardless_of_weight():
    fb = EloPoissonPredictor(gmax=7)
    for w in (0.0, 0.5, 1.0):
        p = MarketOddsPredictor(odds={}, fallback=fb, gmax=7, market_weight=w)
        assert np.allclose(
            p.predict(_match(), _teams()).scoreline.matrix,
            fb.predict(_match(), _teams()).scoreline.matrix,
        )


def test_market_weight_validated_and_surfaced_in_params():
    p = MarketOddsPredictor(gmax=7, market_weight=0.25, match_draw=True)
    assert p.params["market_weight"] == 0.25
    assert p.params["match_draw"] is True
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            MarketOddsPredictor(gmax=7, market_weight=bad)


def test_pool_log_linear_endpoints_and_normalisation():
    rng = np.random.default_rng(0)
    a = rng.random((8, 8))
    a /= a.sum()
    b = rng.random((8, 8))
    b /= b.sum()
    assert np.allclose(pool_log_linear(a, b, 1.0), a)
    assert np.allclose(pool_log_linear(a, b, 0.0), b)
    half = pool_log_linear(a, b, 0.5)
    assert half.sum() == pytest.approx(1.0)
    assert (half > 0).all()


def test_match_draw_reproduces_full_triple():
    # With match_draw=True all three de-vigged components are matched, not just the balance.
    p_home, p_draw, p_away = 0.5, 0.3, 0.2
    dist = expand_1x2_to_scoreline(p_home, p_draw, p_away, gmax=7, match_draw=True)
    assert dist.p_home_win() == pytest.approx(p_home, abs=0.02)
    assert dist.p_draw() == pytest.approx(p_draw, abs=0.02)
    assert dist.p_away_win() == pytest.approx(p_away, abs=0.02)


def test_match_draw_false_unchanged_and_high_draw_price_means_fewer_goals():
    # Default path byte-identical to the historical fixed-total expansion.
    legacy = expand_1x2_to_scoreline(0.5, 0.3, 0.2, total_goals=2.6, gmax=7)
    again = expand_1x2_to_scoreline(0.5, 0.3, 0.2, total_goals=2.6, gmax=7, match_draw=False)
    assert np.array_equal(legacy.matrix, again.matrix)
    # A market pricing the draw high implies a tighter (lower-total) match than one pricing
    # it low, at the same win balance.
    drawy = expand_1x2_to_scoreline(0.40, 0.35, 0.25, gmax=7, match_draw=True)
    open_ = expand_1x2_to_scoreline(0.475, 0.20, 0.325, gmax=7, match_draw=True)
    assert sum(drawy.expected_goals()) < sum(open_.expected_goals())


def test_match_draw_goal_scale_lifts_totals_after_the_solve():
    base = expand_1x2_to_scoreline(0.5, 0.3, 0.2, gmax=7, match_draw=True)
    lifted = expand_1x2_to_scoreline(0.5, 0.3, 0.2, gmax=7, match_draw=True, goal_scale=1.3)
    assert sum(lifted.expected_goals()) > sum(base.expected_goals())


def test_odds_loader_optional(tmp_path):
    # No file configured -> empty.
    prov = FileDataProvider("t", "f", "r", odds_file=None)
    assert prov.get_odds() == {}
    # Missing file -> empty.
    prov2 = FileDataProvider("t", "f", "r", odds_file=tmp_path / "nope.csv")
    assert prov2.get_odds() == {}
    # Real file -> de-vigged Odds1X2 keyed by match_id.
    csv_path = tmp_path / "odds.csv"
    csv_path.write_text(
        "match_id,odds_home,odds_draw,odds_away\nM1,2.0,4.0,4.0\nM2,1.5,4.0,6.0\n"
    )
    prov3 = FileDataProvider("t", "f", "r", odds_file=csv_path)
    odds = prov3.get_odds()
    assert set(odds) == {"M1", "M2"}
    assert isinstance(odds["M1"], Odds1X2)
    assert odds["M1"].p_home == pytest.approx(0.5)
