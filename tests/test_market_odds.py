"""MarketOddsPredictor + odds loading / de-vig tests (spec §6.2.6)."""

from datetime import datetime, timezone

import numpy as np
import pytest

from tippspiel.data.base import Odds1X2
from tippspiel.data.file_provider import FileDataProvider, _devig_proportional
from tippspiel.model.stages import Stage
from tippspiel.model.types import Match, Team, TeamRef
from tippspiel.predictors.elo_poisson import EloPoissonPredictor
from tippspiel.predictors.expansion import expand_1x2_to_scoreline
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
