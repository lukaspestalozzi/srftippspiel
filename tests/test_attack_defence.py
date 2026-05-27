"""Tests for the attack/defence rating model + predictor (offline — no network)."""

from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import tippspiel
from tippspiel.config import Config, PredictorConfig, ReportConfig, SimulationConfig, StrategyConfig
from tippspiel.elo import build_model, run_forward_pass
from tippspiel.elo.attack_defence import AttackDefenceElo
from tippspiel.elo.config import EloConfig
from tippspiel.elo.matches import HistoricalMatch
from tippspiel.model.stages import Stage
from tippspiel.model.types import Match, Team, TeamRef
from tippspiel.pipeline import build_predictor
from tippspiel.predictors.attack_defence import AttackDefencePoissonPredictor
from tippspiel.predictors.elo_poisson import EloPoissonPredictor, scoreline_from_rates

REPO = Path(tippspiel.__file__).parent.parent

# c=0, ha=0 so exp(...) = 1.0 for both sides => clean hand-checkable gradient step.
_FLAT = EloConfig(model="attack_defence", learning_rate=0.1, base_log_rate=0.0, ad_home_advantage=0.0)


def _m(home="A", away="B", hs=2, as_=0, neutral=True, weight=1.0):
    return HistoricalMatch(
        date=date(2020, 1, 1), home=home, away=away, home_score=hs, away_score=as_,
        tournament="Friendly", neutral=neutral, weight=weight,
    )


# --------------------------------------------------------------------------- SGD update
def test_update_gradient_step_signs_and_magnitude():
    m = AttackDefenceElo(_FLAT)
    m.seed("A")
    m.seed("B")
    m.update(_m("A", "B", 2, 0))  # lam_h = lam_a = 1.0; eh = +1, ea = -1; lr = 0.1
    atk_a, def_a = m.attack_defence("A")
    atk_b, def_b = m.attack_defence("B")
    assert atk_a == pytest.approx(0.1)   # home overscored -> attack up
    assert def_a == pytest.approx(0.1)   # home conceded < expected -> defence up
    assert atk_b == pytest.approx(-0.1)  # away underscored -> attack down
    assert def_b == pytest.approx(-0.1)  # away conceded > expected -> defence down


def test_recency_weight_scales_step():
    m = AttackDefenceElo(_FLAT)
    m.seed("A")
    m.seed("B")
    m.update(_m("A", "B", 2, 0, weight=0.5))
    assert m.attack_defence("A")[0] == pytest.approx(0.05)  # half the lr step


def test_shrinkage_pulls_toward_zero():
    cfg = replace(_FLAT, ad_shrinkage=0.1)
    m = AttackDefenceElo(cfg)
    m.seed("A")
    m.seed("B")
    m.update(_m("A", "B", 2, 0))
    # raw step would be 0.1; shrinkage multiplies the touched ratings by (1 - 0.1)
    assert m.attack_defence("A")[0] == pytest.approx(0.1 * 0.9)


def test_forward_pass_deterministic_regardless_of_order():
    ms = [_m("A", "B", 3, 1), _m("B", "C", 0, 0, neutral=False), _m("C", "A", 1, 2)]
    a = run_forward_pass(list(ms), AttackDefenceElo(_FLAT)).attack_defence_ratings()
    b = run_forward_pass(list(reversed(ms)), AttackDefenceElo(_FLAT)).attack_defence_ratings()
    assert a == b


def test_build_model_dispatch():
    assert isinstance(build_model(EloConfig(model="attack_defence")), AttackDefenceElo)


# --------------------------------------------------------------------------- predictor
def _match(stage=Stage.GROUP, venue=None):
    return Match(
        match_id="M", stage=stage, home=TeamRef(team_id="AAA"), away=TeamRef(team_id="BBB"),
        kickoff=datetime(2026, 6, 1, tzinfo=timezone.utc), group="A", venue_country=venue,
    )


def _teams(atk_a=0.3, def_a=-0.2, atk_b=0.0, def_b=0.0):
    return {
        "AAA": Team("AAA", "Alpha", 1500.0, None, attack=atk_a, defence=def_a),
        "BBB": Team("BBB", "Beta", 1500.0, None, attack=atk_b, defence=def_b),
    }


def test_predictor_goal_rates_from_attack_defence():
    import math

    pred = AttackDefencePoissonPredictor(base_log_rate=0.3, home_advantage=0.0, rho=0.0)
    lam_h, lam_a = pred.goal_rates(0.3, -0.2, 0.0, 0.0)  # atk_h, def_h, atk_a, def_a
    assert lam_h == pytest.approx(math.exp(0.3 + 0.3 - 0.0))   # c + atk_h - def_a
    assert lam_a == pytest.approx(math.exp(0.3 + 0.0 + 0.2))   # c + atk_a - def_h


def test_predictor_distribution_normalised_and_host_bonus():
    pred = AttackDefencePoissonPredictor(base_log_rate=0.3, home_advantage=0.2, rho=-0.1)
    dist = pred.predict(_match(), _teams()).scoreline
    assert dist.matrix.sum() == pytest.approx(1.0)
    # Host bonus raises the home win probability.
    p_no_host = pred.predict(_match(venue=None), _teams()).scoreline.p_home_win()
    p_host = pred.predict(_match(venue="AAA"), _teams()).scoreline.p_home_win()
    assert p_host > p_no_host


def test_knockout_goal_scale_applies():
    pred = AttackDefencePoissonPredictor(base_log_rate=0.3, ko_goal_scale=1.5)
    teams = _teams()
    g = pred.predict(_match(stage=Stage.GROUP), teams).scoreline.cell(0, 0)
    k = pred.predict(_match(stage=Stage.FINAL), teams).scoreline.cell(0, 0)
    assert k < g  # scaling goal rates up lowers P(0:0)


def test_missing_ratings_fall_back_to_base_rate():
    pred = AttackDefencePoissonPredictor(base_log_rate=0.3, home_advantage=0.0, rho=0.0)
    teams = {"AAA": Team("AAA", "Alpha", 1500.0), "BBB": Team("BBB", "Beta", 1500.0)}  # no atk/def
    dist = pred.predict(_match(), teams).scoreline
    assert dist.matrix.sum() == pytest.approx(1.0)
    assert dist.p_home_win() == pytest.approx(dist.p_away_win(), abs=1e-9)  # symmetric


# --------------------------------------------------------------------------- shared scoreline refactor
def test_scoreline_from_rates_matches_elo_predictor():
    ep = EloPoissonPredictor(rho=-0.1, gmax=7)
    assert (ep.scoreline_matrix(1.6, 1.1) == scoreline_from_rates(1.6, 1.1, 7, -0.1)).all()


# --------------------------------------------------------------------------- build_predictor dispatch
def test_build_predictor_dispatch():
    cfg = Config(
        predictor=PredictorConfig(name="attack_defence_poisson", params={"rho": -0.1}),
        strategy=StrategyConfig(name="expected_points"),
        simulation=SimulationConfig(iterations=10, seed=1, penalty_model="coin_flip"),
        report=ReportConfig(output_dir="output", display_timezone="UTC"),
    )
    assert isinstance(build_predictor(cfg), AttackDefencePoissonPredictor)


# --------------------------------------------------------------------------- emit attack/defence
def test_emit_teams_csv_writes_attack_defence(tmp_path):
    from tippspiel.config import load_tournament
    from tippspiel.data.file_provider import FileDataProvider
    from tippspiel.elo.names import normalize
    from tippspiel.pipeline import _emit_teams_csv

    bundle = load_tournament(REPO / "configs" / "wc2022.yaml")
    teams = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file).get_teams()
    pairs = {normalize(teams[0].name): (0.42, -0.15)}
    ratings = {normalize(teams[0].name): 1600.0}
    out = tmp_path / "teams_ad.csv"
    _emit_teams_csv(bundle, ratings, out, pairs=pairs)

    emitted = {t.team_id: t for t in
               FileDataProvider(out, bundle.fixtures_file, bundle.results_file).get_teams()}
    assert emitted[teams[0].team_id].attack == pytest.approx(0.42)
    assert emitted[teams[0].team_id].defence == pytest.approx(-0.15)
    # A team without a computed pair has no attack/defence.
    assert emitted[teams[5].team_id].attack is None
