"""Rank-optimising strategy + field model tests (spec §6.3.3)."""

from pathlib import Path

import numpy as np
import pytest

import tippspiel
from tippspiel.config import load_config, load_tournament, select_predictor
from tippspiel.data.file_provider import FileDataProvider
from tippspiel.model.scoreline import ScorelineDistribution
from tippspiel.pipeline import _predict_tippable, build_predictor, build_strategy
from tippspiel.strategy.rank_optimizing import (
    PredictorDerivedFieldModel,
    RankOptimizingStrategy,
    SlateComparison,
    comparison_from_params,
    field_score_moments,
    optimize_slate,
)

REPO = Path(tippspiel.__file__).parent.parent


def _dist(cells: dict[tuple[int, int], float], gmax: int) -> ScorelineDistribution:
    m = np.zeros((gmax + 1, gmax + 1))
    for (h, a), p in cells.items():
        m[h, a] = p
    return ScorelineDistribution(m)


def _pred(match_id, dist):
    from tippspiel.model.types import MatchPrediction

    return MatchPrediction(match_id=match_id, scoreline=dist, predictor_name="test")


def _match(match_id, *, knockout=False):
    from datetime import datetime, timezone

    from tippspiel.model.stages import Stage
    from tippspiel.model.types import Match, TeamRef

    return Match(
        match_id=match_id,
        stage=Stage.R16 if knockout else Stage.GROUP,
        home=TeamRef(team_id="AAA"),
        away=TeamRef(team_id="BBB"),
        kickoff=datetime(2026, 6, 1, tzinfo=timezone.utc),
        group=None if knockout else "A",
    )


# --------------------------------------------------------------------------- field model
def test_field_distribution_sums_to_one():
    for cells in (
        {(2, 0): 0.5, (1, 0): 0.3, (0, 0): 0.1, (1, 1): 0.1},
        {(0, 0): 0.2, (1, 1): 0.2, (1, 0): 0.2, (0, 1): 0.2, (2, 1): 0.2},
        {(3, 1): 1.0},
    ):
        dist = _dist(cells, gmax=4)
        fd = PredictorDerivedFieldModel().opponent_tip_distribution(_pred("m", dist))
        assert sum(fd.values()) == pytest.approx(1.0)
        assert all(p >= 0.0 for p in fd.values())


def test_field_expert_fraction_concentrates_on_popular_tip():
    dist = _dist({(2, 0): 0.5, (1, 0): 0.3, (0, 0): 0.1, (1, 1): 0.1}, gmax=3)
    concentrated = PredictorDerivedFieldModel(
        expert_fraction=0.95, temperature=1.0
    ).opponent_tip_distribution(_pred("m", dist))
    spread = PredictorDerivedFieldModel(
        expert_fraction=0.2, temperature=3.0
    ).opponent_tip_distribution(_pred("m", dist))
    # The modal/EV-optimal scoreline (2:0) gets more field mass under a more expert field.
    assert concentrated[(2, 0)] > spread[(2, 0)]


def test_field_score_moments_hand_computed():
    field = {(1, 0): 0.5, (0, 1): 0.5}
    # Result 1:0. Tip 1:0 scores 10 (exact); tip 0:1 scores 0. mean=5, var=25.
    mean, var = field_score_moments(field, 1, 0, weight=1)
    assert mean == pytest.approx(5.0)
    assert var == pytest.approx(25.0)


# --------------------------------------------------------------------------- optimiser
def test_invariants_rank_never_raises_ev_and_never_lowers_pwin():
    # A realistic mix: a clear favourite, a coin-flip, and a draw-ish match.
    preds = {
        "m1": _pred("m1", _dist({(2, 0): 0.5, (1, 0): 0.3, (0, 0): 0.1, (1, 1): 0.1}, 3)),
        "m2": _pred("m2", _dist({(1, 0): 0.28, (0, 1): 0.28, (1, 1): 0.22, (2, 1): 0.22}, 3)),
        "m3": _pred("m3", _dist({(0, 0): 0.3, (1, 1): 0.3, (2, 0): 0.2, (0, 2): 0.2}, 3)),
    }
    fixtures = [_match("m1"), _match("m2"), _match("m3")]
    comp = optimize_slate(preds, fixtures, PredictorDerivedFieldModel(),
                          pool_size=50_000, n_worlds=2000, seed=1)
    assert comp.rank_total_ev <= comp.ev_total_ev + 1e-9   # EV is per-match maximal
    assert comp.rank_p_win >= comp.ev_p_win - 1e-12        # ascent starts from EV slate


def test_deterministic_for_same_seed():
    preds = {"m1": _pred("m1", _dist({(2, 0): 0.4, (0, 2): 0.35, (1, 1): 0.25}, 3))}
    fixtures = [_match("m1")]
    a = optimize_slate(preds, fixtures, PredictorDerivedFieldModel(), n_worlds=1500, seed=7)
    b = optimize_slate(preds, fixtures, PredictorDerivedFieldModel(), n_worlds=1500, seed=7)
    assert a.rank_slate == b.rank_slate
    assert a.rank_p_win == pytest.approx(b.rank_p_win)


def test_contrarian_bets_the_upset_against_a_favourite_heavy_field():
    # Home favoured 55/45, but the field piles onto the home win. In a large pool, matching
    # the crowd can't win outright; betting the away upset wins the upset worlds.
    dist = _dist({(2, 0): 0.3, (1, 0): 0.25, (0, 1): 0.25, (0, 2): 0.2}, gmax=3)
    preds = {"m1": _pred("m1", dist)}
    fixtures = [_match("m1")]
    field = PredictorDerivedFieldModel(expert_fraction=0.9, temperature=1.0)
    comp = optimize_slate(preds, fixtures, field, pool_size=200_000, n_worlds=4000, seed=3)
    ev_h, ev_a = comp.ev_slate["m1"]
    rk_h, rk_a = comp.rank_slate["m1"]
    assert (ev_h > ev_a)              # EV tip is the home win
    assert (rk_h < rk_a)              # rank tip bets the away upset
    assert comp.n_diff == 1
    assert comp.rank_p_win > comp.ev_p_win


# --------------------------------------------------------------------------- integration
def test_integration_through_pipeline_on_wc2026_groups():
    cfg = select_predictor(load_config(REPO / "config.yaml"), "elo_poisson")
    bundle = load_tournament(REPO / "config.yaml")
    provider = FileDataProvider(
        bundle.teams_file, bundle.fixtures_file, bundle.results_file,
        bundle.thirds_allocation_file,
    )
    teams = {t.team_id: t for t in provider.get_teams()}
    fixtures = provider.get_fixtures()
    results = {r.match_id: r for r in provider.get_results()}
    predictor = build_predictor(cfg)
    predictions = _predict_tippable(fixtures, teams, set(results), predictor)

    strat = RankOptimizingStrategy(
        pool_size=200_000, n_worlds=400, seed=cfg.simulation.seed,
        bonus_question_configs=bundle.bonus_questions,
    )
    tipset = strat.generate_tips(predictions, None, fixtures)
    # One tip per tippable group fixture, all within the predicted goal grid.
    assert set(tipset.tips) == set(predictions)
    for mid, tip in tipset.tips.items():
        g = predictions[mid].scoreline.gmax
        assert 0 <= tip.tip_home <= g and 0 <= tip.tip_away <= g


def test_cli_strategy_override():
    from tippspiel.cli import _override_strategy

    cfg = load_config(REPO / "config.yaml")
    assert cfg.strategy.name == "expected_points"
    assert _override_strategy(cfg, None) is cfg          # no-op
    overridden = _override_strategy(cfg, "rank_optimizing")
    assert overridden.strategy.name == "rank_optimizing"
    assert cfg.strategy.name == "expected_points"        # original unchanged (frozen)


def test_comparison_from_params_groups_and_empty():
    preds = {
        "m1": _pred("m1", _dist({(2, 0): 0.5, (1, 0): 0.3, (0, 0): 0.1, (1, 1): 0.1}, 3)),
        "m2": _pred("m2", _dist({(1, 0): 0.28, (0, 1): 0.28, (1, 1): 0.22, (2, 1): 0.22}, 3)),
    }
    fixtures = [_match("m1"), _match("m2")]
    comp = comparison_from_params(preds, fixtures, {"n_worlds": 500}, seed=1)
    assert isinstance(comp, SlateComparison)
    assert set(comp.ev_slate) == {"m1", "m2"}
    # No tippable fixtures -> no comparison.
    assert comparison_from_params({}, fixtures, {}, seed=1) is None


def test_report_context_carries_per_model_strategy_summary(tmp_path):
    import dataclasses

    from tippspiel.pipeline import run_combined_pipeline
    from tippspiel.report.html_writer import ReportWriter

    cfg = load_config(REPO / "config.yaml")
    bundle = load_tournament(REPO / "config.yaml")
    # Smaller world count keeps the rank optimiser snappy.
    cfg = dataclasses.replace(
        cfg, strategy=dataclasses.replace(cfg.strategy, params={"n_worlds": 600})
    )
    context = run_combined_pipeline(cfg, bundle, simulate=False)["context"]

    summary = context["strategy_summary"]
    assert summary is not None
    # One row per model, each with the expected per-model fields.
    assert {r["model_name"] for r in summary["rows"]} == {
        "elo_poisson", "attack_defence_poisson",
    }
    for row in summary["rows"]:
        assert row["n_tippable"] == 72
        assert 0 <= row["n_diff"] <= row["n_tippable"]

    # Every tippable group fixture carries one tip row per model with a contrarian flag.
    blocks = [fx for g in context["groups"] for fx in g["fixtures"]]
    tippable = [fx for fx in blocks if fx["tippable"]]
    assert len(tippable) == 72
    for fx in tippable:
        assert len(fx["tip_rows"]) == 2
        for row in fx["tip_rows"]:
            assert isinstance(row["contrarian"], bool)

    html = ReportWriter().render(context)
    assert 'id="strategy"' in html
    assert "Strategy summary" in html
    if any(r["n_diff"] > 0 for r in summary["rows"]):
        assert "contrarian" in html


def test_build_strategy_dispatches_rank_optimizing():
    import dataclasses

    cfg = load_config(REPO / "config.yaml")
    bundle = load_tournament(REPO / "config.yaml")
    cfg = dataclasses.replace(
        cfg,
        strategy=dataclasses.replace(cfg.strategy, name="rank_optimizing", params={"n_worlds": 200}),
    )
    strat = build_strategy(cfg, bundle)
    assert isinstance(strat, RankOptimizingStrategy)
    assert strat.n_worlds == 200
