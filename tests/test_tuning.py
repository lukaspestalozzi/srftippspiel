"""Parameter-tuning harness tests (blended objective)."""

from pathlib import Path

import pytest

import tippspiel
from tippspiel.config import load_config, load_tournament
from tippspiel.data.file_provider import FileDataProvider
from tippspiel.report.tuning import (
    _blended_key,
    _default_params,
    _evaluate,
    build_market_grid,
    build_tuning,
)

REPO = Path(tippspiel.__file__).parent.parent
WOMENSEURO = REPO / "configs" / "womenseuro2025.yaml"

# Tiny grid so the test is fast (4 combos over one 31-match benchmark).
TINY_GRID = {
    "mu": [2.4, 2.6],
    "k": [0.0015],
    "rho": [0.0],
    "host_elo_bonus": [0],
    "ko_goal_scale": [1.0, 1.2],
}

# Tiny market grid: Elo axes pinned, only the blend weight swept (2 combos).
TINY_MARKET_GRID = {
    "mu": [2.6], "k": [0.0015], "rho": [0.0], "host_elo_bonus": [0],
    "ko_goal_scale": [1.0], "alpha": [0.0],
    "market_weight": [0.0, 1.0],
    "total_goals": [2.6],
    "match_draw": [False],
}


def _benchmark(config, with_odds=False):
    bundle = load_tournament(config)
    prov = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file,
                            odds_file=bundle.odds_file if with_odds else None)
    teams = {t.team_id: t for t in prov.get_teams()}
    return (bundle, teams, prov.get_fixtures(),
            {r.match_id: r for r in prov.get_results()}, prov.get_odds())


def test_build_tuning_produces_ranked_leaderboard():
    cfg = load_config(WOMENSEURO)
    data = build_tuning(cfg, [_benchmark(WOMENSEURO)], grid=TINY_GRID, top=5)[1]

    # The 4 grid combos, plus the base-config default if it isn't already one of them.
    assert data["grid_size"] in (4, 5)
    assert "ko_goal_scale" in data["recommended_params"]
    board = data["leaderboard"]
    assert board and board[0]["params"] == data["recommended_params"]
    # Leaderboard is sorted by the blended key (non-decreasing).
    keys = [(_blended_key(r)) for r in board]
    assert keys == sorted(keys)
    rec = data["recommended_metrics"]
    assert 0.0 <= rec["mean_rps"] <= 1.0
    assert rec["mean_nll"] > 0.0
    # Leave-one-out has an entry per benchmark (here just one).
    assert set(data["leave_one_out"]) == {"womenseuro2025"}


def test_market_sweep_ranks_blend_weights_and_default_baseline():
    cfg = load_config(WOMENSEURO)
    bench = _benchmark(WOMENSEURO, with_odds=True)
    assert bench[4], "womenseuro2025 must have a committed odds.csv for this test"
    data = build_tuning(cfg, [bench], grid=TINY_MARKET_GRID, top=5)[1]

    # 2 market combos + the pure-Elo base-config default appended as the baseline.
    assert data["grid_size"] == 3
    boards = {row["params"].get("market_weight") for row in data["leaderboard"]}
    assert {0.0, 1.0, None} <= boards
    # w=0 must reproduce the pure-Elo metrics of the same Elo params exactly: the blend
    # degrades to the fallback on every match.
    elo_params = {k: v[0] for k, v in TINY_MARKET_GRID.items()
                  if k not in ("market_weight", "total_goals", "match_draw")}
    pure = _evaluate(elo_params, [bench])
    by_mw = {row["params"].get("market_weight"): row for row in data["leaderboard"]}
    assert by_mw[0.0]["mean_rps"] == pytest.approx(pure["mean_rps"], abs=1e-12)
    # w=1 (pure market) must actually differ from pure Elo on an odds-backed benchmark.
    assert by_mw[1.0]["mean_rps"] != pytest.approx(pure["mean_rps"], abs=1e-12)


def test_market_sweep_without_odds_degrades_to_pure_elo():
    cfg = load_config(WOMENSEURO)
    bundle, teams, fixtures, results, _odds = _benchmark(WOMENSEURO, with_odds=False)
    bench = (bundle, teams, fixtures, results, {})  # no odds snapshot
    data = build_tuning(cfg, [bench], grid=TINY_MARKET_GRID, top=5)[1]
    rps = {row["params"].get("market_weight"): row["mean_rps"]
           for row in data["leaderboard"]}
    # With no odds every blend weight is the same pure-Elo predictor.
    assert rps[0.0] == pytest.approx(rps[1.0], abs=1e-12)


def test_build_market_grid_pins_elo_axes_and_handles_fallback_params():
    grid = build_market_grid({"fallback_params": {"mu": 2.8, "k": 0.002}, "total_goals": 2.6})
    assert grid["mu"] == [2.8] and grid["k"] == [0.002]
    assert all(len(grid[k]) == 1 for k in ("mu", "k", "rho", "host_elo_bonus",
                                           "ko_goal_scale", "alpha"))
    assert set(grid["market_weight"]) == {0.0, 0.25, 0.5, 0.75, 1.0}
    assert grid["match_draw"] == [False, True]


def test_default_params_lift_fallback_of_market_odds_config():
    # The live wc2026 config is market_odds with the tuned Elo params nested under
    # fallback_params; the plain `tune` baseline must reflect those, not the code defaults.
    cfg = load_config(REPO / "config.yaml")
    assert cfg.predictor.name == "market_odds"  # guard: the scenario under test
    d = _default_params(cfg)
    fb = cfg.predictor.params["fallback_params"]
    assert d["k"] == fb["k"] and d["rho"] == fb["rho"] and d["alpha"] == fb["alpha"]


def test_blended_key_prefers_calibration_then_points():
    low_rps = {"mean_rps": 0.180, "model_pct": 40.0}
    high_rps = {"mean_rps": 0.190, "model_pct": 99.0}
    # Lower RPS wins even with far fewer points.
    assert _blended_key(low_rps) < _blended_key(high_rps)
    # Near-equal RPS (rounds to the same) -> more points wins.
    a = {"mean_rps": 0.19001, "model_pct": 44.0}
    b = {"mean_rps": 0.19002, "model_pct": 45.0}
    assert _blended_key(b) < _blended_key(a)
