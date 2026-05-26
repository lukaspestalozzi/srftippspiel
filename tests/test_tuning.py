"""Parameter-tuning harness tests (blended objective)."""

from pathlib import Path

import tippspiel
from tippspiel.config import load_config, load_tournament
from tippspiel.data.file_provider import FileDataProvider
from tippspiel.report.tuning import _blended_key, build_tuning

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


def _benchmark(config):
    bundle = load_tournament(config)
    prov = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file)
    teams = {t.team_id: t for t in prov.get_teams()}
    return (bundle, teams, prov.get_fixtures(),
            {r.match_id: r for r in prov.get_results()})


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


def test_blended_key_prefers_calibration_then_points():
    low_rps = {"mean_rps": 0.180, "model_pct": 40.0}
    high_rps = {"mean_rps": 0.190, "model_pct": 99.0}
    # Lower RPS wins even with far fewer points.
    assert _blended_key(low_rps) < _blended_key(high_rps)
    # Near-equal RPS (rounds to the same) -> more points wins.
    a = {"mean_rps": 0.19001, "model_pct": 44.0}
    b = {"mean_rps": 0.19002, "model_pct": 45.0}
    assert _blended_key(b) < _blended_key(a)
