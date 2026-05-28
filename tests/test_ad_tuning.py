"""Staged A/D tuner — Stage 1 (generation) × Stage 2 (predictor) + reality check."""

from pathlib import Path

import pytest

import tippspiel
from tippspiel.config import load_config, load_tournament
from tippspiel.data.file_provider import FileDataProvider
from tippspiel.elo import get_results_csv
from tippspiel.elo.config import load_elo_config
from tippspiel.pipeline import _resolve_as_of
from tippspiel.report.ad_tuning import build_ad_tuning

REPO = Path(tippspiel.__file__).parent.parent
EURO2016 = REPO / "configs" / "euro2016.yaml"

# Tiny grids — 2 gen × 2 predictor points × 1 benchmark keeps the test in the ~few seconds range.
TINY_GEN = {
    "learning_rate": [0.03, 0.06],
    "lookback_years": [200],
    "recency_decay": [False],
    "ad_home_advantage": [0.0],
}
TINY_PRED = {
    "base_log_rate": [0.3],
    "home_advantage": [0.0],
    "rho": [-0.10, 0.0],
    "ko_goal_scale": [1.2],
}


@pytest.fixture(scope="module")
def historical_text():
    try:
        return get_results_csv(load_elo_config({}), cache_only=True)
    except Exception:
        pytest.skip("historical results cache not available; run `tippspiel build-elo` first")


def _benchmark(config_path: Path):
    bundle = load_tournament(config_path)
    prov = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file)
    fixtures = prov.get_fixtures()
    results = {r.match_id: r for r in prov.get_results()}
    as_of = _resolve_as_of(None, bundle)
    return (bundle, fixtures, results, as_of)


def test_build_ad_tuning_has_two_stages_and_reality_check(historical_text):
    cfg = load_config(EURO2016)
    md, data = build_ad_tuning(
        cfg, [_benchmark(EURO2016)],
        predictor_grid=TINY_PRED, generation_grid=TINY_GEN, top=5,
        historical_text=historical_text,
    )

    assert data["predictor"] == "attack_defence_poisson"
    assert data["benchmarks"] == ["euro2016"]
    assert md.strip().startswith("# A/D parameter tuning")

    # Stage 1 — generation params.
    s1 = data["stage1_generation"]
    assert s1["grid_size"] in (2, 3)  # 2 grid points + maybe the default
    assert set(s1["recommended_params"]) == {
        "learning_rate", "lookback_years", "recency_decay", "ad_home_advantage",
    }
    board1 = s1["leaderboard"]
    assert board1 and 0.0 <= board1[0]["mean_rps"] <= 1.0

    # Stage 2 — predictor params.
    s2 = data["stage2_predictor"]
    assert s2["grid_size"] in (2, 3)
    assert set(s2["recommended_params"]) == {
        "base_log_rate", "home_advantage", "rho", "ko_goal_scale",
    }
    board2 = s2["leaderboard"]
    assert board2 and 0.0 <= board2[0]["mean_rps"] <= 1.0

    # Combined recommendation glues the two layers.
    combined = data["combined_recommended_params"]
    assert combined["generation"] == s1["recommended_params"]
    assert combined["predictor"] == s2["recommended_params"]

    # Reality check is present for both default and recommended, with a valid verdict.
    rc = data["reality_check"]
    for variant in ("default", "recommended"):
        block = rc[variant]
        assert block["verdict"]["status"] in ("PASS", "WARN", "FAIL")
        assert block["pooled"]["matches"] > 0
        # Mean goals are finite, positive numbers in a plausible range.
        mg = block["pooled"]["mean_goals"]
        assert 0.5 < mg["actual_total"] < 8.0
        assert 0.5 < mg["predicted_total"] < 8.0
        # Tendency split sums to 1.0 on each side.
        for side in ("predicted", "actual"):
            assert abs(sum(block["pooled"]["tendency_split"][side].values()) - 1.0) < 1e-9


def test_run_ad_tuning_writes_outputs(tmp_path, historical_text, monkeypatch):
    # End-to-end: monkey-patch get_results_csv to return the cached text so the
    # pipeline writer path is exercised without re-fetching.
    import dataclasses

    import tippspiel.report.ad_tuning as ad_tuning
    from tippspiel.pipeline import run_ad_tuning

    monkeypatch.setattr(ad_tuning, "get_results_csv", lambda *a, **k: historical_text)
    monkeypatch.setattr(ad_tuning, "DEFAULT_GENERATION_GRID", TINY_GEN)
    monkeypatch.setattr(ad_tuning, "DEFAULT_PREDICTOR_GRID", TINY_PRED)

    cfg = load_config(EURO2016)
    cfg = dataclasses.replace(cfg, report=dataclasses.replace(cfg.report, output_dir=str(tmp_path)))
    result = run_ad_tuning(cfg, [EURO2016], top=5)
    assert result["paths"]["markdown"].exists()
    assert result["paths"]["json"].exists()
    assert result["data"]["predictor"] == "attack_defence_poisson"
