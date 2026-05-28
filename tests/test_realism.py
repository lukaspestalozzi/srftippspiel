"""Realism / reality-check helpers for the A/D tuning report."""

from pathlib import Path

import tippspiel
from tippspiel.config import load_config, load_tournament, select_predictor
from tippspiel.data.file_provider import FileDataProvider
from tippspiel.pipeline import build_predictor, ratings_file
from tippspiel.report.realism import (
    _expected_goals,
    reality_check_one,
    reality_pooled,
    verdict_of,
)

REPO = Path(tippspiel.__file__).parent.parent
EURO2016 = REPO / "configs" / "euro2016.yaml"


def _benchmark_with_predictor():
    cfg = select_predictor(load_config(EURO2016), "attack_defence_poisson")
    bundle = load_tournament(EURO2016)
    predictor = build_predictor(cfg)
    prov = FileDataProvider(
        ratings_file(predictor, bundle), bundle.fixtures_file, bundle.results_file,
    )
    teams = {t.team_id: t for t in prov.get_teams()}
    fixtures = prov.get_fixtures()
    results = {r.match_id: r for r in prov.get_results()}
    return bundle, teams, fixtures, results, predictor


def test_expected_goals_matches_p_distribution_marginals():
    # Build a scoreline distribution from a real match and confirm E[H] + E[A] are sensible.
    bundle, teams, fixtures, results, predictor = _benchmark_with_predictor()
    match = next(m for m in fixtures if m.participants_known)
    dist = predictor.predict(match, teams).scoreline
    eh, ea = _expected_goals(dist)
    # Both rates strictly positive, total in a plausible international-football range.
    assert eh > 0 and ea > 0
    assert 0.5 < eh + ea < 8.0
    # Marginal consistency: sum_h h * p_home_goals(h) == E[H].
    eh_marg = sum(h * dist.p_home_goals(h) for h in range(dist.gmax + 1))
    ea_marg = sum(a * dist.p_away_goals(a) for a in range(dist.gmax + 1))
    assert abs(eh - eh_marg) < 1e-9
    assert abs(ea - ea_marg) < 1e-9


def test_reality_check_one_has_expected_structure_and_actuals_match():
    bundle, teams, fixtures, results, predictor = _benchmark_with_predictor()
    rc = reality_check_one(bundle, teams, fixtures, results, predictor)

    # Per-tournament shape.
    for key in ("tournament", "matches", "mean_goals", "tendency_split",
                "scoreline_tvd", "tip_composition", "top5_scorelines", "pool_points"):
        assert key in rc
    assert rc["tournament"] == "euro2016"
    assert rc["matches"] == len(results)

    # Actual mean goals match the raw results exactly (sanity-check the aggregation).
    total = sum(r.home_goals + r.away_goals for r in results.values())
    assert abs(rc["mean_goals"]["actual_total"] - total / len(results)) < 1e-9

    # Tendency split rows each sum to 1.0 (probability-of-outcome aggregate).
    for side in ("predicted", "actual"):
        s = sum(rc["tendency_split"][side].values())
        assert abs(s - 1.0) < 1e-9

    # Top-5 lists are length 5 (or shorter if the matrix has fewer non-zero cells).
    assert len(rc["top5_scorelines"]["predicted"]) <= 5
    assert len(rc["top5_scorelines"]["actual"]) <= 5
    # Predicted cells are probabilities; actual cells are integer counts.
    for cell in rc["top5_scorelines"]["predicted"]:
        assert 0.0 <= cell["prob"] <= 1.0
    for cell in rc["top5_scorelines"]["actual"]:
        assert isinstance(cell["count"], int) and cell["count"] >= 1

    # Hit rates are valid probabilities; modal-share is in (0, 1].
    tc = rc["tip_composition"]
    assert 0.0 <= tc["exact_hit_rate"] <= 1.0
    assert 0.0 <= tc["tendency_hit_rate"] <= 1.0
    assert 0.0 < tc["modal_tip"]["share"] <= 1.0


def test_reality_pooled_matches_weighted_aggregates():
    bundle, teams, fixtures, results, predictor = _benchmark_with_predictor()
    rc = reality_check_one(bundle, teams, fixtures, results, predictor)
    pooled = reality_pooled([rc])

    # With a single tournament, pooled equals per-tournament (matches-weighted of one row).
    assert pooled["matches"] == rc["matches"]
    assert abs(pooled["mean_goals"]["actual_total"]
               - rc["mean_goals"]["actual_total"]) < 1e-9
    assert abs(pooled["scoreline_tvd_weighted"] - rc["scoreline_tvd"]) < 1e-9


def test_verdict_passes_on_well_behaved_inputs():
    pooled = {
        "matches": 100,
        "mean_goals": {"delta_total": 0.05},
        "tendency_delta_max": 0.02,
        "tip_composition": {"modal_share_weighted": 0.3},
        "scoreline_tvd_weighted": 0.15,
    }
    v = verdict_of(pooled)
    assert v["status"] == "PASS"


def test_verdict_flags_concentrated_tips_and_drifted_goals():
    pooled = {
        "matches": 100,
        "mean_goals": {"delta_total": 0.45},  # WARN range
        "tendency_delta_max": 0.04,
        "tip_composition": {"modal_share_weighted": 0.9},  # FAIL range
        "scoreline_tvd_weighted": 0.2,
    }
    v = verdict_of(pooled)
    # Modal share triggers FAIL; goals drift triggers WARN; FAIL dominates.
    assert v["status"] == "FAIL"
    assert any("modal tip share" in r for r in v["reasons"])
    assert any("mean goals" in r for r in v["reasons"])
