"""Verification backtest tests: deterministic scoring + internal consistency (spec §10)."""

from pathlib import Path

import tippspiel
from tippspiel.config import load_config, load_tournament
from tippspiel.data.file_provider import FileDataProvider
from tippspiel.pipeline import build_predictor
from tippspiel.report.backtest import build_verification
from tippspiel.strategy.expected_points import score_tip

REPO = Path(tippspiel.__file__).parent.parent
WOMENSEURO_CONFIG = REPO / "configs" / "womenseuro2025.yaml"


def test_score_tip_hand_cases():
    assert score_tip(2, 1, 2, 1, 1) == 10   # exact: 5 + 1 + 1 + 3
    assert score_tip(1, 0, 3, 1, 1) == 5    # right tendency only
    assert score_tip(0, 2, 1, 0, 1) == 0    # wrong tendency, nothing right
    assert score_tip(2, 1, 2, 1, 2) == 20   # knockout doubling
    assert score_tip(1, 1, 2, 2, 1) == 8    # draw tendency (5) + goal diff (3)
    assert score_tip(2, 2, 2, 2, 1) == 10   # exact draw


def test_verify_totals_are_internally_consistent():
    cfg = load_config(WOMENSEURO_CONFIG)
    bundle = load_tournament(WOMENSEURO_CONFIG)
    prov = FileDataProvider(bundle.teams_file, bundle.fixtures_file,
                            bundle.results_file)
    teams = {t.team_id: t for t in prov.get_teams()}
    fixtures = prov.get_fixtures()
    results = {r.match_id: r for r in prov.get_results()}
    md, data = build_verification(bundle, teams, fixtures, results, build_predictor(cfg))

    s = data["summary"]
    assert s["all"]["matches"] == 31  # 24 group + 7 knockout
    assert s["group"]["matches"] + s["knockout"]["matches"] == s["all"]["matches"]
    for key in ("all", "group", "knockout"):
        t = s[key]
        assert 0 <= t["model"] <= t["max"]
        assert 0 <= t["naive"] <= t["max"]
    # Per-stage points partition the overall totals.
    assert s["group"]["model"] + s["knockout"]["model"] == s["all"]["model"]
    assert s["group"]["max"] + s["knockout"]["max"] == s["all"]["max"]
    # Knockout matches are worth double (7 matches x 20 max).
    assert s["knockout"]["max"] == 7 * 20
    assert "Verification backtest" in md
    assert len(data["matches"]) == 31
