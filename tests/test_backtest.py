"""Verification backtest tests: deterministic scoring + internal consistency (spec §10)."""

from pathlib import Path

import pytest

import tippspiel
from tippspiel.config import load_config, load_tournament
from tippspiel.data.file_provider import FileDataProvider
from tippspiel.pipeline import build_predictor
from tippspiel.report.backtest import build_verification
from tippspiel.strategy.expected_points import score_tip

REPO = Path(tippspiel.__file__).parent.parent
WOMENSEURO_CONFIG = REPO / "configs" / "womenseuro2025.yaml"


def _verify(config):
    cfg = load_config(config)
    bundle = load_tournament(config)
    prov = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file)
    teams = {t.team_id: t for t in prov.get_teams()}
    fixtures = prov.get_fixtures()
    results = {r.match_id: r for r in prov.get_results()}
    return build_verification(bundle, teams, fixtures, results, build_predictor(cfg))


def test_score_tip_hand_cases():
    assert score_tip(2, 1, 2, 1, 1) == 10   # exact: 5 + 1 + 1 + 3
    assert score_tip(1, 0, 3, 1, 1) == 5    # right tendency only
    assert score_tip(0, 2, 1, 0, 1) == 0    # wrong tendency, nothing right
    assert score_tip(2, 1, 2, 1, 2) == 20   # knockout doubling
    assert score_tip(1, 1, 2, 2, 1) == 8    # draw tendency (5) + goal diff (3)
    assert score_tip(2, 2, 2, 2, 1) == 10   # exact draw


def test_verify_totals_are_internally_consistent():
    md, data = _verify(WOMENSEURO_CONFIG)

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


# (config, expected total matches, expected knockout matches) for the seeded benchmarks.
_BENCHMARKS = [
    (REPO / "configs" / "wc2022.yaml", 64, 16),     # 48 group + 16 knockout
    (REPO / "configs" / "euro2024.yaml", 51, 15),   # 36 group + 15 knockout (no 3rd-place game)
]


@pytest.mark.parametrize("config,n_matches,n_ko", _BENCHMARKS, ids=lambda v: getattr(v, "stem", v))
def test_seeded_benchmarks_score_consistently(config, n_matches, n_ko):
    md, data = _verify(config)
    s = data["summary"]
    assert s["all"]["matches"] == n_matches
    assert s["knockout"]["matches"] == n_ko
    assert len(data["matches"]) == n_matches
    for key in ("all", "group", "knockout"):
        t = s[key]
        assert 0 <= t["model"] <= t["max"]
        assert 0 <= t["naive"] <= t["max"]
    assert s["group"]["model"] + s["knockout"]["model"] == s["all"]["model"]
    assert s["group"]["max"] + s["knockout"]["max"] == s["all"]["max"]
    assert s["knockout"]["max"] == n_ko * 20  # knockout exact = 20 pts each
