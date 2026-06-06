"""Tests for the Claude diagnostic report (model-introspection tool)."""

import json
from pathlib import Path

import pytest

import tippspiel
from tippspiel.config import load_config, load_tournament
from tippspiel.data.file_provider import FileDataProvider
from tippspiel.pipeline import _predict_tippable, build_predictor, build_strategy
from tippspiel.report.diagnostics import DiagnosticsWriter, build_diagnostics
from tippspiel.simulation.simulator import TournamentSimulator
from tippspiel.strategy.expected_points import ExpectedPointsStrategy

REPO = Path(tippspiel.__file__).parent.parent


def _load():
    cfg = load_config(REPO / "config.yaml")
    bundle = load_tournament(REPO / "config.yaml")
    prov = FileDataProvider(bundle.teams_file, bundle.fixtures_file,
                            bundle.results_file, bundle.thirds_allocation_file)
    teams = {t.team_id: t for t in prov.get_teams()}
    return cfg, bundle, teams, prov.get_fixtures(), prov


@pytest.fixture(scope="module")
def diag():
    cfg, bundle, teams, fixtures, prov = _load()
    predictor = build_predictor(cfg)
    strategy = build_strategy(cfg, bundle)
    outcome = TournamentSimulator(fixtures, teams, {}, predictor, prov.get_thirds_allocation(),
                                  iterations=3000, seed=7).run()
    preds = _predict_tippable(fixtures, teams, set(), predictor)
    tipset = strategy.generate_tips(preds, outcome, fixtures)
    md, data = build_diagnostics(cfg, bundle, teams, fixtures, {}, preds, tipset, outcome, predictor)
    return {"md": md, "data": data, "n_tippable": len(preds)}


def test_markdown_has_all_sections(diag):
    md = diag["md"]
    for header in (
        "# Claude Diagnostic Report",
        "## 2. Predictor behaviour",
        "## 3. Offensive / defensive Elo",
        "## 4. Per-fixture detail",
        "## 5. Simulation diagnostics",
        "## 6. Bonus-question diagnostics",
        "## 7. Validation / anomaly summary",
    ):
        assert header in md


def test_tip_frequency_and_fixture_rows_account_for_every_tippable(diag):
    pb = diag["data"]["predictor_behaviour"]
    assert sum(d["count"] for d in pb["tip_frequency"]) == diag["n_tippable"]
    assert len(diag["data"]["fixtures"]) == diag["n_tippable"]


def test_low_scoreline_behaviour_is_explained():
    # Under strict EV (realism_tolerance=0) tips cluster on 1:0/0:1; the diagnostic must answer the
    # headline "why always 1:0 / 0:1?" via the tendency-dominance note. (The default config sets a
    # realism tolerance that mitigates the clustering, so this exercises the strict-EV path.)
    cfg, bundle, teams, fixtures, prov = _load()
    predictor = build_predictor(cfg)
    strict = ExpectedPointsStrategy(bundle.bonus_questions, realism_tolerance=0.0)
    preds = _predict_tippable(fixtures, teams, set(), predictor)
    tipset = strict.generate_tips(preds, None, fixtures)
    _md, data = build_diagnostics(cfg, bundle, teams, fixtures, {}, preds, tipset, None, predictor)
    notes = " ".join(data["predictor_behaviour"]["notes"]).lower()
    assert "1:0" in notes and "tendency" in notes


def test_no_failing_anomaly_checks_on_default_data(diag):
    statuses = [a["status"] for a in diag["data"]["anomalies"]]
    assert "FAIL" not in statuses
    # The sim invariants must be present and passing.
    assert {"Sum(wins_title) ~ 1", "Sum(reach_r32) is integer"} <= {a["name"] for a in diag["data"]["anomalies"]}


def test_json_sidecar_round_trips(diag, tmp_path):
    paths = DiagnosticsWriter().write(diag["md"], diag["data"], tmp_path)
    loaded = json.loads(paths["json"].read_text())
    assert loaded["meta"]["simulated"] is True
    assert loaded["fixtures"] and loaded["anomalies"]


def test_no_sim_mode_degrades_gracefully():
    cfg, bundle, teams, fixtures, _ = _load()
    predictor = build_predictor(cfg)
    strategy = build_strategy(cfg, bundle)
    preds = _predict_tippable(fixtures, teams, set(), predictor)
    tipset = strategy.generate_tips(preds, None, fixtures)
    md, data = build_diagnostics(cfg, bundle, teams, fixtures, {}, preds, tipset, None, predictor)
    assert data["simulation"] is None
    assert "Simulation skipped" in md
    bonus = {b["id"]: b for b in data["bonus"]}
    assert bonus["champion"]["available"] is False          # needs the simulation
    assert bonus["top_scorer_goals"]["available"] is True   # fixed historical prior
