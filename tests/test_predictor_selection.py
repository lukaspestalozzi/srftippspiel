"""Predictor is a required CLI parameter (no config default); each model reads its own
ratings file (elo_poisson -> teams.csv; attack_defence_poisson -> teams_attack_defence.csv)."""

import dataclasses
from pathlib import Path

import pytest

import tippspiel
from tippspiel.cli import main
from tippspiel.config import load_config, load_tournament, select_predictor
from tippspiel.pipeline import ratings_file
from tippspiel.predictors.attack_defence import AttackDefencePoissonPredictor
from tippspiel.predictors.elo_poisson import EloPoissonPredictor

REPO = Path(tippspiel.__file__).parent.parent
WC2026 = REPO / "config.yaml"


def test_config_has_no_default_predictor_but_offers_both():
    cfg = load_config(WC2026)
    assert cfg.predictor is None  # no default — must be chosen explicitly
    assert {"elo_poisson", "attack_defence_poisson"} <= set(cfg.predictors)


def test_select_predictor_sets_active_and_validates():
    cfg = load_config(WC2026)
    selected = select_predictor(cfg, "elo_poisson")
    assert selected.predictor.name == "elo_poisson"
    assert cfg.predictor is None  # original unchanged (frozen dataclass)
    with pytest.raises(ValueError):
        select_predictor(cfg, "no_such_model")


def test_ratings_file_resolves_by_predictor_kind():
    bundle = load_tournament(WC2026)
    # elo_poisson reads the official eloratings snapshot (teams.csv, always present).
    assert ratings_file(EloPoissonPredictor(), bundle) == bundle.teams_file
    # attack_defence_poisson reads the computed two-rating file by convention.
    assert bundle.teams_files["attack_defence"].name == "teams_attack_defence.csv"


def test_ratings_file_missing_raises():
    bundle = load_tournament(WC2026)
    bad = dataclasses.replace(
        bundle, teams_files={**bundle.teams_files, "attack_defence": Path("/no/such/file.csv")}
    )
    with pytest.raises(ValueError):
        ratings_file(AttackDefencePoissonPredictor(), bad)


def test_cli_requires_predictor_for_single_model_commands(capsys):
    # verify/diagnose/tune all build a single Predictor and require --predictor (no default).
    # run/predict do not (they always run every configured predictor side by side).
    for cmd in ("verify", "diagnose", "tune"):
        rc = main([cmd, "--config", str(WC2026)])
        assert rc == 2, f"{cmd} should require --predictor"
        err = capsys.readouterr().err
        assert "--predictor is required" in err, f"{cmd}: {err!r}"


def test_cli_rejects_unknown_predictor(capsys):
    rc = main(["verify", "--config", str(WC2026), "--predictor", "bogus"])
    assert rc == 2
    assert "Unknown predictor" in capsys.readouterr().err
