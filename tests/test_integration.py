"""Full-pipeline integration test (spec §10): a well-formed, self-contained, multi-model report.

The combined pipeline runs every configured predictor and presents four tips per match
(2 ELO models × {EV, rank}) plus a separate outcomes section per model."""

import dataclasses
import re
from pathlib import Path

import pytest

import tippspiel
from tippspiel.config import load_config, load_tournament
from tippspiel.model.types import Result
from tippspiel.pipeline import run_combined_pipeline, write_report

REPO = Path(tippspiel.__file__).parent.parent
BUNDLE = load_tournament(REPO / "config.yaml")


@pytest.fixture(scope="module")
def small_cfg():
    cfg = load_config(REPO / "config.yaml")
    sim = dataclasses.replace(cfg.simulation, iterations=400)
    return dataclasses.replace(cfg, simulation=sim)


def test_predict_only_combined_pipeline(small_cfg):
    result = run_combined_pipeline(small_cfg, BUNDLE, simulate=False)
    # Both configured predictors should run; each gets 72 tippable group fixtures.
    runs = result["runs"]
    assert {r["name"] for r in runs} == {"elo_poisson", "attack_defence_poisson"}
    for r in runs:
        assert r["core"]["outcome"] is None
        assert len(r["core"]["predictions"]) == 72
    # Per-fixture: two model rows, each with an EV and a Rank tip.
    fx0 = result["context"]["groups"][0]["fixtures"][0]
    assert fx0["tippable"] is True
    assert len(fx0["tip_rows"]) == 2
    for row in fx0["tip_rows"]:
        assert {"ev", "rank", "contrarian", "ldw_chart", "heatmap"} <= set(row)


def test_full_combined_pipeline_self_contained_report(tmp_path, small_cfg):
    cfg = dataclasses.replace(
        small_cfg, report=dataclasses.replace(small_cfg.report, output_dir=str(tmp_path))
    )
    result = run_combined_pipeline(cfg, BUNDLE, simulate=True)
    path = write_report(cfg, result["context"])
    html = Path(path).read_text()

    assert html.strip().endswith("</html>")
    assert "Plotly.newPlot" in html
    # Self-contained: no external script/style/image loads.
    assert not re.search(r'<(script|link|img)[^>]*(src|href)=[\"\']https?://', html)
    # Both model labels and a per-model outcomes section for each.
    for label in ("FIFA Elo", "Attack / Defence"):
        assert label in html
        assert f"Outcomes — {label}" in html
    # The 2×2 tip matrix renders for every tippable group fixture.
    assert "tipmatrix" in html
    # Each model produced a champion pick from its own Monte-Carlo outcome.
    for run in result["runs"]:
        outcome = run["core"]["outcome"]
        assert outcome is not None
        assert any(m.get("wins_title", 0.0) > 0 for m in outcome.advancement.values())


def test_played_match_excluded_from_tips(small_cfg):
    """A played match must not receive any tip rows (its result is fixed)."""
    from tippspiel.data.file_provider import FileDataProvider

    orig = FileDataProvider.get_results
    FileDataProvider.get_results = lambda self: [Result("G_A_1", 1, 0)]
    try:
        result = run_combined_pipeline(small_cfg, BUNDLE, simulate=False)
    finally:
        FileDataProvider.get_results = orig

    blocks = [fx for g in result["context"]["groups"] for fx in g["fixtures"]]
    by_id = {fx["match_id"]: fx for fx in blocks}
    assert by_id["G_A_1"]["played"] is True
    assert by_id["G_A_1"]["tip_rows"] == []
    # Every other group fixture stays tippable with two model rows.
    tippable = [fx for fx in blocks if fx["tippable"]]
    assert len(tippable) == 71
    for fx in tippable:
        assert len(fx["tip_rows"]) == 2
