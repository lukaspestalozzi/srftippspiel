"""Full-pipeline integration test (spec §10): a well-formed, self-contained report."""

import dataclasses
import re
from pathlib import Path

import pytest

import tippspiel
from tippspiel.config import load_config
from tippspiel.model.types import Result
from tippspiel.pipeline import run_pipeline, write_report

REPO = Path(tippspiel.__file__).parent.parent


@pytest.fixture(scope="module")
def small_cfg():
    cfg = load_config(REPO / "config.yaml")
    sim = dataclasses.replace(cfg.simulation, iterations=400)
    return dataclasses.replace(cfg, simulation=sim)


def test_predict_only_pipeline(small_cfg):
    result = run_pipeline(small_cfg, simulate=False)
    # 72 group fixtures, all tippable, no simulation-dependent bonus answers required.
    assert len(result["tipset"].tips) == 72
    assert result["outcome"] is None


def test_full_pipeline_self_contained_report(tmp_path, small_cfg):
    cfg = dataclasses.replace(
        small_cfg, report=dataclasses.replace(small_cfg.report, output_dir=str(tmp_path))
    )
    result = run_pipeline(cfg, simulate=True)
    path = write_report(cfg, result["context"])
    html = Path(path).read_text()

    assert html.strip().endswith("</html>")
    assert "Plotly.newPlot" in html  # figures rendered
    # Self-contained: no external script/style/image loads.
    assert not re.search(r'<(script|link|img)[^>]*(src|href)=[\"\']https?://', html)
    # Champion recommendation present.
    assert result["tipset"].bonus_answers.get("champion")
    for section in ("Group-stage fixtures", "Group advancement", "Title odds", "Bonus"):
        assert section in html


def test_played_match_excluded_from_tips(small_cfg):
    # A played match must not receive a tip (its result is fixed).
    cfg = small_cfg
    import tippspiel.pipeline as pl
    from tippspiel.data.file_provider import FileDataProvider

    orig = FileDataProvider.get_results
    FileDataProvider.get_results = lambda self: [Result("G_A_1", 1, 0)]
    try:
        result = run_pipeline(cfg, simulate=False)
    finally:
        FileDataProvider.get_results = orig
    assert "G_A_1" not in result["tipset"].tips
    assert len(result["tipset"].tips) == 71
