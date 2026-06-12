"""Site assembler: the GitHub Pages publish job's `python -m tippspiel.report.site`.

The assembler maps CI artifact directories (run-report, predict-report-<name>, verify-<name>)
onto the static-site layout, with the tournament list derived from the committed config files.
"""

import html
from pathlib import Path

import pytest

import tippspiel
from tippspiel.config import load_tournament
from tippspiel.report.site import build_site, discover_configs

REPO = Path(tippspiel.__file__).parent.parent
BUNDLES = [load_tournament(p) for p in discover_configs(REPO)]
LIVE = next(b for b in BUNDLES if not b.completed)
COMPLETED = [b for b in BUNDLES if b.completed]


@pytest.fixture
def artifacts(tmp_path):
    art = tmp_path / "artifacts"
    (art / "run-report").mkdir(parents=True)
    (art / "run-report" / "report.html").write_text("<html>live report</html>", encoding="utf-8")
    for b in COMPLETED:
        (art / f"predict-report-{b.name}").mkdir()
        (art / f"predict-report-{b.name}" / "report.html").write_text(f"<html>{b.name}</html>", encoding="utf-8")
        (art / f"verify-{b.name}").mkdir()
        (art / f"verify-{b.name}" / "verify.md").write_text(f"# {b.name}\n<pre> & raw md", encoding="utf-8")
    return art


def test_site_layout(artifacts, tmp_path):
    out = tmp_path / "site"
    written = build_site(artifacts, out, commit="abc123def456789")
    assert (out / "index.html").is_file()
    assert (out / LIVE.name / "index.html").read_text(encoding="utf-8") == "<html>live report</html>"
    for b in COMPLETED:
        assert (out / b.name / "report.html").read_text(encoding="utf-8") == f"<html>{b.name}</html>"
        assert (out / b.name / "backtest.html").is_file()
    assert set(written) == {p for p in out.rglob("*") if p.is_file()}


def test_index_links_every_tournament(artifacts, tmp_path):
    out = tmp_path / "site"
    build_site(artifacts, out, commit="abc123def456789")
    index = (out / "index.html").read_text(encoding="utf-8")
    assert f'href="{LIVE.name}/"' in index and html.escape(LIVE.display_name) in index
    for b in COMPLETED:
        assert f'href="{b.name}/report.html"' in index
        assert f'href="{b.name}/backtest.html"' in index
        assert html.escape(b.display_name) in index
    assert "abc123def456" in index  # short commit in the footer


def test_verify_markdown_is_escaped_in_pre(artifacts, tmp_path):
    out = tmp_path / "site"
    build_site(artifacts, out)
    page = (out / COMPLETED[0].name / "backtest.html").read_text(encoding="utf-8")
    assert "&lt;pre&gt; &amp; raw md" in page
    assert "<pre> & raw md" not in page  # the raw markdown must never land unescaped


def test_missing_artifact_fails_loudly(artifacts, tmp_path):
    (artifacts / f"verify-{COMPLETED[0].name}" / "verify.md").unlink()
    with pytest.raises(FileNotFoundError, match=f"verify-{COMPLETED[0].name}"):
        build_site(artifacts, tmp_path / "site")


def test_exactly_one_live_tournament_expected(artifacts, tmp_path):
    completed_only = [p for p in discover_configs(REPO) if load_tournament(p).completed]
    with pytest.raises(ValueError, match="exactly one live"):
        build_site(artifacts, tmp_path / "site", configs=completed_only)
