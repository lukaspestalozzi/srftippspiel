"""Static-site assembler for the GitHub Pages publish job.

Collects the report artifacts that CI already builds (the live tournament's ``run`` report,
plus each completed benchmark's ``predict`` report and ``verify`` backtest) into a single
static site with a generated landing page:

    site/index.html                      landing page (tournament list from the config files)
    site/<live>/index.html               full run report (predictions + Monte Carlo)
    site/<benchmark>/report.html         a-priori tips next to the actual results (predict mode)
    site/<benchmark>/backtest.html       the verify backtest, rendered as a monospace page

The tournament list is derived from ``config.yaml`` + ``configs/*.yaml`` (display name and
``completed`` flag via :func:`tippspiel.config.load_tournament`), never hardcoded. Expected
artifact layout (one directory per CI artifact, as ``actions/download-artifact`` produces):
``run-report/report.html`` for the live tournament, ``predict-report-<name>/report.html`` and
``verify-<name>/verify.md`` per completed one. Missing inputs fail loudly.

CI usage: ``python -m tippspiel.report.site --artifacts <dir> --out site --commit <sha>``.
"""

from __future__ import annotations

import argparse
import html
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..config import TournamentBundle, load_tournament

_REPO = Path(__file__).parent.parent.parent

_PAGE_CSS = """
  :root { --fg:#222; --muted:#666; --line:#e2e2e2; --accent:#2c7fb8; --tip:#e6550d; }
  body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         color: var(--fg); max-width: 860px; margin: 2rem auto; padding: 0 1rem; }
  h1 { margin-bottom: 0.2rem; }
  a { color: var(--accent); }
  .muted { color: var(--muted); }
  .card { border: 1px solid var(--line); border-radius: 8px; padding: 1rem 1.2rem;
          margin: 1rem 0; }
  .card.live { border-color: var(--tip); }
  .badge { font-size: 0.75rem; color: #fff; background: var(--tip); border-radius: 4px;
           padding: 2px 6px; vertical-align: middle; }
  footer { margin-top: 2rem; font-size: 0.85rem; color: var(--muted);
           border-top: 1px solid var(--line); padding-top: 0.6rem; }
  pre { font-size: 0.85rem; line-height: 1.35; overflow-x: auto; }
"""


def discover_configs(repo: Path = _REPO) -> list[Path]:
    """The live config (``config.yaml``) + every benchmark under ``configs/``."""
    return [repo / "config.yaml", *sorted((repo / "configs").glob("*.yaml"))]


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{_PAGE_CSS}</style>\n</head>\n"
        f"<body>\n{body}\n</body>\n</html>\n"
    )


def _footer(commit: str) -> str:
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit_note = f" from commit <code>{html.escape(commit[:12])}</code>" if commit else ""
    return f"<footer>Built {built}{commit_note} by the CI publish job.</footer>"


def _index_html(live: TournamentBundle, completed: list[TournamentBundle], commit: str) -> str:
    cards = [
        '<div class="card live">'
        f'<h2><a href="{live.name}/">{html.escape(live.display_name)}</a> '
        '<span class="badge">live</span></h2>'
        "<p>Recommended tips + bonus answers, with the full Monte-Carlo simulation "
        "(title odds, group qualification, bracket).</p></div>"
    ]
    for b in completed:
        cards.append(
            '<div class="card">'
            f"<h2>{html.escape(b.display_name)}</h2>"
            f'<p><a href="{b.name}/report.html">A-priori tips vs results</a> &middot; '
            f'<a href="{b.name}/backtest.html">Predictor backtest</a></p></div>'
        )
    body = (
        "<h1>Tippspiel reports</h1>\n"
        '<p class="muted">Scoreline tips optimised for expected betting-pool points, plus '
        "predictor backtests against completed tournaments.</p>\n"
        + "\n".join(cards)
        + "\n"
        + _footer(commit)
    )
    return _page("Tippspiel reports", body)


def _backtest_html(bundle: TournamentBundle, verify_md: str, commit: str) -> str:
    title = f"{bundle.display_name} — predictor backtest"
    body = (
        f"<h1>{html.escape(title)}</h1>\n"
        '<p class="muted"><a href="../index.html">&larr; all tournaments</a> &middot; '
        f'<a href="report.html">tips report</a></p>\n'
        f"<pre>{html.escape(verify_md)}</pre>\n" + _footer(commit)
    )
    return _page(title, body)


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"expected artifact file missing: {path}")
    return path


def build_site(artifacts: Path, out: Path, *, commit: str = "",
               configs: list[Path] | None = None) -> list[Path]:
    """Assemble the static site under ``out``; returns the files written."""
    bundles = [load_tournament(p) for p in (configs or discover_configs())]
    live = [b for b in bundles if not b.completed]
    completed = [b for b in bundles if b.completed]
    if len(live) != 1:
        raise ValueError(
            f"expected exactly one live (completed: false) tournament for the run-report "
            f"artifact, found {[b.name for b in live]}"
        )

    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    live_dir = out / live[0].name
    live_dir.mkdir(exist_ok=True)
    written.append(shutil.copyfile(_require(artifacts / "run-report" / "report.html"),
                                   live_dir / "index.html"))

    for b in completed:
        dest = out / b.name
        dest.mkdir(exist_ok=True)
        written.append(shutil.copyfile(
            _require(artifacts / f"predict-report-{b.name}" / "report.html"),
            dest / "report.html"))
        verify_md = _require(artifacts / f"verify-{b.name}" / "verify.md").read_text()
        backtest = dest / "backtest.html"
        backtest.write_text(_backtest_html(b, verify_md, commit))
        written.append(backtest)

    index = out / "index.html"
    index.write_text(_index_html(live[0], completed, commit))
    written.append(index)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, required=True,
                        help="directory of downloaded CI artifacts (one subdir per artifact)")
    parser.add_argument("--out", type=Path, required=True, help="site output directory")
    parser.add_argument("--commit", default="", help="commit SHA for the page footers")
    args = parser.parse_args(argv)
    for path in build_site(args.artifacts, args.out, commit=args.commit):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
