"""Write-side helpers for the match corpus (the counterpart to ``corpus_results.py`` reads).

Used by the maintainer results-fetch tool to record a played score **directly into**
``international_results.csv``: fill the pre-existing ``NA,NA`` row for the match (the common case
for a scheduled tournament — every fixture already sits in the corpus as a future row) or append a
new row, matched by date (±1 day) + the unordered team pair — exactly the join the resolver and the
migration use. Edits are **line-targeted** so the ~49k-row corpus diff stays one changed line per
recorded match.
"""

from __future__ import annotations

import csv
import io
from datetime import date as _date
from datetime import timedelta
from pathlib import Path

from .historical_results_adapter import DEFAULT_CORPUS

_NA = {"", "NA"}


def read_corpus_lines(path: str | Path = DEFAULT_CORPUS) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines(keepends=True)


def write_corpus_lines(path: str | Path, lines: list[str]) -> None:
    Path(path).write_text("".join(lines), encoding="utf-8")


def _fields(line: str) -> list[str]:
    return next(csv.reader([line.rstrip("\n")]))


def _format_row(fields: list[str]) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(fields)
    return buf.getvalue()


def find_corpus_match(
    lines: list[str], *, date_hint: str, home_corpus: str, away_corpus: str
) -> tuple[int, list[str]] | None:
    """Index + parsed fields of the corpus line for this match (date ±1 day, unordered pair)."""
    try:
        target = _date.fromisoformat(date_hint)
    except ValueError:
        target = None
    pair = frozenset((home_corpus, away_corpus))
    for i in range(1, len(lines)):
        if not lines[i].strip():
            continue
        f = _fields(lines[i])
        if len(f) < 5 or frozenset((f[1], f[2])) != pair:
            continue
        if target is not None:
            try:
                if abs((_date.fromisoformat(f[0]) - target).days) > 1:
                    continue
            except ValueError:
                continue
        return i, f
    return None


def set_corpus_score(
    lines: list[str],
    *,
    date_hint: str,
    home_corpus: str,
    away_corpus: str,
    home_goals: int,
    away_goals: int,
    competition: str = "FIFA World Cup",
    country: str = "",
    neutral: bool = True,
) -> tuple[str, str]:
    """Record a score in ``lines`` (mutated in place). Returns ``(corpus_date, action)``.

    ``action`` is ``"filled"`` (an ``NA`` row was scored), ``"exists"`` (already scored — no-op,
    idempotent) or ``"appended"`` (no row existed, e.g. a knockout match whose participants were
    unknown when the corpus was built). The scoreline is oriented to the matched row's home team.
    """
    found = find_corpus_match(
        lines, date_hint=date_hint, home_corpus=home_corpus, away_corpus=away_corpus
    )
    if found is not None:
        idx, f = found
        if f[3] in _NA or f[4] in _NA:
            oh, oa = (home_goals, away_goals) if f[1] == home_corpus else (away_goals, home_goals)
            # The two score fields are the only ",NA,NA," in the row, so a targeted replace keeps
            # every other field byte-identical.
            lines[idx] = lines[idx].replace(",NA,NA,", f",{oh},{oa},", 1)
        return f[0], "filled" if (f[3] in _NA or f[4] in _NA) else "exists"
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    lines.append(_format_row(
        [date_hint, home_corpus, away_corpus, str(home_goals), str(away_goals),
         competition, "", country, "TRUE" if neutral else "FALSE"]
    ))
    return date_hint, "appended"


def latest_results_date(results_csv: str | Path) -> str | None:
    """Max of the thin ``results.csv`` ``date`` column (the latest played corpus date)."""
    path = Path(results_csv)
    if not path.exists():
        return None
    dates: list[str] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            d = (row.get("date") or "").strip()
            if d:
                dates.append(d)
    return max(dates) if dates else None


def snapshot_after(latest_date: str) -> str:
    """The ``offdef.snapshot_date`` for a given latest-played date: the day after (cutoff is
    strictly-earlier, so this folds the played match into the fit)."""
    return (_date.fromisoformat(latest_date) + timedelta(days=1)).isoformat()
