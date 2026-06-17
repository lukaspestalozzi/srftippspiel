"""One-off: convert each tournament's fat results.csv into the thin corpus-reference form.

Old form: ``match_id,home_goals,away_goals,winner_team_id``
New form: ``match_id,date,winner_team_id`` (scoreline read from the corpus by date + teams).

For each played row we locate its corpus match by the fixture's concrete team pair within a
kickoff +/- 1 day window, verify the corpus scoreline reproduces the old result (in either
orientation), and emit the corpus date. Rows whose fixture is not concrete (unplayed live KO),
or that do not resolve to exactly one corpus match, are reported and left for manual fixup
(the script refuses to rewrite a file with unresolved rows unless --force).

Usage:  python scripts/migrate_results_to_corpus_refs.py [--write] [config ...]
Default configs: the live config.yaml + all configs/*.yaml.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tippspiel.config import load_tournament  # noqa: E402
from tippspiel.data.corpus_results import build_corpus_index  # noqa: E402
from tippspiel.data.file_provider import FileDataProvider  # noqa: E402
from tippspiel.data.historical_results_adapter import DEFAULT_CORPUS, corpus_name_for  # noqa: E402

DEFAULT_CONFIGS = ["config.yaml", *sorted(str(p) for p in (REPO / "configs").glob("*.yaml"))]


def _pair_window_index(corpus_path):
    """Index corpus matches by unordered team pair -> list of (date, home_name, hg, ag)."""
    idx: dict[frozenset, list[tuple[str, str, int, int]]] = {}
    with Path(corpus_path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            d = (row.get("date") or "").strip()
            hg, ag = (row.get("home_score") or "").strip(), (row.get("away_score") or "").strip()
            if not d or not hg or not ag or hg == "NA" or ag == "NA":
                continue
            home, away = (row.get("home_team") or "").strip(), (row.get("away_team") or "").strip()
            idx.setdefault(frozenset((home, away)), []).append((d, home, int(hg), int(ag)))
    return idx


def migrate_config(config_path: str, pair_idx, write: bool) -> int:
    bundle = load_tournament(config_path)
    provider = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file)
    fixtures = {m.match_id: m for m in provider.get_fixtures()}
    name_by_id = {t.team_id: t.name for t in provider.get_teams()}

    if not bundle.results_file.exists():
        print(f"[{bundle.name}] no results.csv; skipped")
        return 0

    with bundle.results_file.open(newline="", encoding="utf-8") as fh:
        old_rows = [r for r in csv.DictReader(fh) if r.get("match_id")]

    out_rows: list[dict] = []
    problems: list[str] = []
    for r in old_rows:
        mid = r["match_id"].strip()
        winner = (r.get("winner_team_id") or "").strip()
        if "date" in r and (r.get("date") or "").strip() and not (r.get("home_goals") or "").strip():
            out_rows.append({"match_id": mid, "date": r["date"].strip(), "winner_team_id": winner})
            continue  # already thin
        old_h, old_a = int(r["home_goals"]), int(r["away_goals"])
        fx = fixtures.get(mid)
        if fx is None or not (fx.home.is_concrete and fx.away.is_concrete):
            problems.append(f"{mid}: fixture missing or not concrete (unplayed KO?)")
            continue
        hc = corpus_name_for(name_by_id.get(fx.home.team_id, ""))
        ac = corpus_name_for(name_by_id.get(fx.away.team_id, ""))
        kdate = fx.kickoff.date()
        cands = [
            (d, home, hg, ag)
            for (d, home, hg, ag) in pair_idx.get(frozenset((hc, ac)), [])
            if abs((date.fromisoformat(d) - kdate).days) <= 1
        ]
        if not cands:
            problems.append(f"{mid}: no corpus match for {hc} vs {ac} near {kdate}")
            continue
        # Keep candidates whose scoreline reproduces the old result (either orientation).
        good = []
        for d, home, hg, ag in cands:
            oriented = (hg, ag) if home == hc else (ag, hg)
            if oriented == (old_h, old_a):
                good.append((d, home, hg, ag))
        if len(good) != 1:
            problems.append(
                f"{mid}: {len(good)} corpus rows reproduce {old_h}-{old_a} for {hc} vs {ac} "
                f"near {kdate} (of {len(cands)} pair candidates)"
            )
            continue
        out_rows.append({"match_id": mid, "date": good[0][0], "winner_team_id": winner})

    status = "OK " if not problems else "!! "
    print(f"[{status}{bundle.name}] {len(out_rows)}/{len(old_rows)} rows resolved")
    for p in problems:
        print(f"      - {p}")
    if problems:
        return len(problems)

    if write:
        with bundle.results_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["match_id", "date", "winner_team_id"])
            writer.writeheader()
            writer.writerows(out_rows)
        # Self-check: re-resolving the thin file reproduces the old scorelines exactly.
        corpus_index = build_corpus_index(DEFAULT_CORPUS)
        from tippspiel.data.corpus_results import resolve_corpus_result

        old_by_id = {r["match_id"].strip(): (int(r["home_goals"]), int(r["away_goals"]))
                     for r in old_rows if (r.get("home_goals") or "").strip()}
        for row in out_rows:
            res = resolve_corpus_result(row["match_id"], row["date"],
                                        row["winner_team_id"] or None,
                                        fixtures, name_by_id, corpus_index)
            exp = old_by_id.get(row["match_id"])
            if exp and (res.home_goals, res.away_goals) != exp:
                print(f"      SELF-CHECK FAIL {row['match_id']}: "
                      f"{(res.home_goals, res.away_goals)} != {exp}")
                return 1
        print(f"      written + self-check passed -> {bundle.results_file}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--write", action="store_true", help="rewrite results.csv (else dry-run)")
    args = ap.parse_args(argv)
    pair_idx = _pair_window_index(DEFAULT_CORPUS)
    total = 0
    for cfg in (args.configs or DEFAULT_CONFIGS):
        total += migrate_config(cfg, pair_idx, args.write)
    if total:
        print(f"\n{total} unresolved row(s) across configs; fix naming/aliases or fixtures.")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
