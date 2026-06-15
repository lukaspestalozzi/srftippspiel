#!/usr/bin/env python3
"""Leak-free retrospective: score the tips we *would have published before kickoff* for every
already-played match of a LIVE tournament, against the actual results.

Why this exists
---------------
``tippspiel verify`` backtests the predictor on a *completed* tournament from its single
pre-tournament Elo snapshot. It cannot score a *live* tournament's tips, because the live data
files are mutated every matchday (Elo re-fitted, odds refreshed, results appended). Scoring the
played matches against the *current* files is **leaky**: a team's Elo is bumped up *after* it
wins, so the model looks like it "knew". (On WC2026 matchday 1 that leak was worth +5 pool
points -- 49 vs the true 44.)

This tool reconstructs each match's genuine pre-kickoff tip by replaying the predictor on the
data **as it was committed before that match's result was added**. For every match it finds the
commit that first introduced the result and predicts from that commit's *first parent* -- a
snapshot whose ``results.csv`` provably does not yet contain the match. The result is the honest,
leak-free pool-points the recommended tips actually earned, plus a draw-calibration line.

Re-run it after each matchday to keep the predict -> score -> learn loop closed. Read-only; it
writes nothing and touches only ``git show`` of historical blobs.

Usage
-----
    python scripts/retro_tips.py [--config config.yaml] [--data-dir tournaments/wc2026]
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
import tempfile
from pathlib import Path

# Make the package importable when run as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tippspiel.config import load_config, load_tournament  # noqa: E402
from tippspiel.data.file_provider import FileDataProvider  # noqa: E402
from tippspiel.pipeline import build_predictor  # noqa: E402
from tippspiel.strategy.expected_points import best_tip, score_tip  # noqa: E402

DATA_FILES = ("teams.csv", "fixtures.csv", "results.csv", "odds.csv")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def _blob(commit: str, path: str) -> str | None:
    """Contents of ``path`` at ``commit``, or None if the file did not exist there."""
    res = subprocess.run(
        ["git", "show", f"{commit}:{path}"], capture_output=True, text=True
    )
    return res.stdout if res.returncode == 0 else None


def _result_ids(text: str | None) -> set[str]:
    if not text:
        return set()
    return {row["match_id"] for row in csv.DictReader(io.StringIO(text)) if row.get("match_id")}


def _snapshot_for_each_match(results_rel: str) -> dict[str, str]:
    """Map each played match_id -> the commit to predict it from (the pre-kickoff snapshot).

    Walks the commits that touched results.csv oldest-first; a match introduced by commit C is
    predicted from C's first parent (``C^``), whose results.csv provably lacks that match.
    """
    log = _git("log", "--reverse", "--format=%H", "--", results_rel).split()
    snap: dict[str, str] = {}
    seen: set[str] = set()
    for commit in log:
        ids = _result_ids(_blob(commit, results_rel))
        new = ids - seen
        if new:
            parent = _git("rev-parse", f"{commit}^").strip()
            for mid in new:
                snap[mid] = parent
        seen |= ids
    return snap


def _load_snapshot(commit: str, cfg, data_rel: str):
    """Materialise a commit's tournament data into a temp dir and load teams/fixtures/predictor."""
    tmp = tempfile.mkdtemp(prefix=f"retro_{commit[:7]}_")
    dest = Path(tmp) / data_rel
    dest.mkdir(parents=True, exist_ok=True)
    src_dir = f"tippspiel/data/{data_rel}"
    for fname in DATA_FILES:
        blob = _blob(commit, f"{src_dir}/{fname}")
        if blob is not None:
            (dest / fname).write_text(blob, encoding="utf-8")
    bundle = load_tournament(cfg.config_path, data_root=Path(tmp))
    provider = FileDataProvider(
        bundle.teams_file, bundle.fixtures_file, bundle.results_file,
        bundle.thirds_allocation_file, bundle.odds_file,
    )
    teams = {t.team_id: t for t in provider.get_teams()}
    fixtures = {m.match_id: m for m in provider.get_fixtures()}
    predictor = build_predictor(cfg, odds=provider.get_odds())
    return teams, fixtures, predictor, set(provider.get_odds())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data-dir", default=None,
                    help="tournament data dir relative to tippspiel/data "
                         "(default: from the config's tournament.data_dir)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bundle = load_tournament(args.config)
    data_rel = args.data_dir or str(bundle.teams_file.parent.relative_to(
        bundle.teams_file.parents[2]))  # .../data/<data_rel>/teams.csv
    results_rel = f"tippspiel/data/{data_rel}/results.csv"
    realism = cfg.strategy.realism_tolerance

    actual: dict[str, tuple[int, int]] = {}
    with open(bundle.results_file, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            actual[row["match_id"]] = (int(row["home_goals"]), int(row["away_goals"]))
    if not actual:
        print("No played matches yet -- nothing to retrospect.")
        return 0

    snap = _snapshot_for_each_match(results_rel)
    cache: dict[str, tuple] = {}

    rows = []
    our_pts = naive_pts = max_pts = 0
    exact = tendency = miss = 0
    pred_draw_sum = actual_draws = 0.0
    for mid, (ah, aa) in actual.items():
        commit = snap.get(mid)
        if commit is None:
            print(f"  (skip {mid}: no pre-result snapshot found in git history)")
            continue
        if commit not in cache:
            cache[commit] = _load_snapshot(commit, cfg, data_rel)
        teams, fixtures, predictor, odds_ids = cache[commit]
        m = fixtures[mid]
        weight = m.stage.points_weight
        dist = predictor.predict(m, teams).scoreline
        th, ta, _ = best_tip(dist, weight, realism)
        nh, na, _ = dist.most_likely_scorelines(1)[0]
        pts = score_tip(th, ta, ah, aa, weight)
        npts = score_tip(nh, na, ah, aa, weight)
        our_pts += pts
        naive_pts += npts
        max_pts += 10 * weight
        if (th, ta) == (ah, aa):
            label = "EXACT"
            exact += 1
        elif (th > ta) == (ah > aa) and (th < ta) == (ah < aa):
            label = "tendency"
            tendency += 1
        else:
            label = "MISS"
            miss += 1
        pred_draw_sum += dist.p_draw()
        actual_draws += 1 if ah == aa else 0
        eg = dist.expected_goals()
        name = (f"{teams[m.home.team_id].name}-{teams[m.away.team_id].name}"
                if m.home.is_concrete and m.away.is_concrete else mid)
        ldw = f"{dist.p_home_win()*100:.0f}/{dist.p_draw()*100:.0f}/{dist.p_away_win()*100:.0f}"
        rows.append((mid, name[:26], ldw, f"{eg[0]:.1f}-{eg[1]:.1f}",
                     f"{th}:{ta}", f"{ah}:{aa}", pts, f"{nh}:{na}", npts,
                     "odds" if mid in odds_ids else "elo", label))

    n = len(rows)
    print(f"\nLeak-free pre-kickoff tip retrospective  ({bundle.display_name}, {n} played)\n")
    hdr = ("match", "tie", "LDW%", "xG", "tip", "act", "pts", "naive", "np", "src", "result")
    print(f"{hdr[0]:<8}{hdr[1]:<27}{hdr[2]:<11}{hdr[3]:<9}{hdr[4]:<6}{hdr[5]:<6}"
          f"{hdr[6]:<5}{hdr[7]:<7}{hdr[8]:<4}{hdr[9]:<6}{hdr[10]}")
    print("-" * 100)
    for r in rows:
        print(f"{r[0]:<8}{r[1]:<27}{r[2]:<11}{r[3]:<9}{r[4]:<6}{r[5]:<6}"
              f"{r[6]:<5}{r[7]:<7}{r[8]:<4}{r[9]:<6}{r[10]}")
    print("-" * 100)
    print(f"our tips : {our_pts}/{max_pts} ({our_pts/max_pts*100:.1f}% of max)")
    print(f"naive    : {naive_pts}/{max_pts} ({naive_pts/max_pts*100:.1f}% of max)")
    print(f"exact {exact} | tendency-only {tendency} | miss {miss}")
    print(f"draws    : model expected {pred_draw_sum:.1f}, actually {actual_draws:.0f} "
          f"(of {n}) -- mean P(draw) {pred_draw_sum/n*100:.0f}%, actual {actual_draws/n*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
