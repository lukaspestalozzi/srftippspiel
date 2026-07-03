"""Maintainer tool: record finished matches into the corpus + thin ``results.csv`` + snapshot.

Offline, **not on the runtime path** (mirrors ``espn_odds_fetch``). Takes the same tippable
fixture list the odds fetchers use (:func:`load_tippable_fixtures`) — group matches **and**
knockout matches whose participants the played results have already settled (KO rows store their
participants as structural refs ``W:A``/``R:B``/``3RD:…``, so the raw ``fixtures.csv`` rows would
otherwise be skipped and no KO result ever recorded) — keeps those whose kickoff has passed and
that aren't yet in ``results.csv``, and looks up the full-time score from the same ESPN scoreboard
JSON used for odds (events with ``status.type.state == "post"``).

Under the corpus model a score is recorded in **one reviewed step** (``--write``):
  * the score fills the match's row in ``international_results.csv`` (or appends one), and
  * a thin ``match_id,date,winner_team_id`` row is appended to ``results.csv``, and
  * ``offdef.snapshot_date`` in the config is advanced to the day after the latest played date.

Run **without** ``--write`` first: it prints each candidate (``match_id  HOME h-a AWAY
(corpus_date) [filled|exists|appended]``) so you can **dual-source** the scores (e.g. against
FIFA/Wikipedia) before committing — this tool is one of the two sources, not a replacement.
For a knockout match level after 90' the shootout winner is read from the scoreboard's
``shootoutScore`` and written to ``winner_team_id`` (printed as ``pens:<WINNER>`` for review); it
is left blank and flagged on stderr only when the feed carries no shootout score, for the
maintainer to fill in by hand.

Usage (run from the repo root, network required for the fetch)::

    python -m tippspiel.data.espn_results_fetch wc2026 fifa.world
    python -m tippspiel.data.espn_results_fetch wc2026 fifa.world --write --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

from tippspiel.data.corpus_update import (
    latest_results_date,
    read_corpus_lines,
    set_corpus_score,
    snapshot_after,
    write_corpus_lines,
)
from tippspiel.data.espn_common import (
    REPO,
    date_window,
    fetch_scoreboard,
    find_event,
    load_played_match_ids,
    load_team_names,
    load_teams,
)
from tippspiel.data.fixture_resolve import load_tippable_fixtures
from tippspiel.data.historical_results_adapter import DEFAULT_CORPUS, corpus_name_for

_THIN_FIELDS = ["match_id", "date", "winner_team_id"]


def _score(competitor: dict) -> str | None:
    raw = competitor.get("score")
    if isinstance(raw, dict):
        raw = raw.get("displayValue") or raw.get("value")
    return None if raw is None else str(raw)


def _pens(competitor: dict) -> int | None:
    """The competitor's penalty-shootout score, if the scoreboard carries one (else ``None``)."""
    raw = competitor.get("shootoutScore")
    if isinstance(raw, dict):
        raw = raw.get("displayValue") or raw.get("value")
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return None


def fetch_results(tournament: str, slug: str, *, now: datetime | None = None) -> list[dict]:
    """Return score rows for ``tournament``'s finished, unrecorded matches (network).

    Each row: ``match_id, home_id, away_id, home_goals, away_goals, stage, date (YYYY-MM-DD),
    venue_country, shootout, winner_team_id``. For a knockout match level after 90' the shootout
    winner is read from the scoreboard's ``shootoutScore`` and put in ``winner_team_id``; only a
    shootout the feed doesn't price is left blank for the maintainer. Diagnostics (not-final /
    missing / shootout-needs-manual) go to stderr.

    ``now`` (defaults to the current UTC time) is the cutoff for "kickoff has passed"; tests pass a
    fixed value so the candidate list doesn't depend on the wall clock.
    """
    tdir = REPO / "tippspiel" / "data" / "tournaments" / tournament
    teams = load_teams(tdir)
    played = load_played_match_ids(tdir)
    now = now or datetime.now(timezone.utc)
    candidates = [
        f for f in load_tippable_fixtures(tdir)
        if f["match_id"] not in played
        and datetime.fromisoformat(f["kickoff_utc"].replace("Z", "+00:00")) < now
    ]
    scoreboard = fetch_scoreboard(slug, sorted({d for f in candidates for d in date_window(f["date"])}))

    rows, not_final, missing, shootout_watch = [], [], [], []
    for f in candidates:
        try:
            events = [e for d in date_window(f["date"]) for e in scoreboard.get(d, [])]
            found = find_event(events, teams, f["home_id"], f["away_id"])
            if found is None:
                missing.append(f["match_id"])
                continue
            event, ids = found
            if ((event.get("status") or {}).get("type") or {}).get("state") != "post":
                not_final.append(f["match_id"])
                continue
            comp = (event.get("competitions") or [{}])[0]
            competitors = comp.get("competitors", [])
            scores = {ids.get(c.get("homeAway")): _score(c) for c in competitors}
            pens = {ids.get(c.get("homeAway")): _pens(c) for c in competitors}
            if scores.get(f["home_id"]) is None or scores.get(f["away_id"]) is None:
                missing.append(f["match_id"])
                continue
            hg, ag = int(scores[f["home_id"]]), int(scores[f["away_id"]])
            shootout = f["stage"] != "GROUP" and hg == ag
            winner_team_id = ""
            if shootout:
                ph, pa = pens.get(f["home_id"]), pens.get(f["away_id"])
                if ph is not None and pa is not None and ph != pa:
                    winner_team_id = f["home_id"] if ph > pa else f["away_id"]
                else:
                    shootout_watch.append(f["match_id"])  # no usable shootout score -> by hand
            rows.append({
                "match_id": f["match_id"], "home_id": f["home_id"], "away_id": f["away_id"],
                "home_goals": hg, "away_goals": ag, "stage": f["stage"],
                "date": f["kickoff_utc"][:10], "venue_country": f.get("venue_country", ""),
                "shootout": shootout, "winner_team_id": winner_team_id,
            })
        except Exception:  # noqa: BLE001 — one bad fixture shouldn't abort the run
            missing.append(f["match_id"])
    if not_final:
        print(f"# kicked off but not yet final (state != post): {not_final}", file=sys.stderr)
    if missing:
        print(f"# no finished-match scoreboard entry found: {missing}", file=sys.stderr)
    if shootout_watch:
        print(f"# level after 90' in a knockout match but no shootout score in the feed -- fill "
              f"winner_team_id by hand once the shootout result is known: {shootout_watch}",
              file=sys.stderr)
    return rows


def record_results(
    tdir: str | Path,
    score_rows: list[dict],
    *,
    corpus_path: str | Path = DEFAULT_CORPUS,
    config_path: str | Path | None = None,
    write: bool = False,
) -> dict:
    """Record ``score_rows`` into the corpus + thin ``results.csv`` + ``snapshot_date`` (offline).

    ``tdir`` is the tournament data dir (holds ``teams.csv``/``results.csv``). Each row fills (or
    appends) its corpus match and yields a thin ``results.csv`` row; the proposed
    ``offdef.snapshot_date`` is the day after the latest played date. With ``write=False`` nothing
    is written — the returned plan is printed for dual-source review. Idempotent: corpus rows
    already scored are left untouched and re-runs add no duplicate thin rows.
    """
    tdir = Path(tdir)
    names = load_team_names(tdir)
    results_csv = tdir / "results.csv"
    already = load_played_match_ids(tdir)

    lines = read_corpus_lines(corpus_path)
    plan: list[dict] = []
    for r in score_rows:
        if r["match_id"] in already:
            continue
        hc, ac = corpus_name_for(names[r["home_id"]]), corpus_name_for(names[r["away_id"]])
        neutral = r.get("venue_country", "") not in (r["home_id"], r["away_id"])
        corpus_date, action = set_corpus_score(
            lines, date_hint=r["date"], home_corpus=hc, away_corpus=ac,
            home_goals=r["home_goals"], away_goals=r["away_goals"],
            country=r.get("venue_country", ""), neutral=neutral,
        )
        plan.append({
            "match_id": r["match_id"], "home": hc, "away": ac,
            "home_goals": r["home_goals"], "away_goals": r["away_goals"],
            "corpus_date": corpus_date, "action": action,
            # Knockout shootout winner, auto-read from the feed's shootoutScore; blank only when the
            # feed didn't price it (the maintainer then fills it by hand — flagged on stderr).
            "winner": r.get("winner_team_id", ""),
        })

    # Proposed snapshot = day after the latest played corpus date (existing + newly recorded).
    dates = [p["corpus_date"] for p in plan]
    if (cur := latest_results_date(results_csv)) is not None:
        dates.append(cur)
    snapshot = snapshot_after(max(dates)) if dates else None

    if write and plan:
        write_corpus_lines(corpus_path, lines)
        header = not results_csv.exists() or results_csv.stat().st_size == 0
        with results_csv.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=_THIN_FIELDS)
            if header:
                w.writeheader()
            for p in plan:
                w.writerow({"match_id": p["match_id"], "date": p["corpus_date"],
                            "winner_team_id": p["winner"]})
        if config_path and snapshot:
            from tippspiel.config import write_offdef_snapshot_date
            try:
                write_offdef_snapshot_date(config_path, snapshot)
            except ValueError:
                # Corpus + results are already written; don't abort on a config without the line —
                # tell the maintainer to set it by hand rather than leaving a half-applied state.
                print(f"# WARN: no 'snapshot_date:' in {config_path}; set offdef.snapshot_date = "
                      f"{snapshot} by hand", file=sys.stderr)

    return {"plan": plan, "snapshot": snapshot, "written": bool(write and plan)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Record finished ESPN matches into the corpus.")
    ap.add_argument("tournament")
    ap.add_argument("slug")
    ap.add_argument("--write", action="store_true",
                    help="commit the corpus + thin results + snapshot edits (review first)")
    ap.add_argument("--config", default="config.yaml",
                    help="config whose offdef.snapshot_date is advanced (default: config.yaml)")
    args = ap.parse_args(argv)

    rows = fetch_results(args.tournament, args.slug)
    tdir = REPO / "tippspiel" / "data" / "tournaments" / args.tournament
    result = record_results(tdir, rows, config_path=args.config, write=args.write)
    plan = result["plan"]
    if not plan:
        print("no new finished matches to record.", file=sys.stderr)
        return 0
    for p in plan:
        pens = f" pens:{p['winner']}" if p["winner"] else ""
        print(f"{p['match_id']:<8} {p['home']} {p['home_goals']}-{p['away_goals']} {p['away']}{pens}"
              f"  ({p['corpus_date']}) [{p['action']}]")
    verb = "recorded" if result["written"] else "would record"
    print(f"# {verb} {len(plan)} match(es); snapshot_date -> {result['snapshot']}"
          f"{'' if result['written'] else '  (dry-run; pass --write to commit)'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
