"""Maintainer tool: list finished-but-unrecorded matches as candidate ``results.csv`` rows.

Offline, **not on the runtime path** (mirrors ``espn_odds_fetch``). Reads ``fixtures.csv``, finds
fixtures whose kickoff has passed and that aren't yet in ``results.csv``, and looks up the
full-time score from the same ESPN scoreboard JSON used for odds (events with
``status.type.state == "post"``).

Prints each candidate row (``match_id,home_goals,away_goals,winner_team_id``) to stdout for the
maintainer to **dual-source** (e.g. against FIFA/Wikipedia) before committing -- this tool is one
of the two sources, not a replacement for dual-sourcing. Pass ``--write`` to append the rows to
``results.csv`` directly (still review/dual-source first).

Knockout matches level after 90' may have gone to penalties; ``winner_team_id`` is always left
blank -- these are flagged on stderr for the maintainer to fill in by hand.

Usage (run from the repo root, network required)::

    python -m tippspiel.data.espn_results_fetch wc2026 fifa.world
    python -m tippspiel.data.espn_results_fetch wc2026 fifa.world --write
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone

from tippspiel.data.espn_common import (
    REPO,
    fetch_scoreboard,
    find_event,
    load_concrete_fixtures,
    load_played_match_ids,
    load_teams,
)

_FIELDNAMES = ["match_id", "home_goals", "away_goals", "winner_team_id"]


def _score(competitor: dict) -> str | None:
    raw = competitor.get("score")
    if isinstance(raw, dict):
        raw = raw.get("displayValue") or raw.get("value")
    return None if raw is None else str(raw)


def fetch_results(tournament: str, slug: str, write: bool = False) -> list[dict]:
    """Return candidate ``results.csv`` rows for ``tournament``'s finished, unrecorded matches."""
    tdir = REPO / "tippspiel" / "data" / "tournaments" / tournament
    teams = load_teams(tdir)
    played = load_played_match_ids(tdir)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    candidates = [
        f for f in load_concrete_fixtures(tdir)
        if f["match_id"] not in played and f["kickoff_utc"] < now
    ]

    dates = sorted({f["date"] for f in candidates})
    scoreboard = fetch_scoreboard(slug, dates)

    rows_out, not_final, missing, shootout_watch = [], [], [], []
    final_detail: dict[str, str] = {}
    for f in candidates:
        try:
            found = find_event(scoreboard.get(f["date"], []), teams, f["home_id"], f["away_id"])
            if found is None:
                missing.append(f["match_id"])
                continue
            event, ids = found
            status_type = (event.get("status") or {}).get("type") or {}
            if status_type.get("state") != "post":
                not_final.append(f["match_id"])
                continue
            comp = (event.get("competitions") or [{}])[0]
            scores: dict[str, str] = {}
            for c in comp.get("competitors", []):
                rid = ids.get(c.get("homeAway"))
                score = _score(c)
                if rid and score is not None:
                    scores[rid] = score
            if f["home_id"] not in scores or f["away_id"] not in scores:
                missing.append(f["match_id"])
                continue
            home_goals, away_goals = scores[f["home_id"]], scores[f["away_id"]]
            if f["stage"] != "GROUP" and home_goals == away_goals:
                shootout_watch.append(f["match_id"])
            rows_out.append({
                "match_id": f["match_id"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "winner_team_id": "",
            })
            final_detail[f["match_id"]] = status_type.get("shortDetail") or "FT"
        except Exception:  # noqa: BLE001 — one bad fixture shouldn't abort the run
            missing.append(f["match_id"])
            continue

    writer = csv.DictWriter(sys.stdout, fieldnames=_FIELDNAMES, lineterminator="\n")
    for row in rows_out:
        writer.writerow(row)
    if rows_out:
        confirmed = ", ".join(
            f"{r['match_id']} {r['home_goals']}-{r['away_goals']} ({final_detail[r['match_id']]})"
            for r in rows_out
        )
        print(f"# confirmed final (status.state==post): {confirmed}", file=sys.stderr)
    if shootout_watch:
        print(f"# level after 90' in a knockout match -- fill winner_team_id by hand: "
              f"{shootout_watch}", file=sys.stderr)
    if not_final:
        print(f"# kicked off but not yet final (state != post): {not_final}", file=sys.stderr)
    if missing:
        print(f"# no finished-match scoreboard entry found: {missing}", file=sys.stderr)

    if write and rows_out:
        out = tdir / "results.csv"
        with out.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=_FIELDNAMES).writerows(rows_out)
        print(f"appended {len(rows_out)} rows to {out}", file=sys.stderr)
    return rows_out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python -m tippspiel.data.espn_results_fetch <tournament> "
                 "<espn_league_slug> [--write]")
    fetch_results(sys.argv[1], sys.argv[2], write="--write" in sys.argv[3:])
