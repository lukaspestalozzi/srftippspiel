"""Maintainer tool: build a committed ``odds.csv`` snapshot from ESPN's public odds feed.

Offline, **not on the runtime path** (mirrors ``eloratings_adapter`` / ``odds_adapter``). ESPN's
undocumented JSON API exposes per-match 1X2 prices from real sportsbooks as structured data, so
there is no HTML scraping / language-model extraction step (and therefore no risk of hallucinated
numbers). We read the *moneyline* trio (home / draw / away), which every usable provider quotes,
convert American → decimal, and write the repo's ``odds.csv`` schema
(``match_id,odds_home,odds_draw,odds_away``, raw decimal; de-vig happens at load).

Fixtures already decided in ``results.csv`` are skipped -- ``odds.csv`` only carries upcoming
matches.

Provenance and guards:
- Source: ``site.api.espn.com`` (scoreboard, for fixtures + team names) and
  ``sports.core.api.espn.com`` (per-event odds). Public, no key.
- The legacy provider ``2000`` ("Bet 365") quotes a 3-way *decimal* market that is frequently
  garbage (e.g. a 350/1 home price for a near-even match); it is **skipped**. We use the moneyline
  providers (DraftKings, Bet365, Caesars, …) and take the first whose home/draw/away trio is
  complete and whose de-vig booksum is sane (``1.0 <= sum(1/dec) <= 1.35``).
- Prices are oriented by *team identity* (ESPN home team → repo ``home_ref``), not by assuming the
  two sources agree on which side is "home".
- A malformed/unexpected scoreboard or odds entry for one fixture is skipped (counted as
  "without odds"), never aborts the whole run.

Usage (run from the repo root, network required)::

    python -m tippspiel.data.espn_odds_fetch wc2026 fifa.world
    python -m tippspiel.data.espn_odds_fetch wc2022 fifa.world
    python -m tippspiel.data.espn_odds_fetch euro2024 uefa.euro
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tippspiel.data.espn_common import (
    REPO,
    date_window,
    fetch_scoreboard,
    find_event,
    get_json,
    load_teams,
)
from tippspiel.data.fixture_resolve import load_tippable_fixtures


def _american_to_decimal(american: float) -> float:
    a = float(american)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def _trio_from_item(item: dict) -> tuple[float, float, float] | None:
    """A sane (home, draw, away) decimal trio from a moneyline provider item, or ``None``."""
    if str(item.get("provider", {}).get("id")) == "2000":
        return None  # legacy 3-way decimal feed — unreliable, skip

    def ml(side: str):
        cur = (item.get(side) or {}).get("current") or {}
        return (cur.get("moneyLine") or {}).get("american")

    home_am, away_am = ml("homeTeamOdds"), ml("awayTeamOdds")
    draw_am = (item.get("drawOdds") or {}).get("moneyLine")
    if home_am is None or away_am is None or draw_am is None:
        return None
    try:
        h = _american_to_decimal(str(home_am).replace("+", ""))
        a = _american_to_decimal(str(away_am).replace("+", ""))
        d = _american_to_decimal(draw_am)
    except (ValueError, ZeroDivisionError):
        return None
    if min(h, d, a) <= 1.0:
        return None
    booksum = 1 / h + 1 / d + 1 / a
    if not (1.0 <= booksum <= 1.35):
        return None
    return h, d, a


def _odds_for_event(slug: str, event_id: str) -> tuple[float, float, float] | None:
    url = (f"https://sports.core.api.espn.com/v2/sports/soccer/leagues/{slug}"
           f"/events/{event_id}/competitions/{event_id}/odds")
    try:
        data = get_json(url)
    except Exception:  # noqa: BLE001
        return None
    for item in data.get("items", []):
        trio = _trio_from_item(item)
        if trio is not None:
            return trio
    return None


def fetch_odds(tournament: str, slug: str, out_path: str | Path | None = None) -> int:
    """Fetch ESPN odds for ``tournament``'s fixtures and write ``odds.csv``. Returns rows written."""
    tdir = REPO / "tippspiel" / "data" / "tournaments" / tournament
    teams = load_teams(tdir)
    # Group matches plus any knockout fixtures the played results have already settled.
    fixtures = load_tippable_fixtures(tdir)

    # ESPN files a match under its *local* date, which can precede a late-UTC kickoff at a
    # western venue — search the same ±1-day window the results fetcher uses, or evening games
    # in the Americas (e.g. WC2026 M85, kickoff 03:00Z) never get priced.
    dates = sorted({d for f in fixtures for d in date_window(f["date"])})
    scoreboard = fetch_scoreboard(slug, dates)

    rows_out, missing = [], []
    for f in fixtures:
        try:
            events = [e for d in date_window(f["date"]) for e in scoreboard.get(d, [])]
            found = find_event(events, teams, f["home_id"], f["away_id"])
            if found is None:
                missing.append(f["match_id"])
                continue
            event, ids = found
            event_id = event.get("id")
            if event_id is None:
                missing.append(f["match_id"])
                continue
            trio = _odds_for_event(slug, event_id)
            if trio is None:
                missing.append(f["match_id"])
                continue
            h, draw, a = trio
            # Orient by team identity: ESPN home team may differ from repo home_ref.
            if ids.get("home") == f["home_id"]:
                odds_home, odds_away = h, a
            else:
                odds_home, odds_away = a, h
            rows_out.append({
                "match_id": f["match_id"],
                "odds_home": f"{odds_home:.2f}",
                "odds_draw": f"{draw:.2f}",
                "odds_away": f"{odds_away:.2f}",
            })
        except Exception:  # noqa: BLE001 — one bad fixture shouldn't abort the run
            missing.append(f["match_id"])
            continue

    out = Path(out_path) if out_path else (tdir / "odds.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["match_id", "odds_home", "odds_draw", "odds_away"])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"{tournament}: wrote {len(rows_out)} rows to {out} "
          f"({len(missing)} fixtures without odds: {missing[:8]}{'…' if len(missing) > 8 else ''})")
    return len(rows_out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch ESPN 1X2 odds -> odds.csv schema")
    ap.add_argument("tournament", help="repo tournament name, e.g. wc2026")
    ap.add_argument("slug", help="ESPN soccer league slug, e.g. fifa.world")
    ap.add_argument("--out", default=None, help="output path (default: <data_dir>/odds.csv)")
    args = ap.parse_args()
    fetch_odds(args.tournament, args.slug, args.out)
