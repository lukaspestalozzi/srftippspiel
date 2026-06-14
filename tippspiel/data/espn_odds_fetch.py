"""Maintainer tool: build a committed ``odds.csv`` snapshot from ESPN's public odds feed.

Offline, **not on the runtime path** (mirrors ``eloratings_adapter`` / ``odds_adapter``). ESPN's
undocumented JSON API exposes per-match 1X2 prices from real sportsbooks as structured data, so
there is no HTML scraping / language-model extraction step (and therefore no risk of hallucinated
numbers). We read the *moneyline* trio (home / draw / away), which every usable provider quotes,
convert American → decimal, and write the repo's ``odds.csv`` schema
(``match_id,odds_home,odds_draw,odds_away``, raw decimal; de-vig happens at load).

Provenance and guards:
- Source: ``site.api.espn.com`` (scoreboard, for fixtures + team names) and
  ``sports.core.api.espn.com`` (per-event odds). Public, no key.
- The legacy provider ``2000`` ("Bet 365") quotes a 3-way *decimal* market that is frequently
  garbage (e.g. a 350/1 home price for a near-even match); it is **skipped**. We use the moneyline
  providers (DraftKings, Bet365, Caesars, …) and take the first whose home/draw/away trio is
  complete and whose de-vig booksum is sane (``1.0 <= sum(1/dec) <= 1.35``).
- Prices are oriented by *team identity* (ESPN home team → repo ``home_ref``), not by assuming the
  two sources agree on which side is "home".

Usage (run from the repo root, network required)::

    python -m tippspiel.data.espn_odds_fetch wc2026 fifa.world
    python -m tippspiel.data.espn_odds_fetch wc2022 fifa.world
    python -m tippspiel.data.espn_odds_fetch euro2024 uefa.euro
"""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import tippspiel

REPO = Path(tippspiel.__file__).parent.parent
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# ESPN display names that differ from the repo's ``teams.csv`` names. Extend as needed.
_ALIASES = {
    "ir iran": "iran",
    "korea republic": "south korea",
    "usa": "united states",
    "côte d'ivoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "cabo verde": "cape verde",
    "congo dr": "dr congo",
    "bosnia & herzegovina": "bosnia and herzegovina",
    "turkey": "türkiye",
    "czech republic": "czechia",
}


def _norm(name: str) -> str:
    return _ALIASES.get(name.strip().lower(), name.strip().lower())


def _get_json(url: str, *, retries: int = 4) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=25) as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001 — maintainer tool, best-effort with backoff
            if i == retries - 1:
                raise
            time.sleep(2 ** i)
    return {}


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
        data = _get_json(url)
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
    teams = {}
    with (tdir / "teams.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            teams[_norm(row["name"])] = row["team_id"]

    # Fixtures we want odds for: real, dated group/known matches keyed by team pair + date.
    fixtures = []
    with (tdir / "fixtures.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ko = row.get("kickoff_utc", "")
            home, away = row.get("home_ref", ""), row.get("away_ref", "")
            # Only concrete matchups (group stage / completed KO), not structural refs (W:A …).
            if ":" in home or ":" in away or "T" not in ko:
                continue
            fixtures.append((row["match_id"], ko[:10].replace("-", ""), home, away))

    rows_out, missing = [], []
    dates = sorted({d for _, d, _, _ in fixtures})
    scoreboard: dict[str, list] = {}
    for d in dates:
        url = (f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}"
               f"/scoreboard?dates={d}")
        try:
            scoreboard[d] = _get_json(url).get("events", [])
        except Exception:  # noqa: BLE001
            scoreboard[d] = []

    for match_id, d, home_id, away_id in fixtures:
        event = None
        for e in scoreboard.get(d, []):
            comps = e.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            ids = {}
            for c in comp.get("competitors", []):
                rid = teams.get(_norm(c["team"]["displayName"]))
                if rid:
                    ids[c["homeAway"]] = rid
            if {ids.get("home"), ids.get("away")} == {home_id, away_id}:
                event = (e["id"], ids)
                break
        if event is None:
            missing.append(match_id)
            continue
        trio = _odds_for_event(slug, event[0])
        if trio is None:
            missing.append(match_id)
            continue
        h, draw, a = trio
        # Orient by team identity: ESPN home team may differ from repo home_ref.
        if event[1].get("home") == home_id:
            odds_home, odds_away = h, a
        else:
            odds_home, odds_away = a, h
        rows_out.append({
            "match_id": match_id,
            "odds_home": f"{odds_home:.2f}",
            "odds_draw": f"{draw:.2f}",
            "odds_away": f"{odds_away:.2f}",
        })

    out = Path(out_path) if out_path else (tdir / "odds.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["match_id", "odds_home", "odds_draw", "odds_away"])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"{tournament}: wrote {len(rows_out)} rows to {out} "
          f"({len(missing)} fixtures without odds: {missing[:8]}{'…' if len(missing) > 8 else ''})")
    return len(rows_out)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python -m tippspiel.data.espn_odds_fetch <tournament> <espn_league_slug>")
    fetch_odds(sys.argv[1], sys.argv[2])
