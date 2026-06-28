"""Maintainer tool: build an ``odds`` snapshot from Polymarket's match-winner markets.

Offline, **not on the runtime path** (mirrors ``espn_odds_fetch`` / ``odds_adapter``). Polymarket
is a high-liquidity prediction market; its per-match prices carry sharp, late-breaking signal that
a pre-tournament Elo can't. The Gamma API (``gamma-api.polymarket.com``) is public, unauthenticated
JSON, so there is no HTML scraping / language-model extraction step.

A soccer match is one Gamma *event* (slug ``fifwc-<home>-<away>-<date>``) holding three binary
``moneyline`` markets — home-win, draw, away-win — each market's *Yes* price being that outcome's
implied probability (0–1). We read the three Yes-prices, **orient them by team identity** (via each
market's ``groupItemTitle``), normalise to sum 1, and write decimal odds ``1/p`` into the repo's
``odds.csv`` schema (``match_id,odds_home,odds_draw,odds_away`` — de-vig happens at load, and is a
near-identity since these prices are already de-vigged probabilities).

Why slug construction (not search): Gamma's events/markets listings embed full market bodies and run
to tens of MB — unreliable to stream. Single-event fetches (``/events/slug/<slug>``) are tiny and
reliable. The per-team slug code is read from ``/teams?league=<league>`` (it is Polymarket's own
abbreviation, e.g. ``nld``/``prt``/``cvi``, not the repo's FIFA id), and the few order/date
ambiguities are covered by trying both team orders and the match date ±1.

Coverage is partial for not-yet-posted matches (Polymarket creates a match event closer to
kickoff) — those are reported as "without odds" and skipped, never fabricated; re-run periodically.
Knockout fixtures are resolved to concrete participants (``fixture_resolve``) once the bracket is
settled, so they are priced too.

Usage (run from the repo root, network required)::

    python -m tippspiel.data.polymarket_odds_fetch wc2026                 # -> odds_polymarket.csv
    python -m tippspiel.data.polymarket_odds_fetch wc2026 --league fifwc --out odds.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from tippspiel.data.espn_common import UA, load_teams, norm
from tippspiel.data.fixture_resolve import load_tippable_fixtures, tournament_dir

GAMMA = "https://gamma-api.polymarket.com"


def _get_json(url: str, *, retries: int = 4):
    """Full-body GET → parsed JSON, or ``None`` for a 404 (missing slug — not an error).

    Reads the whole body before parsing (streaming ``json.load`` trips on chunked responses
    through some egress proxies) and retries transient failures with exponential backoff. A 404 is
    returned as ``None`` immediately — a not-yet-posted match is expected, not a failure.
    """
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as fh:
                return json.loads(fh.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if i == retries - 1:
                raise
            time.sleep(2**i)
        except Exception:  # noqa: BLE001 — maintainer tool, best-effort with backoff
            if i == retries - 1:
                raise
            time.sleep(2**i)
    return None


def team_codes(league: str, name_to_id: dict[str, str]) -> dict[str, str]:
    """Map repo ``team_id`` -> Polymarket abbreviation, from ``/teams?league=<league>``.

    Polymarket's abbreviation is its own (ISO/FIFA mix); we match its ``name``/``alias`` to the
    repo team via the shared name normaliser so the slug code is exact, not guessed.
    """
    data = _get_json(f"{GAMMA}/teams?league={league}&limit=500") or []
    codes: dict[str, str] = {}
    for t in data:
        abbr = (t.get("abbreviation") or "").strip()
        if not abbr:
            continue
        for nm in (t.get("name"), t.get("alias")):
            rid = name_to_id.get(norm(nm)) if nm else None
            if rid:
                codes[rid] = abbr
    return codes


def _candidate_slugs(league: str, ch: str, ca: str, kickoff: str) -> list[str]:
    """Bare match-event slugs to try: both team orders × the match date and ±1 day."""
    base = datetime.fromisoformat(kickoff)
    slugs: list[str] = []
    for delta in (0, -1, 1):
        d = (base + timedelta(days=delta)).strftime("%Y-%m-%d")
        slugs.append(f"{league}-{ch}-{ca}-{d}")
        slugs.append(f"{league}-{ca}-{ch}-{d}")
    return slugs


def _trio_from_event(event: dict, home_id: str, away_id: str,
                     name_to_id: dict[str, str]) -> tuple[float, float, float] | None:
    """A (p_home, p_draw, p_away) probability triple oriented to the repo teams, or ``None``.

    Each market is a binary Yes/No whose ``groupItemTitle`` is a team name (a win market) or starts
    with "Draw"; its first ``outcomePrices`` entry is the Yes (outcome) probability. We require all
    three and that the two win markets resolve to exactly this fixture's teams (rejects a mis-hit).
    """
    yes: dict[str, float] = {}
    for m in event.get("markets") or []:
        title = (m.get("groupItemTitle") or "").strip()
        try:
            prices = json.loads(m.get("outcomePrices") or "[]")
        except (ValueError, TypeError):
            return None
        if not prices:
            return None
        p = float(prices[0])
        if title.lower().startswith("draw"):
            yes["draw"] = p
        else:
            rid = name_to_id.get(norm(title))
            if rid:
                yes[rid] = p
    if home_id not in yes or away_id not in yes or "draw" not in yes:
        return None
    ph, pd, pa = yes[home_id], yes["draw"], yes[away_id]
    total = ph + pd + pa
    # Each outcome must be a real probability and the implied book sane (small Polymarket overround).
    if min(ph, pd, pa) <= 0.0 or not (0.90 <= total <= 1.20):
        return None
    return ph / total, pd / total, pa / total


def _event_for_fixture(league: str, codes: dict[str, str], fixture: dict,
                       name_to_id: dict[str, str]) -> tuple[float, float, float] | None:
    ch, ca = codes.get(fixture["home_id"]), codes.get(fixture["away_id"])
    if not ch or not ca:
        return None
    for slug in _candidate_slugs(league, ch, ca, fixture["kickoff_utc"]):
        event = _get_json(f"{GAMMA}/events/slug/{slug}")
        if not event:
            continue
        if isinstance(event, list):
            event = event[0] if event else None
        if not isinstance(event, dict):
            continue
        trio = _trio_from_event(event, fixture["home_id"], fixture["away_id"], name_to_id)
        if trio is not None:
            return trio
    return None


def fetch_odds(tournament: str, league: str = "fifwc",
               out_path: str | Path | None = None) -> int:
    """Fetch Polymarket odds for ``tournament``'s tippable fixtures; write ``odds_polymarket.csv``.

    Returns rows written. ``league`` is Polymarket's league slug (``fifwc`` for the men's World Cup).
    """
    tdir = tournament_dir(tournament)
    name_to_id = load_teams(tdir)
    codes = team_codes(league, name_to_id)
    fixtures = load_tippable_fixtures(tdir)

    rows_out, missing = [], []
    for f in fixtures:
        try:
            trio = _event_for_fixture(league, codes, f, name_to_id)
        except Exception:  # noqa: BLE001 — one bad fixture shouldn't abort the run
            trio = None
        if trio is None:
            missing.append(f["match_id"])
            continue
        ph, pd, pa = trio
        rows_out.append({
            "match_id": f["match_id"],
            "odds_home": f"{1.0 / ph:.2f}",
            "odds_draw": f"{1.0 / pd:.2f}",
            "odds_away": f"{1.0 / pa:.2f}",
        })

    out = Path(out_path) if out_path else (tdir / "odds_polymarket.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["match_id", "odds_home", "odds_draw", "odds_away"])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"{tournament}: wrote {len(rows_out)} rows to {out} "
          f"({len(missing)} fixtures without odds: {missing[:8]}{'…' if len(missing) > 8 else ''})")
    return len(rows_out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch Polymarket match-winner odds -> odds.csv schema")
    ap.add_argument("tournament", help="repo tournament name, e.g. wc2026")
    ap.add_argument("--league", default="fifwc", help="Polymarket league slug (default: fifwc)")
    ap.add_argument("--out", default=None, help="output path (default: <data_dir>/odds_polymarket.csv)")
    args = ap.parse_args(argv)
    fetch_odds(args.tournament, args.league, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
