---
name: update-odds
description: >-
  Refresh or add bookmaker odds (odds.csv) for a tippspiel tournament from ESPN's
  public odds feed, so the report's "Market-odds tip" line populates with real
  market data. Use when the user wants to update/refresh odds, add odds for a new
  or in-progress tournament, fill in newly-posted WC2026 matches, or fix/regenerate
  an odds.csv. Covers the ESPN JSON endpoints, the espn_odds_fetch maintainer tool,
  per-tournament league slugs, validation, and the known gotchas.
---

# Update the odds data

`odds.csv` (`match_id,odds_home,odds_draw,odds_away`, raw decimal, de-vigged at load) feeds the
`MarketOddsPredictor` and the report's per-fixture market-odds tip. This skill regenerates it
from **ESPN's public JSON feed** — real sportsbook moneylines as *structured data*, so there is no
HTML scraping and no language-model extraction step (and therefore no risk of fabricated numbers).

The committed tool that does all of this is `tippspiel/data/espn_odds_fetch.py`. In the normal
case you just run it; the rest of this file explains how it works and how to extend/debug it.

## Quick start

Run from the repo root (network required). Args are `<tournament_dir_name> <espn_league_slug>`:

```bash
python -m tippspiel.data.espn_odds_fetch wc2026 fifa.world      # the live report
python -m tippspiel.data.espn_odds_fetch wc2022 fifa.world      # benchmark
python -m tippspiel.data.espn_odds_fetch euro2024 uefa.euro
python -m tippspiel.data.espn_odds_fetch euro2020 uefa.euro
python -m tippspiel.data.espn_odds_fetch womenseuro2025 uefa.weuro
```

It writes `tippspiel/data/tournaments/<name>/odds.csv` and prints rows-written + which fixtures had
no odds. Then validate and commit (see "After fetching").

### League slugs (the second arg)

| Tournament family            | ESPN slug      |
|------------------------------|----------------|
| FIFA World Cup (men)         | `fifa.world`   |
| UEFA Euro (men)              | `uefa.euro`    |
| UEFA Women's Euro            | `uefa.weuro`   |

For a new tournament, confirm the slug by probing the scoreboard on a known match date (see
"Probing / debugging"). The `dates=YYYYMMDD` scoreboard returns that date's events regardless of
season year, so the same slug works across editions.

## How it works (the successful approach)

Two undocumented but public ESPN endpoints, fetched with a browser `User-Agent` (a plain default
UA can get a 503):

1. **Scoreboard** — fixtures + team names + ESPN home/away + event ids, one call per match date:
   ```
   https://site.api.espn.com/apis/site/v2/sports/soccer/<slug>/scoreboard?dates=YYYYMMDD
   ```
   The scoreboard's own `odds` field is `null` — it is used **only** to map a (date, team-pair) to
   an ESPN `event_id` and to learn which side ESPN calls home.

2. **Per-event odds** — the actual prices, one call per event:
   ```
   https://sports.core.api.espn.com/v2/sports/soccer/leagues/<slug>/events/<id>/competitions/<id>/odds
   ```
   This returns ~12 provider items (DraftKings, Bet365, Caesars, …).

The tool then, for each repo fixture (real, dated matchup — structural KO refs like `W:A` are
skipped): finds the event whose two teams map to the fixture's `home_ref`/`away_ref`, picks the
first usable provider trio, converts American → decimal, orients by **team identity**, validates,
and writes the row.

### Reading a usable 1X2 trio (the key insight)

- **Use the moneyline trio, not the 3-way decimal market.** Home/away come from
  `homeTeamOdds.current.moneyLine.american` and `awayTeamOdds.current.moneyLine.american`; the draw
  comes from `drawOdds.moneyLine` (a bare American number). Convert American→decimal:
  `d = 1 + a/100` if `a>0` else `1 + 100/(-a)`.
- **Skip provider id `2000`** ("Bet 365" legacy). It is the *only* provider exposing a ready-made
  3-way *decimal* market, but its values are frequently garbage (it quoted Qatar at 350/1 and
  Ecuador at 1/250 for a near-even match). The tool hard-skips it.
- **Take the first provider whose trio is complete and sane** — all three present, each decimal
  `> 1.0`, and de-vig booksum `sum(1/dec)` in `[1.0, 1.35]`. Different tournaments surface different
  providers (WC2026 came from DraftKings, WC2022 from Bet365's moneyline item), so never hardcode a
  provider — iterate and validate.
- **Orient by team, not by position.** ESPN's "home" may differ from the repo's `home_ref`. Map
  each price to its team id and assign `odds_home` to the repo's home team's price.

### Team-name mapping

ESPN display names are matched to repo `team_id` via `teams.csv` names plus an `_ALIASES` dict in
the tool (e.g. `IR Iran→Iran`, `Korea Republic→South Korea`, `Côte d'Ivoire→Ivory Coast`,
`Czech Republic→Czechia`). If a fixture is reported as "without odds" but the match clearly exists,
the usual cause is an unmapped name — add the ESPN spelling to `_ALIASES`.

## After fetching

```bash
tippspiel validate-data --config <config>   # schema/consistency incl. odds.csv
pytest -q                                    # full suite, must stay green
```

Wire the tournament's config once (idempotent): add under its `tournament:` block

```yaml
  odds_file: odds.csv     # ESPN public odds snapshot (moneylines, de-vigged at load)
```

`config.yaml` (WC2026) and `configs/{wc2022,euro2024,euro2020,womenseuro2025}.yaml` already have it.
The market-odds tip renders via a dedicated reporting pass **regardless of the active
predictor** — you do NOT need to switch `predictor.name` to `market_odds` for the tip to show.

Commit the `odds.csv` file(s) (and any `_ALIASES` edit). These are committed static snapshots, like
Elo in `teams.csv`.

### Spot-check before committing

- Favorite sanity: e.g. WC2026 `G_A_1` (Mexico v South Africa) should be `1.49,4.30,7.00` — heavy
  home favorite. A flipped or flat trio means an orientation or mapping bug.
- Optional: confirm the backtest still behaves with
  `build_verification(...)` under `predictor.name=market_odds` vs `elo_poisson` (compare
  `calibration.all.mean_rps`).

## Gotchas & known limits

- **Coverage is partial for not-yet-played matches.** Bookmakers post later group matches closer to
  kickoff, so an in-progress/future tournament (e.g. WC2026 ~40/72) will legitimately miss rows.
  Missing fixtures simply render no odds tip — that's expected, not a bug. **Re-run this skill
  periodically** as more matches are priced.
- **Old tournaments may have no odds.** `wc2018` returned 0 rows (ESPN has none that far back) — no
  `odds.csv` is committed for it, and its config has no `odds_file`. Don't commit an empty file.
- **The market predictor is not a guaranteed accuracy win**, because the scoreline expander
  (`expand_1x2_to_scoreline`) matches only the home−away balance and re-implies the draw from
  `total_goals`, **discarding the real de-vigged draw price**. So the default predictor stays
  `elo_poisson`; the market tips are a reference line. Fixing this (a full-triple expansion) is the
  real lever for a calibration lift — out of scope for an odds refresh.
- **Network/UA**: the tool sets a browser UA and retries with exponential backoff. If ESPN starts
  blocking, the `WebFetch`/scrape fallbacks are NOT reliable (oddsportal 503s; JS-rendered books
  yield empty HTML and risk hallucinated extraction) — prefer fixing the JSON path.

## Probing / debugging a single match by hand

```bash
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
# 1) find the event id for a date
curl -sS -A "$UA" "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=20221120" \
  | python -c "import sys,json; [print(e['id'], e['name']) for e in json.load(sys.stdin)['events']]"
# 2) list providers + their moneyline trios for that event id
curl -sS -A "$UA" "https://sports.core.api.espn.com/v2/sports/soccer/leagues/fifa.world/events/633790/competitions/633790/odds" \
  | python -c "import sys,json
for it in json.load(sys.stdin)['items']:
    ml=lambda k:((it.get(k) or {}).get('current') or {}).get('moneyLine',{}).get('american')
    print(it['provider'].get('id'), it['provider'].get('name'), 'H',ml('homeTeamOdds'),'D',(it.get('drawOdds') or {}).get('moneyLine'),'A',ml('awayTeamOdds'))"
```
