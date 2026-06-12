---
name: update-odds
description: >-
  Refresh a live tippspiel tournament's data after a matchday: the played-match
  results (results.csv), the bookmaker odds (odds.csv, from ESPN's public feed), and
  the Elo ratings (teams.csv base elo + fitted att/def). Use when the user wants to
  update/refresh odds, add the latest results, update Elo after games are played,
  fill in newly-posted WC2026 matches, or otherwise bring a live tournament's inputs
  up to date. Covers sourcing results from the web, the ESPN JSON endpoints +
  espn_odds_fetch tool, the eloratings.net TSV data feed (World.tsv / latest.tsv)
  with the update formula as fallback, re-running fit-offdef, validation, and the
  known gotchas.
---

# Update live-tournament data (results, odds, Elo)

When a matchday has been played in a **live** tournament (e.g. `wc2026`, `completed: false`),
three input files need refreshing, in this order:

1. **`results.csv`** — add the full-time scores of the games that have kicked off.
2. **`odds.csv`** — re-fetch market odds (newly-priced upcoming matches appear; played ones drop).
3. **`teams.csv`** Elo — bump the base `elo` of the teams that played, then re-run `fit-offdef`.

Sourcing convention (decided 2026-06): **fetch the real-world data from the web** (results via
search, odds via the ESPN feed below, Elo via the eloratings.net TSV feed), and for Elo update the
**base `elo` and re-run `fit-offdef`** so att/def shift too. Never invent numbers — every value
must trace to a fetched source or the published eloratings formula.

The report already renders a played match with its **pre-match prediction (tip, forecast, data
table) plus the actual result added** — so once `results.csv` is updated, no report change is
needed; played matches keep showing "as before" with the real score layered on
(`tippspiel/pipeline.py` `_fixture_block`, `report.html.j2`).

The detailed odds machinery is below; the **results** and **Elo** procedures are in
"Results" and "Elo" near the end.

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

## Results (results.csv)

`results.csv` (`match_id,home_goals,away_goals,winner_team_id`) lists the games already played; the
engine **conditions** on them (the simulator uses the actual scoreline; the report shows them with
their pre-match tip). Schema: group matches leave `winner_team_id` **blank**; it is only needed for
a knockout decided by penalties (the 120-minute scoreline goes in the goal columns, the
shootout winner's `team_id` in the last column). Goals are the **full-time 90-minute** score for
group games.

Procedure:

1. **Find which fixtures have kicked off.** Read `fixtures.csv` and compare each `kickoff_utc` to
   the current time. Only add matches whose kickoff has genuinely passed — adding a not-yet-played
   match injects future info into the sim.
2. **Fetch each full-time score from the web** and **dual-source** it (e.g. ESPN + FIFA/CNN) before
   writing — this is the one place a transcription error silently propagates into standings.
3. **Orient home/away to the fixture.** Match the fixture's `home_ref`/`away_ref`; write
   `home_goals` as the repo home team's goals. Append the row(s) to `results.csv`.

Example (WC2026 matchday 1): `G_A_1` Mexico 2–0 South Africa → `G_A_1,2,0,` and `G_A_2`
South Korea 2–1 Czechia → `G_A_2,2,1,`.

## Elo (teams.csv base elo + att/def)

After a matchday only the teams that **played** have moved on eloratings.net; the rest are
unchanged from the committed snapshot. Don't rewrite all 48 rows — update just the movers, in
place, preserving the `att_elo`/`def_elo` columns.

**Fetch the ratings from eloratings.net's TSV data feed.** The site's HTML is JavaScript-rendered
(fetching a page URL returns an empty shell), but the JS loads plain tab-separated files that are
directly fetchable with a browser `User-Agent` — the same trick as the ESPN feed:

```bash
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
# Current ratings, one row per team (no header): col1 rank, col3 team code, col4 CURRENT RATING
curl -sS -A "$UA" "https://www.eloratings.net/World.tsv" \
  | awk -F'\t' '$3=="MX"||$3=="ZA"||$3=="KR"||$3=="CZ"{print $3, $4}'
# Recent matches with the Elo exchange applied (no header):
#   cols 1-3 date (Y M D) · 4-5 home/away team code · 6-7 goals · 8 competition (WC, F, …)
#   · 9 host code (blank = home team hosts) · 10 POINTS EXCHANGED · 11-12 NEW home/away rating
#   · 13-14 rank change · 15-16 new rank
curl -sS -A "$UA" "https://www.eloratings.net/latest.tsv" | head -20
```

Other useful files on the same host: `<year>.tsv` (that year's rating table, e.g. `2026.tsv`),
`<year>_results.tsv` (all matches of a year), `en.teams.tsv` (team code → name(s), with aliases).
Team codes are eloratings' own two-letter codes (mostly ISO-3166: `MX` Mexico, `ZA` South Africa,
`KR` South Korea, `CZ` Czechia, `EN` England, `PT` Portugal…) — map via `en.teams.tsv`, not by
guessing. Use `latest.tsv` to confirm a match was processed and lift the **new ratings**
(cols 11-12) straight into `teams.csv`.

**Lag caveat:** the feed updates only after eloratings processes a match — typically same-day, but
a game that finished hours ago may not be in `latest.tsv` yet (matchday 1: MEX–RSA was in the feed
the next morning; KOR–CZE, finished ~04:00Z, was not). For an unprocessed match, apply
eloratings.net's own published formula to the committed pre-match ratings (auditable and exactly
reproducible — the formula's output matched the feed's processed MEX–RSA row to the point), then
re-verify against the feed on the next refresh:

```
new = old + K · G · (W − We)
We  = 1 / (10^(−dr/400) + 1)           # expected result for the team
dr  = own_rating − opp_rating + H      # H = +100 for the home team, 0 at a neutral venue
W   = 1 win / 0.5 draw / 0 loss
G   = 1 (goal diff ≤ 1) · 1.5 (gd 2) · 1.75 (gd 3) · 1.75+(gd−3)/8 (gd ≥ 4)
K   = 60 World Cup · 50 continental final · 40 WC/continental qualifier · 20 friendly
```

Host advantage `H` applies when a team plays in its own country (a host nation), **not** for a
neutral-venue match between two visitors. Round to the nearest integer; the two sides exchange
equal and opposite points. Once the feed catches up, `latest.tsv` cols 11-12 are the ground truth —
if a computed value disagrees with a processed row, the feed wins.

Worked example (WC2026 matchday 1):
- **MEX 2–0 RSA** (gd 2 → G=1.5; Mexico host → H=+100): MEX 1875→**1881**, RSA 1517→**1511**.
- **KOR 2–1 CZE** (gd 1 → G=1; neutral): KOR 1758→**1786**, CZE 1740→**1712**.

Then:

1. **Edit the movers' `elo` in `teams.csv`** in place; update `tournament.elo_source` in the config
   to the new snapshot date + what changed.
2. **Re-run `fit-offdef` so att/def reflect the games.** The fitter is deterministic for a fixed
   corpus+snapshot, so it only moves if the corpus grows: the WC2026 fixtures already sit in
   `tippspiel/data/historical/international_results.csv` as future rows with `NA,NA` scores (the
   adapter drops `NA`), so **fill in the played matches' scores there** (find the rows by date/teams
   — note the corpus uses full names like "Czech Republic"), and set `offdef.snapshot_date` in the
   config to a date **after** them (the default snapshot = day-before-first-kickoff is for
   leak-free *backtests*; a live tournament is not a benchmark, so fold the played games in). Run
   `tippspiel fit-offdef` to rewrite `att_elo`/`def_elo` for all teams.

```bash
tippspiel fit-offdef          # after editing teams.csv elo + corpus scores + offdef.snapshot_date
tippspiel validate-data       # schema/consistency
tippspiel run                 # regenerate output/report.html; eyeball played matches
pytest -q && ruff check tippspiel tests
```

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
