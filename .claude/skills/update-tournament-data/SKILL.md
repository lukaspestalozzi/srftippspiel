---
name: update-tournament-data
description: >-
  Refresh a live tippspiel tournament's data after a matchday: the played-match
  results (results.csv), the bookmaker odds (odds.csv, from ESPN's public feed), and
  the Elo ratings (teams.csv base elo + fitted att/def). Use when the user wants to
  update/refresh odds, add the latest results, update Elo after games are played,
  fill in newly-posted WC2026 matches, or otherwise bring a live tournament's inputs
  up to date. Covers sourcing results from the web, the ESPN JSON endpoints +
  espn_odds_fetch tool, appending played scores to the match corpus and re-running
  fit-ratings (which derives elo + att/def from the corpus — no more eloratings.net
  fetching), validation, and the known gotchas.
---

# Update live-tournament data (results, odds, Elo)

When a matchday has been played in a **live** tournament (e.g. `wc2026`, `completed: false`),
refresh the inputs in this order:

1. **The match corpus** `tippspiel/data/historical/international_results.csv` — fill in the
   full-time score of each played match (the rows are usually already present with `NA,NA`;
   set `home_score,away_score`). This is now the single source of truth for both Elo and scorelines.
2. **`results.csv`** — add a **thin** row per played match: `match_id,date,winner_team_id` (the
   `date` is the corpus match date; `winner_team_id` only for a knockout penalty shootout). The
   scoreline is read from the corpus, never restated here.
3. **`odds.csv`** — re-fetch market odds (`espn_odds_fetch.py` filters fixtures already in
   `results.csv`).
4. **`teams.csv`** Elo — advance `offdef.snapshot_date` to the day after the latest played corpus
   date, then run **`tippspiel fit-ratings --config config.yaml`**. With `elo.source: corpus`
   (wc2026) this recomputes `elo` + `att_elo` + `def_elo` from the grown corpus in one step — **no
   eloratings.net fetch**.

Sourcing convention: **fetch real-world results from the web** and write them into the corpus;
`fit-ratings` derives every Elo column. Never invent numbers — every score must trace to a fetched
source. (The eloratings.net TSV flow below is retained only as a deprecated fallback / the
`eloratings_diff` calibration check.)

The report already renders a played match with its **pre-match prediction (tip, forecast, data
table) plus the actual result added** — so once `results.csv` is updated, no report change is
needed; played matches keep showing "as before" with the real score layered on
(`tippspiel/pipeline.py` `_fixture_block`, `report.html.j2`).

The detailed odds machinery is below; the **results** and **Elo** procedures are in
"Results" and "Elo" near the end.

## Workflow & environment (read first)

This skill is run repeatedly against the same branch, in a fresh container each time. Get these
right before touching data, or you'll waste a pass reconciling.

- **Branch & base.** **The task prompt's branch instruction always wins** — if the prompt names a
  branch, use that one and skip the default below. Default (no branch given in the prompt): work on
  `claude/update-data`. If `origin/claude/update-data` already exists, **start from it**
  (`git reset --hard origin/claude/update-data` onto a local branch of the same name) and build on
  top — a prior run may have a *partial* commit there (e.g. results added but a feed was
  unreachable, so odds/base-elo are still stale). Do **not** branch off some other dev branch and
  replace that work; layer onto it so the push fast-forwards. Fall back to `origin/main` only if the
  branch doesn't exist yet.
- **Environment setup.** `pip install -e ".[dev]"` (PyYAML, needed by `config.py`, is included).
  Run tests with **`python -m pytest -q`** — a bare `pytest` on PATH may be an isolated uv-tool
  install without the project's deps and will fail collection with `ModuleNotFoundError: No module
  named 'yaml'`.
- **Network egress may block the feeds.** Some environments allowlist outbound hosts; a blocked
  fetch returns `403 Host not in allowlist` (curl and WebFetch alike). The hosts this skill needs
  are `site.api.espn.com`, `sports.core.api.espn.com` (odds) and `www.eloratings.net` (Elo). If one
  is blocked: **skip just that sub-step, leave the committed snapshot untouched** (don't write an
  empty file, don't hand-estimate), note it in the commit/PR body, and let the next run pick it up.
  Never fabricate a value to work around a blocked feed.
- **Commit gate.** Commit only if something actually changed (a fixture was added, odds changed, or
  Elo changed). If no fixture has been played and odds+Elo are unchanged, make no commit. After
  pushing, open a **draft** PR if none exists; if a PR already exists for the branch, **update its
  body** to match the final branch state rather than leaving a stale prior-run description.
- **Checking CI after pushing.** Don't enumerate full workflow-run/job payloads just to see whether
  the push is green — `actions_get get_workflow_run` repeats the full repo metadata twice, and
  `list_workflow_runs` unfiltered can return ~hundreds of KB. (A "check-suite status for the SHA"
  via `pull_request_read get_status` is **not useful here** — it returned `total_count: 0` while
  Actions CI was actively running.) Instead:
  1. `actions_list list_workflow_runs` filtered to `branch: <pushed-branch>` with `per_page: 1` to
     get the latest run's id for this push.
  2. `actions_list list_workflow_jobs` for that run id with `workflow_jobs_filter: {filter:
     "latest"}` — a compact per-job `conclusion` list (build, tests, the Pages-`publish` job, etc.)
     without per-step logs.
  Only fetch individual job logs (`actions_get get_workflow_job` / log endpoint) if one of those
  jobs reports `failure` and you need to find which step broke.

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
python -m pytest -q                          # full suite, must stay green (NOT bare `pytest` — see Workflow)
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

## Results (corpus score + thin results.csv)

`results.csv` is now **thin**: `match_id,date,winner_team_id`. The scoreline lives in the corpus
(`international_results.csv`) and is resolved at load time by the match date + the fixture's teams
(`data/corpus_results.py`); the simulator still conditions on the resolved score and the report
still shows it with its pre-match tip. `winner_team_id` is **blank** except for a knockout decided
by penalties (the shootout winner's `team_id`; the 120-minute scoreline stays in the corpus).

Procedure:

1. **Find which fixtures have kicked off.** Read `fixtures.csv` and compare each `kickoff_utc` to
   the current time. Only add matches whose kickoff has genuinely passed.
2. **Fetch each full-time score from the web** and **dual-source** it (e.g. ESPN + FIFA/CNN), then
   **write the score into the corpus** `international_results.csv` (the WC2026 rows are usually
   already present with `NA,NA` — set `home_score,away_score`; match by date + teams). A
   transcription error here propagates into both standings and the Elo fit, so dual-source it.
3. **Add the thin `results.csv` row**: `match_id,<corpus_date>,<winner_or_blank>`. Orientation is
   handled automatically (the resolver re-orients the corpus score to the fixture's home/away), so
   you do not restate goals. Corpus dates are **local** match dates and can differ from
   `kickoff_utc` by a day — use the corpus row's date.

Example (WC2026 matchday 1): corpus gets `2026-06-11,Mexico,South Africa,2,0,FIFA World Cup,...`,
and `results.csv` gets `G_A_1,2026-06-11,`. Run `tippspiel validate-data` — it asserts every
results row resolves to exactly one corpus match.

## Elo (corpus-derived via fit-ratings)

Elo is no longer fetched. After writing the matchday's scores into the corpus (above), advance
`offdef.snapshot_date` in `config.yaml` to the day after the latest played corpus date and run:

```bash
tippspiel fit-ratings --config config.yaml
```

With `elo.source: corpus` this recomputes **`elo` + `att_elo` + `def_elo`** for all 48 teams from
the grown corpus (deterministic, single command). The scalar `elo` uses the calibrated
World-Football-Elo defaults (`k_scale 1.4`, `home_advantage 60`) in the `elo:` block.

**Optional calibration check (offline, network):** `python -m tippspiel.data.eloratings_diff
wc2026` compares the corpus-fitted `elo` against eloratings.net (expect Spearman ~0.98 + a benign
uniform offset; the predictor uses Elo *differences*). It no longer drives the update — it only
confirms the fit hasn't drifted. The legacy eloratings TSV fetch below is a deprecated fallback.

A one-line summary is **always** printed to stderr, even when nothing moved, e.g. `"0 movers; 2
played teams already up-to-date; 1 not yet processed (unmapped or no rating yet): CV"` — silent
stdout + that summary is the normal "feed hasn't processed this match yet" lag case below, not a
failure. If the summary is missing entirely, something went wrong (network error, bad tournament
name).

### Fetching by hand / debugging

The site's HTML is JavaScript-rendered (fetching a page URL returns an empty shell), but the JS
loads plain tab-separated files that are directly fetchable with a browser `User-Agent` — the same
trick as the ESPN feed, and what `eloratings_diff` does under the hood:

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
guessing (this is what `eloratings_diff` does). Use `latest.tsv` to confirm a match was processed
and lift the **new ratings** (cols 11-12) straight into `teams.csv` if `eloratings_diff` can't
resolve a team's code (e.g. a new tournament's roster needs an alias added).

**Lag caveat — do NOT compute Elo yourself.** The feed updates only after eloratings processes a
match — typically same-day, but a game that finished hours ago may not be in `latest.tsv`/`World.tsv`
yet (matchday 1: MEX–RSA was in the feed the next morning; KOR–CZE, finished ~04:00Z, was not). If a
played match has **not** been processed yet, leave that team's `elo` at its **last fetched value**
(the committed snapshot) and move on — do not apply the eloratings formula to estimate it. Eloratings
is the single source of truth for the rating; an estimated value would diverge from it (their model
folds in factors beyond the headline K·G·(W−We), and a later-corrected score or competition code
would compound the drift). Re-run the fetch on the next refresh and pick up the real value then.

1. **Edit the movers' `elo` in `teams.csv`** in place; update `tournament.elo_source` in the config
   to the new snapshot date + what changed.
2. **Re-run `fit-offdef` so att/def reflect the games.** The fitter is deterministic for a fixed
   corpus+snapshot, and **base `elo` is NOT an input** — att/def only move when the *corpus* grows
   (so on a run that only updates base elo, fit-offdef is a confirming no-op). The WC2026 fixtures
   already sit in `tippspiel/data/historical/international_results.csv` as future rows with `NA,NA`
   scores (the adapter drops `NA`), so **fill in the played matches' scores there** (find the rows by
   date/teams — note the corpus uses full names like "Czech Republic").

   **Corpus dates are local match dates; `fixtures.csv`'s `kickoff_utc` is UTC.** A late UTC kickoff
   can fall on the *previous* local day at a western-hemisphere venue — e.g. WC2026 `G_A_2`
   (KOR–CZE) has `kickoff_utc=2026-06-12T02:00:00Z` but its `international_results.csv` row is dated
   `2026-06-11` (Zapopan, Mexico, UTC-6). When deciding `offdef.snapshot_date`, compare against the
   **corpus dates** of the matches you just filled in, not their `kickoff_utc`.

   Then check `offdef.snapshot_date`: the cutoff is **strictly earlier** (`date >= snapshot_date` is
   dropped), so it must be the day **after** the latest played match's *corpus* date. **Only change
   it if a newer matchday was added**; if the current value already covers the just-filled games (and
   excludes the not-yet-played ones), leave it — "bumping" it past an unplayed matchday is wrong.
   (The default snapshot = day-before-first-kickoff is for leak-free *backtests*; a live tournament
   is not a benchmark, so fold the played games in.)

   **Check the cutoff mechanically before committing to it:** `tippspiel fit-offdef --dry-run`
   prints every corpus match within 5 days of `snapshot_date`, marked `IN`/`OUT` by the same
   `< snapshot_date` rule the fitter uses — confirm the just-filled games are `IN` and the
   not-yet-played ones are `OUT` (writes nothing). Then run `tippspiel fit-offdef` for real to
   rewrite `att_elo`/`def_elo` for all teams.

```bash
tippspiel fit-offdef          # after editing teams.csv elo + corpus scores + offdef.snapshot_date
tippspiel validate-data       # schema/consistency
tippspiel run                 # regenerate output/report.html; eyeball played matches
python -m pytest -q && ruff check tippspiel tests
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
