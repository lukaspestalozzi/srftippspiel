---
name: update-tournament-data
description: >-
  Refresh a live tippspiel tournament's data after a matchday: the played-match
  results (into the match corpus + a thin results.csv), the bookmaker odds
  (odds.csv, from ESPN's public feed), and the Elo ratings (corpus-derived via
  fit-ratings). Use when the user wants to update/refresh odds, add the latest
  results, update Elo after games are played, fill in newly-posted WC2026 matches,
  or otherwise bring a live tournament's inputs up to date. Covers the
  espn_results_fetch and espn_odds_fetch tools, fit-ratings (elo + att/def from the
  corpus — no eloratings.net fetching), validation, the commit gate and the gotchas.
---

# Update live-tournament data (results, odds, Elo)

When a matchday has been played in a **live** tournament (e.g. `wc2026`, `completed: false`),
refresh its inputs. **One committed corpus is the single source of truth** for both Elo and
scorelines: scores live in `tippspiel/data/historical/international_results.csv`, `results.csv` is
a **thin** reference (`match_id,date,winner_team_id`) resolved against it at load time, and
`fit-ratings` derives every Elo column from it. Two tools do the fetching; you mostly just run them.

## The pipeline (one matchday)

Run from the repo root. For wc2026 the ESPN slug is `fifa.world`.

```bash
# 1. RESULTS — dry-run first, then review/dual-source the printed scores, then --write.
python -m tippspiel.data.espn_results_fetch wc2026 fifa.world                      # dry-run (prints plan)
python -m tippspiel.data.espn_results_fetch wc2026 fifa.world --write --config config.yaml

# 2. VALIDATE THE RESULTS NOW — fail-fast gate before spending the odds/Elo fetches. A recorded
#    result that can't be read back (e.g. a knockout join issue) surfaces here with a clear message
#    instead of as an opaque traceback inside the odds fetch.
tippspiel validate-data --config config.yaml

# 3. ODDS — fetch both sources into sidecars, then blend into the consumed odds.csv
python -m tippspiel.data.espn_odds_fetch wc2026 fifa.world --out tippspiel/data/tournaments/wc2026/odds_espn.csv
python -m tippspiel.data.polymarket_odds_fetch wc2026                                  # -> odds_polymarket.csv
python -m tippspiel.data.odds_consensus tippspiel/data/tournaments/wc2026/odds.csv \
    tippspiel/data/tournaments/wc2026/odds_espn.csv tippspiel/data/tournaments/wc2026/odds_polymarket.csv

# 4. ELO (elo + att/def, from the grown corpus)
tippspiel fit-ratings --config config.yaml

# 5. VALIDATE (full) + tests
tippspiel validate-data --config config.yaml
python -m pytest -q          # NOT bare `pytest` — see Environment
ruff check tippspiel tests

# 6. COMMIT only if something actually changed (see Commit gate)
```

Step 1 (`--write`) does the whole "record a result" operation: it fills each match's score into the
corpus (or appends a row), appends the thin `results.csv` row, and **advances
`offdef.snapshot_date`** to the day after the latest played date — no manual two-place transcription
and no manual snapshot bump. Step 4 then recomputes Elo from that grown corpus.

**Idempotence:** every step is a no-op when nothing's new. Re-running `espn_results_fetch` skips
matches already in `results.csv`; `fit-ratings` is deterministic and only moves Elo when the corpus
grew; the odds fetch re-prices only not-yet-kicked-off fixtures and preserves every played /
kicked-off match's committed row verbatim (the **frozen-odds rule**, see the Odds section). So a
run with no freshly-finished match and unmoved markets makes no data change and should produce
**no commit**.

The report renders a played match with its **pre-match prediction (tip, forecast, data table) plus
the actual result** — so once `results.csv` is updated, no report change is needed
(`tippspiel/pipeline.py` `_fixture_block`, `report.html.j2`).

## Workflow & environment (read first)

This skill is run repeatedly against the same branch, in a fresh container each time.

- **Branch & base.** **The task prompt's branch instruction always wins** — read it first and follow
  it literally; it overrides every default below, including whether a PR is opened.
  - *Prompt says commit straight to `main` (no branch, no PR):* `git checkout main && git reset --hard
    origin/main`, do the work, commit, and push to `main`. Do **not** create a feature branch or a PR
    in this mode; the "draft PR" step under the commit gate does not apply.
  - *Prompt names a feature branch:* use exactly that branch (start from `origin/<branch>` if it
    exists, else `origin/main`), and follow the commit gate's draft-PR step.
  - *No branch in the prompt (default):* work on `claude/update-data`; if `origin/claude/update-data`
    exists, **start from it** (`git reset --hard origin/claude/update-data`) and layer on top — a prior
    run may have a *partial* commit. Fall back to `origin/main` only if the branch doesn't exist.
- **Environment setup.** `pip install -e ".[dev]"` (PyYAML, needed by `config.py`, is included). Run
  tests with **`python -m pytest -q`** — a bare `pytest` on PATH may be an isolated uv-tool install
  without the project's deps and fails collection with `ModuleNotFoundError: No module named 'yaml'`.
- **Network egress may block the feeds.** Some environments allowlist outbound hosts; a blocked
  fetch returns `403 Host not in allowlist`. The hosts this skill needs are `site.api.espn.com`,
  `sports.core.api.espn.com` and `gamma-api.polymarket.com` (Polymarket odds). If a feed is blocked
  it degrades cleanly — `espn_results_fetch` reports
  the fixtures as "no finished-match scoreboard entry found" and records nothing; the odds fetch
  writes nothing new. **Skip that sub-step, leave the committed snapshot untouched, note it in the
  commit, and let the next run pick it up.** Never fabricate a value to work around a blocked feed.
- **Commit gate.** Commit only if something actually changed (a result was recorded, odds changed, or
  Elo changed — check `git status`). If nothing has been played and odds+Elo are unchanged, make no
  commit. After pushing to a **feature branch**, open a **draft** PR if none exists; if a PR exists,
  **update its body** to match the final branch state rather than leaving a stale prior-run
  description. In **commit-straight-to-`main`** mode (see Branch & base) there is no PR — skip this
  step entirely.
- **Checking CI after pushing.** Don't enumerate full workflow-run/job payloads. (`pull_request_read
  get_status` is **not useful** — it returned `total_count: 0` while Actions CI was running.) Instead:
  1. **Find the run id once.** `actions_list list_workflow_runs` filtered to `branch: <pushed-branch>`.
     This call is only to learn the id of the run whose `head_sha` matches your pushed commit — its
     per-run objects are stripped down (they carry `status` but **not `conclusion`**, and `per_page`
     may be ignored), so don't try to read the result from here.
  2. **Poll that run by id** with `actions_get get_workflow_run <run_id>` — a single compact object
     that **does** carry both `status` and `conclusion`. Re-fetch this same call until `status` is
     `completed`, then read `conclusion` (`success` / `failure`).
  3. Only if the run **failed**, `actions_list list_workflow_jobs` for that run id with
     `workflow_jobs_filter: {filter: "latest"}` to find which job, then fetch that job's logs.
     A green run needs no jobs call.
  **Waiting for completion.** CI takes a few minutes and this environment can't poll GitHub from bash
  (no token — the MCP tool holds the auth), and foreground `sleep` is blocked. So don't busy-loop:
  arm a one-shot timer with the **Monitor** tool (`command: "sleep 150; echo recheck"`, `timeout_ms`
  a bit above the sleep) and, when it fires, re-run the step-2 `actions_get`. Repeat the timer if
  still `in_progress`. Never spin on `actions_get` back-to-back.
  **Payload overflow:** this repo fans out to ~16 parallel CI jobs, so the step-1 `list` call can
  exceed the tool's token limit and save its JSON to a file instead, printing the path. Don't Read the
  whole file — extract just the run id for your sha, e.g.
  `python3 -c "import json; d=json.load(open('<path>')); print([(r['id'], r['status']) for r in d['workflow_runs'] if r['head_sha'].startswith('<sha7>')])"`.
  (The step-2 `get_workflow_run` object is small and never overflows.)

## Results — `espn_results_fetch`

Records finished, unrecorded matches. **Always run the dry-run first** (no `--write`): it prints one
line per candidate —

```
G_A_3    Czech Republic 2-1 South Africa  (2026-06-18) [filled]
# would record 1 match(es); snapshot_date -> 2026-06-19  (dry-run; pass --write to commit)
```

— so you can **dual-source** each score against a second source (FIFA / Wikipedia / a major outlet)
before committing. This tool is one of the two sources, not a replacement: a transcription error
propagates into both standings and the Elo fit. Once verified, re-run with
`--write --config config.yaml`.

How it works (under the hood):
- The candidate list comes from `fixture_resolve.load_tippable_fixtures` — the **same resolver the
  ESPN and Polymarket odds fetchers use** — so it covers **group matches and knockout matches whose
  participants the played results already settle** (KO rows store participants as structural refs
  `W:A`/`R:B`/`3RD:74:…`, resolved to concrete teams from the results so far). It keeps those whose
  `kickoff_utc` has passed and that aren't yet in `results.csv`, looks up the full-time score from
  the ESPN scoreboard JSON (events with `status.type.state == "post"`; in-play or not-yet-final
  matches are reported and skipped), and maps ESPN names to repo `team_id`s. *(All three fetchers
  share `load_tippable_fixtures`; an earlier divergence — the results fetcher alone using the
  raw-fixtures helper — was exactly why knockout results were never found.)*
- On `--write` it fills the match's row in the corpus (matched by date **±1 day** + the unordered
  team pair, oriented to that row's home team), appends the thin `results.csv` row, and advances
  `offdef.snapshot_date`. Code: `tippspiel/data/espn_results_fetch.py` (`record_results`) +
  `tippspiel/data/corpus_update.py` (`set_corpus_score`).
- **Knockout shootouts:** for a knockout match level after 90' the shootout winner is read
  automatically from the scoreboard's `shootoutScore` and written to `winner_team_id`. It's left
  blank (and flagged on stderr) **only** when the feed doesn't carry the shootout score — then fill
  it in `results.csv` by hand. The 120-minute scoreline stays in the corpus; the thin row carries
  the winner. Always eyeball the printed `pens:<WINNER>` against a second source before `--write`.

`tippspiel validate-data` then asserts every `results.csv` row resolves to exactly one corpus match.

### Gotchas (results)

- **Corpus dates are *local* match dates; `kickoff_utc` is UTC.** A late UTC kickoff can fall on the
  previous local day at a western venue (e.g. WC2026 `G_A_2` KOR–CZE is `…T02:00:00Z` but dated
  `2026-06-11` in the corpus). The tool's ±1-day join handles the offset, and the thin row + the
  computed `snapshot_date` use the **corpus** date, so you never reason about this by hand.
- **`snapshot_date` is computed, not chosen:** it becomes the day after the latest played corpus date
  (the cutoff is strictly-earlier, so the just-played games are folded into the live fit while
  unplayed ones stay out). Don't bump it past an unplayed matchday.
- **Future knockout rows aren't in the corpus yet** (participants TBD), so a KO match takes the
  `[appended]` path with `neutral` inferred from the venue; group matches are always `[filled]`.
- **A recorded KO result is read back by resolving the bracket, not from its fixture row.** A live
  KO fixture keeps structural-ref participants (`W:A`/`3RD:74:…`), so `FileDataProvider.get_results`
  resolves a played KO match's two teams in layers (group results → R32 → R16 → …, reusing the
  report's `resolve_known_participants`) before joining its corpus scoreline. `fixtures.csv` stays
  declarative — you never hand-edit refs to concrete teams. If `validate-data` reports a KO row as
  unresolved, the cause is an earlier round's result missing/incorrect, not the KO row itself.

## Odds — two sources, blended into one `odds.csv`

`odds.csv` (`match_id,odds_home,odds_draw,odds_away`, raw decimal, de-vigged at load) feeds the
`MarketOddsPredictor` and the report's per-fixture market-odds tip. For the live wc2026 it is the
**consensus** of two market sources, each fetched into its own committed sidecar then blended:

- **ESPN** (`espn_odds_fetch.py`) — real sportsbook moneylines from ESPN's public JSON feed →
  `odds_espn.csv`.
- **Polymarket** (`polymarket_odds_fetch.py`) — match-winner prices from the Polymarket Gamma API
  (high-liquidity prediction market, sharp/late signal) → `odds_polymarket.csv`.
- **Consensus** (`odds_consensus.py`) — de-vigs each source to probabilities, averages them per
  match (a fixture present in only one source passes through), writes the blended `odds.csv`.

Both fetchers are structured-JSON only (no HTML scraping / language-model extraction, so no
fabricated numbers) and both now price **knockout** fixtures too: they build their fixture list from
`fixture_resolve.load_tippable_fixtures`, which resolves KO participants from the played results once
the bracket is settled (group matches + certain KO matchups; still-open slots are skipped).

**Frozen-odds rule — played matches are never updated.** A match that is recorded in `results.csv`
or has already kicked off is *not* re-priced (an in-play or post-match quote is not a pre-match
odd), and its existing committed row is preserved **verbatim** by every write — both fetchers and
the consensus (`fixture_resolve.frozen_match_ids` / `write_odds_preserving_frozen`). So the odds
files *accumulate* each match's frozen pre-match snapshot over the tournament while only future
fixtures move; the report/diagnostic keep their market view of played matches. Don't hand-delete
played rows, and expect `git diff` after a refresh to only ever touch not-yet-kicked-off rows.

```bash
D=tippspiel/data/tournaments/wc2026
python -m tippspiel.data.espn_odds_fetch wc2026 fifa.world --out $D/odds_espn.csv
python -m tippspiel.data.polymarket_odds_fetch wc2026                       # --league fifwc (default)
python -m tippspiel.data.odds_consensus $D/odds.csv $D/odds_espn.csv $D/odds_polymarket.csv
```

Commit all three (`odds.csv` + both sidecars): the sidecars are the provenance and feed the
diagnostic's **"Market source agreement"** subsection (ESPN vs Polymarket per-fixture gap). If one
feed is blocked/empty, run the consensus over whichever sidecar(s) you have and note it — never
fabricate a value.

### ESPN source

The ESPN fetch reads the *moneyline* trio (home/draw/away) from real sportsbooks. Code:
`tippspiel/data/espn_odds_fetch.py`.

### League slugs (the second arg)

| Tournament family      | ESPN slug    |
|------------------------|--------------|
| FIFA World Cup (men)   | `fifa.world` |
| UEFA Euro (men)        | `uefa.euro`  |
| UEFA Women's Euro      | `uefa.weuro` |

### How it works

Two public ESPN endpoints, fetched with a browser `User-Agent` (a plain default UA can get a 503):

1. **Scoreboard** — fixtures + team names + ESPN home/away + event ids, one call per match date:
   `https://site.api.espn.com/apis/site/v2/sports/soccer/<slug>/scoreboard?dates=YYYYMMDD`. Its own
   `odds` field is `null` — used only to map a (date, team-pair) to an `event_id` and learn ESPN's
   home side.
2. **Per-event odds** — the prices, one call per event:
   `https://sports.core.api.espn.com/v2/sports/soccer/leagues/<slug>/events/<id>/competitions/<id>/odds`
   (~12 provider items).

For each repo fixture (real, dated matchup — structural KO refs like `W:A` are skipped) the tool
finds the matching event, picks the first usable provider trio, converts American → decimal, orients
by **team identity**, validates, and writes the row.

### Reading a usable 1X2 trio (the key insight)

- **Use the moneyline trio, not the 3-way decimal market.** Home/away from
  `homeTeamOdds.current.moneyLine.american` and `awayTeamOdds.current.moneyLine.american`; draw from
  `drawOdds.moneyLine`. Convert American→decimal: `d = 1 + a/100` if `a>0` else `1 + 100/(-a)`.
- **Skip provider id `2000`** ("Bet 365" legacy) — the only ready-made 3-way *decimal* market, but
  frequently garbage. The tool hard-skips it.
- **Take the first provider whose trio is complete and sane** — all three present, each decimal
  `> 1.0`, de-vig booksum `sum(1/dec)` in `[1.0, 1.35]`. Never hardcode a provider — iterate/validate.
- **Orient by team, not by position.** Map each price to its team id; assign `odds_home` to the
  repo's home team's price.

### Team-name mapping

ESPN names → repo `team_id` via `teams.csv` names plus an `_ALIASES` dict in the tool (e.g.
`IR Iran→Iran`, `Korea Republic→South Korea`, `Czech Republic→Czechia`). If a fixture is "without
odds" but clearly exists, the usual cause is an unmapped name — add the ESPN spelling to `_ALIASES`.

### Polymarket source

Code: `tippspiel/data/polymarket_odds_fetch.py`. The Polymarket Gamma API
(`gamma-api.polymarket.com`, public, no key) models each match as one *event*
(slug `fifwc-<home>-<away>-<date>`) holding three binary `moneyline` markets — home-win, draw,
away-win — each market's *Yes* `outcomePrices[0]` being that outcome's implied probability.

- **Discovery is by exact slug, not search/listing.** Gamma's events/markets *listings* embed full
  market bodies and run to tens of MB (unreliable to stream); single-event fetches
  (`/events/slug/<slug>`) are tiny and reliable. The per-team slug code is read from
  `/teams?league=fifwc` (Polymarket's own abbreviation, e.g. `nld`/`prt`/`cvi` — **not** the repo's
  FIFA id), mapped to the repo team by normalised name. Order/date ambiguity is covered by trying
  both team orders × the match date ±1.
- **Orient by team identity** (each market's `groupItemTitle`), normalise the trio to sum 1, write
  `1/p` as decimal odds. A 404 (match not posted yet) is skipped, not retried.
- **Team-name mapping** reuses the shared `ALIASES`/`norm` in `espn_common.py`. A team that comes up
  "without odds" but clearly exists is usually a missing name alias **or** the match isn't posted yet.

### Spot-check the odds before committing

- Favorite sanity: e.g. WC2026 `G_A_1` (Mexico v South Africa) ≈ `1.49,4.30,7.00` (heavy home
  favorite). A flipped or flat trio means an orientation or mapping bug.
- Cross-source sanity: ESPN and Polymarket should broadly agree — the diagnostic's "Market source
  agreement" subsection (`tippspiel diagnose`) reports the per-fixture gap; a large mean gap flags a
  mapping/orientation bug in one source.

### Probing Polymarket by hand

```bash
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
curl -sS -A "$UA" "https://gamma-api.polymarket.com/teams?league=fifwc&limit=500" \
  | python -c "import sys,json;[print(t['abbreviation'],t['name']) for t in json.load(sys.stdin)]"
curl -sS -A "$UA" "https://gamma-api.polymarket.com/events/slug/fifwc-fra-swe-2026-06-30" \
  | python -c "import sys,json;e=json.load(sys.stdin);print([(m['groupItemTitle'],m['outcomePrices']) for m in e['markets']])"
```

**Caveat — live wc2026 only.** Polymarket has no historical data, so unlike ESPN it cannot be added
to the `verify`/`tune` benchmarks (completed-benchmark configs stay pure-Elo). It's a live-only
enhancement, validated via the diagnostic's source-agreement check rather than a backtest. The egress
host it needs is `gamma-api.polymarket.com` (a blocked feed degrades cleanly — skip it, blend the
sources you have).

## Elo — `fit-ratings`

Elo is **no longer fetched**. After results are recorded (which grew the corpus and bumped
`snapshot_date`), one deterministic command recomputes **`elo` + `att_elo` + `def_elo`** for the
whole field:

```bash
tippspiel fit-ratings --config config.yaml
```

wc2026's config sets `elo.source: corpus`, so the scalar `elo` is a World-Football-Elo fit (calibrated
`k_scale 1.4`, `home_advantage 60`) over the corpus up to `snapshot_date`; the completed benchmarks
keep `elo.source: external` (their committed snapshot is untouched). If the corpus didn't grow this
run, `teams.csv` won't change (idempotent). Background: see `CLAUDE.md` "One corpus, two derivations".

**Optional calibration check (offline, network):** `python -m tippspiel.data.eloratings_diff wc2026`
compares the corpus-fitted `elo` against eloratings.net (expect Spearman ~0.98 + a benign uniform
offset — the predictor uses Elo *differences*). It does **not** drive the update; it only confirms
the fit hasn't drifted. `eloratings_adapter.py` is a deprecated fallback.

## Gotchas & known limits (odds)

- **Coverage is partial for not-yet-played matches.** Bookmakers post later group matches closer to
  kickoff, so an in-progress tournament will legitimately miss rows. Missing fixtures render no odds
  tip — expected, not a bug. **Re-run periodically** as more matches are priced.
- **Old tournaments may have no odds.** `wc2018` returned 0 rows (ESPN has none that far back) — no
  `odds.csv` is committed for it. Don't commit an empty file.
- **Network/UA**: the tools set a browser UA and retry with backoff. If ESPN starts blocking, the
  `WebFetch`/scrape fallbacks are NOT reliable (oddsportal 503s; JS-rendered books yield empty HTML
  and risk hallucinated extraction) — prefer fixing the JSON path.

## Probing / debugging a single match by hand

```bash
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
# find the event id for a date
curl -sS -A "$UA" "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=20221120" \
  | python -c "import sys,json; [print(e['id'], e['name']) for e in json.load(sys.stdin)['events']]"
# list providers + their moneyline trios for that event id
curl -sS -A "$UA" "https://sports.core.api.espn.com/v2/sports/soccer/leagues/fifa.world/events/633790/competitions/633790/odds" \
  | python -c "import sys,json
for it in json.load(sys.stdin)['items']:
    ml=lambda k:((it.get(k) or {}).get('current') or {}).get('moneyLine',{}).get('american')
    print(it['provider'].get('id'), it['provider'].get('name'), 'H',ml('homeTeamOdds'),'D',(it.get('drawOdds') or {}).get('moneyLine'),'A',ml('awayTeamOdds'))"
```
