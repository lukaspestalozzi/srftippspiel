# srftippspiel — FIFA World Cup 2026 Tippspiel Predictor

A command-line tool that recommends what to **tip for every fixture** of the FIFA World
Cup 2026 betting pool, plus bonus-question answers, to **maximise expected pool points**.
It is re-runnable at any point during the tournament: results already played are read from
a file and all downstream predictions and simulations are conditioned on them.

## What it does

1. **Predicts** each match as a full scoreline distribution (Elo-Poisson model, with an
   optional per-team offensive/defensive goal-volume layer fitted from match history).
2. **Optimises** the tip per match to maximise expected pool points (not just the most
   likely scoreline — those differ, and the difference is where the edge is).
3. **Simulates** the whole tournament 50,000 times (Monte Carlo) to get group-advancement
   and title probabilities and the recommended World Champion.
4. **Reports** everything in a single self-contained, offline-openable `report.html` with
   interactive charts.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+. Dependencies: numpy, plotly, jinja2, pyyaml (+ pytest for tests).

## Usage

```bash
tippspiel validate-data           # check input files for schema/consistency errors
tippspiel predict                 # group-stage predictions + tips only (Phase 1, no sim)
tippspiel run                     # full pipeline: predict + 50k simulations + report
tippspiel verify                  # backtest the predictor on a completed tournament (pool points + calibration)
tippspiel tune                    # sweep predictor params vs the completed-tournament backtests
tippspiel fit-offdef              # fit per-team offensive/defensive Elo from history into teams.csv
tippspiel run --config configs/womenseuro2025.yaml   # run for a different tournament
```

The report is written to `output/report.html` (configurable). A full `run` completes in a
few seconds.

## Multiple tournaments & verification

The CLI is **multi-tournament**. Each tournament is **one config file** — `config.yaml`
(default, FIFA World Cup 2026) and `configs/<name>.yaml` for the rest — carrying the engine
settings plus a `tournament:` block (data folder, metadata, bonus questions). Select one with
`--config <file>`. The engine derives the format from the data (group count/size from
`fixtures.csv`; the knockout chain + thirds from the knockout fixtures' references), so it
handles different shapes — the 48-team / best-8-thirds / R32-first World Cup 2026, the
32-team **World Cup 2022**, the 24-team / best-thirds **Euro 2024**, and the 16-team / no-thirds
/ QF-first **UEFA Women's Euro 2025** — without code changes. Add a tournament by dropping in a
data folder + a config file.

`tippspiel verify --config configs/<completed>.yaml` backtests predictor accuracy: it tips every
actual match a-priori from the pre-tournament Elo snapshot and totals the **pool points** the
tips would have scored against the real results, against a naive most-likely-scoreline baseline
and the per-match maximum, plus **calibration** (tendency RPS + scoreline NLL). Five completed
tournaments ship as seeded benchmarks — `womenseuro2025`, `wc2022`, `euro2024`, `wc2018` and
`euro2020`; the model beats the naive baseline on all five. Output: `output/verify.{md,json}`.

`tippspiel tune` sweeps the predictor parameters (`mu`, `k`, `rho`, `host_elo_bonus`,
`ko_goal_scale`, `alpha`) over those benchmarks and writes a leaderboard
(`output/tune.{md,json}`), ranking by calibration with pool points as the tie-break, plus a
leave-one-tournament-out generalisation check. The shipped config parameters are the tuned result.

## Offensive/defensive Elo

A single Elo rating sets *who* wins but makes every match expect the same total goals. To add
the goal-**volume** dimension — a Spain–Norway shoot-out vs. an Italy–Greece stalemate — each
team also carries `att_elo` (attack) and `def_elo` (defence), fitted by `tippspiel fit-offdef`
from the full international match-goal history (1872–present, committed under
`tippspiel/data/historical/`) using an online, Elo-style update on goals scored/conceded
(FIFA-importance-weighted). The predictor folds them in as a symmetric volume term with a tunable
weight `alpha` (0 = pure Elo); two strong attacks → higher-scoring, two stingy defences → tighter.
`fit-offdef` snapshots ratings as of the day before kickoff and writes them into `teams.csv`.

## Configuration

Each tournament config file holds the model parameters, MC iterations + seed, display
timezone and bonus questions, plus the `tournament:` block. The MC seed is mandatory and
surfaced in the report: same seed + same inputs ⇒ identical output. The predictor params were
**tuned** via `tippspiel tune`: `k` (Elo→goal-rate sensitivity), `rho` (Dixon-Coles low-score
correction; negative lifts draws over 1:0/0:1), `host_elo_bonus` (applied when a team plays in
its own country), and `ko_goal_scale` (knockout goal-rate multiplier, since knockout results
are the 120-minute scoreline).

A separate `strategy:` block holds **`realism_tolerance`**. Pure expected-points maximisation
tips a shutout (one team scores 0) in ~89% of matches — correct for points, but unrealistic,
since the tendency (5) and goal-diff (3) terms dwarf the per-team goal terms (1+1). With a small
tolerance, the optimiser picks — among scorelines within that many pool-points of the EV optimum
— the one closest to the model's expected score, flipping e.g. `1:0`→`2:1` (same winner and
margin) when the model expects goals. `0` reproduces strict EV; `~0.15` lifts both-teams-score
tips to a realistic ~50% at a negligible (<2%) points cost.

## Scoring rules implemented

Exact-scoreline tips. Per group match: tendency 5, home-goal +1, away-goal +1, goal-diff
+3 (exact score = 10). Knockout matches: identical structure, all values doubled (exact
score = 20). Knockout tips are the result after 120 minutes; a shootout counts as a draw.
World Champion bonus = 50 points.

## Data files (`tippspiel/data/tournaments/<name>/`)

| File | Contents |
|---|---|
| `teams.csv` | teams: `team_id, name, elo, elo_trend` (+ optional `att_elo, def_elo` from `fit-offdef`) |
| `fixtures.csv` | all matches. Group rows use concrete teams; knockout rows use concrete teams for a completed event, else structured references — `W:A`/`R:B` (group winner/runner-up), `3RD:74:ABCDF` (a best-third filling slot 74 from the listed groups), `WIN:M101`/`LOSE:M101` (winner/loser of a match). The bracket is derived from these. |
| `results.csv` | played matches (append rows as the tournament runs; full for a completed event) |
| `thirds_allocation.json` | *optional* — explicit third-place combination→slot table (FIFA "Annex C"); absent ⇒ constraint-respecting bipartite fallback |
| `odds.csv` | *optional* — pre-match bookmaker 1X2 odds: `match_id, odds_home, odds_draw, odds_away` (raw decimal odds, de-vigged at load). Feeds the market-odds predictor and the report's per-fixture **Market-odds tip**. Rows are per-match optional; a missing match falls back to Elo. |

The tournament's display name, `completed` flag, data folder, Elo source and `bonus_questions`
live in its **config file** (`config.yaml` / `configs/<name>.yaml`), not in the data folder.

`data/eloratings_adapter.py` converts an eloratings.net `World.tsv` export into
`teams.csv`; `data/odds_adapter.py` converts a raw bookmaker 1X2 export into `odds.csv`;
`data/historical_results_adapter.py` loads the international-match corpus for `fit-offdef`. Elo
ratings and odds change over time — refresh both shortly before kickoff (11 June 2026).

### Data provenance & the one remaining approximation (snapshot, June 2026)

The teams, group draw (A–L), the 6 play-off winners (Czechia, Bosnia, Sweden, Türkiye,
DR Congo, Iraq) and the **full match schedule** — group *and* knockout dates, UTC kickoff
times and host countries — are taken per-match from the official post-draw fixture list.
**Elo ratings** are an eloratings.net snapshot (~5 June 2026) and **odds** an ESPN moneyline
snapshot (de-vigged at load); both move continuously, so refresh them shortly before kickoff
(11 June 2026).

One modelling approximation remains:

- **Third-placed → Round-of-32 allocation table.** The official FIFA "Annex C" table
  (mapping each of the 495 combinations of 8 qualifying third-placed groups to bracket
  slots) could not be fully confirmed. `fixtures.csv` therefore encodes only the **confirmed**
  structure (R32 pairings, the 8 receiving slots, and each slot's allowed source groups via the
  `3RD:<slot>:<groups>` refs). The simulator resolves slots with a **deterministic,
  constraint-respecting bipartite matching** (`simulation/bracket.py`) — a documented
  approximation. Drop in a `thirds_allocation.json` sidecar (and point `thirds_allocation_file`
  at it in the config) from the official table to override it.

## Accuracy note

The Elo-Poisson model is a reasonable forecaster but **will not systematically out-predict
the betting market**. The **market-odds predictor** (Phase 3) closes that gap where you supply
an `odds.csv` snapshot: it uses de-vigged bookmaker 1X2 odds for those fixtures and falls back
to Elo elsewhere, and the report shows its pick as a per-fixture **Market-odds tip**. This
tool's edge over casual pool participants is **correct probability-to-scoreline optimisation**
and **simulating the bracket for the champion bonus**. Elo ratings and odds are snapshots and
change continuously.

## Architecture

A pipeline with a designed-for-extension predictor seam:

- **`Predictor`** (`predictors/`) — match → scoreline distribution. Ships
  `EloPoissonPredictor` (multiplicative goal rates, optional Dixon-Coles low-score
  correction; host-venue advantage configurable via `host_elo_bonus`, default 0) and
  `MarketOddsPredictor` (de-vigged bookmaker 1X2 odds expanded to a scoreline where an
  `odds.csv` snapshot supplies them, Elo fallback otherwise).
- **`ExpectedPointsStrategy`** (`strategy/`) — turns the whole slate of predictions +
  tournament outcome into a complete set of tips, picking the scoreline that maximises expected
  pool points for each fixture and the modal answer for each bonus question.

The `TournamentSimulator` (`simulation/`) runs the vectorised Monte Carlo: group standings
with exact FIFA tiebreakers (criteria 1–4; criterion 5 via a named seeded random
tiebreak), best-8 third-placed selection, bracket assembly, and knockout progression.

### Phase 3 (implemented)

`MarketOddsPredictor` is implemented.
Activate the market predictor per tournament with `predictor: { name: market_odds, params: {
total_goals: 2.6, gmax: 7, ko_goal_scale: 1.0, fallback_params: { ... Elo params ... } } }` and
`tournament.odds_file: odds.csv`. Committing sourced pre-tournament odds for the `verify`
benchmarks then quantifies the calibration/pool-points lift over the Elo baseline.

## Testing

```bash
pytest
```

Covers the EV optimiser (hand-computed cases), Poisson/Dixon-Coles, Elo→goal-rate
positivity, standings tiebreakers (incl. head-to-head), third-placed selection, bracket
assembly, Monte Carlo reproducibility + convergence, partial-state conditioning, and a
full-pipeline self-contained-report integration test.
