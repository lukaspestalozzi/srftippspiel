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
tippspiel predict                 # group-stage predictions + tips only (no sim)
tippspiel run                     # full pipeline: predict + 50k simulations + report
tippspiel verify                  # backtest the predictor on a completed tournament
tippspiel tune                    # sweep predictor params vs the completed-tournament backtests
tippspiel fit-ratings             # fit scalar + offensive/defensive Elo from the corpus into teams.csv
tippspiel run --config configs/womenseuro2025.yaml   # run for a different tournament
```

The report is written to `output/report.html` (configurable). A full `run` completes in a
few seconds.

## Published reports

The latest reports are published to **<https://lukaspestalozzi.github.io/srftippspiel/>**:
the WC 2026 `run` report, plus each completed benchmark tournament's a-priori tips
report and predictor backtest. CI rebuilds and deploys the site on every push to `main`
(and on ready-for-review PRs, so changes can be checked on the live URL before merging) —
see the `publish` job in `.github/workflows/ci.yml`.

## Multiple tournaments & verification

The CLI is **multi-tournament**. Each tournament is **one config file** — `config.yaml`
(default, FIFA World Cup 2026) and `configs/<name>.yaml` for the rest — carrying the engine
settings plus a `tournament:` block (data folder, metadata, bonus questions). Select one with
`--config <file>`. The engine derives the format from the data (group count/size from
`fixtures.csv`; the knockout chain + thirds from the knockout fixtures' references), so it
handles the 48-team WC 2026, the 32-team **WC 2022**, the 24-team **Euro 2024** and the
16-team **Women's Euro 2025** without code changes. Add a tournament by dropping in a data
folder + a config file.

`tippspiel verify --config configs/<completed>.yaml` backtests predictor accuracy: it tips every
actual match a-priori from the pre-tournament Elo snapshot and totals the **pool points** the
tips would have scored against the real results, against a naive most-likely-scoreline baseline
and the per-match maximum, plus **calibration** (tendency RPS + scoreline NLL). Five completed
tournaments ship as seeded benchmarks — `womenseuro2025`, `wc2022`, `euro2024`, `wc2018` and
`euro2020`; the model beats the naive baseline on all five. Output: `output/verify.{md,json}`.

`tippspiel tune` sweeps the predictor parameters over those benchmarks and writes a leaderboard
(`output/tune.{md,json}`), ranking by calibration with pool points as the tie-break, plus a
leave-one-tournament-out generalisation check. The shipped config parameters are the tuned result.

## Offensive/defensive Elo

A single Elo rating sets *who* wins but makes every match expect the same total goals. To add
the goal-**volume** dimension — a Spain–Norway shoot-out vs. an Italy–Greece stalemate — each
team also carries `att_elo` (attack) and `def_elo` (defence), fitted by `tippspiel fit-ratings`
from the full international match-goal history (1872–present, committed under
`tippspiel/data/historical/`) using an online, Elo-style update on goals scored/conceded
(FIFA-importance-weighted). The predictor folds them in as a symmetric volume term with a tunable
weight `alpha` (0 = pure Elo); two strong attacks → higher-scoring, two stingy defences → tighter.
`fit-ratings` snapshots ratings as of the day before kickoff and writes them into `teams.csv`. The
same command (with `elo.source: corpus`) also derives the scalar base `elo` from that one corpus,
so the live tournament no longer fetches eloratings.net; completed benchmarks keep their committed
snapshot (`elo.source: external`). Per-tournament `results.csv` is thin and references the corpus.

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
margin) when the model expects goals. `0` reproduces strict EV; the config uses `0.05`, which
lifts both-teams-score tips to a more realistic rate at essentially no points cost (the
benchmark sweep in `docs/matchday-retrospective.md`).

## Scoring rules implemented

Exact-scoreline tips. Per group match: tendency 5, home-goal +1, away-goal +1, goal-diff
+3 (exact score = 10). Knockout matches: identical structure, all values doubled (exact
score = 20). Knockout tips are the result after 120 minutes; a shootout counts as a draw.
World Champion bonus = 50 points.

## Data files (`tippspiel/data/tournaments/<name>/`)

| File | Contents |
|---|---|
| `teams.csv` | teams: `team_id, name, elo` (+ optional `att_elo, def_elo`) — all three columns written by `fit-ratings` |
| `fixtures.csv` | all matches. Group rows use concrete teams; knockout rows use concrete teams for a completed event, else structured references — `W:A`/`R:B` (group winner/runner-up), `3RD:74:ABCDF` (a best-third filling slot 74 from the listed groups), `WIN:M101`/`LOSE:M101` (winner/loser of a match). The bracket is derived from these. |
| `results.csv` | played matches, **thin**: `match_id, date, winner_team_id` — the scoreline is read from the match corpus by date + teams (`winner_team_id` only for a knockout penalty shootout). Inline `home_goals, away_goals` are still accepted (synthetic/legacy data). |
| `thirds_allocation.json` | *optional* — explicit third-place combination→slot table (FIFA "Annex C"); absent ⇒ constraint-respecting bipartite fallback |
| `odds.csv` | *optional* — pre-match bookmaker 1X2 odds: `match_id, odds_home, odds_draw, odds_away` (raw decimal, de-vigged at load). Feeds the market-odds predictor and the report's per-fixture **Market-odds tip**. Rows are per-match optional; a missing match falls back to Elo. |

The tournament's display name, `completed` flag, data folder, Elo source and `bonus_questions`
live in its **config file**, not in the data folder. Adapters convert source data into these
files: `data/odds_adapter.py` (bookmaker 1X2 export → `odds.csv`) and
`data/historical_results_adapter.py` (the international-match corpus that feeds `fit-ratings`;
`data/eloratings_adapter.py` is a deprecated fallback now that scalar Elo is corpus-derived).
Odds are a per-fixture snapshot and a live tournament's Elo moves as the corpus grows — while a
tournament is running, refresh both after each matchday (see the `update-tournament-data` skill).

### Provenance & the thirds-allocation fallback

The teams, group draw (A–L), the 6 play-off winners (Czechia, Bosnia, Sweden, Türkiye,
DR Congo, Iraq) and the **full match schedule** (group and knockout dates, UTC kickoff times,
host countries) come per-match from the official post-draw fixture list. **Elo ratings** are
corpus-derived (World-Football-Elo fitted from the committed international-match corpus) and
**odds** an ESPN + Polymarket consensus snapshot (de-vigged at load).

One modelling detail is worth noting — the **third-placed → Round-of-32 allocation table.** When
the official FIFA "Annex C" table isn't pinned, `fixtures.csv` encodes only the confirmed
structure (R32 pairings, the 8 receiving slots, each slot's allowed source groups via
`3RD:<slot>:<groups>` refs) and the simulator resolves slots with a deterministic,
constraint-respecting bipartite matching (`simulation/bracket.py`). WC2026 **ships the official
allocation** as a `thirds_allocation.json` sidecar (pointed at by `thirds_allocation_file` in the
config), so its bracket is exact; the bipartite fallback is only used for an unplayed best-thirds
format whose official table isn't yet confirmed.

## Accuracy note

The Elo-Poisson model is a reasonable forecaster but **will not systematically out-predict
the betting market**. The **market-odds predictor** closes that gap where you supply an
`odds.csv` snapshot: it uses de-vigged bookmaker 1X2 odds for those fixtures and falls back to
Elo elsewhere, surfaced as a per-fixture **Market-odds tip**. This tool's edge over casual pool
participants is **correct probability-to-scoreline optimisation** and **simulating the bracket
for the champion bonus**.

## Architecture

A pipeline with a designed-for-extension predictor seam:

- **`Predictor`** (`predictors/`) — match → scoreline distribution. Ships `EloPoissonPredictor`
  (multiplicative goal rates, optional Dixon-Coles low-score correction; host-venue advantage via
  `host_elo_bonus`) and `MarketOddsPredictor` (de-vigged bookmaker 1X2 odds expanded to a scoreline
  where an `odds.csv` snapshot supplies them, Elo fallback otherwise). Activate the latter with
  `predictor: { name: market_odds, params: { total_goals: 2.6, gmax: 7, ko_goal_scale: 1.0,
  fallback_params: { ...Elo params... } } }` + `tournament.odds_file: odds.csv`.
- **`ExpectedPointsStrategy`** (`strategy/`) — turns the whole slate of predictions + tournament
  outcome into a complete set of tips, picking the scoreline that maximises expected pool points
  for each fixture and the modal answer for each bonus question.
- **`TournamentSimulator`** (`simulation/`) — vectorised Monte Carlo: group standings with exact
  FIFA tiebreakers (criteria 1–4; criterion 5 via a named seeded random tiebreak), best-thirds
  selection, bracket assembly, and knockout progression.

## Testing

```bash
pytest
```

Covers the EV optimiser (hand-computed cases), Poisson/Dixon-Coles, Elo→goal-rate
positivity, standings tiebreakers (incl. head-to-head), third-placed selection, bracket
assembly, Monte Carlo reproducibility + convergence, partial-state conditioning, and a
full-pipeline self-contained-report integration test.
