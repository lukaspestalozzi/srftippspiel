# srftippspiel — FIFA World Cup 2026 Tippspiel Predictor

A command-line tool that recommends what to **tip for every fixture** of the FIFA World
Cup 2026 betting pool, plus bonus-question answers, to **maximise expected pool points**.
It is re-runnable at any point during the tournament: results already played are read from
a file and all downstream predictions and simulations are conditioned on them.

## What it does

1. **Predicts** each match as a full scoreline distribution (Elo-Poisson model).
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
tippspiel verify                  # backtest the predictor on a completed tournament (pool points)
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
and the per-match maximum. `womenseuro2025`, `wc2022` and `euro2024` ship as seeded benchmarks;
the model beats the naive baseline. Output: `output/verify.{md,json}`.

## Configuration

Each tournament config file holds the model parameters, MC iterations + seed, display
timezone and bonus questions, plus the `tournament:` block. The MC seed is mandatory and
surfaced in the report: same seed + same inputs ⇒ identical output.

## Scoring rules implemented

Exact-scoreline tips. Per group match: tendency 5, home-goal +1, away-goal +1, goal-diff
+3 (exact score = 10). Knockout matches: identical structure, all values doubled (exact
score = 20). Knockout tips are the result after 120 minutes; a shootout counts as a draw.
World Champion bonus = 50 points.

## Data files (`tippspiel/data/tournaments/<name>/`)

| File | Contents |
|---|---|
| `teams.csv` | teams: `team_id, name, elo, elo_trend` |
| `fixtures.csv` | all matches. Group rows use concrete teams; knockout rows use concrete teams for a completed event, else structured references — `W:A`/`R:B` (group winner/runner-up), `3RD:74:ABCDF` (a best-third filling slot 74 from the listed groups), `WIN:M101`/`LOSE:M101` (winner/loser of a match). The bracket is derived from these. |
| `results.csv` | played matches (append rows as the tournament runs; full for a completed event) |
| `thirds_allocation.json` | *optional* — explicit third-place combination→slot table (FIFA "Annex C"); absent ⇒ constraint-respecting bipartite fallback |

The tournament's display name, `completed` flag, data folder, Elo source and `bonus_questions`
live in its **config file** (`config.yaml` / `configs/<name>.yaml`), not in the data folder.

`data/eloratings_adapter.py` converts an eloratings.net `World.tsv` export into
`teams.csv`. Elo ratings change after every international match — refresh `teams.csv`
shortly before kickoff (11 June 2026).

### ⚠️ Known data risks (best-effort snapshot, May 2026)

This snapshot was sourced best-effort from public sources and **should be verified before
relying on exact output**:

- **Third-placed → Round-of-32 allocation table.** The official FIFA "Annex C" table
  (mapping each of the 495 combinations of 8 qualifying third-placed groups to bracket
  slots) could not be fully confirmed. `fixtures.csv` therefore encodes only the **confirmed**
  structure (R32 pairings, the 8 receiving slots, and each slot's allowed source groups via the
  `3RD:<slot>:<groups>` refs). The simulator resolves slots with a **deterministic,
  constraint-respecting bipartite matching** (`simulation/bracket.py`) — a documented
  approximation. Drop in a `thirds_allocation.json` sidecar (and point `thirds_allocation_file`
  at it in the config) from the official table to override it.
- **Elo ratings** are from an eloratings.net mirror (snapshot ~25 May 2026). Panama,
  Czechia and Canada Elo values are estimates pending verification.
- **Kickoff times** are converted from a published UK-time schedule; exact minutes and a
  few matchday-3 pairings are best-effort. Dates and host venues are confirmed.

The 48 teams, group draw (A–L), and the 6 playoff winners (Czechia, Bosnia, Sweden,
Türkiye, DR Congo, Iraq) are confirmed.

## Accuracy note

The Elo-Poisson model is a reasonable forecaster but **will not systematically out-predict
the betting market**. A market-odds predictor (Phase 3, stubbed) would be the
higher-accuracy option. This tool's edge over casual pool participants is **correct
probability-to-scoreline optimisation** and **simulating the bracket for the champion
bonus**, not a superior forecast. Elo ratings are a snapshot and change continuously.

## Architecture

A pipeline with two designed-for-extension seams:

- **`Predictor`** (`predictors/`) — match → scoreline distribution. Ships
  `EloPoissonPredictor` (multiplicative goal rates, optional Dixon-Coles low-score
  correction). Host-venue advantage is configurable (`host_elo_bonus`, default 0).
- **`TipStrategy`** (`strategy/`) — the whole slate of predictions + tournament outcome →
  a complete set of tips. Ships `ExpectedPointsStrategy`.

The `TournamentSimulator` (`simulation/`) runs the vectorised Monte Carlo: group standings
with exact FIFA tiebreakers (criteria 1–4; criterion 5 via a named seeded random
tiebreak), best-8 third-placed selection, bracket assembly, and knockout progression.

### Phase 3 (stubbed, not implemented)

`MarketOddsPredictor`, `RankOptimizingStrategy` and `FieldModel` are abstract/`NotImplementedError`
stubs. The interfaces already accommodate them, so Phase 3 slots in without refactoring.

## Testing

```bash
pytest
```

Covers the EV optimiser (hand-computed cases), Poisson/Dixon-Coles, Elo→goal-rate
positivity, standings tiebreakers (incl. head-to-head), third-placed selection, bracket
assembly, Monte Carlo reproducibility + convergence, partial-state conditioning, and a
full-pipeline self-contained-report integration test.
