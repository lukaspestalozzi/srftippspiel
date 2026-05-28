# srftippspiel — FIFA World Cup 2026 Tippspiel Predictor

A command-line tool that recommends what to **tip for every fixture** of the FIFA World
Cup 2026 betting pool, plus bonus-question answers, to **maximise expected pool points**.
It is re-runnable at any point during the tournament: results already played are read from
a file and all downstream predictions and simulations are conditioned on them.

## What it does

1. **Predicts** each match as a full scoreline distribution under **two** rating models —
   the official eloratings.net snapshot (`elo_poisson`) and a computed attack/defence
   model (`attack_defence_poisson`).
2. **Optimises** the tip under **two** meta-strategies — EV-optimal (maximise own expected
   points) and pool-rank (maximise the probability of winning the pool against a modelled
   field). **Four tips per match: 2 models × 2 strategies.**
3. **Simulates** the whole tournament 50,000 times **per model** (Monte Carlo) to get
   group-advancement and title probabilities, the recommended World Champion, and the
   other bonus answers — separately for each model.
4. **Reports** everything in a single self-contained, offline-openable, mobile-friendly
   `report.html` with a 2×2 tip matrix per fixture, a per-model outcomes section, and
   interactive charts.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+. Dependencies: numpy, plotly, jinja2, pyyaml (+ pytest for tests).

## Usage

```bash
tippspiel validate-data                              # check input files for schema/consistency errors
tippspiel predict                                    # combined multi-model report, no simulation
tippspiel run                                        # combined multi-model report + 50k Monte Carlo per model
tippspiel verify --predictor elo_poisson             # backtest one predictor against a completed tournament
tippspiel tune --predictor elo_poisson               # sweep elo_poisson params (~seconds)
tippspiel tune --predictor attack_defence_poisson    # staged A/D sweep + reality check (~few minutes)
tippspiel run --config configs/euro2016.yaml         # the combined report for a different tournament
```

`run` and `predict` always run **every** configured predictor and present both meta-strategies
side by side — no `--predictor` flag needed. The single-model commands `verify` and `diagnose`
still **require** `--predictor` (no default): `elo_poisson` uses the official eloratings snapshot
(`teams.csv`); `attack_defence_poisson` uses the computed two-rating file
(`teams_attack_defence.csv`). The report is written to `output/report.html` (configurable). A full
`run` takes a few seconds longer than a single-model pass (it runs the simulator once per model).

## Multiple tournaments & verification

The CLI is **multi-tournament**. Each tournament is **one config file** — `config.yaml`
(default, FIFA World Cup 2026) and `configs/<name>.yaml` for the rest — carrying the engine
settings plus a `tournament:` block (data folder, metadata, bonus questions). Select one with
`--config <file>`. The engine derives the format from the data (group count/size from
`fixtures.csv`; the knockout chain + thirds from the knockout fixtures' references), so it
handles different shapes — the 48-team / best-8-thirds / R32-first World Cup 2026, the
32-team **World Cup 2022**, and the 24-team / best-thirds **Euro 2024** and **Euro 2016** —
without code changes. Add a tournament by dropping in a data folder + a config file.

`tippspiel verify --config configs/<completed>.yaml --predictor <model>` backtests predictor
accuracy: it tips every actual match a-priori from the pre-tournament ratings snapshot and totals
the **pool points** the tips would have scored against the real results, against a naive
most-likely-scoreline baseline and the per-match maximum, plus **calibration** (tendency RPS +
scoreline NLL). Five completed men's tournaments ship as seeded benchmarks — `euro2016`, `wc2022`,
`euro2024`, `wc2018` and `euro2020`; the model beats the naive baseline on all five. Output:
`output/verify.{md,json}`.

`tippspiel tune --predictor <model>` sweeps that model's parameters against the same
benchmarks and writes a leaderboard (`output/tune.{md,json}`), ranking by calibration with
pool points as the tie-break, plus a leave-one-tournament-out generalisation check. The
shipped config parameters are the tuned result. For `elo_poisson` it's a flat grid over
`mu`, `k`, `rho`, `host_elo_bonus`, `ko_goal_scale`. For `attack_defence_poisson` it's a
**staged** sweep: Stage 1 sweeps the rating-generation knobs in the `elo:` block
(`learning_rate`, `lookback_years`, `recency_decay`, `ad_home_advantage`), regenerating the
per-team `(attack, defence)` ratings via a forward pass each grid point; Stage 2 sweeps the
predictor knobs (`base_log_rate`, `home_advantage`, `rho`, `ko_goal_scale`). The report
includes a **reality check** comparing predicted vs actual mean goals/match, tendency
split, scoreline frequencies, and tip composition for each completed tournament — so it's
visible whether the tuner's RPS-optimal point trades realism for calibration (e.g.
concentrating tips on 1:0).

## Configuration

Each tournament config file holds the model parameters, MC iterations + seed, display
timezone and bonus questions, plus the `tournament:` block. The MC seed is mandatory and
surfaced in the report: same seed + same inputs ⇒ identical output. The predictor params were
**tuned** via `tippspiel tune`: `k` (Elo→goal-rate sensitivity), `rho` (Dixon-Coles low-score
correction; negative lifts draws over 1:0/0:1), `host_elo_bonus` (applied when a team plays in
its own country), and `ko_goal_scale` (knockout goal-rate multiplier, since knockout results
are the 120-minute scoreline).

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
  a complete set of tips. Ships `ExpectedPointsStrategy` (maximise own expected points) and
  `RankOptimizingStrategy` (maximise the probability of *winning* the pool — see below).

The `TournamentSimulator` (`simulation/`) runs the vectorised Monte Carlo: group standings
with exact FIFA tiebreakers (criteria 1–4; criterion 5 via a named seeded random
tiebreak), best-8 third-placed selection, bracket assembly, and knockout progression.

### Rank-optimising strategy (anticipating the field)

In a large pool (~200,000 entrants), the EV-maximising slate scores well but rarely *wins* —
thousands of sharp entrants converge on the same EV-optimal scorelines. `RankOptimizingStrategy`
instead maximises `P(rank ≤ top_n)` by modelling how the field tips (`FieldModel`, default
`PredictorDerivedFieldModel`) and deliberately taking contrarian variance where it raises the
win probability. The combined `run`/`predict` report **always shows both strategies** for every
configured predictor — field-model params (`pool_size`, `top_n`, `expert_fraction`, `temperature`)
come from the config's `strategy:` block. `tippspiel diagnose` is single-strategy and uses the
`name` from the config (default `expected_points`); it reports an EV-vs-rank comparison (estimated
win probability, expected points, and the contrarian deviations). The field model is derived from
the predictor (no real pool-tip data exists) and is a documented assumption.

### Phase 3 (partially implemented)

`RankOptimizingStrategy` and `FieldModel` are implemented (above). `MarketOddsPredictor` remains
an abstract/`NotImplementedError` stub; its interface already accommodates Phase 3 without
refactoring.

## Testing

```bash
pytest
```

Covers the EV optimiser (hand-computed cases), Poisson/Dixon-Coles, Elo→goal-rate
positivity, standings tiebreakers (incl. head-to-head), third-placed selection, bracket
assembly, Monte Carlo reproducibility + convergence, partial-state conditioning, and a
full-pipeline self-contained-report integration test.
