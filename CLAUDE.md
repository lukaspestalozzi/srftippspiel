# CLAUDE.md

Operational notes for working in this repo. (User-facing overview lives in `README.md`.)

## What this is

`tippspiel` is a CLI that recommends scoreline tips + bonus answers for a football tournament
betting pool, optimised to **maximise expected pool points**. Pipeline: predict each match as
a scoreline distribution (Elo-Poisson) → pick the EV-maximising tip → Monte-Carlo the whole
bracket 50k× → report. Re-runnable mid-tournament: played results in the tournament's
`results.csv` condition everything downstream. Originally built for the FIFA World Cup 2026;
now **multi-tournament** (see below).

## Commands

```bash
pip install -e ".[dev]"     # setup (Python 3.11+)
pytest -q                   # run the full suite (~5s); ALWAYS run before committing
tippspiel validate-data     # schema/consistency check on input files
tippspiel predict           # group-stage tips only, no simulation
tippspiel run               # full pipeline + output/report.html (pool-facing)
tippspiel diagnose          # the Claude diagnostic report (see below) -> output/diagnostic.{md,json}
tippspiel diagnose --no-sim # fast, predictor-only (skips Monte Carlo)
tippspiel verify            # backtest the predictor against a completed tournament -> output/verify.{md,json}
```

Each tournament is **one config file**, selected with `--config <file>` (default `config.yaml`
= FIFA World Cup 2026; further tournaments live under `configs/<name>.yaml`). A config file
carries the engine defaults **and** a `tournament:` block (name, display name, `completed`
flag, `data_dir`, Elo source, optional `thirds_allocation_file`) plus `bonus_questions:`.

## Tournaments layer

A tournament's data lives under `tippspiel/data/tournaments/<name>/`: `teams.csv`,
`fixtures.csv`, `results.csv` (+ an optional `thirds_allocation.json` sidecar). The
**format is derived from the data**, not configured: group count/size from `fixtures.csv`,
and the knockout chain + whether thirds qualify **from the knockout fixtures themselves** —
KO rows reference group placings or earlier matches via structured refs in `home_ref`/`away_ref`:
`W:A` (winner of group A), `R:B` (runner-up of B), `3RD:74:ABCDF` (a best-placed third fills
slot 74, drawn from the listed allowed groups), `WIN:M101` / `LOSE:M101` (winner/loser of a
match). A completed tournament may instead list concrete KO participants. The engine is
format-agnostic — it supports 48-team/12-group/best-8-thirds/R32-first (WC 2026),
32-team/8-group/no-thirds/R16-first (WC 2022), 24-team/6-group/best-thirds/R16-first (Euro 2024)
and 16-team/4-group/no-thirds/QF-first (Women's Euro 2025) alike. Add a tournament by dropping
in a data folder + a config file; no engine code changes. The third-place combination->slot
table (FIFA "Annex C") is the **only** optional sidecar, needed only for an unplayed best-thirds
format once the official table is confirmed; absent, a constraint-respecting bipartite matching
over each slot's allowed groups is used (`tippspiel/simulation/bracket.py`).

`tippspiel verify --config configs/<completed>.yaml` is the **predictor-accuracy backtest**: it
predicts every actual match a-priori from the pre-tournament Elo snapshot, tips it, and totals the
pool points scored vs the actual results (with the naive most-likely tip as a baseline, and the
per-match max). `womenseuro2025`, `wc2022` and `euro2024` are the seeded benchmarks; the model
beats the naive baseline overall. Code: `tippspiel/report/backtest.py`; scoring helper `score_tip`
in `strategy/expected_points.py`.

## The diagnostic report — my primary analysis tool

`tippspiel diagnose` writes **`output/diagnostic.md`** (readable, fixed-width tables) and
**`output/diagnostic.json`** (full raw data). It exists *for me, not the pool* — the
machine-optimised counterpart to the human `report.html`. Use it to **analyse, verify, and
validate model output, improve the prediction models, and answer the user's ad-hoc
questions** about model behaviour. **It is mine to evolve freely** — extend or add sections
whenever a new question recurs (code: `tippspiel/report/diagnostics.py`).

It contains: run/config header; **predictor-behaviour** stats (recommended-tip frequency,
tendency split, EV-component breakdown, optimal-vs-naive gap, plain-English interpretation
notes); a **per-fixture table** (L/D/W, top-3 cells, recommended vs naive tip + EV split);
**simulation diagnostics** (invariants, title odds, group qualification); **bonus
calibration** vs `historical_stats.py`; and an **automated PASS/WARN/FAIL anomaly block**.

Workflow: to answer "why does the model do X?", run `diagnose` and read `diagnostic.md`
first; for exact numbers or custom aggregation, read `diagnostic.json`. Example already
answered by the report: *"why always 1:0 / 0:1?"* → the recommended-tip frequency + EV
breakdown show the 5-pt tendency term dominates EV at ~1.3 goals/side, so the optimiser
picks the lowest-total scoreline capturing the dominant tendency.

## Architecture map

- `tippspiel/model/` — frozen dataclasses (`types.py`), `Stage` enum + scoring weights
  (`stages.py`), `ScorelineDistribution` (`scoreline.py`).
- `tippspiel/predictors/` — `EloPoissonPredictor` (Phase-1/2); `market_odds.py` is a Phase-3 stub.
- `tippspiel/strategy/` — `expected_points.py` (EV optimiser; `ev_components()` is the
  reusable EV breakdown), `bonus.py` (bonus questions). `rank_optimizing.py` is a stub.
- `tippspiel/simulation/` — vectorised NumPy `TournamentSimulator` + standings/thirds/bracket.
- `tippspiel/report/` — `html_writer.py`/`charts.py`/templates (pool report) and
  `diagnostics.py` (my report).
- `tippspiel/pipeline.py` — orchestration. `_run_core(cfg, bundle, ...)` returns the raw
  objects shared by all reports; `run_pipeline()`/`write_diagnostics()`/`write_verification()`
  build the HTML report / diagnostic / backtest respectively.
- `tippspiel/config.py` — engine config (`load_config`) + tournament resolution from the same
  config file (`load_tournament` → `TournamentBundle`).
- `tippspiel/report/backtest.py` — the `verify` historical backtest.
- `tippspiel/data/tournaments/<name>/` — per-tournament data (teams/fixtures/results + optional
  `thirds_allocation.json`); `historical_stats.py` holds sourced reference stats (top-scorer
  prior + validation bands). Config files: `config.yaml` + `configs/<name>.yaml`.

## Conventions & gotchas

- **Elo → goal rates are multiplicative**, never additive: `λ = (μ/2)·exp(±k·Δelo)`, so both
  rates stay strictly positive for any matchup. There's a regression test guarding this.
- **Simulator is fully vectorised and seed-deterministic**: same seed + inputs ⇒ identical
  output. Keep new aggregations as array ops; don't add per-iteration Python loops.
- **Format-agnostic engine**: the simulator/bracket derive groups, the thirds count, and the
  knockout `stage_chain` from the data. Advancement metrics are stage-keyed `reach_<stage>`
  (e.g. `reach_r32`/`reach_r16`/.../`wins_title`; `reach_qf` is the first KO round for a
  16-team event). The `Bracket` is built from the knockout fixtures (`Bracket(ko_matches,
  group_letters, thirds_allocation)`): first round = KO fixtures filled from group standings,
  progression = those filled from earlier matches; third-place slots + their allowed groups come
  from the `3RD:<slot>:<groups>` refs. Don't hardcode group counts or stage names.
- **Goal tallies use the 120-minute scoreline only** (penalty-shootout goals excluded).
- **Bonus questions are exact-match scored** → recommend the **mode** (argmax). The strategy
  bonus loop already does this generically; add a question by subclassing `BonusQuestion`,
  registering it in `_BONUS_REGISTRY`, and listing it in `config.yaml`.
- **Validation is against reality**: `historical_stats.py` holds sourced figures that drive
  both the top-scorer prior and `tests/test_historical_validation.py`. Update it (with
  sources) rather than hardcoding magic numbers elsewhere.
- All tunables live in `config.yaml`; nothing model-related is hardcoded in logic.
- Tests live in `tests/`; mirror the `REPO = Path(tippspiel.__file__).parent.parent` fixture
  pattern and keep simulation iterations small (a few thousand) for speed.

## Phase status

Phase 1 (group tips + report) and Phase 2 (Monte Carlo engine + bonus questions) are
implemented. Phase 3 (`MarketOddsPredictor`, `RankOptimizingStrategy`, `FieldModel`) is
interface stubs only — implement the seams, don't refactor around them.
