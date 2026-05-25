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

Every command takes `--tournament <name>` (default `wc2026`, set in `config.yaml`).

## Tournaments layer

A tournament is a self-contained bundle under `tippspiel/data/tournaments/<name>/`:
`teams.csv`, `fixtures.csv`, `results.csv`, `bracket_map.json`, `tournament.yaml` (display
name, data file names, `completed` flag, tournament-specific `bonus_questions`, Elo source).
The **format is derived from the data**, not configured: group count/size from `fixtures.csv`,
the knockout chain + whether thirds qualify from `bracket_map.json`. The engine is
format-agnostic — it supports 48-team/12-group/best-8-thirds/R32-first (WC 2026) and
16-team/4-group/no-thirds/QF-first (Women's Euro 2025) alike. Add a tournament by dropping in a
new bundle; no engine code changes.

`tippspiel verify --tournament <completed>` is the **predictor-accuracy backtest**: it predicts
every actual match a-priori from the pre-tournament Elo snapshot, tips it, and totals the pool
points scored vs the actual results (with the naive most-likely tip as a baseline, and the
per-match max). `womenseuro2025` is the seeded benchmark (England won; the model beats the
naive baseline overall). Code: `tippspiel/report/backtest.py`; scoring helper `score_tip` in
`strategy/expected_points.py`.

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
- `tippspiel/config.py` — global engine config (`load_config`) + per-tournament bundle
  resolution (`resolve_tournament`, `available_tournaments`, `TournamentBundle`).
- `tippspiel/report/backtest.py` — the `verify` historical backtest.
- `tippspiel/data/tournaments/<name>/` — per-tournament data bundles; `historical_stats.py`
  holds sourced reference stats (top-scorer prior + validation bands).

## Conventions & gotchas

- **Elo → goal rates are multiplicative**, never additive: `λ = (μ/2)·exp(±k·Δelo)`, so both
  rates stay strictly positive for any matchup. There's a regression test guarding this.
- **Simulator is fully vectorised and seed-deterministic**: same seed + inputs ⇒ identical
  output. Keep new aggregations as array ops; don't add per-iteration Python loops.
- **Format-agnostic engine**: the simulator/bracket derive groups, the thirds count, and the
  knockout `stage_chain` from the data. Advancement metrics are stage-keyed `reach_<stage>`
  (e.g. `reach_r32`/`reach_r16`/.../`wins_title`; `reach_qf` is the first KO round for a
  16-team event). `bracket_map.json` uses a generic `first_round` (with `stage`) + `progression`
  schema; `_meta.third_place_slots` empty ⇒ no thirds. Don't hardcode group counts or stage
  names.
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
