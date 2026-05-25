# CLAUDE.md

Operational notes for working in this repo. (User-facing overview lives in `README.md`.)

## What this is

`tippspiel` is a CLI that recommends scoreline tips + bonus answers for the FIFA World Cup
2026 betting pool, optimised to **maximise expected pool points**. Pipeline: predict each
match as a scoreline distribution (Elo-Poisson) → pick the EV-maximising tip → Monte-Carlo
the whole bracket 50k× → report. Re-runnable mid-tournament: played results in
`tippspiel/data/results.csv` condition everything downstream.

## Commands

```bash
pip install -e ".[dev]"     # setup (Python 3.11+)
pytest -q                   # run the full suite (~8s); ALWAYS run before committing
tippspiel validate-data     # schema/consistency check on input files
tippspiel predict           # group-stage tips only, no simulation
tippspiel run               # full pipeline + output/report.html (pool-facing)
tippspiel diagnose          # the Claude diagnostic report (see below) -> output/diagnostic.{md,json}
tippspiel diagnose --no-sim # fast, predictor-only (skips Monte Carlo)
```

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
- `tippspiel/pipeline.py` — orchestration. `_run_core()` returns the raw objects shared by
  both reports; `run_pipeline()` builds HTML context, `write_diagnostics()` builds my report.
- `tippspiel/data/` — input data files + `historical_stats.py` (sourced reference stats).

## Conventions & gotchas

- **Elo → goal rates are multiplicative**, never additive: `λ = (μ/2)·exp(±k·Δelo)`, so both
  rates stay strictly positive for any matchup. There's a regression test guarding this.
- **Simulator is fully vectorised and seed-deterministic**: same seed + inputs ⇒ identical
  output. Keep new aggregations as array ops; don't add per-iteration Python loops.
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
