# CLAUDE.md

Operational notes for working in this repo. (User-facing overview lives in `README.md`.)

## What this is

`tippspiel` is a CLI that recommends scoreline tips + bonus answers for a football tournament
betting pool. Every match is predicted as a scoreline distribution under **both** rating models
(`elo_poisson` = eloratings.net snapshot, `attack_defence_poisson` = computed attack/defence) and
tipped under **both** meta-strategies — `expected_points` (maximise own EV) and `rank_optimizing`
(maximise P(win the pool) against a modelled field). The Monte Carlo (50k× per model) drives
advancement/title odds and bonus answers, separately per model. The combined `run`/`predict` writes
one self-contained, mobile-friendly `output/report.html` carrying **four tips per fixture** (2 models
× 2 strategies) plus per-model outcome sections. Re-runnable mid-tournament: played results in the
tournament's `results.csv` condition everything downstream. Originally built for the FIFA World Cup
2026; now **multi-tournament** (see below).

## Commands

```bash
pip install -e ".[dev]"     # setup (Python 3.11+)
pytest -q                   # run the full suite (~5s); ALWAYS run before committing
tippspiel validate-data     # schema/consistency check on input files
tippspiel predict                                    # combined report (every predictor × EV/rank), no simulation
tippspiel run                                        # combined report + Monte Carlo per model -> output/report.html (pool-facing)
tippspiel diagnose --predictor elo_poisson           # single-model Claude diagnostic report -> output/diagnostic.{md,json}
tippspiel diagnose --predictor elo_poisson --no-sim  # fast, predictor-only (skips Monte Carlo)
tippspiel verify --predictor elo_poisson             # backtest a predictor vs a completed tournament -> output/verify.{md,json}
tippspiel tune              # sweep elo_poisson params vs the completed-tournament backtests -> output/tune.{md,json}
tippspiel build-elo         # recompute ratings from ~25y of historical results -> output/elo.{md,json}
```

**`run` and `predict` run every configured predictor automatically** — they take no
`--predictor`/`--strategy` (any value passed is silently ignored), and the resulting `report.html`
shows all models side by side with both meta-strategies per match. The **single-model** commands
`diagnose` and `verify` still **require** `--predictor` (no default; they error without it).
Choices: `elo_poisson` (reads the official eloratings snapshot `teams.csv`) or
`attack_defence_poisson` (reads the computed `teams_attack_defence.csv`). Each config's
`predictors:` block carries params for **both** models; `tune` always sweeps `elo_poisson`,
`build-elo`/`validate-data` need no predictor.

Each tournament is **one config file**, selected with `--config <file>` (default `config.yaml`
= FIFA World Cup 2026; further tournaments live under `configs/<name>.yaml`). A config file
carries the engine defaults (`predictors:` map, strategy, simulation, report, `elo:`) **and** a
`tournament:` block (name, display name, `completed` flag, `data_dir`, Elo source, optional
`thirds_allocation_file`) plus `bonus_questions:`.

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
32-team/8-group/no-thirds/R16-first (WC 2022) and 24-team/6-group/best-thirds/R16-first
(Euro 2024, Euro 2016) alike (a 16-team/4-group/no-thirds/QF-first format also works). Add a
tournament by dropping
in a data folder + a config file; no engine code changes. The third-place combination->slot
table (FIFA "Annex C") is the **only** optional sidecar, needed only for an unplayed best-thirds
format once the official table is confirmed; absent, a constraint-respecting bipartite matching
over each slot's allowed groups is used (`tippspiel/simulation/bracket.py`).

`tippspiel verify --config configs/<completed>.yaml --predictor <model>` is the
**predictor-accuracy backtest**: it predicts every actual match a-priori from the pre-tournament
ratings snapshot, tips it, and totals the pool points scored vs the actual results (with the naive
most-likely tip as a baseline, and the per-match max). It also reports **calibration** (mean
tendency RPS + scoreline NLL). `euro2016`, `wc2022`, `euro2024`, `wc2018` and `euro2020` are the
seeded benchmarks (all men's tournaments — the historical results dataset is men-only); the model
beats the naive baseline on all five. Code: `tippspiel/report/backtest.py`; scoring helper
`score_tip` in `strategy/expected_points.py`.

`tippspiel tune` sweeps the predictor params (`mu`, `k`, `rho`, `host_elo_bonus`,
`ko_goal_scale`) over the completed-tournament backtests and writes a leaderboard
(`output/tune.{md,json}`). The objective is **blended**: rank by mean RPS (calibration),
tie-break on pool-points % of max; it also reports a leave-one-tournament-out generalisation
check. The current `config.yaml` params are the tuned result. Code: `tippspiel/report/tuning.py`.
Note: knockout results are the **120-minute** scoreline, so `ko_goal_scale` lifts the knockout
goal rate (applied in `EloPoissonPredictor.predict` when `match.stage.is_knockout`); host
advantage applies when a team plays in its own country (`venue_country == home.team_id`).

## Elo builder (`build-elo`)

`tippspiel build-elo` recomputes team Elo from scratch instead of trusting the eloratings.net
snapshot in `teams.csv`. It fetches ~25y of international results at runtime (the
`martj42/international_results` CSV, cached under `~/.cache/tippspiel`), runs the **World Football
Elo** algorithm (logistic expectation, goal-difference multiplier, importance-tiered K from the
`tournament` column, +home advantage on non-neutral ground, zero-sum updates) over a chronological
forward pass, **weighting recent matches more heavily** (a lookback window + a half-life decay on
K — both in the config `elo:` block). Writes a ranking + computed-vs-current comparison to
`output/elo.{md,json}`; with `--write-teams PATH` it emits a teams CSV reusing the tournament's own
rows so the name→id map stays collision-free. For `world_football` it overwrites the `elo` column;
for `attack_defence` it **preserves** the official `elo` and only adds `attack`/`defence` columns
(so the emitted file is a strict superset). By convention each tournament ships **two** ratings
files — `teams.csv` (official eloratings, read by `elo_poisson`) and `teams_attack_defence.csv`
(committed `build-elo --write-teams` output, read by `attack_defence_poisson`); each predictor's
`ratings_kind` resolves its file (`TournamentBundle.teams_files` → `pipeline.ratings_file`), so the
combined `run`/`predict` loads both files in one pass and `diagnose`/`verify` load whichever the
`--predictor` selects — integration is **data only**, no `teams_file` edit needed. Default
`--as-of` is today, or a completed tournament's start date − 1 day (no result leakage). Code: `tippspiel/elo/` (the
`RatingModel` ABC in `ratings.py` is the seam between rating schemes; `world_football.py` is the
single-rating implementation, `attack_defence.py` the two-rating one — see below) +
`tippspiel/report/elo_report.py`. The chronological fold lives in `run_forward_pass` (callers read
either scalar `ratings()` or, for attack/defence, the `(atk, def)` pairs). Country-name→team_id
normalization + aliases live in `elo/names.py`. The reconstruction won't equal eloratings.net
exactly and its spread is tighter (windowing + decay compress it), so **re-run `tune`** before
adopting a computed source for live predictions — a constant offset cancels in `elo_home −
elo_away`, but the narrower spread shifts the optimal `k`.

### Attack/defence model (`model: attack_defence`) — beats eloratings on pool points

The single-rating reconstruction **can't beat the eloratings.net snapshots** on the men's backtests
(best ~0.201 vs 0.194 RPS), and **recency weighting *hurts* accuracy** (full history with decay off
is best — the "weight recent matches more heavily" knob is a net negative for prediction). So the
recommended computed source is the **attack/defence** model (`elo/attack_defence.py`,
`AttackDefenceElo`): two ratings per team (offence + defence) fit online by **SGD on the Poisson
log-likelihood** of match goals — the gradient wrt each log-rate is `(observed − expected)`, giving
`atk[h] += lr·w·(gh−λh); def[a] −= lr·w·(gh−λh)` (and symmetrically for the away goals; `w` is the
recency weight, optional `ad_shrinkage` regularises). It maps natively onto the Poisson predictor:
`AttackDefencePoissonPredictor` (`predictors/attack_defence.py`, `attack_defence_poisson`) sets
`λ_home = exp(c + atk_home − def_away + ha·host)`, `λ_away = exp(c + atk_away − def_home)` and reuses
the shared `scoreline_from_rates` matrix builder. Ratings ride in new optional `Team.attack`/
`Team.defence` columns (emitted by `build-elo --write-teams` when the model is attack/defence;
absent ⇒ predictor falls back to the base rate). **Result on the 4 men's completed tournaments
(full history, recency off, `learning_rate=0.03`, `ad_home_advantage=0`, predictor `rho=-0.10`,
`ko_goal_scale=1.2`): pooled pool-points %max 46.4 vs eloratings 44.5 (+1.9pp), winning 3 of 4
(loses only WC2022, the famous-upset edition), at parity RPS (0.1965 vs 0.1942).** The dataset is
men-only, so all shipped tournaments are men's. Each tournament's `elo:` block carries the
attack/defence generation settings (`model: attack_defence, recency_decay: false,
lookback_years: 200, learning_rate: 0.03, ad_home_advantage: 0`) and its config's `predictors:`
block carries the `attack_defence_poisson` params (`rho: -0.1, ko_goal_scale: 1.2`). Regenerate the
committed `teams_attack_defence.csv` with `build-elo --write-teams`, then run any command with
`--predictor attack_defence_poisson` — the file is auto-resolved (no `teams_file` edit).

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
  reusable EV breakdown), `bonus.py` (bonus questions), `rank_optimizing.py`
  (`RankOptimizingStrategy` + `PredictorDerivedFieldModel` — see "Rank-optimising strategy").
- `tippspiel/simulation/` — vectorised NumPy `TournamentSimulator` + standings/thirds/bracket.
- `tippspiel/report/` — `html_writer.py`/`charts.py`/templates (pool report) and
  `diagnostics.py` (my report).
- `tippspiel/pipeline.py` — orchestration. `_run_core(cfg, bundle, ...)` returns the raw
  objects for single-model paths; `run_combined_pipeline()` iterates every configured predictor
  (`_model_run` per predictor + `comparison_from_params` for each model's EV/rank slates) and
  feeds `_build_combined_context` → the multi-model `report.html`. `write_diagnostics()` /
  `write_verification()` build the diagnostic / backtest off `_run_core` (still single-model).
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

## Rank-optimising strategy

`RankOptimizingStrategy` (`strategy/rank_optimizing.py`) tips to **win the pool**, not to
maximise own EV. The combined `run`/`predict` report **always shows both meta-strategies** for
every predictor — field-model params (`pool_size`, `top_n`, `expert_fraction`, `temperature`) come
from the config's `strategy:` block (consumed by `comparison_from_params`); the `name` field there
only matters for `diagnose`, which uses the single configured strategy via `build_strategy` (default
stays `expected_points`). It models the field via `PredictorDerivedFieldModel`
(`expert_fraction`, `temperature`), samples `n_worlds` joint result vectors (each tippable
match drawn independently from its predicted scoreline — pre-tournament that's the group
matches, which are independent), and maximises `P(rank<=top_n)` for a `pool_size`-participant
pool. Objective: **conditioned on a sampled result vector the field is i.i.d.**, so opponent
totals ≈ `Normal(mean_r, var_r)` (per-match moments via `field_score_moments`); the number of
opponents beating me ≈ `Poisson(pool_size·SF((s_me-mean_r)/sd_r))`, and `P(rank<=top_n)` is the
mean Poisson-CDF over worlds. A coordinate-ascent optimiser starts from the EV slate and only
deviates where contrarian variance strictly raises the win probability. The simulator is
**unchanged** (the group slate needs no joint per-iteration scorelines); bonus/champion answers
are carried over as the EV-optimal (modal) picks (contrarian bonus optimisation is out of
scope). Diagnostic §7 ("Rank-optimisation comparison") shows EV-vs-rank `P(win)`, total
`E[points]`, and the contrarian deviations — it runs regardless of the active strategy. Reuses
`score_tip`/`best_tip`/`ev_components` from `expected_points.py`. No external pool-tip data
exists, so the field model is a documented predictor-derived assumption.

## Phase status

Phase 1 (group tips + report) and Phase 2 (Monte Carlo engine + bonus questions) are
implemented. Phase 3: `RankOptimizingStrategy` + `FieldModel` are now implemented (see
"Rank-optimising strategy"); `MarketOddsPredictor` remains an interface stub — implement the
seam, don't refactor around it.
