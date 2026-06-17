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
tippspiel tune              # sweep predictor params vs the completed-tournament backtests -> output/tune.{md,json}
tippspiel fit-ratings       # fit scalar + offensive/defensive Elo from the corpus -> teams.csv elo/att_elo/def_elo
                            #   (was `fit-offdef`, kept as a deprecated alias)
```

Each tournament is **one config file**, selected with `--config <file>` (default `config.yaml`
= FIFA World Cup 2026; further tournaments live under `configs/<name>.yaml`). A config file
carries the engine defaults **and** a `tournament:` block (name, display name, `completed`
flag, `data_dir`, Elo source, optional `thirds_allocation_file`) plus `bonus_questions:`.

## Tournaments layer

A tournament's data lives under `tippspiel/data/tournaments/<name>/`: `teams.csv`,
`fixtures.csv`, `results.csv` (+ optional `thirds_allocation.json` and `odds.csv` sidecars).
`odds.csv` (`match_id,odds_home,odds_draw,odds_away`, raw decimal odds, de-vigged at load)
feeds the `MarketOddsPredictor` and the report's per-fixture "Market-odds tip"; wire it up with
`tournament.odds_file: odds.csv` + `predictor.name: market_odds`. The
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
per-match max). It also reports **calibration** (mean tendency RPS + scoreline NLL).
`womenseuro2025`, `wc2022`, `euro2024`, `wc2018` and `euro2020` are the seeded benchmarks; the
model beats the naive baseline on all five. Code: `tippspiel/report/backtest.py`; scoring helper
`score_tip` in `strategy/expected_points.py`.

`verify` is for **completed** tournaments. To score a **live** tournament's already-played tips
leak-free, use `python scripts/retro_tips.py` (default `--config config.yaml`): it replays the
predictor on the data **as committed before each result was added** (the introducing commit's first
parent) and totals the pool points the recommended tips actually earned, vs naive. Scoring played
matches against the *current* files is leaky (a winner's Elo is bumped up *after* the result —
worth +5 pts on WC2026 matchday 1). Re-run it each matchday to keep the predict→score→learn loop
closed; findings + the standing "don't overfit to a single matchday" cautions live in
`docs/matchday-retrospective.md`.

`tippspiel tune` sweeps the predictor params (`mu`, `k`, `rho`, `host_elo_bonus`,
`ko_goal_scale`, `alpha`) over the completed-tournament backtests and writes a leaderboard
(`output/tune.{md,json}`). The objective is **blended**: rank by mean RPS (calibration),
tie-break on pool-points % of max; it also reports a leave-one-tournament-out generalisation
check. The current `config.yaml` params are the tuned result. Code: `tippspiel/report/tuning.py`.
`tippspiel tune --market` instead sweeps the model×market blend axes (`market_weight`,
`total_goals`, `match_draw`) with the Elo params pinned to the config's values, loading each
benchmark's committed `odds.csv` (a benchmark without one contributes pure-Elo metrics).
Note: knockout results are the **120-minute** scoreline, so `ko_goal_scale` lifts the knockout
goal rate (applied in `EloPoissonPredictor.predict` when `match.stage.is_knockout`); host
advantage applies when a team plays in its own country (`venue_country == home.team_id`).

## One corpus, two derivations (Elo + results)

The committed match corpus `tippspiel/data/historical/international_results.csv` (Mart Jürisoo's
men's dataset, 1872–present) is the **single source of truth** for both Elo and tournament
scorelines. `tippspiel fit-ratings` computes *all three* `teams.csv` columns from it at the
tournament's `snapshot_date`: the scalar `elo` (World-Football-Elo update, `training/scalar_elo.py`)
and `att_elo`/`def_elo` (below). Per-tournament `results.csv` is **thin** —
`match_id,date,winner_team_id` — and its scoreline is **resolved from the corpus** at load time by
date + the fixture's teams (`data/corpus_results.py`, called from `FileDataProvider.get_results`;
`winner_team_id` is the knockout penalty-shootout winner only). `get_results` is dual-mode: a row
with inline `home_goals,away_goals` is used as-is (synthetic test data), else it's a corpus
reference. `scripts/migrate_results_to_corpus_refs.py` did the one-off conversion.

**Scalar Elo source is per-config** via `elo.source: corpus | external` (default `external`):
`config.yaml` (live wc2026) uses `corpus` → **no more eloratings.net fetching**. The completed
benchmarks and `womenseuro2025` use `external` → `fit-ratings` writes only att/def and leaves the
committed `elo` untouched. Why: a from-scratch corpus fit is measurably *less* calibrated than
eloratings on the backtests (~2pp/tournament, ~4% worse RPS — eloratings encodes more signal), so
the `verify`/`tune` baselines stay on their frozen external snapshots; and the men's corpus can't
rate women's teams at all. The corpus-Elo defaults (`k_scale 1.4`, `home_advantage 60` in the
`elo:` block, vs canonical 1.0/100) are **calibrated against those backtests** to close the live
gap as far as a from-scratch fit can. `eloratings_diff.py` is now an offline calibration check
(corpus-Elo vs eloratings, Spearman ~0.98 on wc2026), not a refresh path.

## Offensive/defensive Elo (goal-volume layer)

A single scalar `Team.elo` fixes the goal **ratio** (who wins) but pins every match's **total**
goals to `mu`. Per-team `att_elo`/`def_elo` add the missing volume dimension. `tippspiel
fit-ratings` learns them from the full international match-goal history (Mart Jürisoo's dataset,
committed at `tippspiel/data/historical/international_results.csv`, 1872–present) with an online,
Elo-style update on **goals** (`tippspiel/training/offdef_elo.py`): for each match, chronologically,
`att += k·w·(goals_scored − λ̂)` and `def -= k·w·(goals_conceded − λ̂)` where `λ̂ = (μ/2)·exp(att−def+γ)`
— i.e. SGD on the Poisson NLL. Matches are FIFA-importance-weighted (`w`: friendly ×0.5, qualifier
×2.5, continental ×3, World Cup ×4; `tippspiel/data/historical_results_adapter.py`). `fit-ratings`
snapshots ratings as of the **day before the tournament's first kickoff** (so `verify` stays
leak-free) and writes the `att_elo,def_elo` columns into that tournament's `teams.csv`; they
default to 0 when absent. Convention: higher `att_elo` = scores more; higher `def_elo` = concedes
fewer (stingier). Ratings are zero-centred over the field, so the average matchup still expects `mu`.

The predictor blends them as a **symmetric volume term** weighted by `alpha`:
`vol = ((att_h+att_a) − (def_h+def_a))/2`, added to *both* sides' log-rate. The Elo tendency term
`k·Δ` is left at full strength (the scalar Elo calibrates win/draw/loss better than the fitted
ratings do — backtested), so off/def only moves the goal **total**: two strong attacks → high-scoring,
two stingy defences → tight. `alpha=0` reproduces the pure-Elo model exactly. Fit hyperparameters
(`k_att`, `k_def`, `gamma_home`, `epochs`, weight tiers) live in an optional `offdef:` config block;
`alpha` is in `predictor.params` and is what `tune` optimises.

## The diagnostic report — my primary analysis tool

`tippspiel diagnose` writes **`output/diagnostic.md`** (readable, fixed-width tables) and
**`output/diagnostic.json`** (full raw data). It exists *for me, not the pool* — the
machine-optimised counterpart to the human `report.html`. Use it to **analyse, verify, and
validate model output, improve the prediction models, and answer the user's ad-hoc
questions** about model behaviour. **It is mine to evolve freely** — extend or add sections
whenever a new question recurs (code: `tippspiel/report/diagnostics.py`).

It contains: run/config header; **predictor-behaviour** stats (recommended-tip frequency,
tendency split, EV-component breakdown, optimal-vs-naive gap, plain-English interpretation
notes); an **offensive/defensive Elo** section (most attack- vs defence-minded sides, rating
extremes, att/def-vs-Elo correlation); a **model-vs-market value check** (pure-model 1X2 vs the
de-vigged market per odds-backed fixture, Bächinger-style "WERT" flags where the model exceeds
the market by >7pp, mean-gap drift alarm); a **per-fixture table** (L/D/W, top-3 cells,
recommended vs naive tip + EV split); **simulation diagnostics** (invariants, title odds, group
qualification); **bonus calibration** vs `historical_stats.py`; and an **automated
PASS/WARN/FAIL anomaly block**.

Workflow: to answer "why does the model do X?", run `diagnose` and read `diagnostic.md`
first; for exact numbers or custom aggregation, read `diagnostic.json`. Example already
answered by the report: *"why always 1:0 / 0:1?"* → the recommended-tip frequency + EV
breakdown show the 5-pt tendency term dominates EV at ~1.3 goals/side, so the optimiser
picks the lowest-total scoreline capturing the dominant tendency. This shutout bias is now
**tunably mitigated** by `strategy.realism_tolerance` (see Conventions) without abandoning EV.

## Architecture map

- `tippspiel/model/` — frozen dataclasses (`types.py`), `Stage` enum + scoring weights
  (`stages.py`), `ScorelineDistribution` (`scoreline.py`).
- `tippspiel/predictors/` — `EloPoissonPredictor` (Phase-1/2) and `MarketOddsPredictor`
  (Phase-3): de-vigged bookmaker 1X2 odds expanded to a scoreline (`expansion.py`) where an
  `odds.csv` snapshot supplies them, falling back to Elo for every other (and synthetic) matchup.
  `market_weight` (0..1, default 1 = pure market) log-linearly pools the odds-implied and Elo
  matrices cell-wise where odds exist — the model×market ensemble; `match_draw: true` makes the
  expansion also match the de-vigged draw price (solving the per-match total-goals level via a
  nested bisection) instead of assuming `total_goals`.
- `tippspiel/strategy/` — `expected_points.py` (`ExpectedPointsStrategy`, the EV optimiser;
  `ev_components()` is the reusable EV breakdown; `best_tip()` takes a `realism_tolerance`) and
  `bonus.py` (bonus questions).
- `tippspiel/simulation/` — vectorised NumPy `TournamentSimulator` + standings/thirds/bracket.
- `tippspiel/report/` — `html_writer.py`/`charts.py`/templates (pool report),
  `diagnostics.py` (my report), and `site.py` — the static-site assembler behind CI's
  `publish` job (GitHub Pages, <https://lukaspestalozzi.github.io/srftippspiel/>): landing
  page + live `run` report + per-benchmark `predict` report & `verify` backtest. Deploys on
  push to `main` **and on ready (non-draft) PRs** (one site per repo — a PR deploy replaces
  the live site until the next `main` deploy; that's deliberate, for pre-merge checks).
- `tippspiel/pipeline.py` — orchestration. `_run_core(cfg, bundle, ...)` returns the raw
  objects shared by all reports; `run_pipeline()`/`write_diagnostics()`/`write_verification()`
  build the HTML report / diagnostic / backtest respectively.
- `tippspiel/config.py` — engine config (`load_config`) + tournament resolution from the same
  config file (`load_tournament` → `TournamentBundle`).
- `tippspiel/report/backtest.py` — the `verify` historical backtest.
- `tippspiel/training/` — offline model-fitting (not the hot path). `offdef_elo.py` is the
  online Elo-for-goals fitter behind `fit-ratings`; `offdef_elo.py`'s sibling `scalar_elo.py` is
  the World-Football-Elo scalar fitter (`elo.source: corpus`).
- `tippspiel/data/tournaments/<name>/` — per-tournament data (teams/fixtures/results + optional
  `thirds_allocation.json`); `historical_stats.py` holds sourced reference stats (top-scorer
  prior + validation bands). `data/historical/international_results.csv` is the committed
  international-match corpus + `historical_results_adapter.py` loads/weights/name-maps it for the
  off/def fit. Config files: `config.yaml` + `configs/<name>.yaml`.

## Conventions & gotchas

- **Elo → goal rates are multiplicative**, never additive: `λ = (μ/2)·exp(±k·Δelo + vol)`, so
  both rates stay strictly positive for any matchup. There's a regression test guarding this.
  `vol` is the off/def goal-volume term (see "Offensive/defensive Elo" above); `alpha=0` drops it.
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
- **`strategy.realism_tolerance` (config) trades a sliver of EV for realistic tips.** Pure
  EV-maximisation tips ~89% shutouts (one team scores 0) because the 5-pt tendency + 3-pt
  goal-diff terms dominate the 1-pt goal terms. `best_tip` instead picks, among scorelines
  within `realism_tolerance` pool-points of the EV optimum, the one nearest the model's expected
  scoreline — flipping e.g. 1:0→2:1 (same tendency + margin) when the model expects goals.
  `0` = legacy strict EV (byte-identical, ~25% both-teams-score tips on the benchmarks). The
  config uses `0.05`: the sweep in `docs/matchday-retrospective.md` shows it is the points sweet
  spot (45.33% of max vs strict-EV's 45.24% over the 261 benchmark matches) while lifting BTS tips
  to ~38%, whereas `0.15` pays ~0.8pp of points for extra realism (BTS ~69%) you don't need. It
  only affects pool **points**, never RPS/NLL (those come from the distribution, not the tip), so
  `tune` (RPS-primary) deliberately excludes it — set it by direct points-cost measurement.
- **Validation is against reality**: `historical_stats.py` holds sourced figures that drive
  both the top-scorer prior and `tests/test_historical_validation.py`. Update it (with
  sources) rather than hardcoding magic numbers elsewhere.
- All tunables live in `config.yaml`; nothing model-related is hardcoded in logic.
- Tests live in `tests/`; mirror the `REPO = Path(tippspiel.__file__).parent.parent` fixture
  pattern and keep simulation iterations small (a few thousand) for speed.
- **Pre-existing issues are always in scope — fix them as soon as you detect them.** If you
  notice a lint warning, dead code, broken test, stale comment, or any other defect while
  working — even if it predates your change and is unrelated to your task — fix it in the same
  pass rather than leaving it. Run `ruff check tippspiel tests` and keep it clean.

## Phase status

All three phases are implemented. Phase 1 = group tips + report; Phase 2 = Monte Carlo engine +
bonus questions; Phase 3 = `MarketOddsPredictor` (de-vigged 1X2 odds → scoreline, Elo fallback;
activated per tournament via an `odds.csv` snapshot + `predictor.name: market_odds`, surfaced as
the report's per-fixture "Market-odds tip"), with three blend knobs — `market_weight` (model×market
pool), `match_draw` (draw-matching expansion) and `divergence_threshold` (gate the blend to fixtures
where model and market disagree) — plus the diagnostic model-vs-market value check. `tippspiel tune
--market` sweeps the blend axes with the Elo params pinned. ESPN-sourced `odds.csv` snapshots are
committed for wc2026 and 4 of the 5 `verify` benchmarks (womenseuro2025, wc2022, euro2024,
euro2020); **wc2018 has none** (ESPN's archive doesn't go that far back — sourcing it via
`odds_adapter.convert_odds_export` is an open optional task).

**Decision (2026-06):** on the backtests the blend is at best neutral — it slightly improves
calibration but costs pool points, and the targeted `divergence_threshold` variant recovers the
points without being a measured win. So the **completed-benchmark configs stay `elo_poisson`** to
keep `verify`/`tune` baselines pure-Elo, while the **live wc2026 `config.yaml` uses `market_odds`**
(`market_weight 0.5, match_draw, divergence_threshold 0.15`, tuned Elo params as `fallback_params`)
— a user call justified by live odds carrying lineup/team news a pre-tournament Elo can't, not by
the archives. Full numbers: `output/tune.{md,json}` from `tune --market`. **Leak caveat:** archived
KO odds are priced *after* the group stage, so the leak-clean market-vs-Elo comparison is `verify`'s
**group-stage** calibration split (the Elo snapshot is strictly pre-tournament; the KO odds are not).
