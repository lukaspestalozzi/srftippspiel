---
name: tune-ad-model
description: Tune the attack/defence predictor's parameters (both the elo-block rating-generation knobs and the predictor knobs) against the completed-tournament backtests, sanity-check the result with the realism block, and decide whether to adopt the recommendation. Use when the user asks to tune the A/D model, re-tune parameters, refresh the A/D defaults, evaluate the A/D parameter grid, or investigate whether the shipped defaults are still optimal.
---

# Tune the attack/defence model — staged sweep with reality check

The `attack_defence_poisson` predictor has **two parameter groups** that act at different
layers, so tuning is **staged**:

- **Generation** params (`elo:` config block, `model: attack_defence`) — shape the per-team
  `(attack, defence)` ratings via the historical forward pass that produces
  `teams_attack_defence.csv`. Keys: `learning_rate`, `lookback_years`, `recency_decay`,
  `ad_home_advantage`.
- **Predictor** params (`predictors.attack_defence_poisson` block) — how the predictor turns
  those ratings into a scoreline distribution per match. Keys: `base_log_rate`,
  `home_advantage`, `rho`, `ko_goal_scale`.

The objective is **blended**: rank by mean RPS (tendency calibration), tie-break on pool-points
% of max. Code paths: `tippspiel/report/ad_tuning.py` (orchestration) and
`tippspiel/report/realism.py` (reality check).

## Workflow

### 1. Make sure the historical results cache is populated

```bash
ls -la ~/.cache/tippspiel/results.csv
```

If absent, run `tippspiel build-elo --config config.yaml` once to fetch and cache (~25y of
international results, ~3 MB).

### 2. Run the staged sweep

```bash
tippspiel tune --predictor attack_defence_poisson
```

Cost: ~minutes (24 generation × 5 tournaments forward passes + 72 predictor × 5
verifications). Stage 2 reuses Stage 1's cached per-tournament `teams` dicts, so it's fast.

Stdout shows:
- Stage 1 (generation): default → best mean RPS, recommended params.
- Stage 2 (predictor): default → best mean RPS at the Stage-1-best gen point, recommended params.
- Combined headline (mean RPS, model pts, %max, exact hits / matches).
- Reality-check verdict: PASS / WARN / FAIL.

Full leaderboards + per-tournament breakdowns land in `output/tune.md` and `output/tune.json`.

### 3. Read `output/tune.md` and decide whether to adopt the recommendation

Always inspect **both** the leaderboard ranking AND the reality-check block before adopting.
The blended objective is calibration-first, so the tuner can pick an RPS-optimal point that
trades realism (e.g. concentrates tips on 1:0) for a tiny RPS gain.

**Decision checklist** (apply in order):

1. **Headline gain.** Is `recommended_metrics.mean_rps` materially better than
   `default_metrics.mean_rps`? "Materially" ≈ ≥ 0.001 absolute (the rounding floor of the
   blended key). A 0.0001 win is noise. If no, keep the defaults — say so explicitly.
2. **Pool-points sanity.** Did `model_pct` drop noticeably? The objective allows this (RPS
   primary), but a > 1 pp drop with a sub-0.001 RPS gain is a bad trade — keep the defaults.
3. **Reality-check verdict.** `PASS` is OK to adopt. `WARN` means at least one drift threshold
   was crossed — read the reasons in `reality_check.recommended.verdict.reasons` and the
   per-tournament tables; only adopt if the user agrees the drift is acceptable. `FAIL` means
   don't adopt without explicit user sign-off.
4. **Tip-composition diversity.** Compare `modal_share` between default and recommended in the
   reality-check pooled rows. A jump of > 10 pp toward a single scoreline (typically 1:0) is a
   regression in tip variety even if the RPS technically improved. Flag this to the user.
5. **LOO generalisation.** In `leave_one_out`, check that no single tournament's held-out
   `heldout_mean_rps` is sharply worse than the in-grid average for that tournament. Sharp
   per-tournament regressions on hold-out mean the recommendation is overfit.

If the recommendation passes the checklist, proceed to step 4. Otherwise report the
trade-off honestly and ask whether to keep defaults or pick a different point on the
leaderboard.

### 4. Adopt the recommendation

Two changes ship together (configs are normally identical across tournaments):

**a. Update the `attack_defence_poisson` block** in `config.yaml` AND every
`configs/<name>.yaml` with the Stage 2 recommended values.

**b. Update the `elo:` block** (same files) with the Stage 1 recommended values, then
**regenerate** every tournament's `teams_attack_defence.csv`:

```bash
for cfg in config.yaml configs/{euro2016,wc2022,euro2024,wc2018,euro2020}.yaml; do
  # data_dir is in the tournament block of each config; the path matches it
  name=$(basename "$cfg" .yaml | sed 's/config/wc2026/')
  tippspiel build-elo --config "$cfg" \
    --write-teams "tippspiel/data/tournaments/$name/teams_attack_defence.csv"
done
```

Commit the config + every regenerated `teams_attack_defence.csv` together — the predictor
auto-resolves the file by `ratings_kind`, no other code changes needed.

### 5. Verify

Run the full suite plus one end-to-end check:

```bash
pytest -q
tippspiel verify --config configs/euro2016.yaml --predictor attack_defence_poisson
```

The backtest numbers should match (within rounding) the new tune's per-tournament row for
that benchmark. Confirm with the user before committing.

## Key facts to remember

- `tippspiel tune` is **single-model** (`--predictor` required, no default — same contract as
  `verify`/`diagnose`). `--predictor elo_poisson` runs the classic flat grid; only
  `--predictor attack_defence_poisson` triggers the staged A/D path.
- The reality check compares predicted vs **actual** distributions of the completed
  tournaments — no external data needed. Metrics: mean goals/match, tendency split (H/D/A),
  top-5 scoreline frequencies, tip composition (modal-tip share), aggregate-matrix total
  variation distance. Per-metric thresholds in `tippspiel/report/realism.py` map drifts to
  `PASS`/`WARN`/`FAIL`.
- Generation params controlling the forward pass come from the **tournament's** `elo:` block
  (`tippspiel/elo/config.py:EloConfig`). The tuner uses the first benchmark's elo block as
  the base and overrides each generation grid point via `dataclasses.replace`.
- The synthesised per-tournament `teams` dicts (Stage 1 output) **preserve** the official
  `Team.elo` from the tournament's own `teams.csv` and only inject `attack`/`defence` from
  the forward pass — same convention as `build-elo --write-teams` for the attack/defence
  model.
- The blended objective lives in `tippspiel/report/tuning.py:_blended_key` and is shared
  between the elo and A/D paths.

## When NOT to use this skill

- For `elo_poisson` tuning (no staging, no reality check) — just run
  `tippspiel tune --predictor elo_poisson` directly and read `output/tune.md`.
- For one-off `verify` runs against a single completed tournament — use
  `tippspiel verify --predictor attack_defence_poisson --config configs/<name>.yaml`.
- For tuning the rank-optimising / EV strategy params — those live in the `strategy:` block
  and aren't covered by `tune` (the combined `run`/`predict` report already shows both
  strategies for inspection).
