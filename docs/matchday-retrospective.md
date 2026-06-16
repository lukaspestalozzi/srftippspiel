# WC2026 tip retrospective — matchday 1 (13 played)

_Analysis of the first 13 played World Cup 2026 matches: what we tipped, how it scored, and what
(if anything) to change for the next round. Reproduce with `python scripts/retro_tips.py`._

## Method (leak-free)

Each played match is scored against the tip we **would have published before kickoff** — not the
tip the current data implies. This matters: the live data files are re-fitted every matchday, so a
team's Elo is bumped *after* it wins. Scoring the played matches against today's files flatters the
model by **+5 points (49 vs the true 44)** — almost all of it USA, promoted from 1726→1780 Elo
*because* they won 4:1. `scripts/retro_tips.py` instead replays the predictor on the data as it was
committed **before each result was added** (the introducing commit's first parent, whose
`results.csv` provably lacks that match).

## Scoreboard

| | pool points | % of max |
|---|---|---|
| **Our recommended tips** | **44 / 130** | **33.8%** |
| Naive most-likely tip | 34 / 130 | 26.2% |

2 exact hits · 3 correct-tendency · 8 misses. We beat the naive baseline by **+29% relative** — the
EV-optimisation is doing its job — but 33.8% is below the model's ~44–45% backtest norm. The gap is
**variance, not a broken model** (see below).

### Per-match (leak-free)

| match | model L/D/W | xG | tip | actual | pts | verdict |
|---|---|---|---|---|---|---|
| Mexico–South Africa | 80/15/5 | 2.4–0.5 | 2:0 | 2:0 | **10** | exact — confident & right |
| South Korea–Czechia | 38/28/34 | 1.4–1.3 | 2:1 | 2:1 | **10** | exact on a coin-flip — `realism_tolerance` win |
| Haiti–Scotland | 12/21/68 | 0.9–2.3 | 1:2 | 0:1 | 8 | strong away read, right margin |
| Sweden–Tunisia | 46/28/26 | 1.5–1.1 | 2:1 | 5:1 | 6 | tendency + away goal |
| Germany–Curaçao | 91/7/2 | 3.5–0.5 | 3:0 | 7:1 | 5 | tendency only (blowout margin unknowable) |
| Canada–Bosnia | 63/23/13 | 1.9–0.7 | 2:1 | 1:1 | 1 | draw upset |
| Brazil–Morocco | 50/32/18 | 1.3–0.7 | 1:0 | 1:1 | 1 | draw upset |
| United States–Paraguay | 27/32/42 | 0.9–1.2 | 0:1 | 4:1 | 1 | **host rated underdog** (see below) |
| Netherlands–Japan | 41/28/31 | 1.4–1.2 | 2:1 | 2:2 | 1 | draw upset (naive 1:1 would've scored 8) |
| Spain–Cape Verde | 94/5/1 | 3.7–0.4 | 4:0 | 0:0 | 1 | 94%-fav held to 0:0 — shock |
| Qatar–Switzerland | 2/7/91 | 0.6–3.6 | 0:4 | 1:1 | 0 | 91%-fav drew — shock |
| Australia–Türkiye | 21/27/52 | 1.0–1.7 | 1:2 | 2:0 | 0 | underdog-home win |
| Ivory Coast–Ecuador | 20/33/47 | 0.7–1.2 | 0:1 | 1:0 | 0 | underdog-home win |

## What went right

- **EV optimisation beats naive** (+10 points / +29% relative) — the value is in tipping
  EV-optimal scorelines, not the modal one.
- **`realism_tolerance=0.15` earned its keep here**: it nudged 1:0→2:1 and caught *two* exact 2:1s
  (Korea–Czechia, +9 over strict-EV's 1:0). On this batch the tolerance scored **44 vs strict-EV's
  42** (though on the 261-match benchmark set it costs ~0.7pp — see "non-changes").
- **The market blend behaved as designed**: it made **zero** difference to all 13 played tips
  (identical to pure Elo), because `divergence_threshold=0.15`/`market_weight=0.5` only nudge where
  model and market disagree. It's conservative, not broken.

## What went wrong — and why it's mostly variance

Eight misses, but the diagnosis matters more than the count:

1. **An abnormal draw cluster.** 5 of 13 (38%) were draws vs football's ~24% norm — including two
   shocks where the model was near-max confident (Spain 94%→0:0, Switzerland 91%→1:1). The EV
   optimiser tips a decisive scoreline whenever there's a favourite, so on a draw it banks ~1 point
   while a 1:1 tip would bank 8 (for a draw, tendency **and** goal-difference both pay, since the
   diff is always 0). The two max-confidence shocks alone cost ~18 points of upside.

2. **Underdog-home / form upsets.** USA (4:1), Australia (2:0) and Ivory Coast (1:0) all beat the
   side the model favoured. USA is the most instructive: pre-kickoff **eloratings.net rated Paraguay
   1834 vs host USA 1726** — a 108-point gap the `host_elo_bonus=40` couldn't close — so the model
   made a World Cup host a home underdog. They won 4:1.

## Recommended changes for the next round

### Do — operational (highest leverage, low risk)

1. **Refresh odds + Elo immediately before each matchday** (`update-tournament-data` skill). The
   live edge is entirely in fresh inputs: odds carry lineup/news the pre-tournament Elo can't, and
   the USA miss was a stale-rating story. The blend already wants to act — it changes 4 of the 59
   upcoming tips (e.g. USA–Australia 1:1→2:1, encoding USA's new form) — so it's only as good as the
   freshness of the odds it pools in.
2. **Re-run `scripts/retro_tips.py` after every matchday.** It closes the predict→score→learn loop
   leak-free; `verify` can't (it's for completed tournaments only). Track whether our %-of-max
   converges to the ~44% backtest norm as the draw variance washes out.

### Don't — the tempting tweaks are overfitting traps (verified on 261 benchmark matches)

3. **Do not boost draw-tipping / make `rho` more negative.** It's the obvious lesson from a
   38%-draw batch, but the model is **already well-calibrated on draws**: across 261 benchmark
   matches it predicts draws at **25.1%** vs an actual **23.8%** (reliability bins line up tightly).
   A "tip the draw when it's the single most-likely outcome" rule scores **identically** to current
   on the benchmarks; a "tip the draw when P(draw)≥0.30" rule scores **worse** (43.6% vs 44.5%). The
   matchday-1 draw cluster is variance.
4. **Do not raise `host_elo_bonus`.** Tempting after the USA miss, but host history is mixed — Qatar
   2022 (host) lost all three; a bigger host term would have hurt there. Mexico (also host) was
   tipped 2:0 *exact*. n=1 bad host match isn't a signal.

### Applied — `realism_tolerance` lowered 0.15 → 0.05

5. The old `0.15` sat at the **bottom** of the points curve. Sweeping it on the 261-match
   benchmark set (total pool points, %max):

   | realism_tol | %max | both-teams-score tip% | shutout tip% |
   |---|---|---|---|
   | 0.00 (strict EV) | 45.24 | 24.5 | 75.5 |
   | **0.05 (now)** | **45.33** | **38.3** | 61.7 |
   | 0.10 | 44.64 | 55.6 | 44.4 |
   | 0.15 (old) | 44.52 | 68.6 | 31.4 |
   | 0.20 | 44.76 | 77.8 | 22.2 |
   | 0.30 | 45.24 | 93.9 | 6.1 |

   **`0.05` is the sweet spot**: highest points (fractionally above strict EV) *and* lifts
   both-teams-score tips from 24.5% → 38.3% — i.e. most of the realism benefit at ~zero points
   cost. The old `0.15` paid ~0.8pp of pool points (~27 points over 261 matches) for extra
   realism past that. The points surface is flat/noisy (full range ~0.8pp) and the per-tournament
   delta vs `0.15` flips sign (3 of 5 favour `0.05`), so this is a **weak** points win, not a
   robust one — but `0.05` dominates `0.15` (same-or-better points, still realistic), so it's
   close to a free change. Reproduce: the sweep loop over `best_tip(dist, w, tol)`.

## Bottom line

The methodology is sound: it beats naive, it's well-calibrated, and the bad-looking 33.8% is a
draw-heavy, two-shock unlucky batch — not a model defect. The biggest real gains for round 2 are
**operational (data freshness)**, and the biggest risk is **overfitting to 13 matches** against the
evidence of 261. Hold the line on the model; keep the inputs fresh; re-score every matchday.
