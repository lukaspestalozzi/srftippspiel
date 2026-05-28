"""Reality check: do the predicted distributions + recommended tips look like real
tournament football?

For each completed-tournament match: predict its scoreline distribution, derive an EV-optimal
tip via ``best_tip``, score it against the actual result, and aggregate (per tournament + pooled):

- **Mean goals/match** — predicted ``E[H] + E[A]`` vs the actual mean.
- **Tendency split** — aggregated predicted ``P(H)/P(D)/P(A)`` vs actual W/D/L rates.
- **Top-5 scoreline frequencies** — predicted aggregate-matrix top cells vs actual histogram.
- **Tip composition** — recommended scoreline histogram (flags the "always 1:0" pathology).
- **Aggregate-matrix total variation distance** to the empirical scoreline frequency.

The ``verdict_of`` helper rolls per-metric drift thresholds into a single ``PASS|WARN|FAIL``,
shaped after the diagnostic anomaly block.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..model.scoreline import ScorelineDistribution
from ..strategy.expected_points import best_tip, score_tip

# Per-metric drift thresholds for the rolled-up verdict.
_WARN_GOALS_DELTA = 0.30
_FAIL_GOALS_DELTA = 0.60
_WARN_TENDENCY_DELTA = 0.06
_FAIL_TENDENCY_DELTA = 0.12
_WARN_MODAL_TIP_SHARE = 0.70
_FAIL_MODAL_TIP_SHARE = 0.85
_WARN_TVD = 0.35
_FAIL_TVD = 0.55


def _expected_goals(dist: ScorelineDistribution) -> tuple[float, float]:
    m = dist.matrix
    rng = np.arange(m.shape[0])
    return float((m.sum(axis=1) * rng).sum()), float((m.sum(axis=0) * rng).sum())


def _aggregate_matrix(dists: list[ScorelineDistribution]) -> np.ndarray:
    if not dists:
        return np.zeros((1, 1))
    gmax = max(d.gmax for d in dists)
    total = np.zeros((gmax + 1, gmax + 1))
    for d in dists:
        m = d.matrix
        total[: m.shape[0], : m.shape[1]] += m
    return total / len(dists)


def _actual_count_matrix(actuals: list[tuple[int, int]], gmax: int) -> np.ndarray:
    n = gmax + 1
    out = np.zeros((n, n))
    for h, a in actuals:
        out[min(h, gmax), min(a, gmax)] += 1
    return out


def _tvd(pred_matrix: np.ndarray, actual_counts: np.ndarray) -> float:
    actual_freq = actual_counts / max(1.0, actual_counts.sum())
    h = max(pred_matrix.shape[0], actual_freq.shape[0])
    w = max(pred_matrix.shape[1], actual_freq.shape[1])
    p = np.zeros((h, w))
    p[: pred_matrix.shape[0], : pred_matrix.shape[1]] = pred_matrix
    a = np.zeros((h, w))
    a[: actual_freq.shape[0], : actual_freq.shape[1]] = actual_freq
    return 0.5 * float(np.abs(p - a).sum())


def _topk_pred(matrix: np.ndarray, k: int = 5) -> list[dict]:
    flat = matrix.ravel()
    idx = np.argsort(flat)[::-1][:k]
    cols = matrix.shape[1]
    return [{"home": int(i // cols), "away": int(i % cols), "prob": float(flat[i])} for i in idx]


def _topk_actual(actuals: list[tuple[int, int]], k: int = 5) -> list[dict]:
    c = Counter(actuals)
    return [{"home": h, "away": a, "count": n} for (h, a), n in c.most_common(k)]


def reality_check_one(bundle, teams, fixtures, results, predictor) -> dict:
    """Per-tournament realism: predicted vs actual distributions, tip composition, etc."""
    by_id = {m.match_id: m for m in fixtures}
    dists: list[ScorelineDistribution] = []
    tips: list[tuple[int, int]] = []
    actuals: list[tuple[int, int]] = []
    pool_model = pool_max = exact_hits = tendency_hits = 0
    e_h_sum = e_a_sum = 0.0
    p_hw = p_dr = p_aw = 0.0
    n = 0

    for mid, actual in results.items():
        match = by_id.get(mid)
        if match is None or not match.participants_known:
            continue
        weight = match.stage.points_weight
        dist = predictor.predict(match, teams).scoreline
        dists.append(dist)
        e_h, e_a = _expected_goals(dist)
        e_h_sum += e_h
        e_a_sum += e_a
        p_hw += dist.p_home_win()
        p_dr += dist.p_draw()
        p_aw += dist.p_away_win()
        th, ta, _ = best_tip(dist, weight)
        tips.append((th, ta))
        actuals.append((actual.home_goals, actual.away_goals))
        pool_model += score_tip(th, ta, actual.home_goals, actual.away_goals, weight)
        pool_max += 10 * weight
        if (th, ta) == (actual.home_goals, actual.away_goals):
            exact_hits += 1
        sign_t = (th > ta) - (th < ta)
        sign_a = ((actual.home_goals > actual.away_goals)
                  - (actual.home_goals < actual.away_goals))
        if sign_t == sign_a:
            tendency_hits += 1
        n += 1

    if n == 0:
        return {"tournament": bundle.name, "matches": 0}

    a_h_wins = sum(1 for (h, a) in actuals if h > a)
    a_draws = sum(1 for (h, a) in actuals if h == a)
    a_a_wins = sum(1 for (h, a) in actuals if h < a)
    a_h_goals = sum(h for (h, _) in actuals)
    a_a_goals = sum(a for (_, a) in actuals)

    agg = _aggregate_matrix(dists)
    actual_counts = _actual_count_matrix(actuals, agg.shape[0] - 1)
    tip_hist = Counter(tips)
    modal_tip, modal_n = tip_hist.most_common(1)[0]
    modal_share = modal_n / n

    pred_total_goals = (e_h_sum + e_a_sum) / n
    actual_total_goals = (a_h_goals + a_a_goals) / n
    tend = {"predicted": {"H": p_hw / n, "D": p_dr / n, "A": p_aw / n},
            "actual": {"H": a_h_wins / n, "D": a_draws / n, "A": a_a_wins / n}}

    return {
        "tournament": bundle.name,
        "display_name": bundle.display_name,
        "matches": n,
        "mean_goals": {
            "predicted_home": e_h_sum / n,
            "predicted_away": e_a_sum / n,
            "predicted_total": pred_total_goals,
            "actual_home": a_h_goals / n,
            "actual_away": a_a_goals / n,
            "actual_total": actual_total_goals,
            "delta_total": pred_total_goals - actual_total_goals,
        },
        "tendency_split": tend,
        "tendency_delta_max": max(
            abs(tend["predicted"][k] - tend["actual"][k]) for k in ("H", "D", "A")
        ),
        "top5_scorelines": {
            "predicted": _topk_pred(agg, 5),
            "actual": _topk_actual(actuals, 5),
        },
        "scoreline_tvd": _tvd(agg, actual_counts),
        "tip_composition": {
            "top": [{"home": h, "away": a, "n": v} for (h, a), v in tip_hist.most_common(5)],
            "modal_tip": {"home": modal_tip[0], "away": modal_tip[1], "share": modal_share},
            "exact_hits": exact_hits,
            "tendency_hits": tendency_hits,
            "exact_hit_rate": exact_hits / n,
            "tendency_hit_rate": tendency_hits / n,
        },
        "pool_points": {
            "model": pool_model, "max": pool_max,
            "pct_max": 100.0 * pool_model / pool_max if pool_max else 0.0,
        },
    }


def reality_pooled(per_tournament: list[dict]) -> dict:
    """Aggregate per-tournament realism into one matches-weighted pooled view."""
    rows = [p for p in per_tournament if p.get("matches")]
    n = sum(p["matches"] for p in rows)
    if n == 0:
        return {"matches": 0}

    def w(path: list[str]) -> float:
        total = 0.0
        for p in rows:
            v = p
            for k in path:
                v = v[k]
            total += float(v) * p["matches"]
        return total / n

    pred_total = w(["mean_goals", "predicted_total"])
    actual_total = w(["mean_goals", "actual_total"])
    tend_pred = {k: w(["tendency_split", "predicted", k]) for k in ("H", "D", "A")}
    tend_act = {k: w(["tendency_split", "actual", k]) for k in ("H", "D", "A")}
    modal_share = w(["tip_composition", "modal_tip", "share"])
    exact_hits = sum(p["tip_composition"]["exact_hits"] for p in rows)
    tendency_hits = sum(p["tip_composition"]["tendency_hits"] for p in rows)
    pool_model = sum(p["pool_points"]["model"] for p in rows)
    pool_max = sum(p["pool_points"]["max"] for p in rows)

    return {
        "matches": n,
        "mean_goals": {
            "predicted_total": pred_total, "actual_total": actual_total,
            "delta_total": pred_total - actual_total,
        },
        "tendency_split": {"predicted": tend_pred, "actual": tend_act},
        "tendency_delta_max": max(abs(tend_pred[k] - tend_act[k]) for k in ("H", "D", "A")),
        "tip_composition": {
            "modal_share_weighted": modal_share,
            "exact_hits": exact_hits, "tendency_hits": tendency_hits,
            "exact_hit_rate": exact_hits / n, "tendency_hit_rate": tendency_hits / n,
        },
        "scoreline_tvd_weighted": w(["scoreline_tvd"]),
        "pool_points": {
            "model": pool_model, "max": pool_max,
            "pct_max": 100.0 * pool_model / pool_max if pool_max else 0.0,
        },
    }


def verdict_of(pooled: dict) -> dict:
    """Roll the per-metric drifts into a single PASS|WARN|FAIL with a reason list."""
    if not pooled.get("matches"):
        return {"status": "PASS", "reasons": ["no matches scored"]}

    reasons: list[str] = []
    status = "PASS"

    def bump(new: str) -> None:
        nonlocal status
        order = {"PASS": 0, "WARN": 1, "FAIL": 2}
        if order[new] > order[status]:
            status = new

    g = abs(pooled["mean_goals"]["delta_total"])
    if g > _FAIL_GOALS_DELTA:
        bump("FAIL"); reasons.append(f"mean goals/match off by {g:.2f} (FAIL > {_FAIL_GOALS_DELTA})")
    elif g > _WARN_GOALS_DELTA:
        bump("WARN"); reasons.append(f"mean goals/match off by {g:.2f} (WARN > {_WARN_GOALS_DELTA})")

    t = pooled["tendency_delta_max"]
    if t > _FAIL_TENDENCY_DELTA:
        bump("FAIL"); reasons.append(f"tendency split off by {t:.2f} (FAIL > {_FAIL_TENDENCY_DELTA})")
    elif t > _WARN_TENDENCY_DELTA:
        bump("WARN"); reasons.append(f"tendency split off by {t:.2f} (WARN > {_WARN_TENDENCY_DELTA})")

    ms = pooled["tip_composition"]["modal_share_weighted"]
    if ms > _FAIL_MODAL_TIP_SHARE:
        bump("FAIL"); reasons.append(f"modal tip share {ms:.0%} (FAIL > {_FAIL_MODAL_TIP_SHARE:.0%})")
    elif ms > _WARN_MODAL_TIP_SHARE:
        bump("WARN"); reasons.append(f"modal tip share {ms:.0%} (WARN > {_WARN_MODAL_TIP_SHARE:.0%})")

    tvd = pooled["scoreline_tvd_weighted"]
    if tvd > _FAIL_TVD:
        bump("FAIL"); reasons.append(f"aggregate-matrix TVD {tvd:.2f} (FAIL > {_FAIL_TVD})")
    elif tvd > _WARN_TVD:
        bump("WARN"); reasons.append(f"aggregate-matrix TVD {tvd:.2f} (WARN > {_WARN_TVD})")

    if not reasons:
        reasons.append("all per-metric drifts within thresholds")
    return {"status": status, "reasons": reasons}
