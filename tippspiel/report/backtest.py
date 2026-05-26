"""Historical verification: how many pool points would the model's tips have scored?

For a completed tournament, predict every actual match from the pre-tournament Elo snapshot
(no result conditioning — the honest "fill in your tip sheet up front" test), pick the
EV-maximising tip, and score it against the real result under the pool rules. The naive
most-likely scoreline is scored too as a baseline, and the per-match maximum (an exact hit,
10*W) bounds the achievable total. Writes output/verify.md (+ verify.json).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..model.scoreline import ScorelineDistribution
from ..model.stages import PTS_EXACT
from ..strategy.expected_points import best_tip, expected_points, score_tip
from .diagnostics import _fixed_table, _json_default

_NLL_EPS = 1e-12


def _tendency_rps(dist: ScorelineDistribution, actual_h: int, actual_a: int) -> float:
    """Ranked probability score over the ordered outcomes (home win, draw, away win).

    RPS = 0.5 * sum over the 2 cumulative thresholds of (cum_pred - cum_obs)^2; 0 is perfect,
    1 is worst. The standard, outlier-robust football tendency-calibration metric.
    """
    preds = [dist.p_home_win(), dist.p_draw(), dist.p_away_win()]
    obs_idx = 0 if actual_h > actual_a else (1 if actual_h == actual_a else 2)
    obs = [1.0 if i == obs_idx else 0.0 for i in range(3)]
    cp = co = 0.0
    total = 0.0
    for i in range(2):  # 2 cumulative thresholds for 3 ordered categories
        cp += preds[i]
        co += obs[i]
        total += (cp - co) ** 2
    return 0.5 * total


def _scoreline_nll(dist: ScorelineDistribution, actual_h: int, actual_a: int) -> float:
    """Negative log-likelihood of the actual scoreline (goals clamped to the grid)."""
    h = min(actual_h, dist.gmax)
    a = min(actual_a, dist.gmax)
    return -math.log(max(dist.cell(h, a), _NLL_EPS))


def build_verification(bundle, teams, fixtures, results, predictor) -> tuple[str, dict]:
    by_id = {m.match_id: m for m in fixtures}
    records: list[dict] = []
    for mid, actual in results.items():
        match = by_id.get(mid)
        if match is None or not match.participants_known:
            continue
        weight = match.stage.points_weight
        dist = predictor.predict(match, teams).scoreline
        mh, ma, _ = best_tip(dist, weight)
        nh, na, _ = dist.most_likely_scorelines(1)[0]
        model_pts = score_tip(mh, ma, actual.home_goals, actual.away_goals, weight)
        naive_pts = score_tip(nh, na, actual.home_goals, actual.away_goals, weight)
        max_pts = PTS_EXACT * weight
        records.append({
            "match_id": mid,
            "stage": match.stage.value,
            "weight": weight,
            "home": match.home.team_id,
            "away": match.away.team_id,
            "actual": (actual.home_goals, actual.away_goals),
            "model_tip": (mh, ma),
            "model_pts": model_pts,
            "naive_tip": (nh, na),
            "naive_pts": naive_pts,
            "max_pts": max_pts,
            "exact_hit": (mh, ma) == (actual.home_goals, actual.away_goals),
            "rps": _tendency_rps(dist, actual.home_goals, actual.away_goals),
            "nll": _scoreline_nll(dist, actual.home_goals, actual.away_goals),
        })

    by_stage_kind = {"group": [], "knockout": []}
    for r in records:
        by_stage_kind["group" if r["weight"] == 1 else "knockout"].append(r)

    def totals(rows):
        return {
            "matches": len(rows),
            "model": sum(r["model_pts"] for r in rows),
            "naive": sum(r["naive_pts"] for r in rows),
            "max": sum(r["max_pts"] for r in rows),
            "exact_hits": sum(1 for r in rows if r["exact_hit"]),
        }

    def calib(rows):
        n = len(rows)
        if n == 0:
            return {"matches": 0, "mean_rps": 0.0, "mean_nll": 0.0}
        return {
            "matches": n,
            "mean_rps": sum(r["rps"] for r in rows) / n,
            "mean_nll": sum(r["nll"] for r in rows) / n,
        }

    summary = {
        "all": totals(records),
        "group": totals(by_stage_kind["group"]),
        "knockout": totals(by_stage_kind["knockout"]),
    }
    calibration = {
        "all": calib(records),
        "group": calib(by_stage_kind["group"]),
        "knockout": calib(by_stage_kind["knockout"]),
    }
    data = {
        "tournament": bundle.display_name,
        "predictor": predictor.name,
        "predictor_params": getattr(predictor, "params", {}),
        "elo_source": bundle.elo_source,
        "summary": summary,
        "calibration": calibration,
        "matches": records,
    }
    return _render(bundle, data), data


def _pct(num, den) -> str:
    return f"{(100.0 * num / den):.1f}%" if den else "n/a"


def _render(bundle, data) -> str:
    s = data["summary"]
    L = [f"# Verification backtest — {data['tournament']}", ""]
    L.append(f"Predictor `{data['predictor']}` {json.dumps(data['predictor_params'])}; "
             f"tips made a-priori from the pre-tournament Elo snapshot ({data['elo_source']}).")
    L.append("Pool points the recommended (EV) tips would have scored vs the naive "
             "most-likely-scoreline tips and the per-match maximum (exact = 10x weight).")
    L.append("")
    rows = []
    for key in ("all", "group", "knockout"):
        t = s[key]
        rows.append([
            key, t["matches"], t["model"], t["naive"], t["max"],
            _pct(t["model"], t["max"]), f"{t['model'] - t['naive']:+d}",
            f"{t['exact_hits']}/{t['matches']}",
        ])
    L.append(_fixed_table(
        ["split", "matches", "model", "naive", "max", "model %max", "vs naive", "exact hits"],
        rows,
    ))
    L.append("")
    cal = data.get("calibration", {})
    if cal:
        L.append("## Calibration")
        L.append("Lower is better. RPS = ranked-probability score on the tendency (0 perfect, "
                 "1 worst); NLL = mean negative log-likelihood of the actual scoreline.")
        crows = [[key, cal[key]["matches"], f"{cal[key]['mean_rps']:.4f}",
                  f"{cal[key]['mean_nll']:.4f}"] for key in ("all", "group", "knockout")]
        L.append(_fixed_table(["split", "matches", "mean RPS", "mean NLL"], crows))
        L.append("")
    L.append("## Per-match")
    L.append("pts columns are model / naive / max pool points for that match.")
    mrows = []
    for r in data["matches"]:
        mrows.append([
            r["match_id"], r["stage"], f"{r['home']}-{r['away']}",
            f"{r['actual'][0]}:{r['actual'][1]}",
            f"{r['model_tip'][0]}:{r['model_tip'][1]}",
            f"{r['naive_tip'][0]}:{r['naive_tip'][1]}",
            f"{r['model_pts']}/{r['naive_pts']}/{r['max_pts']}",
            "yes" if r["exact_hit"] else "",
        ])
    L.append(_fixed_table(
        ["match", "stage", "tie", "actual", "model", "naive", "pts m/n/max", "exact"],
        mrows,
    ))
    L.append("")
    return "\n".join(L)


class VerificationWriter:
    def write(self, markdown: str, data: dict, output_dir: str | Path) -> dict[str, Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / "verify.md"
        json_path = out_dir / "verify.json"
        md_path.write_text(markdown, encoding="utf-8")
        json_path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
        return {"markdown": md_path, "json": json_path}
