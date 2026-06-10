"""Parameter tuning: sweep EloPoisson params against the completed-tournament backtests.

Ranks candidate parameter sets by calibration (mean RPS over the pooled matches), tie-broken
by pool-points % of max — the "blended" objective: calibration first, points as the
tie-break. Reuses ``build_verification`` for per-tournament scoring + calibration, aggregates
across all benchmark tournaments, and reports a leaderboard, the recommended set, and a
leave-one-tournament-out generalisation check. Writes ``output/tune.{md,json}``.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from ..predictors.elo_poisson import EloPoissonPredictor
from ..predictors.market_odds import MarketOddsPredictor
from .backtest import build_verification
from .diagnostics import _fixed_table, _json_default

# Coarse default sweep grid. gmax is fixed (truncation, not a behaviour lever). ``alpha`` is
# the off/def goal-volume weight; it needs att_elo/def_elo populated in the benchmark teams.csv
# (run `tippspiel fit-offdef` per tournament) to have any effect — at alpha 0 it is a no-op.
# Trimmed from the pre-alpha grid (dropped the rarely-optimal extremes) so adding the alpha
# axis keeps the sweep near its original runtime; the currently-tuned set stays reachable.
DEFAULT_GRID: dict[str, list] = {
    "mu": [2.4, 2.6, 2.8, 3.0],
    "k": [0.0015, 0.0020, 0.0025],
    "rho": [-0.10, -0.05, 0.0],
    "host_elo_bonus": [0, 40, 80],
    "ko_goal_scale": [1.1, 1.2, 1.33],
    "alpha": [0.0, 0.5, 0.7],
}
_GMAX = 7
_TUNED_KEYS = ("mu", "k", "rho", "host_elo_bonus", "ko_goal_scale", "alpha")


def build_market_grid(base_params: dict | None = None) -> dict:
    """Grid for ``tune --market``: sweep the model x market blend, Elo params pinned.

    The Elo axes are single-element lists at the base config's tuned values (sweeping them
    jointly with the blend would square the runtime for little gain — they are already tuned
    on the same benchmarks); the swept axes are the blend weight, the expansion's assumed
    total-goals level, and whether the expansion matches the de-vigged draw price.
    """
    p = dict(base_params or {})
    # A market_odds base config keeps its Elo params under fallback_params; lift them.
    for key, val in (p.pop("fallback_params", {}) or {}).items():
        p.setdefault(key, val)
    defaults = _default_params(None)
    grid: dict[str, list] = {k: [p.get(k, defaults[k])] for k in _TUNED_KEYS}
    grid.update({
        "market_weight": [0.0, 0.25, 0.5, 0.75, 1.0],
        "total_goals": [2.4, 2.6, 2.8],
        "match_draw": [False, True],
        # Targeted blend: 0 = blend every odds-backed fixture; >0 keeps the pure model unless
        # the model-vs-market 1X2 gap exceeds the threshold on some outcome.
        "divergence_threshold": [0.0, 0.25],
    })
    return grid


def _iter_grid(grid: dict[str, list]):
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def _default_params(base_cfg) -> dict:
    p = getattr(base_cfg.predictor, "params", {}) if base_cfg else {}
    return {
        "mu": p.get("mu", 2.6),
        "k": p.get("k", 0.0015),
        "rho": p.get("rho", 0.0),
        "host_elo_bonus": p.get("host_elo_bonus", 0.0),
        "ko_goal_scale": p.get("ko_goal_scale", 1.0),
        "alpha": p.get("alpha", 0.0),
    }


def _predictor_for(params: dict, odds: dict):
    """Build the predictor for one param set: pure Elo, or the model x market blend.

    A param set carrying ``market_weight`` selects the blended ``MarketOddsPredictor`` (with
    the Elo keys feeding its fallback); without it the historical pure-Elo path is unchanged.
    The predictor is odds-dependent, so it is built per benchmark.
    """
    p = dict(params)
    market_weight = p.pop("market_weight", None)
    total_goals = p.pop("total_goals", 2.6)
    match_draw = p.pop("match_draw", False)
    divergence_threshold = p.pop("divergence_threshold", 0.0)
    fallback = EloPoissonPredictor(gmax=_GMAX, **p)
    if market_weight is None:
        return fallback
    return MarketOddsPredictor(
        odds=odds, fallback=fallback, total_goals=total_goals, gmax=_GMAX,
        ko_goal_scale=p.get("ko_goal_scale", 1.0),
        market_weight=market_weight, match_draw=match_draw,
        divergence_threshold=divergence_threshold,
    )


def _evaluate(params: dict, benchmarks: list) -> dict:
    """Aggregate pool points + calibration across all benchmark tournaments for one param set.

    ``benchmarks``: list of ``(bundle, teams, fixtures, results, odds)``. A benchmark without
    an odds snapshot contributes pure-Elo metrics to a market sweep (the blend degrades to
    the fallback on every match).
    """
    n = model = naive = mx = exact = 0
    rps_sum = nll_sum = 0.0
    per_tournament: dict[str, dict] = {}
    for bundle, teams, fixtures, results, odds in benchmarks:
        predictor = _predictor_for(params, odds)
        # realism_tolerance is fixed at 0 here: it only shifts pool points (never RPS/NLL), so a
        # swept value would be driven to 0 by the RPS-primary objective. It's set separately.
        _md, data = build_verification(bundle, teams, fixtures, results, predictor,
                                       realism_tolerance=0.0)
        s = data["summary"]["all"]
        c = data["calibration"]["all"]
        n += s["matches"]
        model += s["model"]
        naive += s["naive"]
        mx += s["max"]
        exact += s["exact_hits"]
        rps_sum += c["mean_rps"] * c["matches"]
        nll_sum += c["mean_nll"] * c["matches"]
        per_tournament[bundle.name] = {
            "matches": s["matches"],
            "model": s["model"],
            "naive": s["naive"],
            "max": s["max"],
            "model_pct": (100.0 * s["model"] / s["max"]) if s["max"] else 0.0,
            "mean_rps": c["mean_rps"],
            "mean_nll": c["mean_nll"],
        }
    return {
        "params": params,
        "matches": n,
        "model": model,
        "naive": naive,
        "max": mx,
        "exact_hits": exact,
        "model_pct": (100.0 * model / mx) if mx else 0.0,
        "mean_rps": rps_sum / n if n else 0.0,
        "mean_nll": nll_sum / n if n else 0.0,
        "per_tournament": per_tournament,
    }


def _blended_key(result: dict):
    """Calibration-primary, pool-points tie-break: lower RPS first, then higher points %."""
    return (round(result["mean_rps"], 4), -result["model_pct"])


def _leave_one_out(results: list, names: list[str]) -> dict:
    """For each tournament, pick the best params on the *other* tournaments and report how it
    scores on the held-out one — a guard against overfitting only three events."""
    out: dict[str, dict] = {}
    for held in names:
        others = [t for t in names if t != held]

        def train_key(r, others=others):
            tot_m = sum(r["per_tournament"][t]["matches"] for t in others)
            rps = sum(r["per_tournament"][t]["mean_rps"] * r["per_tournament"][t]["matches"]
                      for t in others) / tot_m if tot_m else 0.0
            tot_max = sum(r["per_tournament"][t]["max"] for t in others)
            pct = (100.0 * sum(r["per_tournament"][t]["model"] for t in others) / tot_max
                   if tot_max else 0.0)
            return (round(rps, 4), -pct)

        chosen = min(results, key=train_key)
        held_metrics = chosen["per_tournament"][held]
        out[held] = {
            "chosen_params": chosen["params"],
            "heldout_mean_rps": held_metrics["mean_rps"],
            "heldout_model_pct": held_metrics["model_pct"],
        }
    return out


def build_tuning(base_cfg, benchmarks: list, grid: dict | None = None,
                 top: int = 15) -> tuple[str, dict]:
    grid = grid or DEFAULT_GRID
    candidates = list(_iter_grid(grid))
    default = _default_params(base_cfg)
    if default not in candidates:
        candidates.append(default)

    results = [_evaluate(p, benchmarks) for p in candidates]
    ranked = sorted(results, key=_blended_key)
    recommended = ranked[0]
    default_result = next(r for r in results if r["params"] == default)
    names = [b[0].name for b in benchmarks]

    data = {
        "benchmarks": names,
        "grid_size": len(candidates),
        "default_params": default,
        "default_metrics": _metrics_only(default_result),
        "recommended_params": recommended["params"],
        "recommended_metrics": _metrics_only(recommended),
        "recommended_per_tournament": recommended["per_tournament"],
        "leaderboard": [_row(r) for r in ranked[:top]],
        "leave_one_out": _leave_one_out(results, names),
    }
    return _render(data), data


def _metrics_only(r: dict) -> dict:
    return {k: r[k] for k in ("matches", "model", "naive", "max", "model_pct",
                              "exact_hits", "mean_rps", "mean_nll")}


def _row(r: dict) -> dict:
    return {"params": r["params"], **_metrics_only(r)}


def _fmt_params(p: dict) -> str:
    s = (f"mu{p.get('mu', 2.6)} k{p.get('k', 0.0015)} rho{p.get('rho', 0.0)} "
         f"host{int(p.get('host_elo_bonus', 0))} ko{p.get('ko_goal_scale', 1.0)} "
         f"a{p.get('alpha', 0.0)}")
    if p.get("market_weight") is not None:
        # Market-blend axes; absent on pure-Elo rows of a mixed leaderboard.
        s += f" mw{p['market_weight']} tg{p['total_goals']} dr{int(bool(p['match_draw']))}"
        if p.get("divergence_threshold"):
            s += f" dt{p['divergence_threshold']}"
    return s


def _render(data: dict) -> str:
    L = ["# Parameter tuning — blended (calibration-primary, pool-points tie-break)", ""]
    L.append(f"Benchmarks: {', '.join(data['benchmarks'])}. "
             f"Swept {data['grid_size']} parameter sets; ranked by mean RPS (lower better), "
             f"tie-broken by model pool-points % of max.")
    L.append("")
    d, r = data["default_metrics"], data["recommended_metrics"]
    L.append("## Default vs recommended")
    L.append(_fixed_table(
        ["set", "params", "mean RPS", "mean NLL", "model", "%max", "exact"],
        [
            ["default", _fmt_params(data["default_params"]), f"{d['mean_rps']:.4f}",
             f"{d['mean_nll']:.4f}", d["model"], f"{d['model_pct']:.1f}", d["exact_hits"]],
            ["recommend", _fmt_params(data["recommended_params"]), f"{r['mean_rps']:.4f}",
             f"{r['mean_nll']:.4f}", r["model"], f"{r['model_pct']:.1f}", r["exact_hits"]],
        ],
    ))
    L.append("")
    L.append("## Recommended — per tournament")
    prows = [[name, m["matches"], f"{m['mean_rps']:.4f}", m["model"], m["naive"], m["max"],
              f"{m['model_pct']:.1f}"]
             for name, m in data["recommended_per_tournament"].items()]
    L.append(_fixed_table(["tournament", "matches", "mean RPS", "model", "naive", "max", "%max"],
                          prows))
    L.append("")
    L.append("## Leaderboard (top by blended rank)")
    lrows = [[i + 1, _fmt_params(row["params"]), f"{row['mean_rps']:.4f}",
              f"{row['mean_nll']:.4f}", row["model"], f"{row['model_pct']:.1f}", row["exact_hits"]]
             for i, row in enumerate(data["leaderboard"])]
    L.append(_fixed_table(
        ["#", "params", "mean RPS", "mean NLL", "model", "%max", "exact"], lrows))
    L.append("")
    L.append("## Leave-one-tournament-out (generalisation check)")
    L.append("Params chosen on the other two tournaments, scored on the held-out one.")
    orows = [[held, _fmt_params(v["chosen_params"]), f"{v['heldout_mean_rps']:.4f}",
              f"{v['heldout_model_pct']:.1f}"]
             for held, v in data["leave_one_out"].items()]
    L.append(_fixed_table(["held-out", "chosen params", "heldout RPS", "heldout %max"], orows))
    L.append("")
    return "\n".join(L)


class TuningWriter:
    def write(self, markdown: str, data: dict, output_dir: str | Path) -> dict[str, Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / "tune.md"
        json_path = out_dir / "tune.json"
        md_path.write_text(markdown, encoding="utf-8")
        json_path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
        return {"markdown": md_path, "json": json_path}
