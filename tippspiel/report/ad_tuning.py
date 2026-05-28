"""Staged tuning of the attack/defence model.

Two parameter groups act at different layers:
- **Generation** (the ``elo:`` config block, model = ``attack_defence``): controls the
  forward-pass that produces per-team ``(attack, defence)`` ratings (the values that ride
  on ``Team.attack`` / ``Team.defence`` in ``teams_attack_defence.csv``).
- **Predictor** (the ``predictors.attack_defence_poisson`` block): how the predictor turns
  those ratings into a scoreline distribution per match.

A flat joint grid would multiply cost prohibitively (each gen-point requires a full forward
pass over ~25y of international results). So we sweep them in two stages, caching the
synthesised per-tournament ``Team`` dicts so Stage 2 doesn't redo any forward passes.

Output structure mirrors ``tuning.py`` plus a Stage 1 + Stage 2 + combined block + a
``reality_check`` against the actual completed tournaments. Markdown rendered with the same
fixed-width tables.
"""

from __future__ import annotations

import csv
import dataclasses
import itertools
from datetime import date
from pathlib import Path

from ..elo import (
    build_model,
    get_results_csv,
    parse_csv_text,
    prepare_matches,
    run_forward_pass,
)
from ..elo.config import EloConfig, load_elo_config
from ..elo.names import normalize
from ..model.types import Team
from ..predictors.attack_defence import AttackDefencePoissonPredictor
from .backtest import build_verification
from .diagnostics import _fixed_table
from .realism import reality_check_one, reality_pooled, verdict_of
from .tuning import _blended_key

# Generation knobs that act before the predictor — they shape the (attack, defence) ratings.
DEFAULT_GENERATION_GRID: dict[str, list] = {
    "learning_rate": [0.01, 0.03, 0.06],
    "lookback_years": [100, 200],
    "recency_decay": [False, True],
    "ad_home_advantage": [0.0, 0.1],
}

# Predictor knobs (consumed by AttackDefencePoissonPredictor.predict).
DEFAULT_PREDICTOR_GRID: dict[str, list] = {
    "base_log_rate": [0.2, 0.3, 0.4],
    "home_advantage": [0.0, 0.1, 0.2],
    "rho": [-0.15, -0.10, -0.05, 0.0],
    "ko_goal_scale": [1.0, 1.2],
}

_GMAX = 7
_GEN_KEYS = tuple(DEFAULT_GENERATION_GRID.keys())
_PRED_KEYS = tuple(DEFAULT_PREDICTOR_GRID.keys())


# ---------------------------------------------------------------------------- helpers

def _iter_grid(grid: dict[str, list]):
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def _freeze(d: dict) -> tuple:
    return tuple(sorted(d.items()))


def _default_predictor_params(base_cfg) -> dict:
    p: dict = {}
    if base_cfg is not None:
        entry = getattr(base_cfg, "predictors", {}).get("attack_defence_poisson")
        if entry is not None:
            p = entry.params
    return {
        "base_log_rate": p.get("base_log_rate", 0.3),
        "home_advantage": p.get("home_advantage", 0.0),
        "rho": p.get("rho", -0.1),
        "ko_goal_scale": p.get("ko_goal_scale", 1.2),
    }


def _default_generation_params(base_elo_cfg: EloConfig) -> dict:
    return {
        "learning_rate": base_elo_cfg.learning_rate,
        "lookback_years": base_elo_cfg.lookback_years,
        "recency_decay": base_elo_cfg.recency_decay,
        "ad_home_advantage": base_elo_cfg.ad_home_advantage,
    }


def _build_teams_for_gen(
    bundle, gen_params: dict, base_elo_cfg: EloConfig,
    as_of: date, all_matches,
) -> tuple[dict[str, Team], dict]:
    """Run forward pass with these gen params; synthesise per-tournament ``teams`` dict
    preserving the official ``Team.elo`` from the tournament's own teams.csv."""
    cfg = dataclasses.replace(base_elo_cfg, model="attack_defence", **gen_params)
    matches = prepare_matches(all_matches, as_of, cfg)
    model = run_forward_pass(matches, build_model(cfg))
    pairs = model.attack_defence_ratings()  # {normalized_name: (atk, def)}

    teams: dict[str, Team] = {}
    coverage = 0
    with Path(bundle.teams_file).open(newline="") as fh:
        for row in csv.DictReader(fh):
            tid = (row.get("team_id") or "").strip()
            if not tid:
                continue
            name = row["name"]
            elo = float(row.get("elo") or 0.0)
            ad = pairs.get(normalize(name))
            attack = ad[0] if ad else None
            defence = ad[1] if ad else None
            if ad is not None:
                coverage += 1
            teams[tid] = Team(
                team_id=tid, name=name, elo=elo, elo_trend=None,
                attack=attack, defence=defence,
            )
    return teams, {"coverage": coverage, "total": len(teams)}


def _aggregate(records: list[dict]) -> dict:
    """Aggregate per-tournament verification summaries into one combined result."""
    n = model = naive = mx = exact = 0
    rps_sum = nll_sum = 0.0
    for r in records:
        n += r["matches"]
        model += r["model"]
        naive += r["naive"]
        mx += r["max"]
        exact += r["exact_hits"]
        rps_sum += r["mean_rps"] * r["matches"]
        nll_sum += r["mean_nll"] * r["matches"]
    return {
        "matches": n, "model": model, "naive": naive, "max": mx, "exact_hits": exact,
        "model_pct": (100.0 * model / mx) if mx else 0.0,
        "mean_rps": rps_sum / n if n else 0.0,
        "mean_nll": nll_sum / n if n else 0.0,
    }


def _per_tournament_stats(bundle_name: str, s: dict, c: dict) -> dict:
    return {
        "tournament": bundle_name,
        "matches": s["matches"],
        "model": s["model"], "naive": s["naive"], "max": s["max"],
        "exact_hits": s["exact_hits"],
        "model_pct": (100.0 * s["model"] / s["max"]) if s["max"] else 0.0,
        "mean_rps": c["mean_rps"], "mean_nll": c["mean_nll"],
    }


def _evaluate(
    pred_params: dict, gen_params: dict,
    benchmarks_with_teams: list[tuple],  # [(bundle, teams, fixtures, results)]
) -> dict:
    """Run build_verification across all benchmarks for one (pred, gen) point. Returns the
    aggregated metrics + per-tournament rows + the params, ready for ranking by ``_blended_key``.
    """
    predictor = AttackDefencePoissonPredictor(gmax=_GMAX, **pred_params)
    per_tournament: dict[str, dict] = {}
    rows: list[dict] = []
    for bundle, teams, fixtures, results in benchmarks_with_teams:
        _md, data = build_verification(bundle, teams, fixtures, results, predictor)
        s = data["summary"]["all"]
        c = data["calibration"]["all"]
        stats = _per_tournament_stats(bundle.name, s, c)
        per_tournament[bundle.name] = stats
        rows.append(stats)
    agg = _aggregate(rows)
    return {
        "params": dict(pred_params),
        "gen_params": dict(gen_params),
        "per_tournament": per_tournament,
        **agg,
    }


def _stage_default_metrics(result: dict) -> dict:
    return {k: result[k] for k in
            ("matches", "model", "naive", "max", "model_pct", "exact_hits",
             "mean_rps", "mean_nll")}


# ---------------------------------------------------------------------------- main

def build_ad_tuning(
    base_cfg, benchmarks: list, *,
    predictor_grid: dict | None = None,
    generation_grid: dict | None = None,
    top: int = 15,
    historical_text: str | None = None,
) -> tuple[str, dict]:
    """Stage-1 gen sweep + Stage-2 predictor sweep + combined recommendation + realism.

    ``benchmarks``: list of ``(bundle, fixtures, results, as_of)`` (no ``teams`` — those
    are synthesised per gen-params in Stage 1 and cached for reuse in Stage 2).
    """
    predictor_grid = predictor_grid or DEFAULT_PREDICTOR_GRID
    generation_grid = generation_grid or DEFAULT_GENERATION_GRID

    # The elo: block lives on the TournamentBundle (engine Config has no elo block).
    # Convention is identical settings across tournaments; use the first benchmark's, falling
    # back to library defaults. Each gen-grid point overrides via dataclasses.replace.
    first_bundle = benchmarks[0][0] if benchmarks else None
    base_elo_cfg = (first_bundle.elo if first_bundle and first_bundle.elo
                    else load_elo_config({}))

    text = historical_text if historical_text is not None else get_results_csv(base_elo_cfg)
    all_matches = parse_csv_text(text)

    default_pred = _default_predictor_params(base_cfg)
    default_gen = _default_generation_params(base_elo_cfg)

    teams_cache: dict[tuple, dict[str, Team]] = {}

    def teams_for(bundle, fixtures, results, as_of, gen_params: dict) -> dict[str, Team]:
        key = (_freeze(gen_params), bundle.name)
        if key not in teams_cache:
            teams, _meta = _build_teams_for_gen(
                bundle, gen_params, base_elo_cfg, as_of, all_matches,
            )
            teams_cache[key] = teams
        return teams_cache[key]

    def benchmarks_with(gen_params: dict) -> list[tuple]:
        return [
            (bundle, teams_for(bundle, fixtures, results, as_of, gen_params), fixtures, results)
            for (bundle, fixtures, results, as_of) in benchmarks
        ]

    # --- Stage 1: sweep generation, predictor at config defaults
    stage1_candidates = [dict(p) for p in _iter_grid(generation_grid)]
    if default_gen not in stage1_candidates:
        stage1_candidates.append(default_gen)
    stage1_results = [_evaluate(default_pred, g, benchmarks_with(g)) for g in stage1_candidates]
    stage1_ranked = sorted(stage1_results, key=_blended_key)
    stage1_best = stage1_ranked[0]
    stage1_default = next(r for r in stage1_results if r["gen_params"] == default_gen)

    # --- Stage 2: sweep predictor, gen fixed at stage-1 best
    best_gen = stage1_best["gen_params"]
    bwt_best = benchmarks_with(best_gen)
    stage2_candidates = [dict(p) for p in _iter_grid(predictor_grid)]
    if default_pred not in stage2_candidates:
        stage2_candidates.append(default_pred)
    stage2_results = [_evaluate(p, best_gen, bwt_best) for p in stage2_candidates]
    stage2_ranked = sorted(stage2_results, key=_blended_key)
    stage2_best = stage2_ranked[0]
    stage2_default = next(r for r in stage2_results if r["params"] == default_pred)

    # --- Combined: best gen × best predictor (already computed as stage2_best)
    combined_pred = stage2_best["params"]
    combined_gen = best_gen
    combined_metrics = _stage_default_metrics(stage2_best)

    # --- Reality check: default config vs combined recommendation
    default_pred_obj = AttackDefencePoissonPredictor(gmax=_GMAX, **default_pred)
    combined_pred_obj = AttackDefencePoissonPredictor(gmax=_GMAX, **combined_pred)
    default_bwt = benchmarks_with(default_gen)
    combined_bwt = bwt_best  # already at best_gen

    default_reality_pt = [
        reality_check_one(bundle, teams, fixtures, results, default_pred_obj)
        for (bundle, teams, fixtures, results) in default_bwt
    ]
    combined_reality_pt = [
        reality_check_one(bundle, teams, fixtures, results, combined_pred_obj)
        for (bundle, teams, fixtures, results) in combined_bwt
    ]
    default_pooled = reality_pooled(default_reality_pt)
    combined_pooled = reality_pooled(combined_reality_pt)
    combined_verdict = verdict_of(combined_pooled)
    default_verdict = verdict_of(default_pooled)

    # --- LOO on the combined recommendation
    loo = _leave_one_out(stage2_results, [b[0].name for b in benchmarks])

    data = {
        "predictor": "attack_defence_poisson",
        "benchmarks": [b[0].name for b in benchmarks],
        "stage1_generation": {
            "grid_size": len(stage1_candidates),
            "default_params": default_gen,
            "default_metrics": _stage_default_metrics(stage1_default),
            "recommended_params": stage1_best["gen_params"],
            "recommended_metrics": _stage_default_metrics(stage1_best),
            "leaderboard": [_row_gen(r) for r in stage1_ranked[:top]],
        },
        "stage2_predictor": {
            "grid_size": len(stage2_candidates),
            "default_params": default_pred,
            "default_metrics": _stage_default_metrics(stage2_default),
            "recommended_params": stage2_best["params"],
            "recommended_metrics": _stage_default_metrics(stage2_best),
            "leaderboard": [_row_pred(r) for r in stage2_ranked[:top]],
        },
        "combined_recommended_params": {"generation": combined_gen, "predictor": combined_pred},
        "combined_metrics": combined_metrics,
        "combined_per_tournament": stage2_best["per_tournament"],
        "reality_check": {
            "default": {"per_tournament": default_reality_pt, "pooled": default_pooled,
                        "verdict": default_verdict},
            "recommended": {"per_tournament": combined_reality_pt, "pooled": combined_pooled,
                            "verdict": combined_verdict},
        },
        "leave_one_out": loo,
    }
    return _render(data), data


def _leave_one_out(results: list, names: list[str]) -> dict:
    """For each held-out tournament, pick the best params on the others; report held-out perf."""
    out: dict[str, dict] = {}
    for held in names:
        others = [t for t in names if t != held]

        def train_key(r, others=others):
            tot_m = sum(r["per_tournament"][t]["matches"] for t in others)
            rps = (sum(r["per_tournament"][t]["mean_rps"] * r["per_tournament"][t]["matches"]
                       for t in others) / tot_m) if tot_m else 0.0
            tot_max = sum(r["per_tournament"][t]["max"] for t in others)
            pct = ((100.0 * sum(r["per_tournament"][t]["model"] for t in others) / tot_max)
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


# ---------------------------------------------------------------------------- rendering

def _row_gen(r: dict) -> dict:
    return {"params": r["gen_params"], **_stage_default_metrics(r)}


def _row_pred(r: dict) -> dict:
    return {"params": r["params"], **_stage_default_metrics(r)}


def _fmt_gen(p: dict) -> str:
    return (f"lr{p['learning_rate']} look{int(p['lookback_years'])} "
            f"decay{'Y' if p['recency_decay'] else 'N'} ha{p['ad_home_advantage']}")


def _fmt_pred(p: dict) -> str:
    return (f"c{p['base_log_rate']} ha{p['home_advantage']} "
            f"rho{p['rho']} ko{p['ko_goal_scale']}")


def _render(data: dict) -> str:
    L = ["# A/D parameter tuning — staged "
         "(blended: mean RPS primary, pool-points %max tie-break)", ""]
    L.append(f"Benchmarks: {', '.join(data['benchmarks'])}. "
             f"Stage 1 (generation): {data['stage1_generation']['grid_size']} points; "
             f"Stage 2 (predictor): {data['stage2_predictor']['grid_size']} points.")
    L.append("")

    # Stage 1
    s1 = data["stage1_generation"]
    L.append("## Stage 1 — generation params (predictor held at config defaults)")
    L.append(_fixed_table(
        ["set", "lr/look/decay/ha", "mean RPS", "mean NLL", "model", "%max", "exact"],
        [
            ["default", _fmt_gen(s1["default_params"]),
             f"{s1['default_metrics']['mean_rps']:.4f}",
             f"{s1['default_metrics']['mean_nll']:.4f}",
             s1["default_metrics"]["model"], f"{s1['default_metrics']['model_pct']:.1f}",
             s1["default_metrics"]["exact_hits"]],
            ["recommend", _fmt_gen(s1["recommended_params"]),
             f"{s1['recommended_metrics']['mean_rps']:.4f}",
             f"{s1['recommended_metrics']['mean_nll']:.4f}",
             s1["recommended_metrics"]["model"], f"{s1['recommended_metrics']['model_pct']:.1f}",
             s1["recommended_metrics"]["exact_hits"]],
        ],
    ))
    L.append("")
    L.append("### Stage 1 leaderboard (top by blended rank)")
    lrows = [[i + 1, _fmt_gen(row["params"]), f"{row['mean_rps']:.4f}",
              f"{row['mean_nll']:.4f}", row["model"], f"{row['model_pct']:.1f}", row["exact_hits"]]
             for i, row in enumerate(s1["leaderboard"])]
    L.append(_fixed_table(["#", "lr/look/decay/ha", "mean RPS", "mean NLL",
                            "model", "%max", "exact"], lrows))
    L.append("")

    # Stage 2
    s2 = data["stage2_predictor"]
    L.append("## Stage 2 — predictor params (generation held at Stage 1 best)")
    L.append(_fixed_table(
        ["set", "c/ha/rho/ko", "mean RPS", "mean NLL", "model", "%max", "exact"],
        [
            ["default", _fmt_pred(s2["default_params"]),
             f"{s2['default_metrics']['mean_rps']:.4f}",
             f"{s2['default_metrics']['mean_nll']:.4f}",
             s2["default_metrics"]["model"], f"{s2['default_metrics']['model_pct']:.1f}",
             s2["default_metrics"]["exact_hits"]],
            ["recommend", _fmt_pred(s2["recommended_params"]),
             f"{s2['recommended_metrics']['mean_rps']:.4f}",
             f"{s2['recommended_metrics']['mean_nll']:.4f}",
             s2["recommended_metrics"]["model"], f"{s2['recommended_metrics']['model_pct']:.1f}",
             s2["recommended_metrics"]["exact_hits"]],
        ],
    ))
    L.append("")
    L.append("### Stage 2 leaderboard (top by blended rank)")
    lrows = [[i + 1, _fmt_pred(row["params"]), f"{row['mean_rps']:.4f}",
              f"{row['mean_nll']:.4f}", row["model"], f"{row['model_pct']:.1f}", row["exact_hits"]]
             for i, row in enumerate(s2["leaderboard"])]
    L.append(_fixed_table(["#", "c/ha/rho/ko", "mean RPS", "mean NLL",
                            "model", "%max", "exact"], lrows))
    L.append("")

    # Combined
    L.append("## Combined recommendation (Stage 1 best × Stage 2 best)")
    L.append(f"- generation: `{_fmt_gen(data['combined_recommended_params']['generation'])}`")
    L.append(f"- predictor:  `{_fmt_pred(data['combined_recommended_params']['predictor'])}`")
    cm = data["combined_metrics"]
    L.append(f"- combined: mean RPS {cm['mean_rps']:.4f}, mean NLL {cm['mean_nll']:.4f}, "
             f"model {cm['model']} pts ({cm['model_pct']:.1f}% of max), "
             f"{cm['exact_hits']} exact hits over {cm['matches']} matches.")
    L.append("")
    L.append("### Combined — per tournament")
    prows = [[name, m["matches"], f"{m['mean_rps']:.4f}", m["model"], m["naive"], m["max"],
              f"{m['model_pct']:.1f}"]
             for name, m in data["combined_per_tournament"].items()]
    L.append(_fixed_table(["tournament", "matches", "mean RPS", "model", "naive", "max", "%max"],
                          prows))
    L.append("")

    # Reality check
    L.append("## Reality check vs actual results")
    L.append("Per-tournament rows compare predicted to actual: mean goals/match, tendency "
             "split (H/D/A), exact + tendency hit rate, and modal-tip share (a high share "
             "flags the 'always 1:0' pathology).")

    def _rc_table(label: str, block: dict) -> list[str]:
        out = [f"### {label} — verdict **{block['verdict']['status']}**"]
        for r in block["verdict"]["reasons"]:
            out.append(f"- {r}")
        rows = []
        for p in block["per_tournament"]:
            if not p.get("matches"):
                continue
            mg = p["mean_goals"]
            ts = p["tendency_split"]
            tc = p["tip_composition"]
            rows.append([
                p["tournament"], p["matches"],
                f"{mg['predicted_total']:.2f}/{mg['actual_total']:.2f}",
                (f"{ts['predicted']['H']:.0%}/{ts['predicted']['D']:.0%}/{ts['predicted']['A']:.0%}"
                 f" vs {ts['actual']['H']:.0%}/{ts['actual']['D']:.0%}/{ts['actual']['A']:.0%}"),
                f"{tc['exact_hit_rate']:.0%}/{tc['tendency_hit_rate']:.0%}",
                f"{tc['modal_tip']['home']}:{tc['modal_tip']['away']} ({tc['modal_tip']['share']:.0%})",
                f"{p['scoreline_tvd']:.3f}",
            ])
        po = block["pooled"]
        if po.get("matches"):
            rows.append([
                "POOLED", po["matches"],
                f"{po['mean_goals']['predicted_total']:.2f}/{po['mean_goals']['actual_total']:.2f}",
                (f"{po['tendency_split']['predicted']['H']:.0%}/"
                 f"{po['tendency_split']['predicted']['D']:.0%}/"
                 f"{po['tendency_split']['predicted']['A']:.0%} vs "
                 f"{po['tendency_split']['actual']['H']:.0%}/"
                 f"{po['tendency_split']['actual']['D']:.0%}/"
                 f"{po['tendency_split']['actual']['A']:.0%}"),
                f"{po['tip_composition']['exact_hit_rate']:.0%}/"
                f"{po['tip_composition']['tendency_hit_rate']:.0%}",
                f"share {po['tip_composition']['modal_share_weighted']:.0%}",
                f"{po['scoreline_tvd_weighted']:.3f}",
            ])
        out.append(_fixed_table(
            ["tournament", "matches", "goals/m pred/act",
             "tendency pred vs actual", "exact/tend hit", "modal tip", "TVD"], rows))
        return out

    rc = data["reality_check"]
    L.extend(_rc_table("Default params", rc["default"]))
    L.append("")
    L.extend(_rc_table("Combined recommendation", rc["recommended"]))
    L.append("")

    # LOO
    L.append("## Leave-one-tournament-out (generalisation check, Stage 2 only)")
    L.append("Predictor params chosen on the other tournaments at the Stage-1-best gen point, "
             "scored on the held-out one.")
    orows = [[held, _fmt_pred(v["chosen_params"]), f"{v['heldout_mean_rps']:.4f}",
              f"{v['heldout_model_pct']:.1f}"]
             for held, v in data["leave_one_out"].items()]
    L.append(_fixed_table(["held-out", "chosen params", "heldout RPS", "heldout %max"], orows))
    L.append("")
    return "\n".join(L)
