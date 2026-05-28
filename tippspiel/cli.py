"""tippspiel CLI (spec §8).

    tippspiel run            combined pipeline across every configured predictor: predict,
                              simulate, and emit one multi-model report.html (4 tips/match:
                              2 ELO models × {EV, rank-optimising})
    tippspiel predict        same combined multi-model report, no simulation
    tippspiel diagnose       write the Claude diagnostic report (markdown + JSON, single model)
    tippspiel verify         backtest one predictor against a completed tournament (single model)
    tippspiel validate-data  check input files for schema/consistency errors

Each tournament is one config file; select it with ``--config <file>`` (default
``config.yaml``, FIFA World Cup 2026; further tournaments under ``configs/<name>.yaml``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config, load_tournament
from .data.file_provider import FileDataProvider
from .pipeline import (
    build_elo,
    run_combined_pipeline,
    run_tuning,
    write_diagnostics,
    write_report,
    write_verification,
)
from .simulation.bracket import Bracket
from .strategy.bonus import build_bonus_questions

DEFAULT_CONFIG = "config.yaml"
DEFAULT_BENCHMARKS = [
    "configs/euro2016.yaml",
    "configs/wc2022.yaml",
    "configs/euro2024.yaml",
    "configs/wc2018.yaml",
    "configs/euro2020.yaml",
]
# Commands that build a single Predictor and therefore require an explicit --predictor.
# ``run`` and ``predict`` no longer take one: they run every configured predictor.
# ``tune`` is single-model (each predictor has its own grid + scoring path).
_PREDICTION_COMMANDS = {"verify", "diagnose", "tune"}


def _print_bonus_picks(run, bundle) -> None:
    outcome = run["core"]["outcome"]
    teams = run["core"]["teams"]
    if outcome is None:
        return
    for q in build_bonus_questions(bundle.bonus_questions):
        dist = q.resolve(outcome)
        if not dist:
            continue
        pick, p = max(dist.items(), key=lambda kv: kv[1])
        name = teams[pick].name if pick in teams else pick
        print(f"    {q.question_id}: {name} ({p:.1%})")


def _cmd_predict(cfg, bundle) -> int:
    result = run_combined_pipeline(cfg, bundle, simulate=False)
    path = write_report(cfg, result["context"])
    runs = result["runs"]
    print(f"[{bundle.display_name}] {len(runs)} model(s), no simulation:")
    for run in runs:
        n_tips = len(run["core"]["predictions"])
        print(f"  {run['label']} ({run['name']}, {run['ratings_file']}): "
              f"{n_tips} tippable fixture(s)")
    print(f"Report written to {path}")
    return 0


def _cmd_run(cfg, bundle) -> int:
    result = run_combined_pipeline(cfg, bundle, simulate=True)
    path = write_report(cfg, result["context"])
    runs = result["runs"]
    print(f"[{bundle.display_name}] {len(runs)} model(s):")
    for run in runs:
        core = run["core"]
        outcome = core["outcome"]
        n_tips = len(core["predictions"])
        print(f"  {run['label']} ({run['name']}, {run['ratings_file']}): "
              f"{n_tips} tippable fixture(s)")
        if outcome is not None:
            print(f"    MC {outcome.mc_iterations:,} iters (seed {outcome.mc_seed}), "
                  f"max SE {outcome.mc_standard_error:.4f}")
            _print_bonus_picks(run, bundle)
    print(f"Report written to {path}")
    return 0


def _cmd_diagnose(cfg, bundle, *, simulate: bool) -> int:
    result = write_diagnostics(cfg, bundle, simulate=simulate)
    paths = result["paths"]
    anomalies = result["data"]["anomalies"]
    n_fail = sum(1 for a in anomalies if a["status"] == "FAIL")
    n_warn = sum(1 for a in anomalies if a["status"] == "WARN")
    print(f"[{bundle.display_name}] diagnostic report written to {paths['markdown']} "
          f"(+ {paths['json'].name}).")
    print(f"Anomaly checks: {len(anomalies)} total, {n_fail} FAIL, {n_warn} WARN.")
    for a in anomalies:
        if a["status"] in ("FAIL", "WARN"):
            print(f"  [{a['status']}] {a['name']}: {a['detail']}")
    return 0


def _cmd_verify(cfg, bundle) -> int:
    if not bundle.completed:
        print(f"verify: '{bundle.name}' is not marked completed; results may be partial.",
              file=sys.stderr)
    result = write_verification(cfg, bundle)
    paths = result["paths"]
    s = result["data"]["summary"]["all"]
    pct = (100.0 * s["model"] / s["max"]) if s["max"] else 0.0
    print(f"[{bundle.display_name}] verification written to {paths['markdown']} "
          f"(+ {paths['json'].name}).")
    print(f"Model {s['model']} pts vs naive {s['naive']} pts over {s['matches']} matches "
          f"(max {s['max']}, {pct:.1f}% of max, {s['exact_hits']} exact hits).")
    return 0


def _cmd_tune(cfg, benchmark_configs, top: int) -> int:
    missing = [p for p in benchmark_configs if not Path(p).exists()]
    if missing:
        print(f"tune: benchmark config(s) not found: {missing}", file=sys.stderr)
        return 2
    name = cfg.predictor.name
    if name == "elo_poisson":
        result = run_tuning(cfg, benchmark_configs, top=top)
        return _print_elo_tune(result)
    if name == "attack_defence_poisson":
        from .pipeline import run_ad_tuning
        result = run_ad_tuning(cfg, benchmark_configs, top=top)
        return _print_ad_tune(result)
    print(f"tune: predictor {name!r} has no tuning grid implemented.", file=sys.stderr)
    return 2


def _print_elo_tune(result: dict) -> int:
    paths, data = result["paths"], result["data"]
    dm, rm = data["default_metrics"], data["recommended_metrics"]
    print(f"tuning written to {paths['markdown']} (+ {paths['json'].name}).")
    print(f"benchmarks: {', '.join(data['benchmarks'])}; swept {data['grid_size']} param sets.")
    print(f"default:     RPS {dm['mean_rps']:.4f}  model {dm['model']} pts "
          f"({dm['model_pct']:.1f}% of max)")
    print(f"recommended: RPS {rm['mean_rps']:.4f}  model {rm['model']} pts "
          f"({rm['model_pct']:.1f}% of max)")
    print("recommended params:")
    for key, val in data["recommended_params"].items():
        print(f"  {key}: {val}")
    return 0


def _print_ad_tune(result: dict) -> int:
    paths, data = result["paths"], result["data"]
    s1, s2 = data["stage1_generation"], data["stage2_predictor"]
    cm = data["combined_metrics"]
    print(f"A/D tuning written to {paths['markdown']} (+ {paths['json'].name}).")
    print(f"benchmarks: {', '.join(data['benchmarks'])}.")
    print(f"Stage 1 (generation, {s1['grid_size']} pts): "
          f"default RPS {s1['default_metrics']['mean_rps']:.4f} -> "
          f"best RPS {s1['recommended_metrics']['mean_rps']:.4f}; "
          f"recommended: {s1['recommended_params']}")
    print(f"Stage 2 (predictor,  {s2['grid_size']} pts): "
          f"default RPS {s2['default_metrics']['mean_rps']:.4f} -> "
          f"best RPS {s2['recommended_metrics']['mean_rps']:.4f}; "
          f"recommended: {s2['recommended_params']}")
    print(f"Combined: mean RPS {cm['mean_rps']:.4f}, model {cm['model']} pts "
          f"({cm['model_pct']:.1f}% of max), {cm['exact_hits']} exact hits / "
          f"{cm['matches']} matches.")
    rv = data["reality_check"]["recommended"]["verdict"]
    print(f"Reality check verdict: {rv['status']}")
    for reason in rv["reasons"]:
        print(f"  - {reason}")
    return 0


def _cmd_build_elo(cfg, bundle, args) -> int:
    result = build_elo(
        cfg, bundle,
        as_of=args.as_of, write_teams=args.write_teams,
        top=args.top, cache_only=args.cache_only,
    )
    paths, data = result["paths"], result["data"]
    m = data["meta"]
    print(f"[{bundle.display_name}] computed Elo for {m['n_teams_rated']} teams from "
          f"{m['n_matches_used']} matches (as of {m['as_of']}, lookback {m['lookback_years']}y, "
          + (f"half-life {m['half_life_years']}y)." if m["recency_decay"] else "no decay)."))
    print(f"Report written to {paths['markdown']} (+ {paths['json'].name}).")
    print("Top 5:")
    for r in data["ranking"][:5]:
        tag = f"  [{r['team_id']}]" if r["team_id"] else ""
        print(f"  {r['rank']:>2}. {r['name']:<22} {r['elo']:.1f}{tag}")
    if "teams_path" in result:
        print(f"Emitted {result['teams_written']} computed teams -> {result['teams_path']}")
    return 0


def _cmd_validate(cfg, bundle) -> int:
    errors = validate_data(bundle)
    if errors:
        print(f"validate-data [{bundle.name}]: {len(errors)} problem(s) found:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"validate-data [{bundle.name}]: all input files OK.")
    return 0


def validate_data(bundle) -> list[str]:
    """Format-general input validation: derives expectations from the data + fixtures bracket."""
    errors: list[str] = []
    provider = FileDataProvider(
        bundle.teams_file, bundle.fixtures_file, bundle.results_file,
        bundle.thirds_allocation_file,
    )
    try:
        teams = {t.team_id: t for t in provider.get_teams()}
    except Exception as exc:  # noqa: BLE001
        return [f"teams file unreadable: {exc}"]
    if len(teams) < 2:
        errors.append(f"expected at least 2 teams, found {len(teams)}")

    try:
        fixtures = provider.get_fixtures()
    except Exception as exc:  # noqa: BLE001
        return errors + [f"fixtures file unreadable: {exc}"]
    if not fixtures:
        errors.append("no fixtures found")

    # Concrete team refs must exist in teams.
    for m in fixtures:
        for ref in (m.home, m.away):
            if ref.is_concrete and ref.team_id not in teams:
                errors.append(f"{m.match_id}: unknown team_id {ref.team_id}")

    # Groups: consistent size, full round-robin (C(size, 2) matches each).
    by_group: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for m in fixtures:
        if not m.group:
            continue
        counts[m.group] = counts.get(m.group, 0) + 1
        by_group.setdefault(m.group, set()).update({m.home.team_id, m.away.team_id})
    sizes = {g: len(t) for g, t in by_group.items()}
    if len(set(sizes.values())) > 1:
        errors.append(f"groups have inconsistent sizes: {sizes}")
    for g in sorted(by_group):
        size = sizes[g]
        expected = size * (size - 1) // 2
        if counts[g] != expected:
            errors.append(f"group {g}: expected {expected} matches for {size} teams, found {counts[g]}")

    errors.extend(_validate_knockout(fixtures, sorted(by_group), provider))

    try:
        provider.get_results()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"results file unreadable: {exc}")

    return errors


def _validate_knockout(fixtures, group_letters, provider) -> list[str]:
    """Validate knockout references. A completed tournament may instead list concrete
    knockout participants (no references), in which case there is nothing to derive."""
    errors: list[str] = []
    ko = [m for m in fixtures if m.group is None]
    if not ko:
        return errors
    all_ids = {m.match_id for m in fixtures}
    group_set = set(group_letters)
    referenced = False
    for m in ko:
        for side in (m.home, m.away):
            r = side.ko_ref
            if r is None:
                continue
            referenced = True
            if r.kind in ("winner", "runner_up") and r.group not in group_set:
                errors.append(f"{m.match_id}: references unknown group {r.group!r}")
            if r.kind == "third_pooled":
                bad = [g for g in r.allowed_groups if g not in group_set]
                if bad:
                    errors.append(f"{m.match_id}: third slot allows unknown groups {bad}")
            if r.kind in ("winner_of", "loser_of") and r.match_id not in all_ids:
                errors.append(f"{m.match_id}: references unknown match {r.match_id!r}")
    if referenced and all(m.home.ko_ref and m.away.ko_ref for m in ko):
        try:
            Bracket(ko, group_letters, provider.get_thirds_allocation())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"bracket cannot be assembled from fixtures: {exc}")
    return errors


def _add_common_args(parser: argparse.ArgumentParser, *, with_default: bool) -> None:
    # Top-level carries the real default; a subcommand copy uses SUPPRESS so a value given
    # before the subcommand is not overwritten. A value after the subcommand wins.
    parser.add_argument(
        "--config", metavar="PATH",
        default=(DEFAULT_CONFIG if with_default else argparse.SUPPRESS),
        help=f"tournament config file (default: {DEFAULT_CONFIG}; others under configs/)",
    )
    parser.add_argument(
        "--strategy", metavar="NAME",
        default=(None if with_default else argparse.SUPPRESS),
        help="override the tip strategy from the config (e.g. rank_optimizing); "
             "strategy params still come from the config (or built-in defaults)",
    )
    parser.add_argument(
        "--predictor", metavar="NAME",
        default=(None if with_default else argparse.SUPPRESS),
        help="prediction model: required for verify/diagnose (single-model); ignored by "
             "run/predict (these always run every configured predictor side by side). "
             "Choices: elo_poisson | attack_defence_poisson; params from config 'predictors:'.",
    )


def _override_strategy(cfg, name: str | None):
    """Return ``cfg`` with its strategy name replaced (params preserved). No-op if name is None."""
    if not name:
        return cfg
    from dataclasses import replace

    return replace(cfg, strategy=replace(cfg.strategy, name=name))


def _select_predictor(cfg, command: str, name: str | None):
    """Resolve the active predictor. Prediction commands (verify/diagnose/tune) require
    ``--predictor`` (no default). ``run``/``predict`` run every configured predictor and
    don't need one. ``validate-data``/``build-elo`` don't build a predictor.
    Returns ``(cfg, error_message_or_None)``."""
    from .config import select_predictor

    if command not in _PREDICTION_COMMANDS:
        return cfg, None
    if not name:
        available = ", ".join(sorted(cfg.predictors))
        return cfg, (f"{command}: --predictor is required (no default). "
                     f"Choose one of: {available}.")
    try:
        return select_predictor(cfg, name), None
    except ValueError as exc:
        return cfg, f"{command}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tippspiel")
    _add_common_args(parser, with_default=True)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("run", "full pipeline: predict, simulate, report"),
        ("predict", "group-stage predictions + tips only (no simulation)"),
        ("verify", "backtest the predictor against a completed tournament (pool points)"),
        ("validate-data", "check input files for errors"),
    ]:
        _add_common_args(sub.add_parser(name, help=help_text), with_default=False)
    diag = sub.add_parser("diagnose", help="write the Claude diagnostic report (markdown + JSON)")
    _add_common_args(diag, with_default=False)
    diag.add_argument("--no-sim", action="store_true",
                      help="skip Monte Carlo (fast, predictor-only diagnostics)")
    tune = sub.add_parser("tune", help="sweep predictor params against completed-tournament backtests")
    _add_common_args(tune, with_default=False)
    tune.add_argument("--benchmarks", metavar="PATH", nargs="+", default=DEFAULT_BENCHMARKS,
                      help="completed-tournament config files to tune against "
                           f"(default: {', '.join(DEFAULT_BENCHMARKS)})")
    tune.add_argument("--top", type=int, default=15, help="leaderboard size (default: 15)")
    be = sub.add_parser("build-elo", help="compute World Football Elo from historical results")
    _add_common_args(be, with_default=False)
    be.add_argument("--as-of", metavar="DATE", default=None,
                    help="snapshot date YYYY-MM-DD (default: today, or a completed tournament's "
                         "start date)")
    be.add_argument("--write-teams", metavar="PATH", default=None,
                    help="also emit a teams.csv for the active tournament with the computed elo")
    be.add_argument("--top", type=int, default=30, help="ranking report size (default: 30)")
    be.add_argument("--cache-only", action="store_true",
                    help="use the cached results.csv only; do not fetch")
    args = parser.parse_args(argv)

    config_path = args.config
    if not Path(config_path).exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2
    cfg = load_config(config_path)
    cfg = _override_strategy(cfg, getattr(args, "strategy", None))
    cfg, err = _select_predictor(cfg, args.command, getattr(args, "predictor", None))
    if err:
        print(err, file=sys.stderr)
        return 2
    try:
        bundle = load_tournament(config_path)
    except (ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        if args.command == "predict":
            return _cmd_predict(cfg, bundle)
        if args.command == "run":
            return _cmd_run(cfg, bundle)
        if args.command == "verify":
            return _cmd_verify(cfg, bundle)
        if args.command == "validate-data":
            return _cmd_validate(cfg, bundle)
        if args.command == "diagnose":
            return _cmd_diagnose(cfg, bundle, simulate=not args.no_sim)
        if args.command == "tune":
            return _cmd_tune(cfg, args.benchmarks, args.top)
        if args.command == "build-elo":
            return _cmd_build_elo(cfg, bundle, args)
    except ValueError as exc:
        print(f"{args.command}: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
