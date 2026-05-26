"""tippspiel CLI (spec §8).

    tippspiel run            full pipeline: predict, simulate, report
    tippspiel predict        group-stage predictions + tips only (no simulation)
    tippspiel diagnose       write the Claude diagnostic report (markdown + JSON)
    tippspiel verify         backtest the predictor against a completed tournament (pool points)
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
from .pipeline import run_pipeline, write_diagnostics, write_report, write_verification
from .simulation.bracket import Bracket

DEFAULT_CONFIG = "config.yaml"


def _cmd_predict(cfg, bundle) -> int:
    result = run_pipeline(cfg, bundle, simulate=False)
    path = write_report(cfg, result["context"])
    tips = result["tipset"].tips
    print(f"[{bundle.display_name}] predicted {len(tips)} tippable fixture(s) "
          f"(group stage, no simulation).")
    print(f"Report written to {path}")
    return 0


def _cmd_run(cfg, bundle) -> int:
    result = run_pipeline(cfg, bundle, simulate=True)
    path = write_report(cfg, result["context"])
    tips = result["tipset"].tips
    outcome = result["outcome"]
    print(f"[{bundle.display_name}] predicted {len(tips)} tippable fixture(s).")
    if outcome is not None:
        print(f"Monte Carlo: {outcome.mc_iterations:,} iterations (seed {outcome.mc_seed}), "
              f"max SE {outcome.mc_standard_error:.4f}.")
        answers = result["tipset"].bonus_answers
        if answers:
            print("Recommended bonus answers:")
            for qid, ans in answers.items():
                print(f"  {qid}: {ans}")
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
    args = parser.parse_args(argv)

    config_path = args.config
    if not Path(config_path).exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2
    cfg = load_config(config_path)
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
    except ValueError as exc:
        print(f"{args.command}: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
