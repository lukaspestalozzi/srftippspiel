"""tippspiel CLI (spec §8).

    tippspiel run            full pipeline: predict, simulate, report
    tippspiel predict        group-stage predictions + tips only (no simulation)
    tippspiel validate-data  check input files for schema/consistency errors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .data.file_provider import FileDataProvider
from .pipeline import run_pipeline, write_report

DEFAULT_CONFIG = "config.yaml"


def _cmd_predict(cfg) -> int:
    result = run_pipeline(cfg, simulate=False)
    path = write_report(cfg, result["context"])
    tips = result["tipset"].tips
    print(f"Predicted {len(tips)} tippable fixture(s) (group stage, no simulation).")
    print(f"Report written to {path}")
    return 0


def _cmd_run(cfg) -> int:
    result = run_pipeline(cfg, simulate=True)
    path = write_report(cfg, result["context"])
    tips = result["tipset"].tips
    outcome = result["outcome"]
    print(f"Predicted {len(tips)} tippable fixture(s).")
    if outcome is not None:
        champ = result["tipset"].bonus_answers.get("champion")
        print(f"Monte Carlo: {outcome.mc_iterations:,} iterations (seed {outcome.mc_seed}), "
              f"max SE {outcome.mc_standard_error:.4f}.")
        if champ:
            print(f"Recommended World Champion: {champ}")
    print(f"Report written to {path}")
    return 0


def _cmd_validate(cfg) -> int:
    errors = validate_data(cfg)
    if errors:
        print(f"validate-data: {len(errors)} problem(s) found:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("validate-data: all input files OK.")
    return 0


def validate_data(cfg) -> list[str]:
    errors: list[str] = []
    provider = FileDataProvider(
        cfg.data.teams_file, cfg.data.fixtures_file,
        cfg.data.results_file, cfg.data.bracket_map_file,
    )
    try:
        teams = {t.team_id: t for t in provider.get_teams()}
    except Exception as exc:  # noqa: BLE001
        return [f"teams file unreadable: {exc}"]
    if len(teams) != 48:
        errors.append(f"expected 48 teams, found {len(teams)}")

    try:
        fixtures = provider.get_fixtures()
    except Exception as exc:  # noqa: BLE001
        return errors + [f"fixtures file unreadable: {exc}"]
    if len(fixtures) != 104:
        errors.append(f"expected 104 fixtures, found {len(fixtures)}")

    group_matches = [m for m in fixtures if m.group]
    if len(group_matches) != 72:
        errors.append(f"expected 72 group matches, found {len(group_matches)}")

    # Concrete team refs must exist in teams.
    for m in fixtures:
        for ref in (m.home, m.away):
            if ref.is_concrete and ref.team_id not in teams:
                errors.append(f"{m.match_id}: unknown team_id {ref.team_id}")

    # Each group: exactly 4 teams and 6 matches (full round-robin).
    by_group: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for m in group_matches:
        counts[m.group] = counts.get(m.group, 0) + 1
        by_group.setdefault(m.group, set()).update({m.home.team_id, m.away.team_id})
    for g, n in sorted(counts.items()):
        if n != 6:
            errors.append(f"group {g}: expected 6 matches, found {n}")
        if len(by_group[g]) != 4:
            errors.append(f"group {g}: expected 4 teams, found {len(by_group[g])}")

    try:
        provider.get_results()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"results file unreadable: {exc}")
    try:
        bm = provider.get_bracket_map()
        if "r32" not in bm or len(bm["r32"]) != 16:
            errors.append("bracket map: expected 16 R32 slots")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"bracket map unreadable: {exc}")

    return errors


def _add_config_arg(parser: argparse.ArgumentParser, *, with_default: bool) -> None:
    # Top-level carries the real default (so `tippspiel run` works with a local
    # config.yaml and the default shows in --help). The subcommand copy uses SUPPRESS so
    # that, when --config is given before the subcommand, the subparser does not overwrite
    # it with the default. Either position works; a value after the subcommand wins.
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=(DEFAULT_CONFIG if with_default else argparse.SUPPRESS),
        help=f"path to config file (default: {DEFAULT_CONFIG} in the current directory)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tippspiel")
    _add_config_arg(parser, with_default=True)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("run", "full pipeline: predict, simulate, report"),
        ("predict", "group-stage predictions + tips only (no simulation)"),
        ("validate-data", "check input files for errors"),
    ]:
        _add_config_arg(sub.add_parser(name, help=help_text), with_default=False)
    args = parser.parse_args(argv)
    config_path = args.config  # always set: top-level default or a subcommand override

    if not Path(config_path).exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2
    cfg = load_config(config_path)

    if args.command == "predict":
        return _cmd_predict(cfg)
    if args.command == "run":
        return _cmd_run(cfg)
    if args.command == "validate-data":
        return _cmd_validate(cfg)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
