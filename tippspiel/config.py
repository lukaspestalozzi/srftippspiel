"""Config loading + validation.

Each tournament is one self-contained config file (``config.yaml`` is the default,
FIFA World Cup 2026; further tournaments live under ``configs/<name>.yaml``). A config file
carries both the engine defaults (predictor / simulation / report) and a
``tournament:`` block describing the tournament's data files, metadata and bonus questions.
Select a tournament with ``--config <file>``. The seed is mandatory and surfaced in the
report for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DATA_ROOT = Path(__file__).parent / "data"


@dataclass(frozen=True)
class PredictorConfig:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class SimulationConfig:
    iterations: int
    seed: int
    penalty_model: str


@dataclass(frozen=True)
class ReportConfig:
    output_dir: str
    display_timezone: str


@dataclass(frozen=True)
class StrategyConfig:
    # EV-tolerance for realistic tip selection: among scorelines within this many pool-points of
    # the EV-optimum, pick the one closest to the model's expected scoreline. 0 = strict
    # EV-maximisation (legacy); ~0.15 lifts both-teams-score tips to a realistic rate. See
    # ``best_tip`` in strategy/expected_points.py.
    realism_tolerance: float = 0.0


@dataclass(frozen=True)
class BonusQuestionConfig:
    id: str
    points: int


@dataclass(frozen=True)
class Config:
    predictor: PredictorConfig
    simulation: SimulationConfig
    report: ReportConfig
    strategy: StrategyConfig = StrategyConfig()
    config_path: Path | None = None


@dataclass(frozen=True)
class TournamentBundle:
    """A tournament's data files + bonus questions, parsed from its config file's
    ``tournament:`` block. The knockout bracket is derived from ``fixtures.csv``; the only
    optional sidecar is ``thirds_allocation_file`` (a third-place combination->slot table).
    """

    name: str
    display_name: str
    completed: bool
    teams_file: Path
    fixtures_file: Path
    results_file: Path
    thirds_allocation_file: Path | None = None
    odds_file: Path | None = None
    bonus_questions: list[BonusQuestionConfig] = field(default_factory=list)
    elo_source: str = ""


_VALID_PENALTY_MODELS = {"coin_flip", "elo_weighted"}


def _read(path: str | Path) -> tuple[Path, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path, (yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def load_config(path: str | Path) -> Config:
    path, raw = _read(path)
    try:
        sim = raw["simulation"]
        penalty = sim.get("penalty_model", "coin_flip")
        if penalty not in _VALID_PENALTY_MODELS:
            raise ValueError(
                f"simulation.penalty_model must be one of {_VALID_PENALTY_MODELS}, got {penalty!r}"
            )
        cfg = Config(
            predictor=PredictorConfig(
                name=raw["predictor"]["name"],
                params=dict(raw["predictor"].get("params", {})),
            ),
            simulation=SimulationConfig(
                iterations=int(sim["iterations"]),
                seed=int(sim["seed"]),
                penalty_model=penalty,
            ),
            report=ReportConfig(**raw["report"]),
            strategy=StrategyConfig(**(raw.get("strategy") or {})),
            config_path=path,
        )
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc}") from exc
    return cfg


def load_offdef_block(path: str | Path) -> dict:
    """The optional ``offdef:`` block (off/def Elo fit hyperparameters + weight tiers +
    snapshot date), used by ``tippspiel fit-ratings``. Empty dict when absent — the fitter then
    runs on its built-in defaults. Kept as a raw dict so ``config.py`` stays type-light; the
    fit layer constructs its own dataclasses from it."""
    _path, raw = _read(path)
    return dict(raw.get("offdef", {}) or {})


def load_elo_block(path: str | Path) -> dict:
    """The optional ``elo:`` block (scalar World-Football-Elo fit hyperparameters + K tiers),
    used by ``tippspiel fit-ratings``. Empty dict when absent — the fitter then runs on its
    built-in defaults. The fit's ``snapshot_date`` is shared with the ``offdef:`` block (a single
    cutoff keeps leak-freeness reasoning intact), so it is read from there, not here."""
    _path, raw = _read(path)
    return dict(raw.get("elo", {}) or {})


def write_offdef_snapshot_date(config_path: str | Path, iso_date: str) -> bool:
    """Set ``offdef.snapshot_date`` to ``iso_date`` via a comment-preserving single-line edit.

    Used by the live-update tooling to advance the cutoff to the day after the latest played match
    without round-tripping the YAML (which would drop comments). Returns True if the file changed.
    """
    import re

    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(
        r"(?m)^(\s*snapshot_date:\s*)\S.*?(\s*(?:#.*)?)$",
        rf'\g<1>"{iso_date}"\g<2>',
        text,
    )
    if n == 0:
        raise ValueError(f"no 'snapshot_date:' line found in {config_path}")
    if new == text:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def load_tournament(path: str | Path, *, data_root: Path = _DATA_ROOT) -> TournamentBundle:
    """Parse the ``tournament:`` block (+ ``bonus_questions:``) of a config file.

    Data-file paths are resolved relative to ``<data_root>/<tournament.data_dir>``.
    """
    _path, raw = _read(path)
    try:
        t = raw["tournament"]
    except KeyError as exc:
        raise ValueError(f"Config {path} has no 'tournament:' block") from exc

    data_dir = data_root / t["data_dir"]
    thirds = t.get("thirds_allocation_file")
    odds = t.get("odds_file")
    return TournamentBundle(
        name=t["name"],
        display_name=t.get("display_name", t["name"]),
        completed=bool(t.get("completed", False)),
        teams_file=data_dir / t.get("teams_file", "teams.csv"),
        fixtures_file=data_dir / t.get("fixtures_file", "fixtures.csv"),
        results_file=data_dir / t.get("results_file", "results.csv"),
        thirds_allocation_file=(data_dir / thirds) if thirds else None,
        odds_file=(data_dir / odds) if odds else None,
        bonus_questions=[
            BonusQuestionConfig(id=q["id"], points=int(q["points"]))
            for q in raw.get("bonus_questions", [])
        ],
        elo_source=t.get("elo_source", ""),
    )
