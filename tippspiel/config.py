"""Config loading + validation.

Each tournament is one self-contained config file (``config.yaml`` is the default,
FIFA World Cup 2026; further tournaments live under ``configs/<name>.yaml``). A config file
carries both the engine defaults (predictor / strategy / simulation / report) and a
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
class StrategyConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


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
class BonusQuestionConfig:
    id: str
    points: int


@dataclass(frozen=True)
class Config:
    predictor: PredictorConfig
    strategy: StrategyConfig
    simulation: SimulationConfig
    report: ReportConfig
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
    bonus_questions: list[BonusQuestionConfig] = field(default_factory=list)
    elo_source: str = ""


_VALID_PENALTY_MODELS = {"coin_flip", "elo_weighted"}


def _read(path: str | Path) -> tuple[Path, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path, (yaml.safe_load(path.read_text()) or {})


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
            strategy=StrategyConfig(
                name=raw["strategy"]["name"],
                params=dict(raw["strategy"].get("params", {})),
            ),
            simulation=SimulationConfig(
                iterations=int(sim["iterations"]),
                seed=int(sim["seed"]),
                penalty_model=penalty,
            ),
            report=ReportConfig(**raw["report"]),
            config_path=path,
        )
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc}") from exc
    return cfg


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
    return TournamentBundle(
        name=t["name"],
        display_name=t.get("display_name", t["name"]),
        completed=bool(t.get("completed", False)),
        teams_file=data_dir / t.get("teams_file", "teams.csv"),
        fixtures_file=data_dir / t.get("fixtures_file", "fixtures.csv"),
        results_file=data_dir / t.get("results_file", "results.csv"),
        thirds_allocation_file=(data_dir / thirds) if thirds else None,
        bonus_questions=[
            BonusQuestionConfig(id=q["id"], points=int(q["points"]))
            for q in raw.get("bonus_questions", [])
        ],
        elo_source=t.get("elo_source", ""),
    )
