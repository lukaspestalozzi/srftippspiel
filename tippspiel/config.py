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

from .elo.config import EloConfig, load_elo_config

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
    # Params for every available predictor, keyed by name. The active predictor is NOT defaulted
    # here — it is chosen at runtime via ``--predictor`` and materialised into ``predictor`` by
    # ``select_predictor`` (None until then).
    predictors: dict[str, PredictorConfig]
    strategy: StrategyConfig
    simulation: SimulationConfig
    report: ReportConfig
    predictor: PredictorConfig | None = None
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
    # Ratings file per predictor ``ratings_kind`` (``elo`` -> the official eloratings snapshot,
    # ``attack_defence`` -> the computed two-rating file). Resolved by convention in
    # ``load_tournament``; ``teams_file`` is the fallback for any kind not listed.
    teams_files: dict[str, Path] = field(default_factory=dict)
    thirds_allocation_file: Path | None = None
    bonus_questions: list[BonusQuestionConfig] = field(default_factory=list)
    elo_source: str = ""
    elo: EloConfig | None = None


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
        predictors = {
            name: PredictorConfig(name=name, params=dict(spec or {}))
            for name, spec in raw["predictors"].items()
        }
        if not predictors:
            raise ValueError("config 'predictors:' block defines no predictors")
        cfg = Config(
            predictors=predictors,
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


def select_predictor(cfg: Config, name: str) -> Config:
    """Return ``cfg`` with its active ``predictor`` set from the ``predictors`` map.

    There is no default predictor: the name must be supplied (via ``--predictor``) and must be
    one the config defines, else a clear error lists the available names.
    """
    from dataclasses import replace

    if name not in cfg.predictors:
        available = ", ".join(sorted(cfg.predictors)) or "(none)"
        raise ValueError(f"Unknown predictor {name!r}; config defines: {available}")
    return replace(cfg, predictor=cfg.predictors[name])


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
    teams_file = data_dir / t.get("teams_file", "teams.csv")
    # Convention: official eloratings snapshot in ``teams.csv``, computed attack/defence ratings
    # in ``teams_attack_defence.csv``. A ``tournament.teams_files`` block (kind -> filename) wins.
    teams_files = {
        "elo": teams_file,
        "attack_defence": data_dir / "teams_attack_defence.csv",
    }
    for kind, fname in (t.get("teams_files") or {}).items():
        teams_files[str(kind)] = data_dir / fname
    return TournamentBundle(
        name=t["name"],
        display_name=t.get("display_name", t["name"]),
        completed=bool(t.get("completed", False)),
        teams_file=teams_file,
        fixtures_file=data_dir / t.get("fixtures_file", "fixtures.csv"),
        results_file=data_dir / t.get("results_file", "results.csv"),
        teams_files=teams_files,
        thirds_allocation_file=(data_dir / thirds) if thirds else None,
        bonus_questions=[
            BonusQuestionConfig(id=q["id"], points=int(q["points"]))
            for q in raw.get("bonus_questions", [])
        ],
        elo_source=t.get("elo_source", ""),
        elo=load_elo_config(raw.get("elo")),
    )
