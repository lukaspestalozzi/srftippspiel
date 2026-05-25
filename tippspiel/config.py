"""Config loading + validation.

All defaults live in ``config.yaml``; nothing is hardcoded in logic. The seed is
mandatory and surfaced in the report for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    teams_file: str
    fixtures_file: str
    results_file: str
    bracket_map_file: str


@dataclass(frozen=True)
class PredictorConfig:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class StrategyConfig:
    name: str


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
    data: DataConfig
    predictor: PredictorConfig
    strategy: StrategyConfig
    simulation: SimulationConfig
    report: ReportConfig
    bonus_questions: list[BonusQuestionConfig] = field(default_factory=list)
    config_path: Path | None = None


_VALID_PENALTY_MODELS = {"coin_flip", "elo_weighted"}


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}

    try:
        sim = raw["simulation"]
        penalty = sim.get("penalty_model", "coin_flip")
        if penalty not in _VALID_PENALTY_MODELS:
            raise ValueError(
                f"simulation.penalty_model must be one of {_VALID_PENALTY_MODELS}, got {penalty!r}"
            )
        cfg = Config(
            data=DataConfig(**raw["data"]),
            predictor=PredictorConfig(
                name=raw["predictor"]["name"],
                params=dict(raw["predictor"].get("params", {})),
            ),
            strategy=StrategyConfig(name=raw["strategy"]["name"]),
            simulation=SimulationConfig(
                iterations=int(sim["iterations"]),
                seed=int(sim["seed"]),
                penalty_model=penalty,
            ),
            report=ReportConfig(**raw["report"]),
            bonus_questions=[
                BonusQuestionConfig(id=q["id"], points=int(q["points"]))
                for q in raw.get("bonus_questions", [])
            ],
            config_path=path,
        )
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc}") from exc
    return cfg
