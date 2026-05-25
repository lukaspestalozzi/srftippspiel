"""Config loading + validation.

Global engine defaults (predictor / strategy / simulation / report) live in ``config.yaml``.
Tournament-specific data + bonus questions live in per-tournament bundles under
``tippspiel/data/tournaments/<name>/tournament.yaml`` and are loaded via ``resolve_tournament``.
The seed is mandatory and surfaced in the report for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_TOURNAMENTS_ROOT = Path(__file__).parent / "data" / "tournaments"


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
    predictor: PredictorConfig
    strategy: StrategyConfig
    simulation: SimulationConfig
    report: ReportConfig
    tournament: str = "wc2026"  # default tournament bundle; overridable via --tournament
    config_path: Path | None = None


@dataclass(frozen=True)
class TournamentBundle:
    """A tournament's data files + bonus questions, resolved from its ``tournament.yaml``."""

    name: str
    display_name: str
    completed: bool
    dir: Path
    teams_file: Path
    fixtures_file: Path
    results_file: Path
    bracket_map_file: Path
    bonus_questions: list[BonusQuestionConfig] = field(default_factory=list)
    elo_source: str = ""


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
            tournament=raw.get("tournament", "wc2026"),
            config_path=path,
        )
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc}") from exc
    return cfg


def available_tournaments(root: Path = _TOURNAMENTS_ROOT) -> list[str]:
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "tournament.yaml").exists())


def resolve_tournament(name: str, *, root: Path = _TOURNAMENTS_ROOT) -> TournamentBundle:
    """Load the bundle for ``name`` from ``<root>/<name>/tournament.yaml``."""
    bundle_dir = root / name
    spec_path = bundle_dir / "tournament.yaml"
    if not spec_path.exists():
        raise ValueError(
            f"Unknown tournament {name!r}. Available: {available_tournaments(root)}"
        )
    raw = yaml.safe_load(spec_path.read_text()) or {}
    return TournamentBundle(
        name=raw.get("name", name),
        display_name=raw.get("display_name", name),
        completed=bool(raw.get("completed", False)),
        dir=bundle_dir,
        teams_file=bundle_dir / raw.get("teams_file", "teams.csv"),
        fixtures_file=bundle_dir / raw.get("fixtures_file", "fixtures.csv"),
        results_file=bundle_dir / raw.get("results_file", "results.csv"),
        bracket_map_file=bundle_dir / raw.get("bracket_map_file", "bracket_map.json"),
        bonus_questions=[
            BonusQuestionConfig(id=q["id"], points=int(q["points"]))
            for q in raw.get("bonus_questions", [])
        ],
        elo_source=raw.get("elo_source", ""),
    )
