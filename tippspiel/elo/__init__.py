"""World Football Elo builder: compute team Elo ratings from historical match results.

Pipeline: fetch results.csv -> parse + normalize names -> window + recency-decay -> fold a
``RatingModel`` over the chronological stream -> emit ratings. The ``RatingModel`` ABC is the
seam for a future attack/defence model; ``build_model`` selects the implementation by config.
"""

from .attack_defence import AttackDefenceElo
from .config import EloConfig, load_elo_config
from .fetch import get_results_csv
from .matches import HistoricalMatch, parse_csv_text, prepare_matches
from .ratings import RatingModel, build_ratings, run_forward_pass
from .world_football import WorldFootballElo

__all__ = [
    "EloConfig",
    "load_elo_config",
    "get_results_csv",
    "HistoricalMatch",
    "parse_csv_text",
    "prepare_matches",
    "RatingModel",
    "build_ratings",
    "run_forward_pass",
    "WorldFootballElo",
    "AttackDefenceElo",
    "build_model",
]


def build_model(cfg: EloConfig) -> RatingModel:
    """Construct the rating model named by ``cfg.model``."""
    if cfg.model == "world_football":
        return WorldFootballElo(cfg)
    if cfg.model == "attack_defence":
        return AttackDefenceElo(cfg)
    raise ValueError(f"Unknown elo model: {cfg.model!r}")
