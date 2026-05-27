"""Configuration for the World Football Elo builder (the ``tippspiel build-elo`` tool).

All fields are defaulted, so a tournament config without an ``elo:`` block still works. The
``tier_k`` table maps the historical dataset's ``tournament`` column to an importance K via an
ordered substring scan (first match wins) — qualifier keywords MUST precede the bare
competition names so "FIFA World Cup qualification" lands in the 40 tier, not the 60 tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# Ordered (substring, K) pairs, scanned case-insensitively; first containment match wins.
# World Football Elo (eloratings.net) importance weights.
DEFAULT_TIER_K: list[tuple[str, float]] = [
    ("friendly", 20.0),
    ("qualification", 40.0),
    ("qualifier", 40.0),
    ("nations league", 40.0),
    ("confederations cup", 40.0),
    ("fifa world cup", 60.0),
    ("world cup", 60.0),
    ("copa america", 60.0),
    ("uefa euro", 60.0),
    ("european championship", 60.0),
    ("african cup of nations", 60.0),
    ("afc asian cup", 60.0),
    ("gold cup", 60.0),
    ("olympic", 60.0),
]

DEFAULT_TIER_K_FALLBACK = 30.0


@dataclass(frozen=True)
class EloConfig:
    source_url: str = DEFAULT_SOURCE_URL
    cache_dir: str = "~/.cache/tippspiel"
    cache_max_age_days: float = 1.0
    lookback_years: int = 25
    recency_decay: bool = True
    half_life_years: float = 8.0
    seed_rating: float = 1500.0
    home_advantage: float = 100.0
    model: str = "world_football"
    tier_k: list[tuple[str, float]] = field(default_factory=lambda: list(DEFAULT_TIER_K))
    tier_k_fallback: float = DEFAULT_TIER_K_FALLBACK


def _coerce_tier_k(tier_k) -> list[tuple[str, float]]:
    # Accept either a mapping (insertion order preserved in YAML) or a list of [substring, K].
    items = tier_k.items() if isinstance(tier_k, dict) else tier_k
    return [(str(sub).lower(), float(k)) for sub, k in items]


def load_elo_config(raw_elo: dict | None) -> EloConfig:
    raw = dict(raw_elo or {})
    tier_k = raw.get("tier_k")
    return EloConfig(
        source_url=raw.get("source_url", DEFAULT_SOURCE_URL),
        cache_dir=raw.get("cache_dir", "~/.cache/tippspiel"),
        cache_max_age_days=float(raw.get("cache_max_age_days", 1.0)),
        lookback_years=int(raw.get("lookback_years", 25)),
        recency_decay=bool(raw.get("recency_decay", True)),
        half_life_years=float(raw.get("half_life_years", 8.0)),
        seed_rating=float(raw.get("seed_rating", 1500.0)),
        home_advantage=float(raw.get("home_advantage", 100.0)),
        model=raw.get("model", "world_football"),
        tier_k=_coerce_tier_k(tier_k) if tier_k is not None else list(DEFAULT_TIER_K),
        tier_k_fallback=float(raw.get("tier_k_fallback", DEFAULT_TIER_K_FALLBACK)),
    )
