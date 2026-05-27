"""Historical-match records + parsing/filtering for the Elo builder.

The dataset (martj42/international_results) columns are
``date,home_team,away_team,home_score,away_score,tournament,city,country,neutral``. Country
names are normalized to canonical keys at parse time, so all downstream code is name/id-agnostic.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, replace
from datetime import date, timedelta

from .config import EloConfig
from .names import normalize

_DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class HistoricalMatch:
    date: date
    home: str  # normalized country name, not a team_id
    away: str
    home_score: int
    away_score: int
    tournament: str
    neutral: bool
    weight: float = 1.0  # recency multiplier (0, 1], filled by prepare_matches


def parse_csv_text(text: str) -> list[HistoricalMatch]:
    """Parse the results CSV text into HistoricalMatch records.

    Rows with a missing/blank/non-integer score or unparseable date are skipped (unplayed or
    abandoned fixtures).
    """
    out: list[HistoricalMatch] = []
    for row in csv.DictReader(io.StringIO(text)):
        raw_h = (row.get("home_score") or "").strip()
        raw_a = (row.get("away_score") or "").strip()
        if not raw_h or not raw_a:
            continue
        try:
            home_score, away_score = int(raw_h), int(raw_a)
            match_date = date.fromisoformat((row.get("date") or "").strip())
        except (ValueError, KeyError):
            continue
        out.append(
            HistoricalMatch(
                date=match_date,
                home=normalize(row["home_team"]),
                away=normalize(row["away_team"]),
                home_score=home_score,
                away_score=away_score,
                tournament=(row.get("tournament") or "").strip(),
                neutral=(row.get("neutral") or "").strip().upper() == "TRUE",
            )
        )
    return out


def apply_window(
    matches: list[HistoricalMatch], as_of: date, lookback_years: int
) -> list[HistoricalMatch]:
    """Keep matches with ``cutoff <= date <= as_of`` where cutoff = as_of - lookback_years."""
    cutoff = as_of - timedelta(days=round(_DAYS_PER_YEAR * lookback_years))
    return [m for m in matches if cutoff <= m.date <= as_of]


def _decay_weight(match: HistoricalMatch, as_of: date, cfg: EloConfig) -> float:
    if not cfg.recency_decay:
        return 1.0
    age_years = (as_of - match.date).days / _DAYS_PER_YEAR
    return 0.5 ** (age_years / cfg.half_life_years)


def prepare_matches(
    matches: list[HistoricalMatch], as_of: date, cfg: EloConfig
) -> list[HistoricalMatch]:
    """Apply the lookback window, then attach the recency-decay weight to each match."""
    kept = apply_window(matches, as_of, cfg.lookback_years)
    return [replace(m, weight=_decay_weight(m, as_of, cfg)) for m in kept]
