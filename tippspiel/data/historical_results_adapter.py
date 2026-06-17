"""Adapter for the historical international-match corpus that feeds the off/def Elo fit.

Source: Mart Jürisoo's "International football results from 1872 to present"
(``github.com/martj42/international_results``), committed verbatim at
``tippspiel/data/historical/international_results.csv`` so the fit is reproducible offline
(columns: ``date,home_team,away_team,home_score,away_score,tournament,city,country,neutral``).

Two jobs:
  * ``load_corpus`` — parse the CSV into ``HistMatch`` records, dropping unplayed rows (future
    fixtures carry ``NA`` scores) and anything on/after a ``before`` snapshot date, and tag each
    with a FIFA-style importance weight derived from the ``tournament`` column.
  * ``corpus_name_for`` / ``ratings_for_team`` — resolve a tournament ``teams.csv`` display name
    to the corpus's full country name (a tiny alias table covers the few mismatches), so a fitted
    rating can be looked up per team_id.

The fitter tracks **every** corpus team (minnows included) so opponents' strengths are right;
only the subset that maps to a tournament's teams is exported back into its ``teams.csv``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ..training.offdef_elo import HistMatch, OffDefRating

_HISTORICAL_DIR = Path(__file__).parent / "historical"
DEFAULT_CORPUS = _HISTORICAL_DIR / "international_results.csv"

# Our teams.csv display names vs. the corpus's full country names. Only genuine mismatches
# need listing; everything else matches verbatim.
_NAME_ALIASES = {
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
}


@dataclass(frozen=True)
class WeightTiers:
    """FIFA-Elo-style match-importance weights, by competition tier."""

    friendly: float = 0.5
    qualifier: float = 2.5
    continental: float = 3.0
    world_cup: float = 4.0
    default: float = 1.0


# Major continental (and inter-confederation) finals -> the continental tier. Qualifiers are
# detected by the "qualification" suffix and the Nations Leagues by name, so only the finals
# competitions are enumerated here.
_CONTINENTAL = frozenset({
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "CONCACAF Championship",
    "Oceania Nations Cup",
    "Confederations Cup",
    "Finalissima",
})
_WORLD_CUP = frozenset({"FIFA World Cup"})


@dataclass(frozen=True)
class KTiers:
    """World-Football-Elo K-factor base, by competition tier (used by the scalar-Elo fit).

    These are the canonical eloratings.net tournament weights: a World Cup final match moves
    ratings 3x as much as a friendly. Distinct from ``WeightTiers`` (the off/def goal-fit
    weights) — the two scales are tuned independently."""

    friendly: float = 20.0
    qualifier: float = 40.0
    minor: float = 30.0
    continental: float = 50.0
    world_cup: float = 60.0


def classify_weight(tournament: str, tiers: WeightTiers = WeightTiers()) -> float:
    """Map a corpus ``tournament`` label to its off/def importance weight."""
    t = tournament.strip()
    if t == "Friendly":
        return tiers.friendly
    if "qualification" in t or "Nations League" in t:
        return tiers.qualifier
    if t in _WORLD_CUP:
        return tiers.world_cup
    if t in _CONTINENTAL:
        return tiers.continental
    return tiers.default


def elo_k_importance(tournament: str, tiers: KTiers = KTiers()) -> float:
    """Map a corpus ``tournament`` label to its World-Football-Elo K base.

    Mirrors ``classify_weight``'s bucketing but on the eloratings K scale: World Cup finals
    (60) > continental finals / Confederations Cup (50) > qualifiers + Nations Leagues (40) >
    other minor tournaments (30) > friendlies (20)."""
    t = tournament.strip()
    if t == "Friendly":
        return tiers.friendly
    if "qualification" in t or "Nations League" in t:
        return tiers.qualifier
    if t in _WORLD_CUP:
        return tiers.world_cup
    if t in _CONTINENTAL:
        return tiers.continental
    return tiers.minor


def load_corpus(
    *,
    before: str | None = None,
    corpus_path: str | Path = DEFAULT_CORPUS,
    tiers: WeightTiers = WeightTiers(),
    k_tiers: KTiers = KTiers(),
) -> list[HistMatch]:
    """Load played historical matches as weighted ``HistMatch`` records.

    ``before`` (ISO ``yyyy-mm-dd``), when given, keeps only matches strictly earlier — the
    pre-tournament snapshot cutoff that keeps a ``verify`` backtest free of result leakage.
    Each record carries both the off/def ``weight`` and the scalar-Elo ``k_importance`` so the
    corpus is parsed once and feeds both fitters.
    """
    out: list[HistMatch] = []
    with Path(corpus_path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            date = (row.get("date") or "").strip()
            if not date or (before is not None and date >= before):
                continue
            hg, ag = (row.get("home_score") or "").strip(), (row.get("away_score") or "").strip()
            if not hg or not ag or hg == "NA" or ag == "NA":
                continue  # unplayed (future) fixture
            tournament = row.get("tournament") or ""
            out.append(HistMatch(
                date=date,
                home=(row["home_team"]).strip(),
                away=(row["away_team"]).strip(),
                home_goals=int(hg),
                away_goals=int(ag),
                weight=classify_weight(tournament, tiers),
                neutral=(row.get("neutral") or "").strip().upper() == "TRUE",
                k_importance=elo_k_importance(tournament, k_tiers),
            ))
    return out


def corpus_name_for(display_name: str) -> str:
    """Resolve a teams.csv display name to its corpus country name (alias table + identity)."""
    return _NAME_ALIASES.get(display_name, display_name)


def ratings_for_team(
    display_name: str, ratings: dict[str, OffDefRating]
) -> OffDefRating:
    """Look up a team's fitted rating by display name; (0, 0) if it never appears in the corpus."""
    return ratings.get(corpus_name_for(display_name), OffDefRating(0.0, 0.0))
