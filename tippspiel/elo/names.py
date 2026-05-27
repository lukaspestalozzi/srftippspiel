"""Country-name normalization mapping the historical dataset's names to the repo's team_ids.

The dataset uses full English country names; each tournament's ``teams.csv`` pairs a 3-letter
``team_id`` with its own name string. Both sides are reduced to a canonical key (accent/case/
whitespace-normalized, then aliased). The authoritative name->id map for emission comes from a
SINGLE tournament's teams.csv (``build_name_to_id``); the cross-tournament union
(``build_canonical_map``) is non-injective — e.g. KSA and SAU both name "Saudi Arabia" — so it
is used only for the ranking report, never for emitting a tournament's ratings.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

# normalized name -> canonical normalized name. Keys are in `_basic_normalize` form (the
# accent-stripped, casefolded variant), so e.g. "Türkiye" arrives here as "turkiye".
ALIASES: dict[str, str] = {
    "korea republic": "south korea",
    "korea dpr": "north korea",
    "ir iran": "iran",
    "china pr": "china",
    "cote d'ivoire": "ivory coast",
    "usa": "united states",
    "united states of america": "united states",
    "czech republic": "czechia",
    "turkiye": "turkey",
    "cabo verde": "cape verde",
    "cape verde islands": "cape verde",
    "the gambia": "gambia",
    "congo dr": "dr congo",
    "democratic republic of the congo": "dr congo",
    "republic of ireland": "ireland",
    "bosnia-herzegovina": "bosnia and herzegovina",
    "macedonia": "north macedonia",
    "fyr macedonia": "north macedonia",
}


def _basic_normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name.strip())
    s = "".join(c for c in s if not unicodedata.combining(c))  # strip accents
    return " ".join(s.split()).casefold()  # collapse whitespace + casefold


def normalize(name: str) -> str:
    """Canonical key for a country name: accent/case/whitespace-normalized, then aliased."""
    key = _basic_normalize(name)
    return ALIASES.get(key, key)


def build_name_to_id(teams) -> dict[str, str]:
    """normalized-name -> team_id from ONE tournament's teams (authoritative for emission)."""
    return {normalize(t.name): t.team_id for t in teams}


def build_canonical_map(data_root: str | Path) -> tuple[dict[str, str], list[str]]:
    """Union of all tournaments' (normalized-name -> team_id) pairs, for the ranking report.

    Returns ``(mapping, conflicts)``. A name that resolves to more than one id (KSA/SAU both
    "Saudi Arabia") keeps the first id seen and records the clash in ``conflicts`` — never raises.
    """
    mapping: dict[str, str] = {}
    conflicts: list[str] = []
    tournaments_dir = Path(data_root) / "tournaments"
    for teams_file in sorted(tournaments_dir.glob("*/teams.csv")):
        with teams_file.open(newline="") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("name") or "").strip()
                team_id = (row.get("team_id") or "").strip()
                if not name or not team_id:
                    continue
                key = normalize(name)
                if key in mapping and mapping[key] != team_id:
                    conflicts.append(f"{name!r}: {mapping[key]} vs {team_id}")
                    continue
                mapping.setdefault(key, team_id)
    return mapping, conflicts
