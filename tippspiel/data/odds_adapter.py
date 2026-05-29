"""Adapter: raw bookmaker 1X2 export -> odds.csv schema (spec §6.2.6).

Offline maintainer tool, mirroring ``eloratings_adapter``. Bookmaker exports key matches by
their own ids / team-name pairs; this maps them to our ``match_id`` and writes the committed
``odds.csv`` (``match_id,odds_home,odds_draw,odds_away``, raw decimal odds — de-vigging happens
at load in ``FileDataProvider.get_odds``). Not on the runtime path.
"""

from __future__ import annotations

import csv
from pathlib import Path


def convert_odds_export(
    src: str | Path,
    match_id_map: dict[str, str],
    out_path: str | Path,
    key_col: str = "key",
    home_col: str = "odds_home",
    draw_col: str = "odds_draw",
    away_col: str = "odds_away",
) -> int:
    """Convert a raw 1X2 odds export into odds.csv. Returns rows written.

    ``match_id_map`` maps each source row's ``key_col`` value to our ``match_id``; only rows
    with a mapping are emitted. Raw decimal odds are passed through unchanged.
    """
    rows_out = []
    with Path(src).open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row.get(key_col) or "").strip()
            match_id = match_id_map.get(key)
            if not match_id:
                continue
            rows_out.append(
                {
                    "match_id": match_id,
                    "odds_home": (row.get(home_col) or "").strip(),
                    "odds_draw": (row.get(draw_col) or "").strip(),
                    "odds_away": (row.get(away_col) or "").strip(),
                }
            )

    fieldnames = ["match_id", "odds_home", "odds_draw", "odds_away"]
    with Path(out_path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    return len(rows_out)
