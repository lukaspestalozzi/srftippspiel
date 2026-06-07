"""Adapter: eloratings.net World.tsv-style export -> teams.csv schema (spec §6.1.2).

eloratings.net is the upstream Elo source. Its export columns vary; this adapter maps
the common form (rank, name, rating, ...) to our `team_id,name,elo` schema.
A name->team_id mapping must be supplied because eloratings uses full country names.
"""

from __future__ import annotations

import csv
from pathlib import Path


def convert_world_tsv(
    tsv_path: str | Path,
    name_to_id: dict[str, str],
    out_path: str | Path,
    rating_col: str = "rating",
    name_col: str = "name",
) -> int:
    """Convert an eloratings World.tsv export into teams.csv. Returns rows written.

    Only teams present in ``name_to_id`` (the 48 qualified teams) are emitted.
    """
    rows_out = []
    with Path(tsv_path).open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            name = (row.get(name_col) or "").strip()
            team_id = name_to_id.get(name)
            if not team_id:
                continue
            rows_out.append(
                {
                    "team_id": team_id,
                    "name": name,
                    "elo": (row.get(rating_col) or "").strip(),
                }
            )

    with Path(out_path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["team_id", "name", "elo"])
        writer.writeheader()
        writer.writerows(rows_out)
    return len(rows_out)
