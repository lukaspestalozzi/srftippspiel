"""Elo-builder report: global ranking + computed-vs-current comparison.

Writes ``output/elo.md`` + ``output/elo.json``, mirroring ``report/backtest.py`` (a ``build_*``
returns ``(markdown, data)``; a ``*Writer`` persists both). The computed ratings will not equal
the eloratings.net snapshot exactly — different exact K history, their proprietary tweaks, and a
windowed vs full-history pass — so the comparison is for sanity/scale, not bit-parity.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from ..elo.names import normalize
from .diagnostics import _fixed_table, _json_default


def build_elo_report(
    bundle, ratings, current_teams, canonical_map, conflicts, meta, *, top=30
) -> tuple[str, dict]:
    ranked = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)
    ranking = [
        {
            "rank": i + 1,
            "name": key.title(),
            "team_id": canonical_map.get(key, ""),
            "elo": round(elo, 2),
        }
        for i, (key, elo) in enumerate(ranked[:top])
    ]

    comparison: list[dict] = []
    unmatched: list[str] = []
    for t in current_teams:
        computed = ratings.get(normalize(t.name))
        if computed is None:
            unmatched.append(t.name)
            comparison.append(
                {"team_id": t.team_id, "name": t.name, "current": t.elo,
                 "computed": None, "delta": None}
            )
        else:
            comparison.append(
                {"team_id": t.team_id, "name": t.name, "current": t.elo,
                 "computed": round(computed, 2), "delta": round(computed - t.elo, 2)}
            )
    comparison.sort(key=lambda r: (r["computed"] is None, -(r["computed"] or 0.0)))

    matched = [r for r in comparison if r["computed"] is not None]
    summary = {
        "n_teams": len(current_teams),
        "n_matched": len(matched),
        "current_stdev": round(statistics.pstdev([r["current"] for r in matched]), 2)
        if len(matched) > 1 else 0.0,
        "computed_stdev": round(statistics.pstdev([r["computed"] for r in matched]), 2)
        if len(matched) > 1 else 0.0,
        "mean_abs_delta": round(
            sum(abs(r["delta"]) for r in matched) / len(matched), 2
        ) if matched else 0.0,
    }

    data = {
        "meta": meta,
        "summary": summary,
        "ranking": ranking,
        "comparison": comparison,
        "unmatched": unmatched,
        "conflicts": conflicts,
    }
    return _render(bundle, data), data


def _render(bundle, data) -> str:
    m = data["meta"]
    s = data["summary"]
    L = [f"# Computed Elo — {bundle.display_name}", ""]
    L.append(
        f"World Football Elo (`{m['model']}`) computed from historical results as of "
        f"**{m['as_of']}**. Window {m['lookback_years']}y; "
        + (f"recency half-life {m['half_life_years']}y." if m["recency_decay"]
           else "recency decay off.")
    )
    L.append(
        f"Source: {m['source_url']} (content {m['content_hash']}); "
        f"{m['n_matches_used']} of {m['n_matches_total']} matches in window; "
        f"{m['n_teams_rated']} teams rated."
    )
    L.append("")
    L.append(
        f"Active-tournament scale: current stdev {s['current_stdev']} vs computed "
        f"{s['computed_stdev']}; mean |delta| {s['mean_abs_delta']} over {s['n_matched']}"
        f"/{s['n_teams']} matched teams."
    )
    L.append("")

    L.append(f"## Top {len(data['ranking'])} (global)")
    L.append(_fixed_table(
        ["#", "team", "id", "elo"],
        [[r["rank"], r["name"], r["team_id"], f"{r['elo']:.1f}"] for r in data["ranking"]],
    ))
    L.append("")

    L.append("## Active tournament: computed vs current")
    rows = []
    for r in data["comparison"]:
        computed = f"{r['computed']:.1f}" if r["computed"] is not None else "—"
        delta = f"{r['delta']:+.1f}" if r["delta"] is not None else "—"
        rows.append([r["team_id"], r["name"], f"{r['current']:.1f}", computed, delta])
    L.append(_fixed_table(["id", "team", "current", "computed", "delta"], rows))
    L.append("")

    if data["unmatched"]:
        L.append("## Unmatched names (no computed rating — add an alias in elo/names.py)")
        L.append(", ".join(data["unmatched"]))
        L.append("")
    if data["conflicts"]:
        L.append("## Name->id conflicts in the canonical map (report-only)")
        L.append(", ".join(data["conflicts"]))
        L.append("")
    return "\n".join(L)


class EloReportWriter:
    def write(self, markdown: str, data: dict, output_dir: str | Path) -> dict[str, Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / "elo.md"
        json_path = out_dir / "elo.json"
        md_path.write_text(markdown, encoding="utf-8")
        json_path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
        return {"markdown": md_path, "json": json_path}
