"""Maintainer tool: blend several ``odds.csv`` sources into one consensus ``odds.csv``.

Offline, **not on the runtime path**. The runtime consumes a single ``odds.csv``; this builds it
from one or more committed source snapshots (e.g. ``odds_espn.csv`` + ``odds_polymarket.csv``) so
the predictor sees a sharper market consensus than any one book. Each source row is de-vigged to a
probability triple (the same proportional de-vig the runtime applies at load), the triples are
**averaged per match** (weighted; a match present in only one source passes through unchanged), and
the blended probabilities are written back as decimal odds ``1/p`` — so the runtime's load-time
de-vig is a near-identity and ``odds.csv`` stays the same auditable schema.

**Frozen-odds rule:** when the output file sits in a tournament data dir (``fixtures.csv`` next to
it), rows for matches that are played or have kicked off are preserved **verbatim** from the
existing output instead of being re-blended — the committed consensus price of a started match is
a historical snapshot and must never change, even if the source sidecars later differ.

Usage (run from the repo root)::

    python -m tippspiel.data.odds_consensus tournaments/wc2026/odds.csv \\
        tournaments/wc2026/odds_espn.csv tournaments/wc2026/odds_polymarket.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tippspiel.data.file_provider import read_odds_file
from tippspiel.data.fixture_resolve import frozen_match_ids, write_odds_preserving_frozen


def _read_source(path: Path) -> dict[str, tuple[float, float, float]]:
    """De-vigged (p_home, p_draw, p_away) per match_id from one odds.csv-schema file ({} if absent).

    Uses the shared ``read_odds_file`` reader, so a malformed source fails fast with file context
    rather than silently dropping rows.
    """
    return {mid: (o.p_home, o.p_draw, o.p_away) for mid, o in read_odds_file(path).items()}


def build_consensus(sources: list[str | Path], out_path: str | Path,
                    weights: list[float] | None = None,
                    frozen: set[str] | None = None) -> int:
    """Blend ``sources`` (odds.csv-schema files) into ``out_path``. Returns rows written.

    ``weights`` is one weight per source (default equal). A match present in a subset of sources is
    averaged over just those, so coverage gaps degrade cleanly rather than dropping the fixture.
    ``frozen`` match ids keep their existing ``out_path`` row verbatim instead of being re-blended
    (see the frozen-odds rule in the module docstring); the CLI derives it automatically.
    """
    paths = [Path(s) for s in sources]
    if weights is None:
        weights = [1.0] * len(paths)
    if len(weights) != len(paths):
        raise ValueError("weights must match the number of sources")
    # Negative weights can yield negative/zero blended probabilities (invalid 1/p odds); an all-zero
    # set would silently drop every fixture. Reject both up front.
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    if sum(weights) <= 0:
        raise ValueError("weights must sum to a positive value")
    parsed = [_read_source(p) for p in paths]

    all_ids = sorted({mid for src in parsed for mid in src})
    rows_out = []
    for mid in all_ids:
        acc = [0.0, 0.0, 0.0]
        wsum = 0.0
        for src, w in zip(parsed, weights):
            trio = src.get(mid)
            if trio is None:
                continue
            for k in range(3):
                acc[k] += w * trio[k]
            wsum += w
        if wsum <= 0.0:
            continue
        probs = [a / wsum for a in acc]
        total = sum(probs)
        ph, pd, pa = (p / total for p in probs)
        rows_out.append({
            "match_id": mid,
            "odds_home": f"{1.0 / ph:.2f}",
            "odds_draw": f"{1.0 / pd:.2f}",
            "odds_away": f"{1.0 / pa:.2f}",
        })

    out = Path(out_path)
    total, kept = write_odds_preserving_frozen(out, rows_out, frozen or set())
    print(f"consensus: wrote {total} rows to {out} from {len(paths)} source(s) "
          f"({kept} frozen rows preserved)")
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Blend odds.csv sources into a consensus odds.csv")
    ap.add_argument("out", help="output odds.csv path")
    ap.add_argument("sources", nargs="+", help="source odds.csv files to blend")
    ap.add_argument("--weights", default=None,
                    help="comma-separated weight per source (default: equal)")
    args = ap.parse_args(argv)
    weights = [float(w) for w in args.weights.split(",")] if args.weights else None
    # The consumed odds.csv lives in the tournament data dir; when that's where we're writing,
    # apply the frozen-odds rule (played/kicked-off rows stay verbatim).
    tdir = Path(args.out).parent
    frozen = frozen_match_ids(tdir) if (tdir / "fixtures.csv").exists() else None
    build_consensus(args.sources, args.out, weights, frozen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
