"""Tests for the odds consensus merge (tippspiel/data/odds_consensus.py)."""

import csv

from tippspiel.data.file_provider import _devig_proportional
from tippspiel.data.odds_consensus import build_consensus


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["match_id", "odds_home", "odds_draw", "odds_away"])
        w.writeheader()
        w.writerows(rows)


def _probs(path):
    """De-vigged probability triple per match_id from an odds.csv-schema file."""
    out = {}
    for r in csv.DictReader(path.open()):
        o = _devig_proportional(float(r["odds_home"]), float(r["odds_draw"]), float(r["odds_away"]))
        out[r["match_id"]] = (o.p_home, o.p_draw, o.p_away)
    return out


def test_two_sources_average_in_probability_space(tmp_path):
    # Decimals are 1/p with p summing to 1, so de-vig is identity and the inputs are exact probs.
    # Source A: (0.5, 0.3, 0.2);  Source B: (0.3, 0.3, 0.4)  ->  mean (0.4, 0.3, 0.3).
    a, b, out = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "out.csv"
    _write(a, [{"match_id": "X", "odds_home": "2.0", "odds_draw": "3.3333", "odds_away": "5.0"}])
    _write(b, [{"match_id": "X", "odds_home": "3.3333", "odds_draw": "3.3333", "odds_away": "2.5"}])

    assert build_consensus([a, b], out) == 1
    ph, pd, pa = _probs(out)["X"]
    assert abs(ph - 0.40) < 0.01
    assert abs(pd - 0.30) < 0.01
    assert abs(pa - 0.30) < 0.01


def test_single_source_fixture_passes_through(tmp_path):
    a, b, out = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "out.csv"
    # Y is only in source A -> it should survive unchanged (averaged over the one source present).
    _write(a, [{"match_id": "Y", "odds_home": "1.6667", "odds_draw": "4.0", "odds_away": "6.6667"}])
    _write(b, [{"match_id": "Z", "odds_home": "2.0", "odds_draw": "4.0", "odds_away": "4.0"}])

    assert build_consensus([a, b], out) == 2
    p = _probs(out)
    assert set(p) == {"Y", "Z"}
    ph, pd, pa = p["Y"]
    assert abs(ph - 0.60) < 0.01 and abs(pd - 0.25) < 0.01 and abs(pa - 0.15) < 0.01


def test_missing_source_file_is_skipped(tmp_path):
    a, out = tmp_path / "a.csv", tmp_path / "out.csv"
    _write(a, [{"match_id": "X", "odds_home": "2.0", "odds_draw": "4.0", "odds_away": "4.0"}])
    # A non-existent second source contributes nothing rather than erroring.
    assert build_consensus([a, tmp_path / "nope.csv"], out) == 1
    assert set(_probs(out)) == {"X"}


def test_weights_bias_the_blend(tmp_path):
    a, b, out = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "out.csv"
    _write(a, [{"match_id": "X", "odds_home": "2.0", "odds_draw": "3.3333", "odds_away": "5.0"}])
    _write(b, [{"match_id": "X", "odds_home": "3.3333", "odds_draw": "3.3333", "odds_away": "2.5"}])
    # Weight source A x3 -> blend leans toward A's (0.5, 0.3, 0.2): home prob > the equal-weight 0.4.
    build_consensus([a, b], out, weights=[3.0, 1.0])
    ph, _, _ = _probs(out)["X"]
    assert ph > 0.44
