"""Off/def Elo fitter + historical-corpus adapter tests."""

from __future__ import annotations

import csv

import pytest

from tippspiel.data.historical_results_adapter import (
    classify_weight,
    corpus_name_for,
    load_corpus,
    ratings_for_team,
)
from tippspiel.training.offdef_elo import (
    HistMatch,
    OffDefParams,
    OffDefRating,
    fit_off_def,
    fit_off_def_history,
)


def _match(home, away, hg, ag, *, date="2020-01-01", weight=1.0, neutral=True):
    return HistMatch(date, home, away, hg, ag, weight, neutral)


def _round_robin_with_dominant():
    """A clearly dominant side HI (wins 4:0 vs a pool) plus average opponents drawing 1:1."""
    opps = ["O1", "O2", "O3", "O4"]
    matches = []
    for i, o in enumerate(opps):
        matches.append(_match("HI", o, 4, 0, date=f"2020-01-0{i + 1}"))
    # opponents are mutually average
    for i in range(len(opps)):
        for j in range(i + 1, len(opps)):
            matches.append(_match(opps[i], opps[j], 1, 1, date=f"2020-02-0{i}{j}"))
    return matches


def test_dominant_attacker_gets_high_att_and_def():
    ratings = fit_off_def(_round_robin_with_dominant(), OffDefParams(epochs=5))
    hi = ratings["HI"]
    others = [r for k, r in ratings.items() if k != "HI"]
    # Scores a lot -> attack above the field; concedes nothing -> defence above the field.
    assert hi.att > max(o.att for o in others)
    assert hi.def_ > max(o.def_ for o in others)


def test_ratings_are_zero_centred():
    ratings = fit_off_def(_round_robin_with_dominant())
    n = len(ratings)
    assert sum(r.att for r in ratings.values()) == pytest.approx(0.0, abs=1e-9)
    assert sum(r.def_ for r in ratings.values()) == pytest.approx(0.0, abs=1e-9)
    assert n == 5


def test_fit_is_deterministic():
    matches = _round_robin_with_dominant()
    a = fit_off_def(matches)
    b = fit_off_def(matches)
    assert a == b


def test_input_order_does_not_matter():
    # The fitter sorts by date, so a shuffled list yields identical ratings.
    matches = _round_robin_with_dominant()
    shuffled = list(reversed(matches))
    assert fit_off_def(matches) == fit_off_def(shuffled)


def test_importance_weight_amplifies_update():
    base = [_match("A", "B", 3, 0, date="2020-01-01", weight=0.5)]
    heavy = [_match("A", "B", 3, 0, date="2020-01-01", weight=4.0)]
    a_light = fit_off_def(base, OffDefParams(epochs=1))["A"]
    a_heavy = fit_off_def(heavy, OffDefParams(epochs=1))["A"]
    # A out-scored expectation; the heavier match pushes its attack rating further.
    assert a_heavy.att > a_light.att


def test_no_matches_yields_empty():
    assert fit_off_def([]) == {}


def test_history_endpoint_matches_fit():
    # Recorded on the final epoch + shifted by the final field means, so a tracked team's last
    # point equals its exported (centred) rating.
    matches = _round_robin_with_dominant()
    fitted = fit_off_def(matches)
    hist = fit_off_def_history(matches, track={"HI", "O1"})
    for team in ("HI", "O1"):
        d, att, def_ = hist[team][-1]
        assert att == pytest.approx(fitted[team].att)
        assert def_ == pytest.approx(fitted[team].def_)


def test_history_window_and_tracking():
    matches = _round_robin_with_dominant()
    hist = fit_off_def_history(matches, track={"HI"}, start_date="2020-02-01")
    assert set(hist) == {"HI"}
    # HI's four matches are all in January -> nothing recorded inside the window.
    assert hist["HI"] == []
    dates = [d for d, _, _ in fit_off_def_history(matches, track={"O1"})["O1"]]
    assert dates == sorted(dates) and len(dates) == 4  # O1 plays HI once + 3 peer draws


def test_residual_cap_limits_a_single_blowout():
    capped = fit_off_def([_match("A", "B", 30, 0)], OffDefParams(epochs=1, residual_cap=3.0))
    uncapped = fit_off_def([_match("A", "B", 30, 0)], OffDefParams(epochs=1, residual_cap=50.0))
    assert capped["A"].att < uncapped["A"].att


# --------------------------------------------------------------------------- adapter
def test_weight_classification():
    assert classify_weight("Friendly") == 0.5
    assert classify_weight("FIFA World Cup qualification") == 2.5
    assert classify_weight("UEFA Nations League") == 2.5
    assert classify_weight("FIFA World Cup") == 4.0
    assert classify_weight("UEFA Euro") == 3.0
    assert classify_weight("Copa América") == 3.0
    assert classify_weight("Merdeka Tournament") == 1.0  # minor competitive -> default


def test_name_aliases_resolve():
    assert corpus_name_for("Czechia") == "Czech Republic"
    assert corpus_name_for("Türkiye") == "Turkey"
    assert corpus_name_for("Spain") == "Spain"  # identity for the common case


def test_ratings_for_team_defaults_to_zero():
    ratings = {"Spain": OffDefRating(0.4, 0.2)}
    assert ratings_for_team("Spain", ratings) == OffDefRating(0.4, 0.2)
    assert ratings_for_team("Atlantis", ratings) == OffDefRating(0.0, 0.0)


def test_load_corpus_filters_na_and_snapshot(tmp_path):
    csv_path = tmp_path / "corpus.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "home_team", "away_team", "home_score", "away_score",
                    "tournament", "city", "country", "neutral"])
        w.writerow(["2019-06-01", "Spain", "Italy", "2", "1", "Friendly", "X", "Y", "FALSE"])
        w.writerow(["2030-01-01", "Spain", "Italy", "NA", "NA", "Friendly", "X", "Y", "TRUE"])
        w.writerow(["2025-01-01", "Spain", "Italy", "1", "1", "Friendly", "X", "Y", "TRUE"])
    rows = load_corpus(before="2026-01-01", corpus_path=csv_path)
    assert len(rows) == 2  # NA dropped, both before cutoff kept
    early = next(r for r in rows if r.date == "2019-06-01")
    assert (early.home, early.home_goals, early.weight, early.neutral) == ("Spain", 2, 0.5, False)
    # Snapshot cutoff excludes on/after the date.
    assert all(r.date < "2020-01-01" for r in load_corpus(before="2020-01-01", corpus_path=csv_path))
