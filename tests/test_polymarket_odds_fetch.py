"""Tests for the Polymarket odds fetcher (tippspiel/data/polymarket_odds_fetch.py).

Network is fully mocked: ``_get_json`` is monkeypatched, so these exercise the parsing,
orientation, sanity guards, slug construction and CSV writing without hitting Gamma.
"""

import csv

import tippspiel.data.polymarket_odds_fetch as pm

NAME2ID = {"alpha": "AAA", "beta": "BBB"}


def _event(home="Alpha", away="Beta", p_home="0.60", p_draw="0.25", p_away="0.20"):
    """A Gamma match event: three binary moneyline markets (two teams + draw)."""
    def mk(title, yes):
        return {"groupItemTitle": title, "outcomes": '["Yes", "No"]',
                "outcomePrices": f'["{yes}", "0.5"]'}
    return {"markets": [mk(home, p_home), mk(f"Draw ({home} vs. {away})", p_draw), mk(away, p_away)]}


def test_trio_oriented_by_team_identity():
    ph, pd, pa = pm._trio_from_event(_event(), "AAA", "BBB", NAME2ID)
    # Normalised to sum 1, oriented to the repo home (Alpha) / away (Beta).
    assert abs(ph + pd + pa - 1.0) < 1e-9
    assert ph > pa  # Alpha is the favourite
    assert abs(ph - 0.60 / 1.05) < 1e-6


def test_trio_orientation_independent_of_market_order():
    # Even if the away team's market is listed first, identity (not position) decides home/away.
    ev = _event()
    ev["markets"].reverse()
    ph, pd, pa = pm._trio_from_event(ev, "AAA", "BBB", NAME2ID)
    assert abs(ph - 0.60 / 1.05) < 1e-6
    assert abs(pa - 0.20 / 1.05) < 1e-6


def test_trio_rejects_event_with_wrong_teams():
    # A mis-hit event (teams don't match the fixture) yields no trio.
    assert pm._trio_from_event(_event(home="Gamma", away="Delta"), "AAA", "BBB", NAME2ID) is None


def test_trio_rejects_insane_booksum():
    # Prices that imply a wildly off book (sum far from 1) are rejected.
    assert pm._trio_from_event(_event(p_home="0.9", p_draw="0.9", p_away="0.9"),
                               "AAA", "BBB", NAME2ID) is None


def test_candidate_slugs_cover_orders_and_dates():
    slugs = pm._candidate_slugs("fifwc", "alp", "bet", "2026-06-28T19:00:00+00:00")
    assert "fifwc-alp-bet-2026-06-28" in slugs   # home-first, exact date
    assert "fifwc-bet-alp-2026-06-28" in slugs   # away-first fallback
    assert "fifwc-alp-bet-2026-06-27" in slugs   # date - 1 (local/UTC offset)
    assert "fifwc-alp-bet-2026-06-29" in slugs   # date + 1


def test_team_codes_maps_name_and_alias(monkeypatch):
    monkeypatch.setattr(pm, "_get_json", lambda url: [
        {"name": "Alpha", "abbreviation": "alp"},
        {"name": "Beta Republic", "alias": "Beta", "abbreviation": "bet"},
    ])
    codes = pm.team_codes("fifwc", {"alpha": "AAA", "beta": "BBB"})
    assert codes == {"AAA": "alp", "BBB": "bet"}


def test_fetch_odds_end_to_end(monkeypatch, tmp_path):
    fixtures = [{
        "match_id": "M1", "stage": "R32", "date": "20260628",
        "home_id": "AAA", "away_id": "BBB",
        "kickoff_utc": "2026-06-28T19:00:00+00:00", "venue_country": "USA",
    }]
    monkeypatch.setattr(pm, "load_teams", lambda tdir: dict(NAME2ID))
    monkeypatch.setattr(pm, "load_tippable_fixtures", lambda tdir: fixtures)

    def fake_get(url):
        if "/teams" in url:
            return [{"name": "Alpha", "abbreviation": "alp"},
                    {"name": "Beta", "abbreviation": "bet"}]
        if url.endswith("/events/slug/fifwc-alp-bet-2026-06-28"):
            return _event()
        return None  # 404 for every other candidate slug

    monkeypatch.setattr(pm, "_get_json", fake_get)
    out = tmp_path / "odds_polymarket.csv"
    n = pm.fetch_odds("wc2026", "fifwc", out_path=out)
    assert n == 1

    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    r = rows[0]
    assert r["match_id"] == "M1"
    assert float(r["odds_home"]) < float(r["odds_away"])  # Alpha favourite -> shorter price


def test_fetch_odds_skips_unposted_match(monkeypatch, tmp_path):
    fixtures = [{
        "match_id": "M2", "stage": "R32", "date": "20260701",
        "home_id": "AAA", "away_id": "BBB",
        "kickoff_utc": "2026-07-01T19:00:00+00:00", "venue_country": "USA",
    }]
    monkeypatch.setattr(pm, "load_teams", lambda tdir: dict(NAME2ID))
    monkeypatch.setattr(pm, "load_tippable_fixtures", lambda tdir: fixtures)
    monkeypatch.setattr(pm, "_get_json", lambda url: (
        [{"name": "Alpha", "abbreviation": "alp"}, {"name": "Beta", "abbreviation": "bet"}]
        if "/teams" in url else None  # no event posted yet
    ))
    out = tmp_path / "odds_polymarket.csv"
    assert pm.fetch_odds("wc2026", "fifwc", out_path=out) == 0
    assert list(csv.DictReader(out.open())) == []  # header only, no rows
