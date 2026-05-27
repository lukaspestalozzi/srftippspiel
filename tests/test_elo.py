"""Tests for the World Football Elo builder (offline — no network)."""

from datetime import date, timedelta
from pathlib import Path

import pytest

import tippspiel
from tippspiel.elo.config import EloConfig, load_elo_config
from tippspiel.elo.matches import HistoricalMatch, apply_window, parse_csv_text, prepare_matches
from tippspiel.elo.names import build_canonical_map, normalize
from tippspiel.elo.ratings import build_ratings
from tippspiel.elo.world_football import WorldFootballElo, goal_difference_multiplier

REPO = Path(tippspiel.__file__).parent.parent

SAMPLE_CSV = """date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2010-06-01,Brazil,Argentina,2,1,Friendly,Rio,Brazil,FALSE
2012-06-01,Spain,Italy,4,0,UEFA Euro,Kyiv,Ukraine,TRUE
2014-07-13,Germany,Argentina,1,0,FIFA World Cup,Rio,Brazil,TRUE
2016-06-01,France,Germany,,,UEFA Euro,Paris,France,FALSE
2018-06-01,England,Croatia,1,2,FIFA World Cup,Moscow,Russia,TRUE
"""


def _m(home="A", away="B", hs=1, as_=0, tournament="Friendly", neutral=True, d=None):
    return HistoricalMatch(
        date=d or date(2020, 1, 1), home=home, away=away,
        home_score=hs, away_score=as_, tournament=tournament, neutral=neutral,
    )


# --------------------------------------------------------------------------- algorithm
def test_neutral_one_nil_even_teams_delta_is_half_k():
    cfg = EloConfig()
    model = WorldFootballElo(cfg)
    model.seed("A")
    model.seed("B")
    m = _m("A", "B", 1, 0, tournament="Friendly", neutral=True)  # K=20, G=1, We=0.5
    model.update(m)
    assert model.rating("A") == pytest.approx(1500 + 20 * 0.5)
    assert model.rating("B") == pytest.approx(1500 - 20 * 0.5)


def test_goal_difference_multiplier():
    assert goal_difference_multiplier(0) == 1.0
    assert goal_difference_multiplier(1) == 1.0
    assert goal_difference_multiplier(-1) == 1.0
    assert goal_difference_multiplier(2) == 1.5
    assert goal_difference_multiplier(3) == pytest.approx(14 / 8)
    assert goal_difference_multiplier(5) == pytest.approx(16 / 8)


def test_update_is_zero_sum():
    model = WorldFootballElo(EloConfig())
    model.seed("A")
    model.seed("B")
    before = model.rating("A") + model.rating("B")
    model.update(_m("A", "B", 3, 0, tournament="FIFA World Cup", neutral=False))
    assert model.rating("A") + model.rating("B") == pytest.approx(before)


def test_home_advantage_raises_expected_when_not_neutral():
    model = WorldFootballElo(EloConfig())
    model.seed("A")
    model.seed("B")
    neutral = model.expected(_m("A", "B", neutral=True))
    home = model.expected(_m("A", "B", neutral=False))
    assert neutral == pytest.approx(0.5)
    assert home > neutral


def test_tier_k_mapping_and_qualifier_precedence():
    model = WorldFootballElo(EloConfig())
    assert model.k_for("FIFA World Cup") == 60
    assert model.k_for("FIFA World Cup qualification") == 40  # qualifier beats bare world cup
    assert model.k_for("UEFA Nations League") == 40
    assert model.k_for("Friendly") == 20
    assert model.k_for("Some Random Cup") == 30  # fallback


# --------------------------------------------------------------------------- recency / window
def test_recency_decay_monotonic_and_half_life():
    cfg = EloConfig(half_life_years=8.0)
    as_of = date(2026, 1, 1)
    fresh = _m(d=as_of)
    old = _m(d=as_of - timedelta(days=round(365.25 * 8)))
    older = _m(d=as_of - timedelta(days=round(365.25 * 16)))
    [pf, po, poo] = [m.weight for m in prepare_matches([fresh, old, older], as_of, cfg)]
    assert pf > po > poo
    assert pf == pytest.approx(1.0)
    assert po == pytest.approx(0.5, abs=1e-3)
    assert poo == pytest.approx(0.25, abs=1e-3)


def test_recency_decay_off_is_unity():
    cfg = EloConfig(recency_decay=False)
    as_of = date(2026, 1, 1)
    out = prepare_matches([_m(d=date(2005, 1, 1)), _m(d=as_of)], as_of, cfg)
    assert all(m.weight == 1.0 for m in out)


def test_window_filters_by_lookback():
    as_of = date(2026, 1, 1)
    cutoff = as_of - timedelta(days=round(365.25 * 25))
    inside = _m(d=cutoff)  # boundary kept
    outside = _m(d=cutoff - timedelta(days=1))
    kept = apply_window([inside, outside], as_of, 25)
    assert inside in kept and outside not in kept


# --------------------------------------------------------------------------- names
def test_name_normalization_and_aliases():
    assert normalize("Korea Republic") == normalize("South Korea")
    assert normalize("Côte d'Ivoire") == "ivory coast"
    assert normalize("Türkiye") == "turkey"
    assert normalize("USA") == "united states"
    assert normalize("Czech Republic") == "czechia"


def test_canonical_map_records_conflicts_not_raises():
    mapping, conflicts = build_canonical_map(REPO / "tippspiel" / "data")
    assert mapping  # built something
    # KSA (wc2018/wc2022) and SAU (wc2026) both name Saudi Arabia -> a recorded conflict.
    assert any("Saudi Arabia" in c for c in conflicts)


# --------------------------------------------------------------------------- parse / driver
def test_parse_skips_blank_scores_and_normalizes():
    matches = parse_csv_text(SAMPLE_CSV)
    assert len(matches) == 4  # the France/Germany row has blank scores
    spain = next(m for m in matches if m.home == "spain")
    assert spain.neutral is True and spain.tournament == "UEFA Euro"


def test_build_ratings_is_deterministic_regardless_of_order():
    matches = parse_csv_text(SAMPLE_CSV)
    a = build_ratings(matches, WorldFootballElo(EloConfig()))
    b = build_ratings(list(reversed(matches)), WorldFootballElo(EloConfig()))
    assert a == b


def test_end_to_end_in_memory():
    matches = prepare_matches(parse_csv_text(SAMPLE_CSV), date(2020, 1, 1), EloConfig())
    ratings = build_ratings(matches, WorldFootballElo(EloConfig()))
    assert ratings  # produced ratings, no network touched
    assert all(isinstance(v, float) for v in ratings.values())


# --------------------------------------------------------------------------- config
def test_load_elo_config_defaults_and_overrides():
    assert load_elo_config(None).lookback_years == 25
    cfg = load_elo_config({"lookback_years": 10, "recency_decay": False,
                           "tier_k": {"friendly": 5, "world cup": 99}})
    assert cfg.lookback_years == 10 and cfg.recency_decay is False
    model = WorldFootballElo(cfg)
    assert model.k_for("Friendly") == 5 and model.k_for("FIFA World Cup") == 99


# --------------------------------------------------------------------------- emit
def test_emit_teams_csv_overwrites_only_elo(tmp_path):
    from tippspiel.config import load_tournament
    from tippspiel.data.file_provider import FileDataProvider
    from tippspiel.pipeline import _emit_teams_csv

    bundle = load_tournament(REPO / "configs" / "wc2022.yaml")
    teams = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file).get_teams()
    # Fake computed ratings for a couple of teams, keyed by normalized name.
    ratings = {normalize(teams[0].name): 1801.0, normalize(teams[1].name): 1234.0}
    out = tmp_path / "teams_computed.csv"
    written = _emit_teams_csv(bundle, ratings, out)
    assert written == 2

    emitted = {t.team_id: t for t in
               FileDataProvider(out, bundle.fixtures_file, bundle.results_file).get_teams()}
    assert emitted[teams[0].team_id].elo == pytest.approx(1801.0)
    assert emitted[teams[0].team_id].name == teams[0].name  # name preserved
    # A team with no computed rating keeps its original elo.
    untouched = teams[5]
    assert emitted[untouched.team_id].elo == pytest.approx(untouched.elo)
