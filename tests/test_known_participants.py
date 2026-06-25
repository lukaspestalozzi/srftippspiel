"""Deterministic resolution of already-decided knockout participants (report/tip path)."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

import tippspiel
from tippspiel.config import load_tournament
from tippspiel.data.file_provider import FileDataProvider
from tippspiel.model.stages import Stage
from tippspiel.model.types import Match, Result, TeamRef
from tippspiel.predictors.elo_poisson import EloPoissonPredictor
from tippspiel.simulation.known_participants import (
    compute_group_standings,
    resolve_known_participants,
)
from tippspiel.simulation.simulator import TournamentSimulator

REPO = Path(tippspiel.__file__).parent.parent


@pytest.fixture
def wc2026():
    bundle = load_tournament(REPO / "config.yaml")
    prov = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file,
                            bundle.thirds_allocation_file)
    fixtures = prov.get_fixtures()
    results = {r.match_id: r for r in prov.get_results()}
    teams = {t.team_id: t for t in prov.get_teams()}
    thirds = prov.get_thirds_allocation()
    return fixtures, results, teams, thirds


def _by_id(fixtures):
    return {m.match_id: m for m in fixtures}


def test_finished_groups_fill_their_knockout_slots(wc2026):
    fixtures, results, _teams, thirds = wc2026
    resolved = _by_id(resolve_known_participants(fixtures, results, thirds))

    # Groups A, B, C have finished -> M73 (R:A vs R:B) is fully known and now tippable.
    m73 = resolved["M73"]
    assert (m73.home.team_id, m73.away.team_id) == ("RSA", "CAN")
    assert m73.participants_known

    # The finished-group side of a half-known fixture is filled; the open side stays a reference.
    half = {"M75": ("away", "MAR"), "M76": ("home", "BRA"),
            "M79": ("home", "MEX"), "M85": ("home", "SUI")}
    for mid, (side, team) in half.items():
        m = resolved[mid]
        known, other = (m.home, m.away) if side == "home" else (m.away, m.home)
        assert known.team_id == team
        assert not other.is_concrete  # opponent not decided yet
        assert not m.participants_known


def test_group_standings_finished_group(wc2026):
    fixtures, results, _teams, _thirds = wc2026
    standings = {g.letter: g for g in compute_group_standings(fixtures, results)}
    a = standings["A"]
    assert a.complete
    # Group A finished: MEX 9, RSA 4, KOR 3, CZE 1 — strictly separated, all placings certain.
    assert [(r.team_id, r.points) for r in a.rows] == [
        ("MEX", 9), ("RSA", 4), ("KOR", 3), ("CZE", 1)]
    mex = a.rows[0]
    assert (mex.played, mex.wins, mex.draws, mex.losses) == (3, 3, 0, 0)
    assert (mex.goals_for, mex.goals_against, mex.goal_diff) == (6, 0, 6)
    assert all(r.placing_certain for r in a.rows)


def test_group_standings_in_progress_group(wc2026):
    fixtures, results, _teams, _thirds = wc2026
    standings = {g.letter: g for g in compute_group_standings(fixtures, results)}
    d = standings["D"]  # only 4 of 6 matches played
    assert not d.complete
    assert sum(r.played for r in d.rows) == 2 * sum(
        1 for m in fixtures if m.group == "D" and m.match_id in results)
    assert all(not r.placing_certain for r in d.rows)  # provisional, nothing settled
    # Ranks are a 1..n permutation in points-descending order.
    assert [r.rank for r in d.rows] == [1, 2, 3, 4]
    pts = [r.points for r in d.rows]
    assert pts == sorted(pts, reverse=True)


def test_undetermined_slots_are_left_untouched(wc2026):
    fixtures, results, _teams, thirds = wc2026
    resolved = _by_id(resolve_known_participants(fixtures, results, thirds))
    original = _by_id(fixtures)
    # Groups D-L are unfinished and no third place / knockout has been decided.
    for mid in ("M74", "M88", "M89", "M104"):
        assert resolved[mid].home == original[mid].home
        assert resolved[mid].away == original[mid].away
    # No 3RD slot can be filled until every group is complete.
    for m in resolved.values():
        for side in (m.home, m.away):
            if side.ko_ref and side.ko_ref.kind == "third_pooled":
                assert not (m.home.is_concrete and m.away.is_concrete)


def _group_match(mid, home, away, group="X"):
    return Match(match_id=mid, stage=Stage.GROUP, home=TeamRef(team_id=home),
                 away=TeamRef(team_id=away), kickoff=datetime(2026, 6, 1, tzinfo=timezone.utc),
                 group=group)


def test_unbreakable_tie_leaves_placing_open():
    # T1 and T2 are identical on points/GD/GF and drew head-to-head -> the 1st/2nd placing can
    # only be settled by fair-play/lots, which we cannot derive: both slots must stay references.
    fixtures = [
        _group_match("X1", "AAA", "BBB"),  # 1-1
        _group_match("X2", "AAA", "CCC"),  # 2-0
        _group_match("X3", "AAA", "DDD"),  # 2-0
        _group_match("X4", "BBB", "CCC"),  # 2-0
        _group_match("X5", "BBB", "DDD"),  # 2-0
        _group_match("X6", "CCC", "DDD"),  # 0-0
        Match(match_id="K1", stage=Stage.QF, home=TeamRef.parse("W:X"),
              away=TeamRef.parse("R:X"),
              kickoff=datetime(2026, 6, 30, tzinfo=timezone.utc), group=None),
    ]
    scores = {"X1": (1, 1), "X2": (2, 0), "X3": (2, 0),
              "X4": (2, 0), "X5": (2, 0), "X6": (0, 0)}
    results = {mid: Result(mid, h, a) for mid, (h, a) in scores.items()}
    resolved = _by_id(resolve_known_participants(fixtures, results))
    k1 = resolved["K1"]
    assert not k1.home.is_concrete and k1.home.ko_ref.kind == "winner"
    assert not k1.away.is_concrete and k1.away.ko_ref.kind == "runner_up"


def test_completed_tournament_bracket_is_left_unchanged():
    # A completed tournament lists concrete knockout participants (no references). The resolver
    # must not try to build a Bracket from that fixed bracket (which is rightly rejected) — it is
    # a no-op. Regression for the predict-<completed> CI jobs.
    bundle = load_tournament(REPO / "configs" / "wc2022.yaml")
    prov = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file,
                            bundle.thirds_allocation_file)
    fixtures = prov.get_fixtures()
    results = {r.match_id: r for r in prov.get_results()}
    resolved = resolve_known_participants(fixtures, results, prov.get_thirds_allocation())
    assert len(resolved) == len(fixtures)
    for before, after in zip(fixtures, resolved):
        assert before.home == after.home and before.away == after.away


def test_resolved_team_matches_simulator_certainty(wc2026):
    # Every team the resolver fixes must be the one the simulator advances with probability 1.0
    # (a completed group's standings are identical in every iteration).
    fixtures, results, teams, thirds = wc2026
    resolved = _by_id(resolve_known_participants(fixtures, results, thirds))
    sim = TournamentSimulator(fixtures=fixtures, teams=teams, results=results,
                              predictor=EloPoissonPredictor(), thirds_allocation=thirds,
                              iterations=2000, seed=1)
    outcome = sim.run()
    # M73's two participants are the certain runners-up of groups A and B.
    for tid in (resolved["M73"].home.team_id, resolved["M73"].away.team_id):
        assert outcome.advancement[tid]["group_second"] == 1.0
    assert outcome.advancement["MEX"]["group_winner"] == 1.0  # W:A filled into M79
    assert outcome.advancement["BRA"]["group_winner"] == 1.0  # W:C filled into M76
