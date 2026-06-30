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

    # The whole group stage has finished -> M73 (R:A vs R:B) is fully known and now tippable.
    m73 = resolved["M73"]
    assert (m73.home.team_id, m73.away.team_id) == ("RSA", "CAN")
    assert m73.participants_known

    # A best-placed third fills its slot: M79 = W:A (Mexico) vs a third drawn from the allowed
    # groups (Ecuador, third in Group E).
    m79 = resolved["M79"]
    assert (m79.home.team_id, m79.away.team_id) == ("MEX", "ECU")
    assert m79.participants_known

    # With every group complete, all 16 R32 fixtures (M73-M88) are concrete on both sides.
    for n in range(73, 89):
        m = resolved[f"M{n}"]
        assert m.home.is_concrete and m.away.is_concrete
        assert m.participants_known


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


def test_group_standings_in_progress_group():
    # A group with only some matches played: standings are provisional and no placing is certain.
    # Built from synthetic fixtures because every live wc2026 group has now finished.
    fixtures = [
        _group_match("Y1", "AAA", "BBB", group="Y"),
        _group_match("Y2", "CCC", "DDD", group="Y"),
        _group_match("Y3", "AAA", "CCC", group="Y"),
        _group_match("Y4", "BBB", "DDD", group="Y"),
        _group_match("Y5", "AAA", "DDD", group="Y"),
        _group_match("Y6", "BBB", "CCC", group="Y"),
    ]
    scores = {"Y1": (1, 1), "Y2": (1, 1), "Y3": (0, 0)}  # only 3 of 6 played, all drawn
    results = {mid: Result(mid, h, a) for mid, (h, a) in scores.items()}
    standings = {g.letter: g for g in compute_group_standings(fixtures, results)}
    d = standings["Y"]
    assert not d.complete
    assert sum(r.played for r in d.rows) == 2 * len(scores)
    assert all(not r.placing_certain for r in d.rows)  # provisional, nothing settled
    # Ranks are a 1..n permutation in points-descending order.
    assert [r.rank for r in d.rows] == [1, 2, 3, 4]
    pts = [r.points for r in d.rows]
    assert pts == sorted(pts, reverse=True)


def test_undetermined_slots_are_left_untouched(wc2026):
    fixtures, results, _teams, thirds = wc2026
    resolved = _by_id(resolve_known_participants(fixtures, results, thirds))
    original = _by_id(fixtures)
    # A knockout slot fed by the winner/loser of a match that hasn't been played yet must stay a
    # reference. Checked generically (over every WIN:/LOSE: side whose feeder isn't in the results)
    # so the assertion survives the live wc2026 data moving forward as matchdays are recorded.
    checked = 0
    for mid, m in original.items():
        if m.group is not None:
            continue
        for ref_side, resolved_side in ((m.home, resolved[mid].home), (m.away, resolved[mid].away)):
            r = ref_side.ko_ref
            if r and r.kind in ("winner_of", "loser_of") and r.match_id not in results:
                assert resolved_side == ref_side  # feeder not played -> slot unchanged
                checked += 1
    assert checked > 0  # the live bracket must still have open downstream slots
    # The group stage is complete, so every third-place slot has been allocated to a concrete team;
    # no third_pooled reference remains anywhere in the bracket.
    for m in resolved.values():
        for side in (m.home, m.away):
            assert not (side.ko_ref and side.ko_ref.kind == "third_pooled")


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
