"""Group standings + tiebreaker tests (spec §3.2 / §10)."""

import numpy as np

from tippspiel.simulation.standings import rank_group

# Round-robin layout for 4 teams (each pair once).
LAYOUT = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]


def _rank(home, away):
    hg = np.array([home])
    ag = np.array([away])
    rand = np.zeros((1, 4))
    order, pts, gd, gf = rank_group(hg, ag, LAYOUT, rand)
    return list(order[0]), pts[0], gd[0], gf[0]


def test_tiebreak_level1_points():
    # 0 beats all, 1 beats 2&3, 2 beats 3: strict points ladder.
    order, pts, _, _ = _rank([1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0])
    assert order == [0, 1, 2, 3]
    assert list(pts) == [9, 6, 3, 0]


def test_tiebreak_level2_goal_difference():
    # Teams 0 and 1 tie on 6 points; team 0 has the better goal difference.
    order, _, gd, _ = _rank([1, 0, 0, 1, 3, 1], [0, 0, 1, 0, 0, 0])
    assert order[0] == 0
    assert gd[0] > gd[1]


def test_tiebreak_level4_head_to_head():
    # Teams 0 and 1 are identical on points, GD and goals; team 0 won the head-to-head.
    order, pts, gd, gf = _rank([1, 0, 0, 1, 1, 1], [0, 0, 1, 0, 0, 0])
    assert (pts[0], gd[0], gf[0]) == (pts[1], gd[1], gf[1])  # tied on criteria 1-3
    assert order.index(0) < order.index(1)  # head-to-head puts 0 above 1


def test_random_tiebreak_is_deterministic():
    # Total deadlock: all matches 0-0 -> every team identical. Random tiebreak must be
    # stable for a fixed rand vector.
    hg = np.zeros((1, 6))
    ag = np.zeros((1, 6))
    rand = np.array([[0.4, 0.1, 0.9, 0.2]])
    o1, *_ = rank_group(hg, ag, LAYOUT, rand)
    o2, *_ = rank_group(hg, ag, LAYOUT, rand)
    assert list(o1[0]) == list(o2[0])
    # Lower rand key ranks higher: team 1 (0.1) first, team 2 (0.9) last.
    assert o1[0][0] == 1 and o1[0][-1] == 2
