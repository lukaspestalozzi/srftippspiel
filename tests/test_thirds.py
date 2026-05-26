"""Third-placed ranking + best-8 selection tests (spec §3.3 / §10)."""

import numpy as np

from tippspiel.simulation.thirds import select_best_thirds


def test_nontrivial_8_of_12_cut_by_goal_difference():
    # 9 groups' thirds have 3 points; only 8 advance, so GD must break the 8th/9th place.
    pts = np.array([[3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1]], dtype=float)
    gd = np.array([[5, 5, 5, 5, 5, 5, 5, 5, 0, 9, 9, 9]], dtype=float)
    gf = np.zeros((1, 12))
    rand = np.zeros((1, 12))
    qualified, _ = select_best_thirds(pts, gd, gf, rand)
    q = qualified[0]
    assert q[:8].all()        # the eight 3pt/+5gd groups qualify
    assert not q[8]           # the 3pt/0gd group is cut despite equal points
    assert not q[9:].any()    # the 1pt groups do not qualify despite high GD


def test_exactly_eight_qualify():
    rng = np.random.default_rng(0)
    pts = rng.integers(0, 10, size=(50, 12)).astype(float)
    gd = rng.integers(-5, 6, size=(50, 12)).astype(float)
    gf = rng.integers(0, 8, size=(50, 12)).astype(float)
    rand = rng.random((50, 12))
    qualified, _ = select_best_thirds(pts, gd, gf, rand)
    assert (qualified.sum(axis=1) == 8).all()
