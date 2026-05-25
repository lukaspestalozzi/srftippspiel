"""Third-placed team ranking and best-8 selection (spec §3.3), vectorised.

The 12 third-placed teams are ranked by points, goal difference, goals scored (the
across-group criteria — head-to-head does not apply, as third-placed teams from
different groups have not met). Remaining ties use the seeded random tiebreak. The best
8 advance to the Round of 32.
"""

from __future__ import annotations

import numpy as np

_GD_OFFSET = 1000


def select_best_thirds(
    pts: np.ndarray, gd: np.ndarray, gf: np.ndarray, rand: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (qualified, order).

    pts/gd/gf/rand are shape [N, 12] (one column per group A..L). ``qualified`` is a
    boolean [N, 12] marking the 8 best third-placed groups per iteration; ``order`` is
    [N, 12] group indices ranked best to worst. ``rand`` (uniform in [0,1)) breaks exact
    (pts, gd, gf) ties deterministically — gaps between distinct keys are >= 1, so a
    sub-1 perturbation only reorders genuine ties.
    """
    key = pts * 1e7 + (gd + _GD_OFFSET) * 1e3 + gf + rand
    order = np.argsort(-key, axis=1, kind="stable")
    n = pts.shape[0]
    qualified = np.zeros_like(pts, dtype=bool)
    rows = np.arange(n)[:, None]
    qualified[rows, order[:, :8]] = True
    return qualified, order
