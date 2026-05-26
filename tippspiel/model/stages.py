"""Tournament stages and the per-stage scoring weight.

Pool scoring (spec §3.4): group-stage components are tendency 5, home-goal +1,
away-goal +1, goal-difference +3 (exact score = 5+1+1+3 = 10). Knockout matches use
the identical structure with all values doubled, which we express as a single weight
``W`` applied to the group-stage point values.
"""

from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    GROUP = "GROUP"
    R32 = "R32"
    R16 = "R16"
    QF = "QF"
    SF = "SF"
    THIRD_PLACE = "THIRD_PLACE"
    FINAL = "FINAL"

    @property
    def is_knockout(self) -> bool:
        return self is not Stage.GROUP

    @property
    def points_weight(self) -> int:
        """W in the EV formula: 1 for group matches, 2 for knockout matches."""
        return 2 if self.is_knockout else 1


# Base (group-stage) component point values. Knockout = these × Stage.points_weight.
PTS_TENDENCY = 5
PTS_HOME_GOALS = 1
PTS_AWAY_GOALS = 1
PTS_GOAL_DIFF = 3
PTS_EXACT = PTS_TENDENCY + PTS_HOME_GOALS + PTS_AWAY_GOALS + PTS_GOAL_DIFF  # 10
