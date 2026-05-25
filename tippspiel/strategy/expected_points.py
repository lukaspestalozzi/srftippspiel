"""ExpectedPointsStrategy (spec §6.3.2): pick the scoreline tip maximising expected
pool points for each tippable fixture.

EV of a candidate tip (th, ta) under scoreline distribution P:

    EV = W * [ 5 * P_tendency
             + 1 * P.p_home_goals(th)
             + 1 * P.p_away_goals(ta)
             + 3 * P.p_goal_difference(th - ta) ]

where P_tendency is the L/D/W probability matching sign(th - ta), and W = 1 for group
matches, 2 for knockout matches. This decomposition is exact: the four components are
additive and the exact-score case yields 5+1+1+3 = 10 (×W) automatically.
"""

from __future__ import annotations

from ..model.scoreline import ScorelineDistribution
from ..model.stages import (
    PTS_AWAY_GOALS,
    PTS_GOAL_DIFF,
    PTS_HOME_GOALS,
    PTS_TENDENCY,
    Stage,
)
from ..model.types import Match, MatchPrediction, Tip, TipSet, TournamentOutcome
from .base import TipStrategy
from .bonus import build_bonus_questions


def ev_components(dist: ScorelineDistribution, th: int, ta: int, weight: int) -> dict[str, float]:
    """Expected pool points for tip (th, ta), broken into the four additive scoring terms.

    Returns ``{tendency, home_goals, away_goals, goal_diff, total}`` (each already weighted).
    ``expected_points`` is the ``total``; the breakdown is what the diagnostic report uses to
    show *why* a given tip wins (e.g. the 5-pt tendency term dominating).
    """
    if th > ta:
        p_tendency = dist.p_home_win()
    elif th == ta:
        p_tendency = dist.p_draw()
    else:
        p_tendency = dist.p_away_win()
    comps = {
        "tendency": weight * PTS_TENDENCY * p_tendency,
        "home_goals": weight * PTS_HOME_GOALS * dist.p_home_goals(th),
        "away_goals": weight * PTS_AWAY_GOALS * dist.p_away_goals(ta),
        "goal_diff": weight * PTS_GOAL_DIFF * dist.p_goal_difference(th - ta),
    }
    comps["total"] = comps["tendency"] + comps["home_goals"] + comps["away_goals"] + comps["goal_diff"]
    return comps


def expected_points(dist: ScorelineDistribution, th: int, ta: int, weight: int) -> float:
    return ev_components(dist, th, ta, weight)["total"]


def best_tip(dist: ScorelineDistribution, weight: int) -> tuple[int, int, float]:
    """Enumerate all (th, ta) in [0, gmax]^2, return the EV-maximising tip.

    Deterministic tie-break: (1) higher probability of the exact tipped scoreline,
    then (2) lower total goals th+ta, then (3) lower th.
    """
    best: tuple[float, float, int, int] | None = None  # (-EV, -P_exact, th+ta, th)
    best_th = best_ta = 0
    best_ev = 0.0
    for th in range(dist.gmax + 1):
        for ta in range(dist.gmax + 1):
            ev = expected_points(dist, th, ta, weight)
            key = (-ev, -dist.cell(th, ta), th + ta, th)
            if best is None or key < best:
                best = key
                best_th, best_ta, best_ev = th, ta, ev
    return best_th, best_ta, best_ev


class ExpectedPointsStrategy(TipStrategy):
    name = "expected_points"

    def __init__(self, bonus_question_configs=()) -> None:
        self._bonus_configs = list(bonus_question_configs)

    def generate_tips(
        self,
        predictions: dict[str, MatchPrediction],
        outcome: TournamentOutcome | None,
        fixtures: list[Match],
    ) -> TipSet:
        by_id = {m.match_id: m for m in fixtures}
        tips: dict[str, Tip] = {}
        for match_id, pred in predictions.items():
            match = by_id.get(match_id)
            if match is None:
                continue
            weight = match.stage.points_weight
            th, ta, ev = best_tip(pred.scoreline, weight)
            naive = pred.scoreline.most_likely_scorelines(1)[0]
            naive_ev = expected_points(pred.scoreline, naive[0], naive[1], weight)
            rationale = (
                f"EV {ev:.2f} pts (naive most-likely tip {naive[0]}-{naive[1]}: "
                f"{naive_ev:.2f} pts)"
            )
            tips[match_id] = Tip(match_id, th, ta, ev, rationale)

        bonus_answers: dict[str, str] = {}
        if outcome is not None:
            for q in build_bonus_questions(self._bonus_configs):
                dist = q.resolve(outcome)
                if dist:
                    bonus_answers[q.question_id] = max(dist, key=dist.get)

        return TipSet(tips=tips, bonus_answers=bonus_answers)
