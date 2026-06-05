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
)
from ..model.types import Match, MatchPrediction, Tip, TipSet, TournamentOutcome
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


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def score_tip(th: int, ta: int, actual_h: int, actual_a: int, weight: int) -> int:
    """Pool points a tip (th, ta) scores against a known result, under the same 5/1/1/3 rules
    used by ``ev_components`` (exact score = 10*W). The deterministic counterpart to the EV;
    knockout results should be the 120-minute scoreline (shootouts count as draws)."""
    pts = 0
    if _sign(th - ta) == _sign(actual_h - actual_a):
        pts += PTS_TENDENCY
    if th == actual_h:
        pts += PTS_HOME_GOALS
    if ta == actual_a:
        pts += PTS_AWAY_GOALS
    if (th - ta) == (actual_h - actual_a):
        pts += PTS_GOAL_DIFF
    return weight * pts


def best_tip(
    dist: ScorelineDistribution, weight: int, realism_tolerance: float = 0.0
) -> tuple[int, int, float]:
    """Enumerate all (th, ta) in [0, gmax]^2 and return the recommended tip.

    With ``realism_tolerance == 0`` this is the strict EV-maximiser, tie-broken by
    (1) higher probability of the exact tipped scoreline, then (2) lower total goals, then
    (3) lower th.

    With ``realism_tolerance > 0`` the tip is chosen among the cells whose EV is within
    ``realism_tolerance`` pool-points of the maximum, preferring the one **closest to the model's
    expected scoreline** (L1 distance to ``expected_goals``), then the same EV/probability
    tie-breaks. Because flipping the win/draw/loss tendency costs ~5 pts (>> a sensible
    tolerance), the candidate set is always same-tendency/same-margin, so this only nudges the
    *absolute* goals toward what the model expects — a 1:0 becomes a same-margin 2:1 when the
    model actually expects goals, at a tiny EV cost, while a genuinely tight game stays low.
    """
    gmax = dist.gmax
    evs = {
        (th, ta): expected_points(dist, th, ta, weight)
        for th in range(gmax + 1)
        for ta in range(gmax + 1)
    }
    mx = max(evs.values())
    e_home, e_away = dist.expected_goals()
    candidates = [c for c in evs if evs[c] >= mx - realism_tolerance]

    def _key(cell: tuple[int, int]) -> tuple:
        th, ta = cell
        # Realism term is inert at tolerance 0 (candidates are then exactly the EV-argmax cells),
        # so the legacy EV/probability/total/home tie-break is reproduced byte-for-byte.
        realism = (abs(th - e_home) + abs(ta - e_away)) if realism_tolerance > 0 else 0.0
        return (realism, -evs[cell], -dist.cell(th, ta), th + ta, th)

    best_th, best_ta = min(candidates, key=_key)
    return best_th, best_ta, evs[(best_th, best_ta)]


class ExpectedPointsStrategy:
    name = "expected_points"

    def __init__(self, bonus_question_configs=(), realism_tolerance: float = 0.0) -> None:
        self._bonus_configs = list(bonus_question_configs)
        self._realism_tolerance = realism_tolerance

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
            th, ta, ev = best_tip(pred.scoreline, weight, self._realism_tolerance)
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
