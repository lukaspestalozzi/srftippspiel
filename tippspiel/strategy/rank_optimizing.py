"""RankOptimizingStrategy + FieldModel (spec §6.3.3): tip to *win* the pool, not to
maximise your own expected points.

In a large pool (~200,000 participants) the EV-maximising slate scores well but rarely
wins outright: thousands of sharp participants converge on the same EV-optimal scorelines,
so an EV slate earns the *crowd's* score, not a winning edge. Winning P(rank=1) requires
**deliberate contrarian variance** — tips that differentiate you from the field and pay off
in the worlds where the crowd is wrong.

How this works (the whole-slate ``TipStrategy`` interface already accommodates it):

* Pre-tournament only fixtures with concrete participants are tippable — the group matches —
  and they are mutually independent, each carrying its own ``ScorelineDistribution``. So a
  match's result random variable *is* its predicted scoreline; we sample ``n_worlds`` joint
  result vectors by drawing each match independently.
* ``FieldModel.opponent_tip_distribution`` estimates how the field tips each match. Since no
  real pool-tip data exists, ``PredictorDerivedFieldModel`` derives it from the predictor: a
  mixture of the popular/EV-optimal scorelines and a temperature-flattened copy of the
  predictor's own distribution.
* Objective — **conditional on a sampled result vector ``r``** the field's per-participant
  totals are i.i.d. (everyone sees the same results; only their tip choices differ), so the
  total is ≈ ``Normal(mean_r, var_r)`` with per-match moments from the field model. The
  expected number of opponents who beat me in world ``r`` is ``lambda_r = P * SF((s_me -
  mean_r)/sd_r)``; the number beating me is ≈ ``Poisson(lambda_r)``, so
  ``P(rank <= top_n) = E_r[ poisson_cdf(top_n - 1; lambda_r) ]`` (for ``top_n = 1`` this is
  ``E_r[exp(-lambda_r)]``). This is smooth in ``s_me`` and rewards catching upsets that the
  field misses.
* Optimiser — coordinate ascent from the EV-optimal slate over a small candidate set per
  match (the top scorelines plus the modal scoreline of each tendency, so betting the upset
  is always an option). A change is kept only if it strictly raises the estimated win
  probability, so the slate deviates from EV only where contrarian variance genuinely helps.

Out of scope for this version (documented intentionally): the simulator is unchanged (the
group slate needs no joint per-iteration scorelines); the bonus/champion answers are carried
over as the EV-optimal (modal) picks rather than contrarian-optimised; knockout matches are
not rank-optimised (they only become tippable once concrete, by which point the field is
largely locked).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..model.scoreline import ScorelineDistribution
from ..model.types import Match, MatchPrediction, Tip, TipSet, TournamentOutcome
from .base import TipStrategy
from .bonus import build_bonus_questions
from .expected_points import best_tip, expected_points, score_tip


class FieldModel(ABC):
    @abstractmethod
    def opponent_tip_distribution(
        self, prediction: MatchPrediction
    ) -> dict[tuple[int, int], float]:
        """Estimate the distribution of opponents' tips for a match."""
        ...


class PredictorDerivedFieldModel(FieldModel):
    """Model the field from the predictor (no external pool-tip data exists).

    Each opponent's tip for a match is drawn from a mixture of:

    * an **expert cluster** — mass on the EV-optimal tip and the few most-likely scorelines,
      the tips sharp participants converge on; weighted by their predicted cell probability;
    * a **public cluster** — the predictor's scoreline matrix flattened by ``temperature``
      (``p_ij ** (1/temperature)``, renormalised); ``temperature > 1`` spreads casual tips out.

    ``expert_fraction`` is the weight on the expert cluster. This is a modelling assumption,
    not measured behaviour — the whole point of the strategy is to be robust to it.
    """

    def __init__(
        self,
        *,
        expert_fraction: float = 0.6,
        temperature: float = 1.5,
        expert_scorelines: int = 3,
    ) -> None:
        if not 0.0 <= expert_fraction <= 1.0:
            raise ValueError("expert_fraction must be in [0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.expert_fraction = expert_fraction
        self.temperature = temperature
        self.expert_scorelines = max(1, expert_scorelines)

    def opponent_tip_distribution(
        self, prediction: MatchPrediction
    ) -> dict[tuple[int, int], float]:
        dist = prediction.scoreline
        # Expert cluster: EV-optimal tip + most-likely scorelines, weighted by cell prob.
        expert: dict[tuple[int, int], float] = {}
        bt_h, bt_a, _ = best_tip(dist, 1)  # argmax is weight-independent
        for h, a, p in dist.most_likely_scorelines(self.expert_scorelines):
            expert[(h, a)] = expert.get((h, a), 0.0) + max(p, 1e-9)
        expert[(bt_h, bt_a)] = expert.get((bt_h, bt_a), 0.0) + dist.cell(bt_h, bt_a) + 1e-9
        e_total = sum(expert.values())
        expert = {k: v / e_total for k, v in expert.items()}

        # Public cluster: temperature-flattened scoreline matrix.
        tempered = np.power(dist.matrix, 1.0 / self.temperature)
        tempered = tempered / tempered.sum()

        out: dict[tuple[int, int], float] = {}
        g = dist.gmax
        for h in range(g + 1):
            for a in range(g + 1):
                p = (1.0 - self.expert_fraction) * float(tempered[h, a])
                p += self.expert_fraction * expert.get((h, a), 0.0)
                if p > 0.0:
                    out[(h, a)] = p
        total = sum(out.values())
        return {k: v / total for k, v in out.items()}


# --------------------------------------------------------------------------- field moments
def field_score_moments(
    field_dist: dict[tuple[int, int], float], actual_h: int, actual_a: int, weight: int
) -> tuple[float, float]:
    """Mean and variance of a random opponent's score on one match against a known result,
    given the field's tip distribution. The per-match building block of the field total."""
    mean = 0.0
    m2 = 0.0
    for (th, ta), p in field_dist.items():
        s = score_tip(th, ta, actual_h, actual_a, weight)
        mean += p * s
        m2 += p * s * s
    return mean, max(0.0, m2 - mean * mean)


# --------------------------------------------------------------------------- normal CDF/SF
def _norm_sf(x: np.ndarray) -> np.ndarray:
    """Standard-normal survival function P(Z > x), vectorised (Zelen & Severo, err < 7.5e-8)."""
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * ax)
    d = 0.3989422804014327 * np.exp(-0.5 * ax * ax)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
              + t * (-1.821255978 + t * 1.330274429))))
    upper = d * poly  # P(Z > ax)
    return np.where(x >= 0.0, upper, 1.0 - upper)


def _poisson_cdf(lam: np.ndarray, k: int) -> np.ndarray:
    """P(Poisson(lam) <= k), vectorised over ``lam``, for small integer ``k``."""
    term = np.exp(-lam)  # j = 0
    cum = term.copy()
    for j in range(1, k + 1):
        term = term * lam / j
        cum = cum + term
    return cum


def _candidate_tips(dist: ScorelineDistribution, weight: int, k: int) -> list[tuple[int, int]]:
    """Top-k scorelines + the EV-optimal tip + the modal scoreline of each tendency.

    Including a per-tendency mode guarantees the optimiser can bet the *upset* (away win when
    home is favoured, etc.), which the plain most-likely list would omit."""
    seen: dict[tuple[int, int], None] = {}
    for h, a, _ in dist.most_likely_scorelines(k):
        seen[(h, a)] = None
    bt_h, bt_a, _ = best_tip(dist, weight)
    seen[(bt_h, bt_a)] = None
    best_by_tend: dict[str, tuple[tuple[int, int], float]] = {}
    g = dist.gmax
    for h in range(g + 1):
        for a in range(g + 1):
            tend = "h" if h > a else "d" if h == a else "a"
            p = dist.cell(h, a)
            if tend not in best_by_tend or p > best_by_tend[tend][1]:
                best_by_tend[tend] = ((h, a), p)
    for cell, _ in best_by_tend.values():
        seen[cell] = None
    return list(seen)


def _score_vector(tip: tuple[int, int], weight: int, gmax: int, idx: np.ndarray) -> np.ndarray:
    """Per-world score of ``tip`` against sampled results (flattened cell indices ``idx``)."""
    mat = np.empty((gmax + 1) * (gmax + 1), dtype=np.int32)
    for ah in range(gmax + 1):
        for aa in range(gmax + 1):
            mat[ah * (gmax + 1) + aa] = score_tip(tip[0], tip[1], ah, aa, weight)
    return mat[idx]


def _entropy(dist: ScorelineDistribution) -> float:
    m = dist.matrix.ravel()
    m = m[m > 0]
    return float(-(m * np.log2(m)).sum())


@dataclass
class SlateComparison:
    """The EV-optimal slate vs the rank-optimised slate, with estimated win probabilities."""

    ev_slate: dict[str, tuple[int, int]]
    rank_slate: dict[str, tuple[int, int]]
    ev_p_win: float
    rank_p_win: float
    ev_total_ev: float
    rank_total_ev: float
    pool_size: int
    top_n: int
    n_worlds: int
    diffs: list[dict] = field(default_factory=list)

    @property
    def n_diff(self) -> int:
        return len(self.diffs)


def optimize_slate(
    predictions: dict[str, MatchPrediction],
    fixtures: list[Match],
    field_model: FieldModel,
    *,
    pool_size: int = 200_000,
    top_n: int = 1,
    n_worlds: int = 10_000,
    candidates_per_match: int = 6,
    seed: int = 20260611,
) -> SlateComparison:
    """Optimise the tippable slate for P(rank <= top_n) against the modelled field.

    Returns a :class:`SlateComparison` (the rank-optimised slate plus the EV baseline and
    both win probabilities). Deterministic for a fixed ``seed``."""
    by_id = {m.match_id: m for m in fixtures}
    mids = sorted(mid for mid in predictions if mid in by_id)
    rng = np.random.default_rng(seed)
    W = n_worlds

    field_mean = np.zeros(W)
    field_var = np.zeros(W)
    per: dict[str, dict] = {}
    for mid in mids:
        pred = predictions[mid]
        weight = by_id[mid].stage.points_weight
        dist = pred.scoreline
        g = dist.gmax
        flat = np.asarray(dist.matrix, dtype=float).ravel()
        flat = flat / flat.sum()
        idx = rng.choice(flat.size, size=W, p=flat)

        fd = field_model.opponent_tip_distribution(pred)
        mu_flat = np.empty(flat.size)
        var_flat = np.empty(flat.size)
        for ah in range(g + 1):
            for aa in range(g + 1):
                mu, var = field_score_moments(fd, ah, aa, weight)
                mu_flat[ah * (g + 1) + aa] = mu
                var_flat[ah * (g + 1) + aa] = var
        field_mean += mu_flat[idx]
        field_var += var_flat[idx]

        cands = []
        for tip in _candidate_tips(dist, weight, candidates_per_match):
            cands.append({
                "tip": tip,
                "score_vec": _score_vector(tip, weight, g, idx),
                "ev": expected_points(dist, tip[0], tip[1], weight),
            })
        bt_h, bt_a, _ = best_tip(dist, weight)
        per[mid] = {"cands": cands, "ev_tip": (bt_h, bt_a), "dist": dist}

    field_sd = np.sqrt(np.maximum(field_var, 1e-9))

    def p_win(s_me: np.ndarray) -> float:
        x = (s_me - field_mean) / field_sd
        lam = pool_size * _norm_sf(x)
        return float(np.mean(_poisson_cdf(lam, top_n - 1)))

    # Initial slate: EV-optimal.
    chosen: dict[str, int] = {}
    s_me = np.zeros(W, dtype=np.int64)
    for mid in mids:
        cands = per[mid]["cands"]
        ci = next(i for i, c in enumerate(cands) if c["tip"] == per[mid]["ev_tip"])
        chosen[mid] = ci
        s_me = s_me + cands[ci]["score_vec"]
    ev_slate = {mid: per[mid]["ev_tip"] for mid in mids}
    ev_p_win = p_win(s_me)
    ev_total_ev = sum(per[mid]["cands"][chosen[mid]]["ev"] for mid in mids)

    # Coordinate ascent: switch a match's tip only if it strictly raises P(win).
    for _ in range(5):
        improved = False
        for mid in mids:
            cands = per[mid]["cands"]
            cur = chosen[mid]
            base = s_me - cands[cur]["score_vec"]
            best_i, best_p = cur, p_win(s_me)
            for i, c in enumerate(cands):
                if i == cur:
                    continue
                p = p_win(base + c["score_vec"])
                if p > best_p:
                    best_i, best_p = i, p
            if best_i != cur:
                chosen[mid] = best_i
                s_me = base + cands[best_i]["score_vec"]
                improved = True
        if not improved:
            break

    rank_slate = {mid: per[mid]["cands"][chosen[mid]]["tip"] for mid in mids}
    rank_p_win = p_win(s_me)
    rank_total_ev = sum(per[mid]["cands"][chosen[mid]]["ev"] for mid in mids)

    diffs = []
    for mid in mids:
        if rank_slate[mid] != ev_slate[mid]:
            d = per[mid]["dist"]
            diffs.append({
                "match_id": mid,
                "ev_tip": ev_slate[mid],
                "rank_tip": rank_slate[mid],
                "ldw": [d.p_home_win(), d.p_draw(), d.p_away_win()],
                "entropy": _entropy(d),
            })

    return SlateComparison(
        ev_slate=ev_slate, rank_slate=rank_slate,
        ev_p_win=ev_p_win, rank_p_win=rank_p_win,
        ev_total_ev=ev_total_ev, rank_total_ev=rank_total_ev,
        pool_size=pool_size, top_n=top_n, n_worlds=W, diffs=diffs,
    )


def field_model_from_params(params: dict) -> PredictorDerivedFieldModel:
    """Build the predictor-derived field model from strategy params (shared by report paths)."""
    return PredictorDerivedFieldModel(
        expert_fraction=params.get("expert_fraction", 0.6),
        temperature=params.get("temperature", 1.5),
    )


def comparison_from_params(
    predictions: dict[str, MatchPrediction],
    fixtures: list[Match],
    params: dict,
    *,
    seed: int,
    n_worlds_cap: int = 8000,
) -> SlateComparison | None:
    """EV-vs-rank slate comparison from strategy params; ``None`` when nothing is tippable.

    Shared by the diagnostic (§7) and the HTML report so both surface identical numbers; the
    world count is capped to keep report generation snappy regardless of the active strategy."""
    if not predictions:
        return None
    return optimize_slate(
        predictions, fixtures, field_model_from_params(params),
        pool_size=params.get("pool_size", 200_000),
        top_n=params.get("top_n", 1),
        n_worlds=min(params.get("n_worlds", n_worlds_cap), n_worlds_cap),
        seed=seed,
    )


class RankOptimizingStrategy(TipStrategy):
    name = "rank_optimizing"

    def __init__(
        self,
        field_model: FieldModel | None = None,
        *,
        pool_size: int = 200_000,
        top_n: int = 1,
        n_worlds: int = 10_000,
        candidates_per_match: int = 6,
        seed: int = 20260611,
        bonus_question_configs=(),
    ) -> None:
        self.field_model = field_model or PredictorDerivedFieldModel()
        self.pool_size = pool_size
        self.top_n = top_n
        self.n_worlds = n_worlds
        self.candidates_per_match = candidates_per_match
        self.seed = seed
        self._bonus_configs = list(bonus_question_configs)

    def generate_tips(
        self,
        predictions: dict[str, MatchPrediction],
        outcome: TournamentOutcome | None,
        fixtures: list[Match],
    ) -> TipSet:
        by_id = {m.match_id: m for m in fixtures}
        tips: dict[str, Tip] = {}
        if predictions:
            comp = optimize_slate(
                predictions, fixtures, self.field_model,
                pool_size=self.pool_size, top_n=self.top_n, n_worlds=self.n_worlds,
                candidates_per_match=self.candidates_per_match, seed=self.seed,
            )
            for mid, (th, ta) in comp.rank_slate.items():
                weight = by_id[mid].stage.points_weight
                ev = expected_points(predictions[mid].scoreline, th, ta, weight)
                ev_tip = comp.ev_slate[mid]
                if (th, ta) == ev_tip:
                    rationale = f"rank-opt: EV tip kept; slate P(rank<= {self.top_n}) {comp.rank_p_win:.4g}"
                else:
                    rationale = (
                        f"rank-opt: contrarian {th}:{ta} (EV tip {ev_tip[0]}:{ev_tip[1]}); "
                        f"slate P(rank<= {self.top_n}) {comp.rank_p_win:.4g} vs EV {comp.ev_p_win:.4g}"
                    )
                tips[mid] = Tip(mid, th, ta, ev, rationale)

        # Bonus answers: EV-optimal (modal) picks; contrarian bonus optimisation is out of scope.
        bonus_answers: dict[str, str] = {}
        if outcome is not None:
            for q in build_bonus_questions(self._bonus_configs):
                dist = q.resolve(outcome)
                if dist:
                    bonus_answers[q.question_id] = max(dist, key=dist.get)

        return TipSet(tips=tips, bonus_answers=bonus_answers)
