"""Scalar Elo: a single per-team strength rating fitted from the match-results corpus.

This replaces the externally-fetched eloratings.net rating with one computed in-repo from the
committed international-match corpus, using the **World Football Elo Ratings** update (the same
algorithm eloratings.net publishes), so the result lands on the familiar ~1000–2100 scale the
predictor's ``k`` is tuned for.

Per match, in chronological order::

    dr = (R_home + home_adv) - R_away          # home_adv = 100 unless neutral venue
    We = 1 / (1 + 10 ** (-dr / 400))           # expected home score (0..1)
    W  = 1.0 / 0.5 / 0.0                        # actual home result (win/draw/loss)
    G  = goal-difference multiplier            # 1, 1.5, or (11+gd)/8 for gd >= 3
    delta = K * G * (W - We)                    # K = match importance (friendly 20 .. WC 60)
    R_home += delta ; R_away -= delta           # zero-sum

Unlike the off/def fit, this is a **single chronological pass** (classical Elo's "current
rating" is path-dependent; epochs would corrupt it) and the ratings are **not zero-centred** —
the absolute level *is* the rating. Deterministic: same corpus + params -> same ratings.

Run by ``tippspiel fit-ratings`` (alongside the off/def fit); snapshotting to a cutoff date
keeps a ``verify`` backtest leak-free exactly as the off/def fit does.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .offdef_elo import HistMatch


@dataclass(frozen=True)
class ScalarEloParams:
    """Fit hyperparameters for the World-Football-Elo update.

    Defaults are **calibrated against the completed-tournament backtests**, not the canonical
    eloratings values (1.0 / 100): a from-scratch single-pass fit compresses the rating spread,
    so a higher ``k_scale`` (faster movement, wider current spread) and a lower fitting
    ``home_advantage`` (sharper attribution of qualifier home/away results) recover the pool-point
    gap on the benchmarks. ``k_scale`` multiplies every match's K base (from
    ``HistMatch.k_importance``) — the single knob the calibration sweep optimises."""

    start_rating: float = 1500.0
    home_advantage: float = 60.0
    k_scale: float = 1.4


def _goal_diff_multiplier(gd: int) -> float:
    """World-Football-Elo margin-of-victory multiplier (1 for gd<=1, 1.5 for gd==2, then grows)."""
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def fit_scalar_elo(
    matches: list[HistMatch], params: ScalarEloParams | None = None
) -> dict[str, float]:
    """Fit a scalar Elo rating for every team appearing in ``matches``.

    Deterministic: matches are sorted by date (stable), so the same corpus + params always yield
    the same ratings. Returns ``team_name -> rating`` (not zero-centred)."""
    params = params or ScalarEloParams()
    ordered = sorted(matches, key=lambda m: m.date)
    rating: dict[str, float] = defaultdict(lambda: params.start_rating)

    for m in ordered:
        home_adv = 0.0 if m.neutral else params.home_advantage
        dr = (rating[m.home] + home_adv) - rating[m.away]
        we = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        if m.home_goals > m.away_goals:
            w = 1.0
        elif m.home_goals < m.away_goals:
            w = 0.0
        else:
            w = 0.5
        g = _goal_diff_multiplier(abs(m.home_goals - m.away_goals))
        delta = params.k_scale * m.k_importance * g * (w - we)
        rating[m.home] += delta
        rating[m.away] -= delta

    return dict(rating)
