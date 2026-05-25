"""Historical reference statistics for the bonus-question predictors and their
validation tests (spec §10 — "verification against reality").

All figures below are *post-120-minute* results where applicable (a 0:0 that is later
decided on penalties still counts as a 0:0 here), matching the SRF pool's question wording.
The data is kept small, sourced, and updatable; both the top-scorer prior and the
historical-validation tests read from it so the model is checked against reality rather
than against itself.

Sources (men's senior tournaments):
  - Golden Boot tallies: FIFA / UEFA / CONMEBOL official tournament records, cross-checked
    against the corresponding Wikipedia tournament pages.
  - 0:0 frequencies: tournament match archives (group + knockout 120-min results).
  - Switzerland finishes & goals: FIFA/UEFA match records for the Swiss national team.
"""

from __future__ import annotations

# --- Golden Boot (top scorer) goal tallies --------------------------------------------
# Men's FIFA World Cup. Most comparable to WC2026, so weighted most heavily in the prior.
WORLD_CUP_TOP_SCORER: dict[int, int] = {
    2002: 8,   # Ronaldo (BRA)
    2006: 5,   # Klose (GER)
    2010: 5,   # Müller / Forlán / Villa / Sneijder (tied)
    2014: 6,   # James Rodríguez (COL)
    2018: 6,   # Kane (ENG)
    2022: 8,   # Mbappé (FRA)
}
# Smaller continental tournaments (fewer matches per team) — used to widen the historical
# context band, not to set the WC prior centre.
EURO_TOP_SCORER: dict[int, int] = {2016: 6, 2020: 5, 2024: 3}
COPA_TOP_SCORER: dict[int, int] = {2021: 4, 2024: 4}

# --- 0:0 match frequency (matches ending 0:0 after 120 min / total matches) ------------
# Approximate per-tournament goalless-result rates; the validation only relies on the
# resulting band, not on exact single-tournament counts.
ZERO_ZERO_RATE: dict[str, tuple[int, int]] = {
    "WC2010": (5, 64),
    "WC2014": (1, 64),
    "WC2018": (1, 64),
    "WC2022": (4, 64),
    "EURO2020": (1, 51),
    "EURO2024": (2, 51),
}

# --- Switzerland recent tournament outcomes -------------------------------------------
# (tournament, finishing stage, goals scored over the tournament in 120-min play).
# Stage labels match SwissProgressBonus' German labels.
SWITZERLAND_RESULTS: list[tuple[str, str, int]] = [
    ("WC2014", "Achtelfinal", 7),     # lost R16 to ARG; 7 goals
    ("EURO2016", "Achtelfinal", 2),   # lost R16 to POL on penalties
    ("WC2018", "Achtelfinal", 5),     # lost R16 to SWE
    ("EURO2020", "Viertelfinal", 8),  # lost QF to ESP on penalties
    ("WC2022", "Achtelfinal", 5),     # lost R16 to POR
    ("EURO2024", "Viertelfinal", 5),  # lost QF to ENG on penalties
]


def recent_wc_top_scorer_mean(last_n: int = 3) -> float:
    """Mean Golden Boot tally over the most recent ``last_n`` World Cups."""
    years = sorted(WORLD_CUP_TOP_SCORER)[-last_n:]
    return sum(WORLD_CUP_TOP_SCORER[y] for y in years) / len(years)


def zero_zero_rate_band() -> tuple[float, float]:
    """(min, max) per-match 0:0 rate observed across the listed tournaments."""
    rates = [z / total for z, total in ZERO_ZERO_RATE.values()]
    return min(rates), max(rates)


# --- Top-scorer prior ------------------------------------------------------------------
# WC2026 expands to 48 teams / 104 matches and the winner now plays 8 games (was 7), so a
# deep run offers one extra scoring opportunity. We therefore nudge the historical World
# Cup Golden Boot tallies up by FORMAT_NUDGE, weight more recent World Cups more heavily,
# and apply light triangular smoothing so neighbouring counts share mass. The recommended
# answer is the mode of this distribution (exact-match scoring).
_RECENCY_WEIGHTS: dict[int, float] = {2022: 4, 2018: 3, 2014: 2, 2010: 1, 2006: 1, 2002: 1}
FORMAT_NUDGE = 1


def top_scorer_prior() -> dict[str, float]:
    """Fixed historical prior over the WC2026 Golden Boot goal tally (keyed by count str)."""
    counts: dict[int, float] = {}
    for year, goals in WORLD_CUP_TOP_SCORER.items():
        centre = goals + FORMAT_NUDGE
        w = _RECENCY_WEIGHTS.get(year, 1.0)
        for offset, share in ((-1, 0.25), (0, 0.50), (1, 0.25)):
            counts[centre + offset] = counts.get(centre + offset, 0.0) + w * share
    total = sum(counts.values())
    return {str(k): v / total for k, v in sorted(counts.items())}
