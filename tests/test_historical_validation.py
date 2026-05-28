"""Verify the bonus-question model outputs against recent historical statistics (spec §10).

These guard against the model drifting away from reality: each simulated/derived bonus
distribution is checked against the sourced figures in data/historical_stats.py.
"""

from pathlib import Path

import pytest

import tippspiel
from tippspiel.config import load_config, load_tournament, select_predictor
from tippspiel.data import historical_stats
from tippspiel.data.file_provider import FileDataProvider
from tippspiel.pipeline import build_predictor
from tippspiel.simulation.simulator import TournamentSimulator
from tippspiel.strategy.bonus import (
    SwissProgressBonus,
    TeamTotalGoalsBonus,
    TopScorerGoalsBonus,
    ZeroZeroCountBonus,
)

REPO = Path(tippspiel.__file__).parent.parent


@pytest.fixture(scope="module")
def outcome():
    cfg = select_predictor(load_config(REPO / "config.yaml"), "elo_poisson")
    b = load_tournament(REPO / "config.yaml")
    prov = FileDataProvider(b.teams_file, b.fixtures_file, b.results_file,
                            b.thirds_allocation_file)
    teams = {t.team_id: t for t in prov.get_teams()}
    sim = TournamentSimulator(prov.get_fixtures(), teams, {}, build_predictor(cfg),
                              prov.get_thirds_allocation(), iterations=8000, seed=123)
    return sim.run()


def _mean(dist: dict[str, float]) -> float:
    return sum(int(k) * v for k, v in dist.items())


def _mode(dist: dict[str, float]):
    return max(dist, key=dist.get)


def test_zero_zero_rate_within_historical_band(outcome):
    lo, hi = historical_stats.zero_zero_rate_band()
    mean_count = _mean(ZeroZeroCountBonus().resolve(outcome))
    rate = mean_count / 104  # 104 fixtures in WC2026
    # Model per-match 0:0 rate sits inside the band observed across recent tournaments
    # (small tolerance for Monte Carlo noise and the larger fixture count).
    assert lo * 0.9 <= rate <= hi * 1.1
    assert 2 <= mean_count <= 13


def test_top_scorer_prior_tracks_recent_world_cups(outcome):
    dist = TopScorerGoalsBonus().resolve(outcome)
    recent = historical_stats.recent_wc_top_scorer_mean()  # mean Golden Boot, last 3 WCs
    mean = _mean(dist)
    # The prior is the recent-WC mean nudged up for the 48-team / 104-match format,
    # so it should sit just above the historical mean, never below or wildly high.
    assert recent - 0.5 <= mean <= recent + 2.0
    assert 5 <= int(_mode(dist)) <= 9


def test_switzerland_progress_matches_mid_tier_history(outcome):
    dist = SwissProgressBonus().resolve(outcome)
    p_clear_group = 1.0 - dist.get("Gruppenphase", 0.0)
    # Switzerland have cleared the group in every recent major tournament.
    assert p_clear_group > 0.5
    # ...and historically exit around the first knockout rounds (R16/QF), so the modal
    # WC2026 exit should be one of the early knockout stages.
    assert _mode(dist) in {"Sechzehntelfinal", "Achtelfinal", "Viertelfinal"}


def test_switzerland_goals_in_plausible_band(outcome):
    dist = TeamTotalGoalsBonus().resolve(outcome)
    mean = _mean(dist)
    # Recent Swiss tournament goal totals were ~2-8 over 4-5 games; WC2026 adds more
    # potential matches, so a slightly higher mean is expected but must stay bounded.
    assert 2.0 <= mean <= 9.0
    hist = [g for _, _, g in historical_stats.SWITZERLAND_RESULTS]
    assert min(hist) <= mean <= max(hist) + 3
