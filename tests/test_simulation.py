"""Monte Carlo simulator tests: reproducibility, convergence, partial-state (spec §10)."""

from pathlib import Path

import pytest

import tippspiel
from tippspiel.config import load_config, load_tournament
from tippspiel.data.file_provider import FileDataProvider
from tippspiel.model.types import Result
from tippspiel.pipeline import build_predictor
from tippspiel.simulation.simulator import TournamentSimulator

REPO = Path(tippspiel.__file__).parent.parent


@pytest.fixture(scope="module")
def env():
    cfg = load_config(REPO / "config.yaml")
    b = load_tournament(REPO / "config.yaml")
    prov = FileDataProvider(b.teams_file, b.fixtures_file, b.results_file,
                            b.thirds_allocation_file)
    teams = {t.team_id: t for t in prov.get_teams()}
    return {
        "teams": teams,
        "fixtures": prov.get_fixtures(),
        "predictor": build_predictor(cfg),
        "thirds": prov.get_thirds_allocation(),
    }


def _sim(env, results, n, seed):
    return TournamentSimulator(
        env["fixtures"], env["teams"], results, env["predictor"],
        env["thirds"], iterations=n, seed=seed,
    )


def test_reproducible_for_same_seed(env):
    o1 = _sim(env, {}, 1500, 7).run()
    o2 = _sim(env, {}, 1500, 7).run()
    assert all(
        o1.advancement[t]["wins_title"] == o2.advancement[t]["wins_title"]
        for t in o1.advancement
    )


def test_probabilities_are_valid(env):
    out = _sim(env, {}, 2000, 1).run()
    title_sum = sum(out.advancement[t]["wins_title"] for t in out.advancement)
    qualify_sum = sum(out.advancement[t]["reach_r32"] for t in out.advancement)
    assert title_sum == pytest.approx(1.0, abs=1e-9)   # exactly one champion per iter
    assert qualify_sum == pytest.approx(32.0, abs=1e-9)  # exactly 32 qualifiers per iter
    for a in out.advancement.values():
        assert all(0.0 <= p <= 1.0 for p in a.values())


def test_convergence_stabilises(env):
    small = _sim(env, {}, 3000, 11).run()
    large = _sim(env, {}, 15000, 11).run()
    # A high, stable probability should agree closely across sample sizes.
    assert small.advancement["FRA"]["reach_r32"] == pytest.approx(
        large.advancement["FRA"]["reach_r32"], abs=0.05
    )
    assert large.mc_standard_error < small.mc_standard_error


def test_partial_state_drives_simulation(env):
    # Force all six Group A matches so Mexico wins the group outright; their
    # group-winner probability must then be exactly 1.0, independent of the seed.
    played = {
        "G_A_1": Result("G_A_1", 2, 0),  # MEX beat RSA
        "G_A_2": Result("G_A_2", 1, 1),  # KOR-CZE
        "G_A_3": Result("G_A_3", 0, 0),  # CZE-RSA
        "G_A_4": Result("G_A_4", 2, 0),  # MEX beat KOR
        "G_A_5": Result("G_A_5", 1, 1),  # RSA-KOR
        "G_A_6": Result("G_A_6", 0, 2),  # CZE lost to MEX
    }
    out = _sim(env, played, 1000, 3).run()
    assert out.advancement["MEX"]["group_winner"] == pytest.approx(1.0)
    assert out.advancement["MEX"]["reach_r32"] == pytest.approx(1.0)
