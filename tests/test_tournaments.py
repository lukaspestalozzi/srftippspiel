"""Tournaments layer: config-file resolution + format-general validation (spec §10).

Each tournament is one config file. The format (group count/size, knockout chain, whether
thirds qualify) is derived from the data — fixtures for unplayed tournaments encode the
knockout bracket via structured references; completed tournaments list concrete participants.
"""

from pathlib import Path

import pytest

import tippspiel
from tippspiel.cli import validate_data
from tippspiel.config import load_tournament
from tippspiel.data.file_provider import FileDataProvider
from tippspiel.simulation.bracket import Bracket

REPO = Path(tippspiel.__file__).parent.parent
WC2026 = REPO / "config.yaml"
EURO2016 = REPO / "configs" / "euro2016.yaml"
ALL_CONFIGS = [
    WC2026,
    EURO2016,
    REPO / "configs" / "wc2022.yaml",
    REPO / "configs" / "euro2024.yaml",
    REPO / "configs" / "wc2018.yaml",
    REPO / "configs" / "euro2020.yaml",
]


def _provider(bundle):
    return FileDataProvider(bundle.teams_file, bundle.fixtures_file,
                            bundle.results_file, bundle.thirds_allocation_file)


def test_config_files_resolve_to_expected_tournaments():
    assert load_tournament(WC2026).name == "wc2026"
    assert load_tournament(EURO2016).name == "euro2016"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_tournament(REPO / "configs" / "does_not_exist.yaml")


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda p: p.stem)
def test_validate_data_passes_for_all_tournaments(config):
    assert validate_data(load_tournament(config)) == []


def test_euro2016_format_is_24_teams_best_thirds_r16_first():
    b = load_tournament(EURO2016)
    assert b.completed is True
    teams = b.teams_file.read_text().strip().splitlines()[1:]  # minus header
    assert len(teams) == 24
    fixtures = _provider(b).get_fixtures()
    groups = {m.group for m in fixtures if m.group}
    assert len(groups) == 6
    ko_stages = {m.stage.value for m in fixtures if m.group is None}
    assert "R16" in ko_stages and "R32" not in ko_stages   # round of 16 is the first KO round
    assert "THIRD_PLACE" not in ko_stages                  # no third-place playoff


def test_wc2026_format_is_48_teams_with_thirds_r32_first():
    b = load_tournament(WC2026)
    assert b.completed is False
    prov = _provider(b)
    fixtures = prov.get_fixtures()
    groups = sorted({m.group for m in fixtures if m.group})
    assert len(groups) == 12
    ko = [m for m in fixtures if m.group is None]
    bracket = Bracket(ko, groups, prov.get_thirds_allocation())
    assert bracket.first_round_stage == "R32"
    assert len(bracket.first_round) == 16
    assert len(bracket.third_slots) == 8                  # best-8 thirds advance


def test_bonus_questions_are_tournament_scoped():
    wc = load_tournament(WC2026)
    eu = load_tournament(EURO2016)
    assert {q.id for q in wc.bonus_questions} >= {"champion", "swiss_progress"}
    assert {q.id for q in eu.bonus_questions} == {"champion"}
