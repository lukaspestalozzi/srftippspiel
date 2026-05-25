"""Data provider interfaces (spec §6.1.1, §6.1.6).

File-based providers are the default and recommended baseline: reproducible, no API
keys, no network flakiness, snapshot-able in version control. API-based providers are
optional plugins implementing the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..model.types import Match, Result, Team


class DataProvider(ABC):
    @abstractmethod
    def get_teams(self) -> list[Team]: ...

    @abstractmethod
    def get_fixtures(self) -> list[Match]: ...

    @abstractmethod
    def get_results(self) -> list[Result]: ...


@dataclass(frozen=True)
class Odds1X2:
    p_home: float
    p_draw: float
    p_away: float


class OddsProvider(ABC):
    """Phase 3 odds source. A file-based stub is provided; live fetching is not built."""

    @abstractmethod
    def get_match_odds(self, match_id: str) -> Odds1X2: ...
