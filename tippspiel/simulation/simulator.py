"""TournamentSimulator: vectorised Monte Carlo over the whole tournament (spec §6.4).

Each iteration: sample/use group results, compute standings (§3.2), select best-8 thirds
(§3.3), assemble the Round of 32 (§6.1.4), then play the knockout rounds, resolving draws
with a penalty-shootout model. Aggregated over N iterations into a TournamentOutcome.

Sampling, standings (criteria 1-3), thirds selection, bracket assembly and every knockout
round are vectorised across all N iterations as NumPy array operations so that 50,000
simulations run in seconds. Same seed + inputs => identical output.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from ..model.stages import Stage
from ..model.types import Match, Result, Team, TeamRef, TournamentOutcome
from ..predictors.base import Predictor
from .bracket import Bracket
from .standings import rank_group
from .thirds import select_best_thirds

_ADV_METRICS = [
    "group_winner", "group_second", "group_third", "qualifies_r32",
    "reach_r16", "reach_qf", "reach_sf", "reach_final", "wins_title",
]


class TournamentSimulator:
    def __init__(
        self,
        fixtures: list[Match],
        teams: dict[str, Team],
        results: dict[str, Result],
        predictor: Predictor,
        bracket_map: dict,
        iterations: int = 50000,
        seed: int = 20260611,
        penalty_model: str = "coin_flip",
    ) -> None:
        self.fixtures = fixtures
        self.teams = teams
        self.results = results
        self.predictor = predictor
        self.bracket = Bracket(bracket_map)
        self.n = iterations
        self.seed = seed
        self.penalty_model = penalty_model
        self.rng = np.random.default_rng(seed)

        self.team_ids = list(teams.keys())
        self.idx = {tid: i for i, tid in enumerate(self.team_ids)}
        self.nteams = len(self.team_ids)
        self.elo = np.array([teams[t].elo for t in self.team_ids])
        self.gmax = int(getattr(predictor, "gmax", 7))
        self.k = self.gmax + 1

        self._group_layouts = self._build_group_layouts()
        self._group_pmf_cache: dict[str, np.ndarray] = {}
        self._pair_cdf = self._build_pair_cdf()

    # ----- setup ---------------------------------------------------------------
    def _build_group_layouts(self) -> dict[str, dict]:
        groups: dict[str, list[Match]] = {}
        for m in self.fixtures:
            if m.group:
                groups.setdefault(m.group, []).append(m)
        layouts = {}
        for letter, ms in groups.items():
            ms = sorted(ms, key=lambda m: m.match_id)
            local_ids = sorted({m.home.team_id for m in ms} | {m.away.team_id for m in ms})
            local = {tid: i for i, tid in enumerate(local_ids)}
            layout = [(local[m.home.team_id], local[m.away.team_id]) for m in ms]
            global_of_local = [self.idx[tid] for tid in local_ids]
            layouts[letter] = {
                "matches": ms, "layout": layout, "global": np.array(global_of_local),
            }
        return layouts

    def _flat_pmf(self, dist) -> np.ndarray:
        return dist.matrix.flatten()

    def _build_pair_cdf(self) -> np.ndarray:
        """Precompute flat scoreline CDFs for all ordered team pairs (neutral knockout)."""
        cdf = np.zeros((self.nteams * self.nteams, self.k * self.k))
        now = datetime.now(timezone.utc)
        for a in range(self.nteams):
            for b in range(self.nteams):
                if a == b:
                    continue
                m = Match(
                    match_id=f"_pair_{a}_{b}", stage=Stage.R32,
                    home=TeamRef(team_id=self.team_ids[a]),
                    away=TeamRef(team_id=self.team_ids[b]),
                    kickoff=now, group=None, venue_country=None,
                )
                pred = self.predictor.predict(m, self.teams)
                cdf[a * self.nteams + b] = np.cumsum(self._flat_pmf(pred.scoreline))
        return cdf

    # ----- sampling ------------------------------------------------------------
    def _sample_from_cdf(self, cdf: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.minimum((cdf[None, :] < u[:, None]).sum(axis=1), cdf.shape[0] - 1)
        return idx // self.k, idx % self.k

    def _sample_group(self, letter: str) -> tuple[np.ndarray, np.ndarray]:
        info = self._group_layouts[letter]
        ms = info["matches"]
        hg = np.zeros((self.n, len(ms)), dtype=np.int64)
        ag = np.zeros((self.n, len(ms)), dtype=np.int64)
        for j, m in enumerate(ms):
            if m.match_id in self.results:
                r = self.results[m.match_id]
                hg[:, j] = r.home_goals
                ag[:, j] = r.away_goals
            else:
                pred = self.predictor.predict(m, self.teams)
                cdf = np.cumsum(self._flat_pmf(pred.scoreline))
                u = self.rng.random(self.n)
                hg[:, j], ag[:, j] = self._sample_from_cdf(cdf, u)
        return hg, ag

    # ----- knockout ------------------------------------------------------------
    def _play(self, home: np.ndarray, away: np.ndarray, fixture_id: str):
        if fixture_id in self.results:
            r = self.results[fixture_id]
            hg = np.full(self.n, r.home_goals)
            ag = np.full(self.n, r.away_goals)
            if r.home_goals == r.away_goals and r.winner_team_id:
                w = self.idx[r.winner_team_id]
                winner = np.full(self.n, w)
            else:
                winner = np.where(r.home_goals > r.away_goals, home, away)
        else:
            pair = home * self.nteams + away
            cdf_rows = self._pair_cdf[pair]  # [N, K]
            u = self.rng.random(self.n)
            idx = np.minimum((cdf_rows < u[:, None]).sum(axis=1), self.k * self.k - 1)
            hg, ag = idx // self.k, idx % self.k
            home_win = hg > ag
            away_win = ag > hg
            draw = ~(home_win | away_win)
            if self.penalty_model == "elo_weighted":
                p_home = 1.0 / (1.0 + 10.0 ** (-(self.elo[home] - self.elo[away]) / 400.0))
                pen_home = self.rng.random(self.n) < p_home
            else:
                pen_home = self.rng.random(self.n) < 0.5
            winner = np.where(home_win, home, np.where(away_win, away, np.where(pen_home, home, away)))
        loser = np.where(winner == home, away, home)
        return winner, loser, hg, ag

    # ----- run -----------------------------------------------------------------
    def run(self) -> TournamentOutcome:
        n, nt = self.n, self.nteams
        counts = {m: np.zeros(nt) for m in _ADV_METRICS}
        # Per-iteration tallies for the Swiss-goals and 0:0-count bonus questions.
        team_goals = np.zeros((n, nt), dtype=np.int64)
        zero_zero = np.zeros(n, dtype=np.int64)
        it = np.arange(n)

        winners = np.zeros((n, 12), dtype=np.int64)
        runners = np.zeros((n, 12), dtype=np.int64)
        thirds_team = np.zeros((n, 12), dtype=np.int64)
        thirds_pts = np.zeros((n, 12))
        thirds_gd = np.zeros((n, 12))
        thirds_gf = np.zeros((n, 12))

        for gi, letter in enumerate("ABCDEFGHIJKL"):
            info = self._group_layouts[letter]
            hg, ag = self._sample_group(letter)
            g_global = info["global"]
            for j, (hl, al) in enumerate(info["layout"]):
                team_goals[:, g_global[hl]] += hg[:, j]
                team_goals[:, g_global[al]] += ag[:, j]
                zero_zero += (hg[:, j] == 0) & (ag[:, j] == 0)
            rand = self.rng.random((n, 4))
            order, pts, gd, gf = rank_group(hg, ag, info["layout"], rand)
            w_local, s_local, t_local = order[:, 0], order[:, 1], order[:, 2]
            winners[:, gi] = g_global[w_local]
            runners[:, gi] = g_global[s_local]
            thirds_team[:, gi] = g_global[t_local]
            rows = np.arange(n)
            thirds_pts[:, gi] = pts[rows, t_local]
            thirds_gd[:, gi] = gd[rows, t_local]
            thirds_gf[:, gi] = gf[rows, t_local]
            np.add.at(counts["group_winner"], winners[:, gi], 1)
            np.add.at(counts["group_second"], runners[:, gi], 1)
            np.add.at(counts["group_third"], thirds_team[:, gi], 1)

        rand_thirds = self.rng.random((n, 12))
        qualified, _order = select_best_thirds(thirds_pts, thirds_gd, thirds_gf, rand_thirds)
        slot_group_idx, _mask = self.bracket.assign_thirds(qualified)
        rows = np.arange(n)[:, None]
        third_in_slot = thirds_team[rows, slot_group_idx]  # [N, n_slots]
        slot_pos = {s: p for p, s in enumerate(self.bracket.third_slots)}

        def resolve_spec(spec) -> np.ndarray:
            kind, val = spec
            if kind == "W":
                return winners[:, val]
            if kind == "RU":
                return runners[:, val]
            if kind == "3RD":
                return third_in_slot[:, slot_pos[val]]
            raise ValueError(kind)

        # Round of 32.
        match_winner: dict[str, np.ndarray] = {}
        match_loser: dict[str, np.ndarray] = {}
        opp_dist: dict[str, dict] = {}
        r32_ids = sorted(self.bracket.map["r32"], key=int)
        for num, (hspec, aspec) in zip(r32_ids, self.bracket.r32_specs):
            home = resolve_spec(hspec)
            away = resolve_spec(aspec)
            np.add.at(counts["qualifies_r32"], home, 1)
            np.add.at(counts["qualifies_r32"], away, 1)
            fid = f"M{num}"
            opp_dist[fid] = {"home": self._dist(home), "away": self._dist(away)}
            w, l, hg, ag = self._play(home, away, fid)
            np.add.at(team_goals, (it, home), hg)
            np.add.at(team_goals, (it, away), ag)
            zero_zero += (hg == 0) & (ag == 0)
            match_winner[num] = w
            match_loser[num] = l
            np.add.at(counts["reach_r16"], w, 1)

        stage_metric = {"R16": "reach_qf", "QF": "reach_sf", "SF": "reach_final", "FINAL": "wins_title"}
        for num, hspec, aspec, stage in self.bracket.progression:
            home = match_winner[hspec[1]] if hspec[0] == "WIN" else match_loser[hspec[1]]
            away = match_winner[aspec[1]] if aspec[0] == "WIN" else match_loser[aspec[1]]
            w, l, hg, ag = self._play(home, away, f"M{num}")
            np.add.at(team_goals, (it, home), hg)
            np.add.at(team_goals, (it, away), ag)
            zero_zero += (hg == 0) & (ag == 0)
            match_winner[num] = w
            match_loser[num] = l
            if stage in stage_metric:
                np.add.at(counts[stage_metric[stage]], w, 1)

        return self._aggregate(counts, opp_dist, team_goals, zero_zero)

    def _dist(self, team_arr: np.ndarray) -> dict[str, float]:
        vals, cnt = np.unique(team_arr, return_counts=True)
        return {self.team_ids[int(v)]: float(c) / self.n for v, c in zip(vals, cnt)}

    def _count_dist(self, arr: np.ndarray) -> dict[str, float]:
        """Distribution over integer counts (keyed by the stringified count)."""
        vals, cnt = np.unique(arr, return_counts=True)
        return {str(int(v)): float(c) / self.n for v, c in zip(vals, cnt)}

    def _aggregate(self, counts, opp_dist, team_goals, zero_zero) -> TournamentOutcome:
        advancement: dict[str, dict[str, float]] = {}
        max_se = 0.0
        for i, tid in enumerate(self.team_ids):
            adv = {m: counts[m][i] / self.n for m in _ADV_METRICS}
            advancement[tid] = adv
            for p in adv.values():
                max_se = max(max_se, (p * (1 - p) / self.n) ** 0.5)
        champ = {tid: advancement[tid]["wins_title"] for tid in self.team_ids
                 if advancement[tid]["wins_title"] > 0}
        team_goal_distribution = {
            tid: self._count_dist(team_goals[:, i])
            for i, tid in enumerate(self.team_ids)
        }
        return TournamentOutcome(
            advancement=advancement,
            opponent_distribution=opp_dist,
            bonus_probabilities={"champion": champ},
            mc_iterations=self.n,
            mc_seed=self.seed,
            mc_standard_error=max_se,
            team_goal_distribution=team_goal_distribution,
            zero_zero_distribution=self._count_dist(zero_zero),
        )
