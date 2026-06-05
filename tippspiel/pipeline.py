"""Pipeline orchestration: data -> predictor -> (simulator) -> strategy -> report."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from .config import Config, TournamentBundle
from .data.base import Odds1X2
from .data.file_provider import FileDataProvider
from .model.types import Match, MatchPrediction, Team, TournamentOutcome
from .predictors.base import Predictor
from .predictors.elo_poisson import EloPoissonPredictor
from .predictors.market_odds import MarketOddsPredictor
from .report import charts
from .report.html_writer import ReportWriter
from .strategy.bonus import build_bonus_questions
from .strategy.expected_points import (
    ExpectedPointsStrategy,
    best_tip,
    ev_components,
    expected_points,
)

CAVEATS = (
    "The Elo-Poisson model is a reasonable forecaster but will not systematically "
    "out-predict the betting market. When an odds.csv snapshot is supplied the market-odds "
    "predictor uses de-vigged bookmaker odds for those fixtures (falling back to Elo "
    "elsewhere), which is the higher-accuracy option; the per-fixture \"Market-odds tip\" "
    "shows that path's scoreline. This tool's edge over casual pool participants is correct "
    "probability-to-scoreline optimisation and bracket simulation for the champion bonus. "
    "Elo ratings and odds are snapshots and change over time."
)


def build_predictor(cfg: Config, odds: dict[str, Odds1X2] | None = None) -> Predictor:
    if cfg.predictor.name == "elo_poisson":
        return EloPoissonPredictor(**cfg.predictor.params)
    if cfg.predictor.name == "market_odds":
        params = dict(cfg.predictor.params)
        fallback_params = params.pop("fallback_params", {})
        fallback = EloPoissonPredictor(**fallback_params)
        return MarketOddsPredictor(odds=odds, fallback=fallback, **params)
    raise ValueError(f"Unknown predictor: {cfg.predictor.name}")


def build_strategy(cfg: Config, bundle: TournamentBundle) -> ExpectedPointsStrategy:
    return ExpectedPointsStrategy(bonus_question_configs=bundle.bonus_questions)


def _predict_tippable(
    fixtures: list[Match],
    teams: dict[str, Team],
    played: set[str],
    predictor: Predictor,
) -> dict[str, MatchPrediction]:
    """Predict every fixture that has concrete participants and is not yet played."""
    preds: dict[str, MatchPrediction] = {}
    for m in fixtures:
        if m.match_id in played or not m.participants_known:
            continue
        preds[m.match_id] = predictor.predict(m, teams)
    return preds


def _run_core(cfg: Config, bundle: TournamentBundle, *, simulate: bool) -> dict:
    """Load data, predict, optionally simulate, and generate tips.

    Returns the raw objects shared by the HTML report and the diagnostic report, so neither
    path has to rebuild them (and the diagnostic path can skip the expensive Plotly context).
    """
    provider = FileDataProvider(
        bundle.teams_file,
        bundle.fixtures_file,
        bundle.results_file,
        bundle.thirds_allocation_file,
        bundle.odds_file,
    )
    teams = {t.team_id: t for t in provider.get_teams()}
    fixtures = provider.get_fixtures()
    results = {r.match_id: r for r in provider.get_results()}
    played = set(results)
    odds = provider.get_odds()

    predictor = build_predictor(cfg, odds=odds)
    strategy = build_strategy(cfg, bundle)

    outcome: TournamentOutcome | None = None
    if simulate:
        from .simulation.simulator import TournamentSimulator

        sim = TournamentSimulator(
            fixtures=fixtures,
            teams=teams,
            results=results,
            predictor=predictor,
            thirds_allocation=provider.get_thirds_allocation(),
            iterations=cfg.simulation.iterations,
            seed=cfg.simulation.seed,
            penalty_model=cfg.simulation.penalty_model,
        )
        outcome = sim.run()

    predictions = _predict_tippable(fixtures, teams, played, predictor)
    tipset = strategy.generate_tips(predictions, outcome, fixtures)
    market_predictions = _market_predictions(cfg, fixtures, teams, played, odds)

    return {
        "teams": teams, "fixtures": fixtures, "results": results,
        "predictions": predictions, "tipset": tipset, "outcome": outcome,
        "predictor": predictor, "market_predictions": market_predictions,
        "odds": odds,
    }


def _market_predictions(
    cfg: Config,
    fixtures: list[Match],
    teams: dict[str, Team],
    played: set[str],
    odds: dict[str, Odds1X2],
) -> dict[str, MatchPrediction]:
    """Market-odds scoreline predictions for the report's per-fixture market tips.

    Runs a dedicated ``MarketOddsPredictor`` *regardless of the configured predictor*, over the
    **full** tippable slate (Elo fallback where a fixture has no odds row). Display is gated to
    genuine-odds fixtures separately. Empty when no odds exist at all.
    """
    if not odds:
        return {}
    p = cfg.predictor.params if cfg.predictor.name == "market_odds" else {}
    gmax = int(p.get("gmax", 7))
    predictor = MarketOddsPredictor(
        odds=odds,
        fallback=EloPoissonPredictor(gmax=gmax),
        total_goals=p.get("total_goals", 2.6),
        gmax=gmax,
        ko_goal_scale=p.get("ko_goal_scale", 1.0),
    )
    return _predict_tippable(fixtures, teams, played, predictor)


def run_pipeline(
    cfg: Config,
    bundle: TournamentBundle,
    *,
    simulate: bool,
) -> dict:
    core = _run_core(cfg, bundle, simulate=simulate)
    context = _build_report_context(
        cfg, bundle, core["teams"], core["fixtures"], core["results"],
        core["predictions"], core["tipset"], core["outcome"], core["predictor"],
        core["market_predictions"], core["odds"],
    )
    return {"context": context, "tipset": core["tipset"], "outcome": core["outcome"]}


def write_diagnostics(cfg: Config, bundle: TournamentBundle, *, simulate: bool) -> dict:
    """Run the core pipeline and write the Claude diagnostic report (markdown + JSON)."""
    from .report.diagnostics import DiagnosticsWriter, build_diagnostics

    core = _run_core(cfg, bundle, simulate=simulate)
    markdown, data = build_diagnostics(
        cfg, bundle, core["teams"], core["fixtures"], core["results"],
        core["predictions"], core["tipset"], core["outcome"], core["predictor"],
    )
    paths = DiagnosticsWriter().write(markdown, data, cfg.report.output_dir)
    return {"paths": paths, "data": data}


def write_verification(cfg: Config, bundle: TournamentBundle) -> dict:
    """Backtest the predictor against a completed tournament; write output/verify.{md,json}."""
    from .report.backtest import VerificationWriter, build_verification

    provider = FileDataProvider(
        bundle.teams_file, bundle.fixtures_file, bundle.results_file,
        odds_file=bundle.odds_file,
    )
    teams = {t.team_id: t for t in provider.get_teams()}
    fixtures = provider.get_fixtures()
    results = {r.match_id: r for r in provider.get_results()}
    predictor = build_predictor(cfg, odds=provider.get_odds())
    markdown, data = build_verification(bundle, teams, fixtures, results, predictor)
    paths = VerificationWriter().write(markdown, data, cfg.report.output_dir)
    return {"paths": paths, "data": data}


def run_tuning(base_cfg: Config, benchmark_configs, *, top: int = 15, grid=None) -> dict:
    """Sweep predictor params against completed-tournament backtests; write output/tune.{md,json}."""
    from .config import load_tournament
    from .report.tuning import TuningWriter, build_tuning

    benchmarks = []
    for cfg_path in benchmark_configs:
        bundle = load_tournament(cfg_path)
        provider = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file)
        teams = {t.team_id: t for t in provider.get_teams()}
        fixtures = provider.get_fixtures()
        results = {r.match_id: r for r in provider.get_results()}
        benchmarks.append((bundle, teams, fixtures, results))
    markdown, data = build_tuning(base_cfg, benchmarks, grid=grid, top=top)
    paths = TuningWriter().write(markdown, data, base_cfg.report.output_dir)
    return {"paths": paths, "data": data}


def write_offdef_snapshot(bundle: TournamentBundle, offdef_block: dict | None = None) -> dict:
    """Fit offensive/defensive Elo from the historical corpus and persist it to teams.csv.

    Fits ``att_elo``/``def_elo`` for every team in the corpus from all matches **strictly
    before** the tournament's first kickoff (so a ``verify`` backtest stays leak-free), then
    writes the subset mapping to this tournament's teams back into its ``teams.csv`` (preserving
    the existing columns). Returns a small stats dict for the CLI to print.
    """
    import csv as _csv

    from .data.historical_results_adapter import (
        DEFAULT_CORPUS,
        WeightTiers,
        corpus_name_for,
        load_corpus,
        ratings_for_team,
    )
    from .training.offdef_elo import OffDefParams, OffDefRating, fit_off_def

    block = dict(offdef_block or {})
    params = OffDefParams(
        mu=float(block.get("mu", OffDefParams.mu)),
        k_att=float(block.get("k_att", OffDefParams.k_att)),
        k_def=float(block.get("k_def", OffDefParams.k_def)),
        gamma_home=float(block.get("gamma_home", OffDefParams.gamma_home)),
        residual_cap=float(block.get("residual_cap", OffDefParams.residual_cap)),
        epochs=int(block.get("epochs", OffDefParams.epochs)),
    )
    w = block.get("weights", {}) or {}
    tiers = WeightTiers(
        friendly=float(w.get("friendly", WeightTiers.friendly)),
        qualifier=float(w.get("qualifier", WeightTiers.qualifier)),
        continental=float(w.get("continental", WeightTiers.continental)),
        world_cup=float(w.get("world_cup", WeightTiers.world_cup)),
        default=float(w.get("default", WeightTiers.default)),
    )
    corpus_path = block.get("corpus_file", DEFAULT_CORPUS)

    provider = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file)
    fixtures = provider.get_fixtures()
    snapshot = block.get("snapshot_date") or min(m.kickoff for m in fixtures).date().isoformat()

    matches = load_corpus(before=snapshot, corpus_path=corpus_path, tiers=tiers)
    ratings = fit_off_def(matches, params)

    teams = provider.get_teams()
    by_id: dict[str, OffDefRating] = {}
    unmapped: list[str] = []
    for t in teams:
        by_id[t.team_id] = ratings_for_team(t.name, ratings)
        if corpus_name_for(t.name) not in ratings:
            unmapped.append(t.name)

    _write_teams_csv_with_offdef(bundle.teams_file, by_id, _csv)

    ranked_att = sorted(teams, key=lambda t: by_id[t.team_id].att, reverse=True)
    ranked_def = sorted(teams, key=lambda t: by_id[t.team_id].def_, reverse=True)
    return {
        "snapshot_date": snapshot,
        "corpus_matches": len(matches),
        "corpus_teams": len(ratings),
        "teams_written": len(teams),
        "unmapped": unmapped,
        "top_attack": [(t.name, by_id[t.team_id].att) for t in ranked_att[:5]],
        "top_defence": [(t.name, by_id[t.team_id].def_) for t in ranked_def[:5]],
    }


def _write_teams_csv_with_offdef(teams_file, by_id, _csv) -> None:
    """Rewrite teams.csv with att_elo/def_elo columns appended, preserving existing columns."""
    with open(teams_file, newline="") as fh:
        reader = _csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for col in ("att_elo", "def_elo"):
        if col not in fieldnames:
            fieldnames.append(col)
    for row in rows:
        tid = (row.get("team_id") or "").strip()
        r = by_id.get(tid)
        if r is not None:
            row["att_elo"] = f"{r.att:.4f}"
            row["def_elo"] = f"{r.def_:.4f}"
    with open(teams_file, "w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_report_context(
    cfg, bundle, teams, fixtures, results, predictions, tipset, outcome, predictor,
    market_predictions=None, odds=None,
) -> dict:
    market_predictions = market_predictions or {}
    # The per-fixture market tip: predictions carry the scoreline (for the EV-optimal tip), and
    # ``odds_ids`` gates display to fixtures backed by genuine bookmaker odds (never a silent
    # Elo-fallback duplicate).
    market = {
        "preds": market_predictions,
        "odds_ids": set(odds or {}),
        "odds": odds or {},
    }
    # Off/def goal-volume weight of the active predictor; gates the per-fixture att/def display.
    alpha = float(getattr(predictor, "alpha", 0.0))
    group_fixtures = _group_fixture_blocks(
        teams, fixtures, results, predictions, tipset, market, alpha
    )
    advancement = _advancement_sections(teams, fixtures, outcome)
    knockout_fixtures = _knockout_sections(
        teams, fixtures, results, predictions, tipset, outcome, market, alpha
    )

    title_odds_chart = None
    bracket_html = None
    bonus = []
    if outcome is not None:
        title_rows = sorted(
            ((teams[t].name, m.get("wins_title", 0.0)) for t, m in outcome.advancement.items()),
            key=lambda r: r[1],
            reverse=True,
        )[:20]
        title_odds_chart = charts.title_odds_bar(title_rows)
        bracket_html = _bracket_chart(teams, outcome)
        bonus = _bonus_sections(bundle, teams, tipset, outcome)

    header = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "predictor_name": predictor.name,
        "predictor_params": getattr(predictor, "params", {}),
        "mc_iterations": outcome.mc_iterations if outcome else None,
        "mc_seed": outcome.mc_seed if outcome else None,
        "results_count": len(results),
        "uses_offdef": alpha > 0,
    }
    return {
        "header": header,
        "group_fixtures": group_fixtures,
        "advancement": advancement,
        "knockout_fixtures": knockout_fixtures,
        "title_odds_chart": title_odds_chart,
        "bracket_html": bracket_html,
        "bonus": bonus,
        "caveats": CAVEATS,
    }


def _fixture_block(
    m, teams, results, predictions, tipset, weight, market=None, alpha=0.0
) -> dict:
    name_h = teams[m.home.team_id].name if m.home.is_concrete else m.home.placeholder
    name_a = teams[m.away.team_id].name if m.away.is_concrete else m.away.placeholder
    block = {"match_id": m.match_id, "home": name_h, "away": name_a, "kickoff": m.kickoff,
             "stage": m.stage.value, "group": m.group, "played": m.match_id in results,
             "result": None, "tip": None, "naive": None, "market_tip": None, "data": None,
             "ldw_chart": None, "heatmap": None}
    if block["played"]:
        r = results[m.match_id]
        block["result"] = {"home_goals": r.home_goals, "away_goals": r.away_goals}
        return block
    pred = predictions.get(m.match_id)
    if pred is None:
        return block
    dist = pred.scoreline
    tip = tipset.tips.get(m.match_id)
    rec_h = rec_a = None
    if tip is not None:
        rec_h, rec_a = tip.tip_home, tip.tip_away
        block["tip"] = {"home": tip.tip_home, "away": tip.tip_away, "ev": tip.expected_points}
        nh, na, _ = dist.most_likely_scorelines(1)[0]
        block["naive"] = {"home": nh, "away": na,
                          "ev": expected_points(dist, nh, na, weight)}
    _set_market_tips(block, m, weight, market)
    rec = (rec_h, rec_a) if tip is not None else None
    block["data"] = _fixture_data(m, teams, dist, rec, weight, market, alpha)
    block["ldw_chart"] = charts.ldw_bar(dist, name_h, name_a)
    block["heatmap"] = charts.scoreline_heatmap(dist, rec_h, rec_a)
    return block


def _fixture_data(m, teams, dist, rec, weight, market, alpha=0.0) -> dict:
    """The underlying prediction numbers surfaced in the per-fixture data table — the same
    payload ``diagnostics._fixture_records`` assembles, so a reader can see *why* a tip wins
    (Elo, off/def ratings, exact L/D/W, top scorelines, the EV breakdown, expected goals,
    de-vigged market odds).
    """
    e_home, e_away = dist.expected_goals()
    data = {
        "ldw": {"home": dist.p_home_win(), "draw": dist.p_draw(), "away": dist.p_away_win()},
        "exp_goals": {"home": e_home, "away": e_away, "total": e_home + e_away},
        "top3": [(h, a, p) for h, a, p in dist.most_likely_scorelines(3)],
        "elo": None,
        "offdef": None,
        "rec_components": None,
        "rec_cell_prob": None,
        "market_probs": None,
    }
    # Elo + off/def only when both sides are concrete teams (knockout slots may be placeholders).
    if m.home.is_concrete and m.away.is_concrete:
        th, ta = teams[m.home.team_id], teams[m.away.team_id]
        data["elo"] = {"home": th.elo, "away": ta.elo, "diff": th.elo - ta.elo}
        # Show the att/def ratings the model is actually using (only when alpha>0 and the teams
        # carry fitted ratings), with the net goal-volume effect this matchup gets vs pure Elo.
        if alpha and (th.att_elo or th.def_elo or ta.att_elo or ta.def_elo):
            raw_vol = ((th.att_elo + ta.att_elo) - (th.def_elo + ta.def_elo)) / 2.0
            pct = (math.exp(alpha * raw_vol) - 1.0) * 100.0
            data["offdef"] = {
                "home_name": th.name, "away_name": ta.name,
                "att_home": th.att_elo, "def_home": th.def_elo,
                "att_away": ta.att_elo, "def_away": ta.def_elo,
                "pct": f"{pct:+.0f}% goals",
            }
    if rec is not None:
        rec_h, rec_a = rec
        data["rec_components"] = ev_components(dist, rec_h, rec_a, weight)
        data["rec_cell_prob"] = dist.cell(rec_h, rec_a)
    # De-vigged bookmaker 1X2, only for fixtures backed by genuine odds (same gate as the tip).
    odds = (market or {}).get("odds", {})
    if m.match_id in odds:
        o = odds[m.match_id]
        data["market_probs"] = {"home": o.p_home, "draw": o.p_draw, "away": o.p_away}
    return data


def _set_market_tips(block, m, weight, market) -> None:
    """Attach the EV-optimal market-odds tip to ``block`` — shown only for fixtures backed by
    genuine bookmaker odds, so it's a real market prediction rather than a silent Elo-fallback
    duplicate of the recommended tip. No-op when the fixture has no odds."""
    if not market or m.match_id not in market["odds_ids"]:
        return
    market_pred = market["preds"].get(m.match_id)
    if market_pred is None:
        return
    mdist = market_pred.scoreline
    th, ta, ev = best_tip(mdist, weight)
    block["market_tip"] = {"home": th, "away": ta, "ev": ev}


def _group_fixture_blocks(teams, fixtures, results, predictions, tipset,
                          market=None, alpha=0.0) -> list[dict]:
    """All group-stage fixtures as one globally chronological list (not grouped into per-group
    sections). Each block carries its ``group`` letter, surfaced as a tag in the report."""
    ms = sorted((m for m in fixtures if m.group), key=lambda m: m.kickoff)
    return [_fixture_block(m, teams, results, predictions, tipset, 1, market, alpha) for m in ms]


def _advancement_sections(teams, fixtures, outcome) -> list[dict]:
    """Per-group advancement charts — the one place the report still groups by group, since the
    chart is inherently a per-group standings view. Empty when there is no simulation."""
    if outcome is None:
        return []
    by_group: dict[str, list[Match]] = {}
    for m in fixtures:
        if m.group:
            by_group.setdefault(m.group, []).append(m)
    return [{"letter": letter,
             "advancement_chart": _advancement_chart(letter, by_group[letter], teams, outcome)}
            for letter in sorted(by_group)]


def _advancement_chart(letter, group_matches, teams, outcome):
    team_ids = sorted({tid for m in group_matches for tid in (m.home.team_id, m.away.team_id)})
    rows = []
    for tid in team_ids:
        a = outcome.advancement.get(tid, {})
        win = a.get("group_winner", 0.0)
        second = a.get("group_second", 0.0)
        third = a.get("group_third", 0.0)
        rows.append({
            "team": teams[tid].name,
            "win": win, "second": second, "third": third,
            "eliminated": max(0.0, 1.0 - win - second - third),
            "se": outcome.mc_standard_error,
        })
    rows.sort(key=lambda r: (r["win"] + r["second"]), reverse=True)
    return charts.advancement_stacked_bar(letter, rows)


def _knockout_sections(teams, fixtures, results, predictions, tipset, outcome,
                       market=None, alpha=0.0) -> list[dict]:
    # Emit knockout fixtures in chronological (kickoff) order so the report reads as a timeline.
    # The fixtures file orders them by bracket position (M73..M104), which is not by date.
    ko_matches = sorted((m for m in fixtures if m.group is None), key=lambda m: m.kickoff)
    blocks = []
    for m in ko_matches:
        if m.participants_known or m.match_id in results:
            blocks.append(_fixture_block(m, teams, results, predictions, tipset, 2,
                                         market, alpha))
        else:
            note = f"Participants not yet determined: {m.home.placeholder} vs {m.away.placeholder}."
            blocks.append({"match_id": m.match_id, "stage": m.stage.value,
                           "kickoff": m.kickoff,
                           "home": m.home.placeholder, "away": m.away.placeholder,
                           "played": False, "tip": None, "slot_note": note,
                           "occupants_chart": None})
    return blocks


def _bracket_chart(teams, outcome):
    # Reach-metric keys appear in the advancement dict in stage order (group placements first,
    # then reach_<stage> per knockout round). Derive them so any format renders correctly.
    sample = next(iter(outcome.advancement.values()))
    reach_keys = [k for k in sample if k.startswith("reach_")]
    keys = reach_keys + ["wins_title"]
    labels = ["Reach " + k[len("reach_"):].upper() for k in reach_keys] + ["Champion"]
    top = sorted(outcome.advancement.items(),
                 key=lambda kv: kv[1].get("wins_title", 0.0), reverse=True)[:10]
    rows = [{"team": teams[tid].name, "probs": [a.get(k, 0.0) for k in keys]} for tid, a in top]
    return charts.bracket_progression(rows, labels)


def _bonus_sections(bundle, teams, tipset, outcome) -> list[dict]:
    def name_of(key):
        return teams[key].name if key in teams else key

    out = []
    for q in build_bonus_questions(bundle.bonus_questions):
        dist = q.resolve(outcome)
        if not dist:
            continue
        ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        top_key, top_p = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else (None, 0.0)
        rows = [(name_of(k), p) for k, p in ranked[:8]]
        out.append({
            "question": q.label or q.question_id,
            "points": q.points,
            "answer": name_of(top_key),
            "prob": top_p,
            "runner_up": name_of(runner[0]) if runner[0] is not None else None,
            "runner_up_prob": runner[1],
            "chart": charts.bonus_candidates_bar(q.label or q.question_id, rows),
        })
    return out


def write_report(cfg: Config, context: dict):
    writer = ReportWriter(display_timezone=cfg.report.display_timezone)
    return writer.write(context, cfg.report.output_dir)
