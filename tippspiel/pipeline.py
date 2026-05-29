"""Pipeline orchestration: data -> predictor -> (simulator) -> strategy -> report."""

from __future__ import annotations

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
from .strategy.expected_points import ExpectedPointsStrategy, best_tip, expected_points

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
    }
    groups = _group_sections(
        teams, fixtures, results, predictions, tipset, outcome, market
    )
    knockout_fixtures = _knockout_sections(
        teams, fixtures, results, predictions, tipset, outcome, market
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
    }
    return {
        "header": header,
        "groups": groups,
        "knockout_fixtures": knockout_fixtures,
        "title_odds_chart": title_odds_chart,
        "bracket_html": bracket_html,
        "bonus": bonus,
        "caveats": CAVEATS,
    }


def _fixture_block(
    m, teams, results, predictions, tipset, weight, market=None
) -> dict:
    name_h = teams[m.home.team_id].name if m.home.is_concrete else m.home.placeholder
    name_a = teams[m.away.team_id].name if m.away.is_concrete else m.away.placeholder
    block = {"match_id": m.match_id, "home": name_h, "away": name_a, "kickoff": m.kickoff,
             "stage": m.stage.value, "played": m.match_id in results, "result": None,
             "tip": None, "naive": None, "market_tip": None,
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
    block["ldw_chart"] = charts.ldw_bar(dist, name_h, name_a)
    block["heatmap"] = charts.scoreline_heatmap(dist, rec_h, rec_a)
    return block


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


def _group_sections(teams, fixtures, results, predictions, tipset, outcome,
                    market=None) -> list[dict]:
    by_group: dict[str, list[Match]] = {}
    for m in fixtures:
        if m.group:
            by_group.setdefault(m.group, []).append(m)
    sections = []
    for letter in sorted(by_group):
        ms = sorted(by_group[letter], key=lambda m: m.kickoff)
        blocks = [_fixture_block(m, teams, results, predictions, tipset, 1,
                                 market) for m in ms]
        adv_chart = None
        if outcome is not None:
            adv_chart = _advancement_chart(letter, ms, teams, outcome)
        sections.append({"letter": letter, "fixtures": blocks, "advancement_chart": adv_chart})
    return sections


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
                       market=None) -> list[dict]:
    blocks = []
    for m in fixtures:
        if m.group is not None:  # group matches handled elsewhere
            continue
        if m.participants_known or m.match_id in results:
            blocks.append(_fixture_block(m, teams, results, predictions, tipset, 2,
                                         market))
        else:
            note = f"Participants not yet determined: {m.home.placeholder} vs {m.away.placeholder}."
            blocks.append({"match_id": m.match_id, "stage": m.stage.value,
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
