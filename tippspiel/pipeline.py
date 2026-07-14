"""Pipeline orchestration: data -> predictor -> (simulator) -> strategy -> report."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

from .config import Config, TournamentBundle, load_elo_block, load_offdef_block
from .data.base import Odds1X2
from .data.file_provider import FileDataProvider, read_odds_file
from .model.types import Match, MatchPrediction, Team, TournamentOutcome
from .predictors.base import Predictor
from .predictors.elo_poisson import EloPoissonPredictor
from .predictors.market_odds import MarketOddsPredictor
from .report import charts
from .report.html_writer import ReportWriter
from .simulation.known_participants import compute_group_standings, resolve_known_participants
from .strategy.bonus import build_bonus_questions
from .strategy.expected_points import (
    ExpectedPointsStrategy,
    best_tip,
    ev_components,
    expected_points,
    score_tip,
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
    return ExpectedPointsStrategy(
        bonus_question_configs=bundle.bonus_questions,
        realism_tolerance=cfg.strategy.realism_tolerance,
    )


def _predict_tippable(
    fixtures: list[Match],
    teams: dict[str, Team],
    predictor: Predictor,
) -> dict[str, MatchPrediction]:
    """Predict every fixture with concrete participants — *including already-played ones*.

    A played match's prediction is never consumed by the simulator (which conditions on the
    actual ``results`` scoreline) or by tip scoring; it exists only so the report can keep
    showing the a-priori tip/distribution alongside the real result.
    """
    preds: dict[str, MatchPrediction] = {}
    for m in fixtures:
        if not m.participants_known:
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
    odds = provider.get_odds()
    thirds_allocation = provider.get_thirds_allocation()

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
            thirds_allocation=thirds_allocation,
            iterations=cfg.simulation.iterations,
            seed=cfg.simulation.seed,
            penalty_model=cfg.simulation.penalty_model,
        )
        outcome = sim.run()

    # Current group tables from the played results — the shared calculation step: the report
    # renders them and the knockout resolver reads the certain placings off them to fill slots
    # whose participant is already decided. The simulator above stays on the raw reference
    # fixtures; resolution is for the predict/tip/report path only.
    standings = compute_group_standings(fixtures, results)
    fixtures = resolve_known_participants(fixtures, results, thirds_allocation, standings=standings)

    predictions = _predict_tippable(fixtures, teams, predictor)
    tipset = strategy.generate_tips(predictions, outcome, fixtures)
    market_predictions = _market_predictions(cfg, fixtures, teams, odds)

    return {
        "teams": teams, "fixtures": fixtures, "results": results,
        "predictions": predictions, "tipset": tipset, "outcome": outcome,
        "predictor": predictor, "market_predictions": market_predictions,
        "odds": odds, "standings": standings,
    }


def _market_predictions(
    cfg: Config,
    fixtures: list[Match],
    teams: dict[str, Team],
    odds: dict[str, Odds1X2],
) -> dict[str, MatchPrediction]:
    """Market-odds scoreline predictions for the report's per-fixture market tips.

    Runs a dedicated ``MarketOddsPredictor`` *regardless of the configured predictor*, over the
    **full** tippable slate (Elo fallback where a fixture has no odds row). Display is gated to
    genuine-odds fixtures separately. Empty when no odds exist at all. Deliberately kept at the
    pure-market default (``market_weight=1``): this line is the report's market *reference*;
    a blended copy would just shadow the recommended tip.
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
        match_draw=bool(p.get("match_draw", False)),
    )
    return _predict_tippable(fixtures, teams, predictor)


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
        core["market_predictions"], core["odds"], core["standings"],
    )
    return {"context": context, "tipset": core["tipset"], "outcome": core["outcome"]}


def write_diagnostics(cfg: Config, bundle: TournamentBundle, *, simulate: bool) -> dict:
    """Run the core pipeline and write the Claude diagnostic report (markdown + JSON)."""
    from .report.diagnostics import DiagnosticsWriter, build_diagnostics

    core = _run_core(cfg, bundle, simulate=simulate)
    markdown, data = build_diagnostics(
        cfg, bundle, core["teams"], core["fixtures"], core["results"],
        core["predictions"], core["tipset"], core["outcome"], core["predictor"],
        odds=core["odds"],
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
    markdown, data = build_verification(
        bundle, teams, fixtures, results, predictor,
        realism_tolerance=cfg.strategy.realism_tolerance,
    )
    paths = VerificationWriter().write(markdown, data, cfg.report.output_dir)
    return {"paths": paths, "data": data}


def run_tuning(base_cfg: Config, benchmark_configs, *, top: int = 15, grid=None) -> dict:
    """Sweep predictor params against completed-tournament backtests; write output/tune.{md,json}."""
    from .config import load_tournament
    from .report.tuning import TuningWriter, build_tuning

    benchmarks = []
    for cfg_path in benchmark_configs:
        bundle = load_tournament(cfg_path)
        provider = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file,
                                    odds_file=bundle.odds_file)
        teams = {t.team_id: t for t in provider.get_teams()}
        fixtures = provider.get_fixtures()
        results = {r.match_id: r for r in provider.get_results()}
        # Odds enable the model x market blend axes (`tune --market`); empty when the
        # tournament has no committed snapshot (the sweep then degrades to pure Elo there).
        benchmarks.append((bundle, teams, fixtures, results, provider.get_odds()))
    markdown, data = build_tuning(base_cfg, benchmarks, grid=grid, top=top)
    paths = TuningWriter().write(markdown, data, base_cfg.report.output_dir)
    return {"paths": paths, "data": data}


def write_ratings_snapshot(
    bundle: TournamentBundle,
    offdef_block: dict | None = None,
    elo_block: dict | None = None,
    *,
    dry_run: bool = False,
) -> dict:
    """Fit scalar + offensive/defensive Elo from the historical corpus and persist to teams.csv.

    Fits a scalar World-Football-Elo rating (``elo``) and ``att_elo``/``def_elo`` for every team
    in the corpus from all matches **strictly before** the tournament's first kickoff (so a
    ``verify`` backtest stays leak-free), then writes those three columns for this tournament's
    teams into its ``teams.csv`` (preserving any other columns). One corpus load feeds both fits.
    Returns a small stats dict for the CLI to print.

    ``dry_run=True`` skips the fit and the ``teams.csv`` write entirely; instead it returns the
    corpus matches within 5 days of ``snapshot_date`` tagged ``included`` (corpus date <
    ``snapshot_date``) or not, so a maintainer can mechanically check the cutoff before committing
    to a ``snapshot_date`` value.
    """
    import csv as _csv

    from .data.historical_results_adapter import (
        DEFAULT_CORPUS,
        corpus_name_for,
        load_corpus,
        ratings_for_team,
    )
    from .training.offdef_elo import OffDefRating, fit_off_def
    from .training.scalar_elo import fit_scalar_elo

    block = dict(offdef_block or {})
    eblock = dict(elo_block or {})
    params, tiers, elo_params, k_tiers = _ratings_fit_params(block, eblock)
    corpus_path = block.get("corpus_file", DEFAULT_CORPUS)

    provider = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file)
    fixtures = provider.get_fixtures()
    snapshot = block.get("snapshot_date") or min(m.kickoff for m in fixtures).date().isoformat()

    if dry_run:
        snapshot_date = date.fromisoformat(snapshot)
        near_cutoff = []
        for m in load_corpus(before=None, corpus_path=corpus_path, tiers=tiers, k_tiers=k_tiers):
            try:
                match_date = date.fromisoformat(m.date)
            except ValueError:
                continue
            if abs((match_date - snapshot_date).days) <= 5:
                near_cutoff.append((m.date, m.home, m.away, m.home_goals, m.away_goals,
                                     match_date < snapshot_date))
        near_cutoff.sort(key=lambda row: row[0])
        return {"dry_run": True, "snapshot_date": snapshot, "near_cutoff": near_cutoff}

    matches = load_corpus(before=snapshot, corpus_path=corpus_path, tiers=tiers, k_tiers=k_tiers)
    ratings = fit_off_def(matches, params)
    # `elo.source: corpus` derives the scalar elo from the corpus too; `external` (default) leaves
    # the committed `elo` column untouched (a frozen eloratings/women's snapshot the men's corpus
    # can't reproduce — completed benchmarks and womenseuro2025). Off/def is always corpus-fitted.
    elo_source = str(eblock.get("source", "external")).strip().lower()
    elo = fit_scalar_elo(matches, elo_params) if elo_source == "corpus" else {}

    teams = provider.get_teams()
    by_id: dict[str, OffDefRating] = {}
    elo_by_id: dict[str, float] = {}
    unmapped: list[str] = []
    for t in teams:
        by_id[t.team_id] = ratings_for_team(t.name, ratings)
        corpus_name = corpus_name_for(t.name)
        if corpus_name in elo:
            elo_by_id[t.team_id] = elo[corpus_name]
        if corpus_name not in ratings:
            unmapped.append(t.name)

    _write_teams_csv_with_ratings(bundle.teams_file, by_id, elo_by_id, _csv)

    ranked_att = sorted(teams, key=lambda t: by_id[t.team_id].att, reverse=True)
    ranked_def = sorted(teams, key=lambda t: by_id[t.team_id].def_, reverse=True)
    ranked_elo = sorted(
        (t for t in teams if t.team_id in elo_by_id),
        key=lambda t: elo_by_id[t.team_id], reverse=True,
    )
    return {
        "snapshot_date": snapshot,
        "elo_source": elo_source,
        "corpus_matches": len(matches),
        "corpus_teams": len(ratings),
        "teams_written": len(teams),
        "unmapped": unmapped,
        "top_elo": [(t.name, elo_by_id[t.team_id]) for t in ranked_elo[:5]],
        "top_attack": [(t.name, by_id[t.team_id].att) for t in ranked_att[:5]],
        "top_defence": [(t.name, by_id[t.team_id].def_) for t in ranked_def[:5]],
    }


def _ratings_fit_params(offdef_block: dict, elo_block: dict):
    """Fit parameters + importance tiers from the raw ``offdef:`` / ``elo:`` config blocks.

    Shared by ``fit-ratings`` (which writes the snapshot into teams.csv) and the report's
    rating-history section (which replays the same fit for the trajectories), so both always
    derive ratings from identical hyperparameters."""
    from .data.historical_results_adapter import KTiers, WeightTiers
    from .training.offdef_elo import OffDefParams
    from .training.scalar_elo import ScalarEloParams

    params = OffDefParams(
        mu=float(offdef_block.get("mu", OffDefParams.mu)),
        k_att=float(offdef_block.get("k_att", OffDefParams.k_att)),
        k_def=float(offdef_block.get("k_def", OffDefParams.k_def)),
        gamma_home=float(offdef_block.get("gamma_home", OffDefParams.gamma_home)),
        residual_cap=float(offdef_block.get("residual_cap", OffDefParams.residual_cap)),
        epochs=int(offdef_block.get("epochs", OffDefParams.epochs)),
    )
    elo_params = ScalarEloParams(
        start_rating=float(elo_block.get("start_rating", ScalarEloParams.start_rating)),
        home_advantage=float(elo_block.get("home_advantage", ScalarEloParams.home_advantage)),
        k_scale=float(elo_block.get("k_scale", ScalarEloParams.k_scale)),
    )
    w = offdef_block.get("weights", {}) or {}
    tiers = WeightTiers(
        friendly=float(w.get("friendly", WeightTiers.friendly)),
        qualifier=float(w.get("qualifier", WeightTiers.qualifier)),
        continental=float(w.get("continental", WeightTiers.continental)),
        world_cup=float(w.get("world_cup", WeightTiers.world_cup)),
        default=float(w.get("default", WeightTiers.default)),
    )
    kw = elo_block.get("k_tiers", {}) or {}
    k_tiers = KTiers(
        friendly=float(kw.get("friendly", KTiers.friendly)),
        qualifier=float(kw.get("qualifier", KTiers.qualifier)),
        minor=float(kw.get("minor", KTiers.minor)),
        continental=float(kw.get("continental", KTiers.continental)),
        world_cup=float(kw.get("world_cup", KTiers.world_cup)),
    )
    return params, tiers, elo_params, k_tiers


def _write_teams_csv_with_ratings(teams_file, by_id, elo_by_id, _csv) -> None:
    """Rewrite teams.csv with elo/att_elo/def_elo columns, preserving any other columns."""
    with open(teams_file, newline="", encoding="utf-8") as fh:
        reader = _csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for col in ("elo", "att_elo", "def_elo"):
        if col not in fieldnames:
            fieldnames.append(col)
    for row in rows:
        tid = (row.get("team_id") or "").strip()
        r = by_id.get(tid)
        if r is not None:
            row["att_elo"] = f"{r.att:.4f}"
            row["def_elo"] = f"{r.def_:.4f}"
        if tid in elo_by_id:
            row["elo"] = f"{elo_by_id[tid]:.0f}"
    with open(teams_file, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Memo for the rating-history section: the corpus fit is deterministic in (config, snapshot),
# so repeated pipeline runs in one process (the test suite, site assembly) replay it only once.
_ELO_HISTORY_CACHE: dict[tuple, dict | None] = {}


def _elo_history_section(cfg: Config, bundle: TournamentBundle, teams, fixtures) -> dict | None:
    """Rating trajectories over time for the report's "Elo ratings over time" section.

    Replays the same corpus fits ``fit-ratings`` runs (identical params + snapshot cutoff, via
    ``_ratings_fit_params``) but records each tournament team's rating after every corpus match
    inside the display window, then renders one line chart per rating (scalar Elo, att, def).
    Corpus-fitted tournaments only (``elo.source: corpus``): an external-Elo tournament's
    committed ratings (frozen eloratings / women's snapshots) cannot be reproduced from the
    men's corpus, so the section is omitted there. Returns ``None`` when skipped."""
    if cfg.config_path is None or not cfg.config_path.exists() or not fixtures:
        return None
    eblock = load_elo_block(cfg.config_path)
    if str(eblock.get("source", "external")).strip().lower() != "corpus":
        return None
    block = load_offdef_block(cfg.config_path)
    snapshot = block.get("snapshot_date") or min(m.kickoff for m in fixtures).date().isoformat()
    years = float(cfg.report.elo_history_years)
    start = cfg.report.elo_history_start.strip() or (
        date.fromisoformat(snapshot) - timedelta(days=round(365.25 * years))
    ).isoformat()
    key = (str(cfg.config_path), snapshot, start)
    if key not in _ELO_HISTORY_CACHE:
        _ELO_HISTORY_CACHE[key] = _build_elo_history(cfg, teams, block, eblock, snapshot, start)
    return _ELO_HISTORY_CACHE[key]


def _build_elo_history(cfg, teams, offdef_block, elo_block, snapshot, start) -> dict | None:
    from .data.historical_results_adapter import DEFAULT_CORPUS, corpus_name_for, load_corpus
    from .training.offdef_elo import fit_off_def_history
    from .training.scalar_elo import fit_scalar_elo_history

    params, tiers, elo_params, k_tiers = _ratings_fit_params(offdef_block, elo_block)
    corpus_path = offdef_block.get("corpus_file", DEFAULT_CORPUS)
    matches = load_corpus(before=snapshot, corpus_path=corpus_path, tiers=tiers, k_tiers=k_tiers)

    display = {corpus_name_for(t.name): t.name for t in teams.values()}
    elo_hist = fit_scalar_elo_history(matches, elo_params, track=display, start_date=start)
    offdef_hist = fit_off_def_history(matches, params, track=display, start_date=start)

    # Highlight the strongest 8 sides by current (end-of-window) scalar Elo — the categorical
    # palette ceiling; every other team is a gray legend-toggle trace. Teams absent from the
    # corpus (no trajectory) drop out naturally.
    finals = {name: pts[-1][1] for name, pts in elo_hist.items() if pts}
    if not finals:
        return None
    ranked = sorted(finals, key=lambda n: finals[n], reverse=True)
    highlight = [display[n] for n in ranked[:8]]
    order = ranked[:8] + sorted(ranked[8:], key=lambda n: display[n])

    # Round to teams.csv precision (elo %.0f, att/def %.4f): full float repr would bloat the
    # embedded JSON payload noticeably at a 26-year window without adding visible detail.
    elo_series = [(display[n], [(d, round(v, 1)) for d, v in elo_hist[n]]) for n in order]
    att_series = [(display[n], [(d, round(a, 4)) for d, a, _ in offdef_hist[n]]) for n in order]
    def_series = [(display[n], [(d, round(f, 4)) for d, _, f in offdef_hist[n]]) for n in order]
    return {
        "window_start": start,
        "snapshot": snapshot,
        "highlight": highlight,
        "elo_chart": charts.rating_history_lines(
            elo_series, title="Scalar Elo (World-Football-Elo, corpus fit)",
            ytitle="Elo rating", highlight=highlight, yfmt=".0f",
        ),
        "att_chart": charts.rating_history_lines(
            att_series, title="Attack rating (att_elo — higher = scores more)",
            ytitle="att (log goal-rate vs field)", highlight=highlight, yfmt=".2f",
        ),
        "def_chart": charts.rating_history_lines(
            def_series, title="Defence rating (def_elo — higher = concedes fewer)",
            ytitle="def (log goal-rate vs field)", highlight=highlight, yfmt=".2f",
        ),
    }


def _build_report_context(
    cfg, bundle, teams, fixtures, results, predictions, tipset, outcome, predictor,
    market_predictions=None, odds=None, standings=None,
) -> dict:
    market_predictions = market_predictions or {}
    # The per-fixture market tip: predictions carry the scoreline (for the EV-optimal tip), and
    # ``odds_ids`` gates display to fixtures backed by genuine bookmaker odds (never a silent
    # Elo-fallback duplicate).
    # Per-source de-vigged odds (ESPN, Polymarket) read from the committed sidecars next to the
    # consumed odds.csv, so the folded per-fixture section can show both sources beside the blend.
    src_dir = bundle.odds_file.parent if bundle.odds_file else None
    market = {
        "preds": market_predictions,
        "odds_ids": set(odds or {}),
        "odds": odds or {},
        "sources": {
            "espn": read_odds_file(src_dir / "odds_espn.csv") if src_dir else {},
            "poly": read_odds_file(src_dir / "odds_polymarket.csv") if src_dir else {},
        },
    }
    # Off/def goal-volume weight of the active predictor; gates the per-fixture att/def display.
    alpha = float(getattr(predictor, "alpha", 0.0))
    # The legend is shown only when the rows will actually appear, i.e. alpha>0 AND some team
    # carries a fitted rating (else `fit-offdef` was never run and every row would be empty).
    uses_offdef = alpha > 0 and any(t.att_elo or t.def_elo for t in teams.values())
    # Realism tolerance keeps the report's market-odds tip consistent with the recommended tip.
    realism = cfg.strategy.realism_tolerance
    group_fixtures = _group_fixture_blocks(
        teams, fixtures, results, predictions, tipset, market, alpha, realism
    )
    group_standings = _group_standings_sections(teams, standings or [])
    advancement = _advancement_sections(teams, fixtures, outcome)
    knockout_fixtures = _knockout_sections(
        teams, fixtures, results, predictions, tipset, outcome, market, alpha, realism
    )
    elo_history = _elo_history_section(cfg, bundle, teams, fixtures)

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
        "uses_offdef": uses_offdef,
    }
    return {
        "header": header,
        "group_fixtures": group_fixtures,
        "group_standings": group_standings,
        "advancement": advancement,
        "knockout_fixtures": knockout_fixtures,
        "title_odds_chart": title_odds_chart,
        "bracket_html": bracket_html,
        "bonus": bonus,
        "elo_history": elo_history,
        "caveats": CAVEATS,
    }


def _side_label(side, teams) -> str:
    """Display label for a fixture side: the team name when concrete, else the slot placeholder
    (``Winner Group A`` / ``3rd place (slot 74)`` / ...). A concrete side's ``placeholder`` is
    ``None``, so a partially-resolved knockout fixture must use this rather than ``placeholder``."""
    return teams[side.team_id].name if side.is_concrete else side.placeholder


def _fixture_block(
    m, teams, results, predictions, tipset, weight, market=None, alpha=0.0, realism=0.0
) -> dict:
    name_h = _side_label(m.home, teams)
    name_a = _side_label(m.away, teams)
    block = {"match_id": m.match_id, "home": name_h, "away": name_a, "kickoff": m.kickoff,
             "stage": m.stage.value, "group": m.group, "played": m.match_id in results,
             "result": None, "tip": None, "naive": None, "market_tip": None, "data": None,
             "actual": None, "tip_outcome": None, "ldw_chart": None, "heatmap": None}
    if block["played"]:
        r = results[m.match_id]
        block["result"] = {"home_goals": r.home_goals, "away_goals": r.away_goals}
    # Build the prediction block even for played matches, so the report keeps showing the
    # a-priori tip/distribution next to the real result (added above).
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
    if block["played"]:
        r = results[m.match_id]
        block["actual"] = _actual_likelihood(dist, r.home_goals, r.away_goals)
        if tip is not None:
            block["tip_outcome"] = _tip_outcome(rec_h, rec_a, r.home_goals, r.away_goals,
                                                weight)
    _set_market_tips(block, m, weight, market, realism)
    rec = (rec_h, rec_a) if tip is not None else None
    block["data"] = _fixture_data(m, teams, dist, rec, weight, market, alpha)
    block["ldw_chart"] = charts.ldw_bar(dist, name_h, name_a)
    block["heatmap"] = charts.scoreline_heatmap(dist, rec_h, rec_a)
    return block


def _actual_likelihood(dist, ah: int, aa: int) -> dict:
    """How likely the model considered an actual result: the exact-scoreline cell probability
    plus the probability of the tendency that occurred. The qualitative label keys off the
    tendency probability — exact-score probabilities are uniformly small (~10–15% at best),
    so they can't carry a word on their own. ``cell()`` returns 0.0 for scores beyond the
    distribution grid, so freak scorelines are safe."""
    if ah > aa:
        p_tendency = dist.p_home_win()
    elif ah < aa:
        p_tendency = dist.p_away_win()
    else:
        p_tendency = dist.p_draw()
    if p_tendency >= 0.50:
        label = "expected"
    elif p_tendency >= 0.25:
        label = "plausible"
    else:
        label = "surprising"
    return {"p_exact": dist.cell(ah, aa), "p_tendency": p_tendency, "label": label}


def _tip_outcome(th: int, ta: int, ah: int, aa: int, weight: int) -> dict:
    """How the recommended tip fared against the actual (120-minute) result: the pool points it
    earned out of the 10×weight maximum, tagged exact hit / correct tendency / miss."""
    points = score_tip(th, ta, ah, aa, weight)
    if (th, ta) == (ah, aa):
        label, cls = "exact hit", "exact"
    elif (th > ta) == (ah > aa) and (th < ta) == (ah < aa):
        label, cls = "correct tendency", "tendency"
    else:
        label, cls = "miss", "miss"
    return {"points": points, "max": 10 * weight, "label": label, "cls": cls}


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
        "market_sources": {},
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
    # Per-source de-vigged 1X2 (ESPN / Polymarket) for the same fixture, where that source priced it,
    # so the folded section shows both markets beside the blend.
    src = (market or {}).get("sources", {})
    data["market_sources"] = {
        name: {"home": o.p_home, "draw": o.p_draw, "away": o.p_away}
        for name, table in (("espn", src.get("espn", {})), ("poly", src.get("poly", {})))
        if (o := table.get(m.match_id)) is not None
    }
    return data


def _set_market_tips(block, m, weight, market, realism=0.0) -> None:
    """Attach the EV-optimal market-odds tip to ``block`` — shown only for fixtures backed by
    genuine bookmaker odds, so it's a real market prediction rather than a silent Elo-fallback
    duplicate of the recommended tip. No-op when the fixture has no odds."""
    if not market or m.match_id not in market["odds_ids"]:
        return
    market_pred = market["preds"].get(m.match_id)
    if market_pred is None:
        return
    mdist = market_pred.scoreline
    th, ta, ev = best_tip(mdist, weight, realism)
    points = None
    if block["result"] is not None:
        points = score_tip(th, ta, block["result"]["home_goals"],
                           block["result"]["away_goals"], weight)
    block["market_tip"] = {"home": th, "away": ta, "ev": ev, "points": points}


def _group_fixture_blocks(teams, fixtures, results, predictions, tipset,
                          market=None, alpha=0.0, realism=0.0) -> list[dict]:
    """All group-stage fixtures as one globally chronological list (not grouped into per-group
    sections). Each block carries its ``group`` letter, surfaced as a tag in the report."""
    ms = sorted((m for m in fixtures if m.group), key=lambda m: m.kickoff)
    return [_fixture_block(m, teams, results, predictions, tipset, 1, market, alpha, realism)
            for m in ms]


def _group_standings_sections(teams, standings) -> list[dict]:
    """Per-group current standings tables from the played results (the shared calc step that also
    feeds knockout determination). Groups with no match played yet are omitted so the section is
    hidden pre-tournament. ``qualified`` flags the certain top-2 of a complete group — the placings
    that fill the knockout bracket."""
    sections = []
    for g in standings:
        if not any(r.played for r in g.rows):
            continue
        rows = [{
            "rank": r.rank,
            "team": teams[r.team_id].name if r.team_id in teams else r.team_id,
            "played": r.played,
            "wins": r.wins, "draws": r.draws, "losses": r.losses,
            "points": r.points,
            "gf": r.goals_for, "ga": r.goals_against, "gd": r.goal_diff,
            "qualified": g.complete and r.placing_certain and r.rank <= 2,
        } for r in g.rows]
        sections.append({"letter": g.letter, "complete": g.complete, "rows": rows})
    return sections


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
                       market=None, alpha=0.0, realism=0.0) -> list[dict]:
    # Emit knockout fixtures in chronological (kickoff) order so the report reads as a timeline.
    # The fixtures file orders them by bracket position (M73..M104), which is not by date.
    ko_matches = sorted((m for m in fixtures if m.group is None), key=lambda m: m.kickoff)
    blocks = []
    for m in ko_matches:
        if m.participants_known or m.match_id in results:
            blocks.append(_fixture_block(m, teams, results, predictions, tipset, 2,
                                         market, alpha, realism))
        else:
            label_h = _side_label(m.home, teams)
            label_a = _side_label(m.away, teams)
            note = f"Participants not yet fully determined: {label_h} vs {label_a}."
            blocks.append({"match_id": m.match_id, "stage": m.stage.value,
                           "kickoff": m.kickoff,
                           "home": label_h, "away": label_a,
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
