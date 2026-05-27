"""Pipeline orchestration: data -> predictor -> (simulator) -> strategy -> report."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .config import Config, TournamentBundle
from .data.file_provider import FileDataProvider
from .model.types import Match, MatchPrediction, Result, Team, TournamentOutcome
from .predictors.base import Predictor
from .predictors.elo_poisson import EloPoissonPredictor
from .report import charts
from .report.html_writer import ReportWriter
from .strategy.base import TipStrategy
from .strategy.bonus import build_bonus_questions
from .strategy.expected_points import ExpectedPointsStrategy, expected_points

CAVEATS = (
    "The Elo-Poisson model is a reasonable forecaster but will not systematically "
    "out-predict the betting market; a market-odds predictor (Phase 3) would be the "
    "higher-accuracy option. This tool's edge over casual pool participants is correct "
    "probability-to-scoreline optimisation and bracket simulation for the champion bonus, "
    "not a superior forecast. Elo ratings are a snapshot and change after every match."
)


def build_predictor(cfg: Config) -> Predictor:
    if cfg.predictor.name == "elo_poisson":
        return EloPoissonPredictor(**cfg.predictor.params)
    if cfg.predictor.name == "attack_defence_poisson":
        from .predictors.attack_defence import AttackDefencePoissonPredictor

        return AttackDefencePoissonPredictor(**cfg.predictor.params)
    raise ValueError(f"Unknown predictor: {cfg.predictor.name}")


def build_strategy(cfg: Config, bundle: TournamentBundle) -> TipStrategy:
    if cfg.strategy.name == "expected_points":
        return ExpectedPointsStrategy(bonus_question_configs=bundle.bonus_questions)
    if cfg.strategy.name == "rank_optimizing":
        from .strategy.rank_optimizing import PredictorDerivedFieldModel, RankOptimizingStrategy

        p = cfg.strategy.params
        field_model = PredictorDerivedFieldModel(
            expert_fraction=p.get("expert_fraction", 0.6),
            temperature=p.get("temperature", 1.5),
        )
        return RankOptimizingStrategy(
            field_model=field_model,
            pool_size=p.get("pool_size", 200_000),
            top_n=p.get("top_n", 1),
            n_worlds=p.get("n_worlds", 10_000),
            candidates_per_match=p.get("candidates_per_match", 6),
            seed=cfg.simulation.seed,
            bonus_question_configs=bundle.bonus_questions,
        )
    raise ValueError(f"Unknown strategy: {cfg.strategy.name}")


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
    )
    teams = {t.team_id: t for t in provider.get_teams()}
    fixtures = provider.get_fixtures()
    results = {r.match_id: r for r in provider.get_results()}
    played = set(results)

    predictor = build_predictor(cfg)
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

    return {
        "teams": teams, "fixtures": fixtures, "results": results,
        "predictions": predictions, "tipset": tipset, "outcome": outcome,
        "predictor": predictor,
    }


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
    )
    teams = {t.team_id: t for t in provider.get_teams()}
    fixtures = provider.get_fixtures()
    results = {r.match_id: r for r in provider.get_results()}
    predictor = build_predictor(cfg)
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


def _strategy_comparison(cfg, predictions, fixtures):
    """EV-optimal vs rank-optimised slate for the report: a per-match ``compare_map`` (marking
    where the two strategies deviate) and an aggregate summary. ``(None, {})`` when nothing is
    tippable. Runs regardless of the active strategy so the report always shows both."""
    from .strategy.rank_optimizing import comparison_from_params

    params = cfg.strategy.params if cfg.strategy.name == "rank_optimizing" else {}
    comp = comparison_from_params(predictions, fixtures, params, seed=cfg.simulation.seed)
    if comp is None:
        return None, {}
    compare_map = {}
    for mid, ev in comp.ev_slate.items():
        rk = comp.rank_slate[mid]
        compare_map[mid] = {
            "ev_home": ev[0], "ev_away": ev[1],
            "rank_home": rk[0], "rank_away": rk[1],
            "differs": ev != rk,
        }
    summary = {
        "ev_p_win": comp.ev_p_win, "rank_p_win": comp.rank_p_win,
        "ev_total_ev": comp.ev_total_ev, "rank_total_ev": comp.rank_total_ev,
        "n_diff": comp.n_diff, "n_tippable": len(comp.ev_slate),
        "pool_size": comp.pool_size, "top_n": comp.top_n,
    }
    return summary, compare_map


def _resolve_as_of(as_of: str | None, bundle: TournamentBundle) -> date:
    """Snapshot date for the forward pass. Explicit ``--as-of`` wins; otherwise a completed
    tournament uses its earliest fixture date minus one day (so the pass cannot leak the
    tournament's own results), and an unplayed tournament uses today."""
    if as_of:
        return date.fromisoformat(as_of)
    if bundle.completed:
        provider = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file)
        kickoffs = [m.kickoff for m in provider.get_fixtures()]
        if kickoffs:
            return min(kickoffs).date() - timedelta(days=1)
    return date.today()


def _emit_teams_csv(
    bundle: TournamentBundle,
    ratings: dict[str, float],
    out_path: str | Path,
    pairs: dict[str, tuple[float, float]] | None = None,
) -> int:
    """Write a teams.csv for the active tournament, overwriting the ``elo`` column from the
    computed ratings (team_id/name/elo_trend preserved). When ``pairs`` is given (attack/defence
    model) also write ``attack``/``defence`` columns. Reusing the tournament's own rows keeps the
    name->id mapping authoritative and collision-free. Returns the count of overwritten rows."""
    import csv

    from .elo.names import normalize

    written = 0
    with Path(bundle.teams_file).open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader if (row.get("team_id") or "").strip()]
    if pairs is not None:
        for col in ("attack", "defence"):
            if col not in fieldnames:
                fieldnames.append(col)
    for row in rows:
        key = normalize(row["name"])
        computed = ratings.get(key)
        if computed is not None:
            row["elo"] = f"{computed:.2f}"
            written += 1
        if pairs is not None and key in pairs:
            atk, dfc = pairs[key]
            row["attack"] = f"{atk:.4f}"
            row["defence"] = f"{dfc:.4f}"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return written


def build_elo(
    cfg: Config,
    bundle: TournamentBundle,
    *,
    as_of: str | None = None,
    write_teams: str | None = None,
    top: int = 30,
    cache_only: bool = False,
) -> dict:
    """Fetch + normalize ~25y of results, run the World Football Elo forward pass, write
    ``output/elo.{md,json}``, and optionally emit a computed teams.csv for the active tournament."""
    from .elo import build_model, get_results_csv, parse_csv_text, prepare_matches, run_forward_pass
    from .elo.config import load_elo_config
    from .elo.names import build_canonical_map
    from .report.elo_report import EloReportWriter, build_elo_report

    elo_cfg = bundle.elo or load_elo_config({})
    as_of_date = _resolve_as_of(as_of, bundle)

    text = get_results_csv(elo_cfg, cache_only=cache_only)
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    all_matches = parse_csv_text(text)
    matches = prepare_matches(all_matches, as_of_date, elo_cfg)
    model = run_forward_pass(matches, build_model(elo_cfg))
    ratings = model.ratings()
    pairs = model.attack_defence_ratings() if hasattr(model, "attack_defence_ratings") else None

    provider = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file)
    current_teams = provider.get_teams()
    canonical_map, conflicts = build_canonical_map(bundle.teams_file.parents[2])

    meta = {
        "as_of": as_of_date.isoformat(),
        "model": elo_cfg.model,
        "lookback_years": elo_cfg.lookback_years,
        "recency_decay": elo_cfg.recency_decay,
        "half_life_years": elo_cfg.half_life_years,
        "source_url": elo_cfg.source_url,
        "content_hash": content_hash,
        "n_matches_total": len(all_matches),
        "n_matches_used": len(matches),
        "n_teams_rated": len(ratings),
    }
    markdown, data = build_elo_report(
        bundle, ratings, current_teams, canonical_map, conflicts, meta, top=top
    )
    paths = EloReportWriter().write(markdown, data, cfg.report.output_dir)

    result = {"paths": paths, "data": data, "ratings": ratings}
    if write_teams:
        result["teams_written"] = _emit_teams_csv(bundle, ratings, write_teams, pairs=pairs)
        result["teams_path"] = write_teams
    return result


def _build_report_context(
    cfg, bundle, teams, fixtures, results, predictions, tipset, outcome, predictor
) -> dict:
    strategy_comparison, compare_map = _strategy_comparison(cfg, predictions, fixtures)
    groups = _group_sections(teams, fixtures, results, predictions, tipset, outcome, compare_map)
    knockout_fixtures = _knockout_sections(
        teams, fixtures, results, predictions, tipset, outcome, compare_map
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
        "tournament": bundle.display_name,
        "config": cfg.config_path.name if cfg.config_path else None,
        "predictor_name": predictor.name,
        "predictor_params": getattr(predictor, "params", {}),
        "strategy_name": cfg.strategy.name,
        "strategy_params": getattr(cfg.strategy, "params", {}),
        "elo_source": bundle.elo_source or None,
        "mc_iterations": outcome.mc_iterations if outcome else None,
        "mc_seed": outcome.mc_seed if outcome else None,
        "results_count": len(results),
    }
    return {
        "header": header,
        "strategy_comparison": strategy_comparison,
        "groups": groups,
        "knockout_fixtures": knockout_fixtures,
        "title_odds_chart": title_odds_chart,
        "bracket_html": bracket_html,
        "bonus": bonus,
        "caveats": CAVEATS,
    }


def _fixture_block(m, teams, results, predictions, tipset, weight, compare_map=None) -> dict:
    name_h = teams[m.home.team_id].name if m.home.is_concrete else m.home.placeholder
    name_a = teams[m.away.team_id].name if m.away.is_concrete else m.away.placeholder
    block = {"match_id": m.match_id, "home": name_h, "away": name_a, "kickoff": m.kickoff,
             "stage": m.stage.value, "played": m.match_id in results, "result": None,
             "tip": None, "naive": None, "compare": None, "ldw_chart": None, "heatmap": None}
    if block["played"]:
        r = results[m.match_id]
        block["result"] = {"home_goals": r.home_goals, "away_goals": r.away_goals}
        return block
    pred = predictions.get(m.match_id)
    if pred is None:
        return block
    if compare_map is not None:
        block["compare"] = compare_map.get(m.match_id)
    dist = pred.scoreline
    tip = tipset.tips.get(m.match_id)
    rec_h = rec_a = None
    if tip is not None:
        rec_h, rec_a = tip.tip_home, tip.tip_away
        block["tip"] = {"home": tip.tip_home, "away": tip.tip_away, "ev": tip.expected_points}
        nh, na, _ = dist.most_likely_scorelines(1)[0]
        block["naive"] = {"home": nh, "away": na,
                          "ev": expected_points(dist, nh, na, weight)}
    block["ldw_chart"] = charts.ldw_bar(dist, name_h, name_a)
    block["heatmap"] = charts.scoreline_heatmap(dist, rec_h, rec_a)
    return block


def _group_sections(teams, fixtures, results, predictions, tipset, outcome, compare_map=None) -> list[dict]:
    by_group: dict[str, list[Match]] = {}
    for m in fixtures:
        if m.group:
            by_group.setdefault(m.group, []).append(m)
    sections = []
    for letter in sorted(by_group):
        ms = sorted(by_group[letter], key=lambda m: m.kickoff)
        blocks = [_fixture_block(m, teams, results, predictions, tipset, 1, compare_map) for m in ms]
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


def _knockout_sections(teams, fixtures, results, predictions, tipset, outcome, compare_map=None) -> list[dict]:
    blocks = []
    for m in fixtures:
        if m.group is not None:  # group matches handled elsewhere
            continue
        if m.participants_known or m.match_id in results:
            blocks.append(_fixture_block(m, teams, results, predictions, tipset, 2, compare_map))
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
