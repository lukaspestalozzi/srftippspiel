"""Pipeline orchestration: data -> predictor -> (simulator) -> strategy -> report.

The combined report (``run``/``predict``) runs every configured predictor and presents both
meta-strategies (EV-optimal and pool-rank-optimising) for each — four tips per match — plus
each model's own Monte-Carlo outcomes. Single-model paths (``diagnose``/``verify``/``tune``)
still go through ``_run_core`` / ``build_predictor``."""

from __future__ import annotations

import hashlib
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .config import Config, TournamentBundle, select_predictor
from .data.file_provider import FileDataProvider
from .model.types import Match, MatchPrediction, Result, Team, TournamentOutcome
from .predictors.base import Predictor
from .predictors.elo_poisson import EloPoissonPredictor
from .report import charts
from .report.html_writer import ReportWriter
from .strategy.base import TipStrategy
from .strategy.bonus import build_bonus_questions
from .strategy.expected_points import (
    ExpectedPointsStrategy,
    ev_components,
    expected_points,
)
from .strategy.rank_optimizing import comparison_from_params

# Human-readable display names for the configured predictors. Anything missing falls back
# to the raw name (so future predictors render acceptably without code changes).
_MODEL_LABELS = {
    "elo_poisson": "FIFA Elo",
    "attack_defence_poisson": "Attack / Defence",
}


def _model_label(name: str) -> str:
    return _MODEL_LABELS.get(name, name)

CAVEATS = (
    "The Elo-Poisson model is a reasonable forecaster but will not systematically "
    "out-predict the betting market; a market-odds predictor (Phase 3) would be the "
    "higher-accuracy option. This tool's edge over casual pool participants is correct "
    "probability-to-scoreline optimisation and bracket simulation for the champion bonus, "
    "not a superior forecast. Elo ratings are a snapshot and change after every match."
)


def build_predictor(cfg: Config) -> Predictor:
    if cfg.predictor is None:
        raise ValueError("no predictor selected; pass --predictor (see config 'predictors:')")
    if cfg.predictor.name == "elo_poisson":
        return EloPoissonPredictor(**cfg.predictor.params)
    if cfg.predictor.name == "attack_defence_poisson":
        from .predictors.attack_defence import AttackDefencePoissonPredictor

        return AttackDefencePoissonPredictor(**cfg.predictor.params)
    raise ValueError(f"Unknown predictor: {cfg.predictor.name}")


def ratings_file(predictor: Predictor, bundle: TournamentBundle) -> Path:
    """The teams/ratings CSV the active predictor should read, by its ``ratings_kind``
    (e.g. ``attack_defence`` -> ``teams_attack_defence.csv``). Falls back to the official
    ``teams_file`` for any kind the tournament doesn't list; errors if the resolved file is
    missing so a misconfigured model fails loudly rather than silently using the wrong ratings."""
    kind = getattr(predictor, "ratings_kind", "elo")
    path = bundle.teams_files.get(kind, bundle.teams_file)
    if not Path(path).exists():
        raise ValueError(
            f"ratings file for predictor '{predictor.name}' (kind '{kind}') not found: {path}. "
            f"For attack/defence, generate it with `tippspiel build-elo --write-teams {path}`."
        )
    return Path(path)


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
    predictor = build_predictor(cfg)
    provider = FileDataProvider(
        ratings_file(predictor, bundle),
        bundle.fixtures_file,
        bundle.results_file,
        bundle.thirds_allocation_file,
    )
    teams = {t.team_id: t for t in provider.get_teams()}
    fixtures = provider.get_fixtures()
    results = {r.match_id: r for r in provider.get_results()}
    played = set(results)

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


def _model_run(cfg: Config, bundle: TournamentBundle, *, simulate: bool) -> dict:
    """Predict + optionally simulate for a single (already-selected) predictor. No strategy
    pass — the combined report computes both EV and rank slates externally via
    ``comparison_from_params``, so we skip ``strategy.generate_tips`` here.
    """
    predictor = build_predictor(cfg)
    provider = FileDataProvider(
        ratings_file(predictor, bundle),
        bundle.fixtures_file,
        bundle.results_file,
        bundle.thirds_allocation_file,
    )
    teams = {t.team_id: t for t in provider.get_teams()}
    fixtures = provider.get_fixtures()
    results = {r.match_id: r for r in provider.get_results()}
    played = set(results)

    outcome: TournamentOutcome | None = None
    if simulate:
        from .simulation.simulator import TournamentSimulator

        sim = TournamentSimulator(
            fixtures=fixtures, teams=teams, results=results, predictor=predictor,
            thirds_allocation=provider.get_thirds_allocation(),
            iterations=cfg.simulation.iterations, seed=cfg.simulation.seed,
            penalty_model=cfg.simulation.penalty_model,
        )
        outcome = sim.run()

    predictions = _predict_tippable(fixtures, teams, played, predictor)
    return {
        "teams": teams, "fixtures": fixtures, "results": results,
        "predictions": predictions, "outcome": outcome, "predictor": predictor,
    }


def run_combined_pipeline(
    cfg: Config, bundle: TournamentBundle, *, simulate: bool,
) -> dict:
    """Run every configured predictor (skipping any whose ratings file is missing), compute
    both meta-strategy slates per model, and assemble the combined multi-model report context.

    Returns ``{"context": ..., "runs": [{"name", "label", "params", "ratings_file", "core",
    "comparison"}, ...]}``.
    """
    runs: list[dict] = []
    for name in cfg.predictors:
        try:
            mcfg = select_predictor(cfg, name)
            predictor_probe = build_predictor(mcfg)
            ratings_path = ratings_file(predictor_probe, bundle)
        except ValueError as exc:
            print(f"skipping predictor {name!r}: {exc}", file=sys.stderr)
            continue
        core = _model_run(mcfg, bundle, simulate=simulate)
        comp = comparison_from_params(
            core["predictions"], core["fixtures"],
            cfg.strategy.params, seed=cfg.simulation.seed,
        )
        runs.append({
            "name": name,
            "label": _model_label(name),
            "params": mcfg.predictor.params,
            "ratings_file": ratings_path.name,
            "core": core,
            "comparison": comp,
        })
    if not runs:
        raise ValueError(
            "no predictors with available ratings files in config; "
            f"defined predictors: {', '.join(sorted(cfg.predictors)) or '(none)'}"
        )
    context = _build_combined_context(cfg, bundle, runs, simulate=simulate)
    return {"context": context, "runs": runs}


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

    predictor = build_predictor(cfg)
    provider = FileDataProvider(
        ratings_file(predictor, bundle), bundle.fixtures_file, bundle.results_file,
    )
    teams = {t.team_id: t for t in provider.get_teams()}
    fixtures = provider.get_fixtures()
    results = {r.match_id: r for r in provider.get_results()}
    markdown, data = build_verification(bundle, teams, fixtures, results, predictor)
    paths = VerificationWriter().write(markdown, data, cfg.report.output_dir)
    return {"paths": paths, "data": data}


def run_tuning(base_cfg: Config, benchmark_configs, *, top: int = 15, grid=None) -> dict:
    """Sweep elo_poisson predictor params against completed-tournament backtests;
    write output/tune.{md,json}."""
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


def run_ad_tuning(base_cfg: Config, benchmark_configs, *, top: int = 15) -> dict:
    """Staged sweep of the attack/defence model (generation params × predictor params)
    against completed-tournament backtests; write output/tune.{md,json}.

    Each gen-params point triggers a fresh forward pass over ~25y of international results,
    so this is heavier than ``run_tuning`` (~minutes vs ~seconds). Stage 2 reuses the
    synthesised per-tournament teams cached by Stage 1.
    """
    from .config import load_tournament
    from .report.ad_tuning import build_ad_tuning
    from .report.tuning import TuningWriter

    benchmarks = []
    for cfg_path in benchmark_configs:
        bundle = load_tournament(cfg_path)
        provider = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file)
        fixtures = provider.get_fixtures()
        results = {r.match_id: r for r in provider.get_results()}
        as_of = _resolve_as_of(None, bundle)
        benchmarks.append((bundle, fixtures, results, as_of))
    markdown, data = build_ad_tuning(base_cfg, benchmarks, top=top)
    paths = TuningWriter().write(markdown, data, base_cfg.report.output_dir)
    return {"paths": paths, "data": data}


def _strategy_summary(runs: list[dict]) -> dict | None:
    """Per-model EV-vs-rank summary for the report header section. Returns ``None`` if no run
    has a comparison (i.e. nothing tippable)."""
    rows = []
    pool_size = top_n = None
    for run in runs:
        comp = run["comparison"]
        if comp is None:
            continue
        rows.append({
            "model_name": run["name"],
            "model_label": run["label"],
            "ev_p_win": comp.ev_p_win,
            "rank_p_win": comp.rank_p_win,
            "ev_total_ev": comp.ev_total_ev,
            "rank_total_ev": comp.rank_total_ev,
            "n_diff": comp.n_diff,
            "n_tippable": len(comp.ev_slate),
        })
        pool_size = comp.pool_size
        top_n = comp.top_n
    if not rows:
        return None
    return {"rows": rows, "pool_size": pool_size, "top_n": top_n}


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
    """Write a ratings CSV for the active tournament from the tournament's own teams.csv rows
    (so the name->id mapping stays authoritative and collision-free). Returns the count of rows
    matched to a computed rating.

    Single-rating models (``pairs is None``) overwrite the ``elo`` column. The attack/defence
    model (``pairs`` given) instead PRESERVES the official ``elo`` column untouched and only adds
    ``attack``/``defence``, so the emitted ``teams_attack_defence.csv`` is a strict superset of the
    official ``teams.csv`` (official elo + computed attack/defence)."""
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
        if pairs is None:
            computed = ratings.get(key)
            if computed is not None:
                row["elo"] = f"{computed:.2f}"
                written += 1
        elif key in pairs:
            atk, dfc = pairs[key]
            row["attack"] = f"{atk:.4f}"
            row["defence"] = f"{dfc:.4f}"
            written += 1
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


def _build_combined_context(
    cfg: Config, bundle: TournamentBundle, runs: list[dict], *, simulate: bool,
) -> dict:
    """Assemble the report context for the combined multi-model view.

    Every fixture carries a per-model EV+rank tip (2 models × 2 strategies = 4 tips per match).
    Each simulated run also gets its own outcomes section (title odds, group advancement,
    bracket, bonus picks)."""
    # fixtures/results are identical across runs (only ratings differ), so take the first.
    primary = runs[0]["core"]
    teams = primary["teams"]
    fixtures = primary["fixtures"]
    results = primary["results"]

    groups = _group_sections(fixtures, results, runs)
    knockout_fixtures = _knockout_sections(fixtures, results, runs)
    strategy_summary = _strategy_summary(runs)

    model_outcomes = []
    if simulate:
        for run in runs:
            outcome = run["core"]["outcome"]
            if outcome is None:
                continue
            mteams = run["core"]["teams"]
            title_rows = sorted(
                ((mteams[t].name, m.get("wins_title", 0.0))
                 for t, m in outcome.advancement.items()),
                key=lambda r: r[1], reverse=True,
            )[:20]
            adv_charts = []
            by_group: dict[str, list[Match]] = {}
            for m in fixtures:
                if m.group:
                    by_group.setdefault(m.group, []).append(m)
            for letter in sorted(by_group):
                adv_charts.append({
                    "letter": letter,
                    "chart": _advancement_chart(letter, by_group[letter], mteams, outcome),
                })
            model_outcomes.append({
                "name": run["name"],
                "label": run["label"],
                "ratings_file": run["ratings_file"],
                "mc_iterations": outcome.mc_iterations,
                "mc_seed": outcome.mc_seed,
                "mc_standard_error": outcome.mc_standard_error,
                "title_odds_chart": charts.title_odds_bar(title_rows),
                "advancement_charts": adv_charts,
                "bracket_html": _bracket_chart(mteams, outcome),
                "bonus": _bonus_sections(bundle, mteams, outcome),
            })

    header = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "tournament": bundle.display_name,
        "config": cfg.config_path.name if cfg.config_path else None,
        "models": [
            {"name": r["name"], "label": r["label"],
             "params": r["params"], "ratings_file": r["ratings_file"]}
            for r in runs
        ],
        "strategies": ["expected_points", "rank_optimizing"],
        "elo_source": bundle.elo_source or None,
        "results_count": len(results),
        "simulate": simulate,
    }
    return {
        "header": header,
        "strategy_summary": strategy_summary,
        "groups": groups,
        "knockout_fixtures": knockout_fixtures,
        "model_outcomes": model_outcomes,
        "caveats": CAVEATS,
    }


def _ad_goal_rates(team_id: str, runs: list[dict]) -> tuple[float, float] | None:
    """Per-match goal rate pair (scored, conceded) for ``team_id`` vs an average opponent
    under the attack/defence model. Returns ``None`` if no run carries A/D ratings for it.

    ``exp(c + atk)`` is the team's expected goals scored vs an average opponent,
    ``exp(c - def)`` its expected conceded — the same ``c + atk - def`` decomposition used
    by ``AttackDefencePoissonPredictor`` to set match goal rates (with the opponent's
    contribution set to 0). Stays in concrete goal units, which read intuitively next to Elo.
    """
    import math

    for run in runs:
        team = run["core"]["teams"].get(team_id)
        if team is None or team.attack is None or team.defence is None:
            continue
        c = float(getattr(run["core"]["predictor"], "base_log_rate", 0.0))
        return (math.exp(c + team.attack), math.exp(c - team.defence))
    return None


def _fixture_block(m: Match, results: dict, runs: list[dict], weight: int) -> dict:
    """One match's block for the combined report: per-model EV/Rank tips + charts.

    The block has a single ``tip_rows`` list (one entry per model) so the template can render
    a 2×2 tip matrix (rows = models, cols = EV / Rank) and, on expand, per-model charts."""
    # Names: use the first run's teams (names identical across models).
    teams0 = runs[0]["core"]["teams"]
    name_h = teams0[m.home.team_id].name if m.home.is_concrete else m.home.placeholder
    name_a = teams0[m.away.team_id].name if m.away.is_concrete else m.away.placeholder
    elo_h = teams0[m.home.team_id].elo if m.home.is_concrete else None
    elo_a = teams0[m.away.team_id].elo if m.away.is_concrete else None
    ad_h = _ad_goal_rates(m.home.team_id, runs) if m.home.is_concrete else None
    ad_a = _ad_goal_rates(m.away.team_id, runs) if m.away.is_concrete else None
    block: dict = {
        "match_id": m.match_id, "home": name_h, "away": name_a,
        "elo_home": elo_h, "elo_away": elo_a,
        "ad_home": ad_h, "ad_away": ad_a,
        "kickoff": m.kickoff, "stage": m.stage.value,
        "played": m.match_id in results, "result": None,
        "tip_rows": [], "tippable": False,
        "agree_ev": True, "agree_rank": True, "any_contrarian": False,
    }
    if block["played"]:
        r = results[m.match_id]
        block["result"] = {"home_goals": r.home_goals, "away_goals": r.away_goals}
        return block
    if not m.participants_known:
        return block

    ev_tips: list[tuple[int, int]] = []
    rank_tips: list[tuple[int, int]] = []
    for run in runs:
        pred = run["core"]["predictions"].get(m.match_id)
        comp = run["comparison"]
        if pred is None or comp is None:
            continue
        dist = pred.scoreline
        ev_h, ev_a = comp.ev_slate.get(m.match_id, (0, 0))
        rk_h, rk_a = comp.rank_slate.get(m.match_id, (ev_h, ev_a))
        ev_points = ev_components(dist, ev_h, ev_a, weight)["total"]
        rk_points = ev_components(dist, rk_h, rk_a, weight)["total"]
        contrarian = (ev_h, ev_a) != (rk_h, rk_a)
        block["tip_rows"].append({
            "model_name": run["name"],
            "model_label": run["label"],
            "ev": {"home": ev_h, "away": ev_a, "ev": ev_points},
            "rank": {"home": rk_h, "away": rk_a, "ev": rk_points},
            "contrarian": contrarian,
            "ldw_chart": charts.ldw_bar(dist, name_h, name_a),
            "heatmap": charts.scoreline_heatmap(
                dist, ev_h, ev_a,
                alt_home=rk_h if contrarian else None,
                alt_away=rk_a if contrarian else None,
            ),
        })
        ev_tips.append((ev_h, ev_a))
        rank_tips.append((rk_h, rk_a))
        if contrarian:
            block["any_contrarian"] = True

    block["tippable"] = bool(block["tip_rows"])
    block["agree_ev"] = len(set(ev_tips)) <= 1
    block["agree_rank"] = len(set(rank_tips)) <= 1
    return block


def _group_sections(fixtures: list[Match], results: dict, runs: list[dict]) -> list[dict]:
    by_group: dict[str, list[Match]] = {}
    for m in fixtures:
        if m.group:
            by_group.setdefault(m.group, []).append(m)
    sections = []
    for letter in sorted(by_group):
        ms = sorted(by_group[letter], key=lambda m: m.kickoff)
        blocks = [_fixture_block(m, results, runs, weight=1) for m in ms]
        sections.append({"letter": letter, "fixtures": blocks})
    return sections


def _knockout_sections(fixtures: list[Match], results: dict, runs: list[dict]) -> list[dict]:
    blocks = []
    for m in fixtures:
        if m.group is not None:
            continue
        if m.participants_known or m.match_id in results:
            blocks.append(_fixture_block(m, results, runs, weight=2))
        else:
            note = (f"Participants not yet determined: "
                    f"{m.home.placeholder} vs {m.away.placeholder}.")
            blocks.append({
                "match_id": m.match_id, "stage": m.stage.value,
                "home": m.home.placeholder, "away": m.away.placeholder,
                "played": False, "tippable": False, "slot_note": note,
                "tip_rows": [],
            })
    return blocks


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


def _bonus_sections(bundle, teams, outcome) -> list[dict]:
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
