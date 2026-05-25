"""Pipeline orchestration: data -> predictor -> (simulator) -> strategy -> report."""

from __future__ import annotations

from datetime import datetime, timezone

from .config import Config
from .data.file_provider import FileDataProvider
from .model.types import Match, MatchPrediction, Result, Team, TournamentOutcome
from .predictors.base import Predictor
from .predictors.elo_poisson import EloPoissonPredictor
from .report import charts
from .report.html_writer import ReportWriter
from .strategy.base import TipStrategy
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
    raise ValueError(f"Unknown predictor: {cfg.predictor.name}")


def build_strategy(cfg: Config) -> TipStrategy:
    if cfg.strategy.name == "expected_points":
        return ExpectedPointsStrategy(bonus_question_configs=cfg.bonus_questions)
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


def run_pipeline(
    cfg: Config,
    *,
    simulate: bool,
) -> dict:
    provider = FileDataProvider(
        cfg.data.teams_file,
        cfg.data.fixtures_file,
        cfg.data.results_file,
        cfg.data.bracket_map_file,
    )
    teams = {t.team_id: t for t in provider.get_teams()}
    fixtures = provider.get_fixtures()
    results = {r.match_id: r for r in provider.get_results()}
    played = set(results)

    predictor = build_predictor(cfg)
    strategy = build_strategy(cfg)

    outcome: TournamentOutcome | None = None
    if simulate:
        from .simulation.simulator import TournamentSimulator

        sim = TournamentSimulator(
            fixtures=fixtures,
            teams=teams,
            results=results,
            predictor=predictor,
            bracket_map=provider.get_bracket_map(),
            iterations=cfg.simulation.iterations,
            seed=cfg.simulation.seed,
            penalty_model=cfg.simulation.penalty_model,
        )
        outcome = sim.run()

    predictions = _predict_tippable(fixtures, teams, played, predictor)
    tipset = strategy.generate_tips(predictions, outcome, fixtures)

    context = _build_report_context(
        cfg, teams, fixtures, results, predictions, tipset, outcome, predictor
    )
    return {"context": context, "tipset": tipset, "outcome": outcome}


def _build_report_context(
    cfg, teams, fixtures, results, predictions, tipset, outcome, predictor
) -> dict:
    groups = _group_sections(teams, fixtures, results, predictions, tipset, outcome)
    knockout_fixtures = _knockout_sections(teams, fixtures, results, predictions, tipset, outcome)

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
        bonus = _bonus_sections(cfg, teams, tipset, outcome)

    header = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "predictor_name": predictor.name,
        "predictor_params": getattr(predictor, "params", {}),
        "strategy_name": cfg.strategy.name,
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


def _fixture_block(m, teams, results, predictions, tipset, weight) -> dict:
    name_h = teams[m.home.team_id].name if m.home.is_concrete else m.home.placeholder
    name_a = teams[m.away.team_id].name if m.away.is_concrete else m.away.placeholder
    block = {"match_id": m.match_id, "home": name_h, "away": name_a, "kickoff": m.kickoff,
             "stage": m.stage.value, "played": m.match_id in results, "result": None,
             "tip": None, "naive": None, "ldw_chart": None, "heatmap": None}
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
    block["ldw_chart"] = charts.ldw_bar(dist, name_h, name_a)
    block["heatmap"] = charts.scoreline_heatmap(dist, rec_h, rec_a)
    return block


def _group_sections(teams, fixtures, results, predictions, tipset, outcome) -> list[dict]:
    by_group: dict[str, list[Match]] = {}
    for m in fixtures:
        if m.group:
            by_group.setdefault(m.group, []).append(m)
    sections = []
    for letter in sorted(by_group):
        ms = sorted(by_group[letter], key=lambda m: m.kickoff)
        blocks = [_fixture_block(m, teams, results, predictions, tipset, 1) for m in ms]
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


def _knockout_sections(teams, fixtures, results, predictions, tipset, outcome) -> list[dict]:
    blocks = []
    for m in fixtures:
        if m.group is not None:  # group matches handled elsewhere
            continue
        if m.participants_known or m.match_id in results:
            blocks.append(_fixture_block(m, teams, results, predictions, tipset, 2))
        else:
            note = f"Participants not yet determined: {m.home.placeholder} vs {m.away.placeholder}."
            blocks.append({"match_id": m.match_id, "stage": m.stage.value,
                           "home": m.home.placeholder, "away": m.away.placeholder,
                           "played": False, "tip": None, "slot_note": note,
                           "occupants_chart": None})
    return blocks


def _bracket_chart(teams, outcome):
    top = sorted(outcome.advancement.items(), key=lambda kv: kv[1]["wins_title"], reverse=True)[:10]
    rows = []
    for tid, a in top:
        rows.append({
            "team": teams[tid].name,
            "probs": [a["qualifies_r32"], a["reach_r16"], a["reach_qf"],
                      a["reach_sf"], a["reach_final"], a["wins_title"]],
        })
    return charts.bracket_progression(rows)


def _bonus_sections(cfg, teams, tipset, outcome) -> list[dict]:
    out = []
    for q_cfg in cfg.bonus_questions:
        dist = outcome.bonus_probabilities.get(q_cfg.id)
        if not dist:
            continue
        ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        top_id, top_p = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else (None, 0.0)
        rows = [(teams[t].name if t in teams else t, p) for t, p in ranked[:8]]
        out.append({
            "question": f"World Champion" if q_cfg.id == "champion" else q_cfg.id,
            "points": q_cfg.points,
            "answer": teams[top_id].name if top_id in teams else top_id,
            "prob": top_p,
            "runner_up": (teams[runner[0]].name if runner[0] in teams else runner[0]) if runner[0] else None,
            "runner_up_prob": runner[1],
            "chart": charts.bonus_candidates_bar(
                "Champion candidates" if q_cfg.id == "champion" else q_cfg.id, rows),
        })
    return out


def write_report(cfg: Config, context: dict):
    writer = ReportWriter(display_timezone=cfg.report.display_timezone)
    return writer.write(context, cfg.report.output_dir)
