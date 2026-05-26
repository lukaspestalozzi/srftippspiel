"""Claude diagnostic report — a model-introspection tool (NOT the pool-facing report.html).

A dense, machine-optimised artifact for understanding, verifying, validating, and improving
the prediction models, and for answering ad-hoc questions about model behaviour (e.g. "why
does the predictor always tip 1:0 / 0:1?"). It surfaces the raw internals the pretty report
hides: per-fixture scoreline distributions, the EV decomposition behind each recommended tip,
simulation outputs + invariants, and bonus-question calibration against historical statistics
— plus a block of automated PASS/WARN/FAIL anomaly checks.

Two files are written to the report output dir:
  - diagnostic.md   — readable summary with fixed-width tables (the primary artifact).
  - diagnostic.json — the full raw data (distributions + aggregates) for ad-hoc querying.

This report is intended to evolve freely; add/extend sections whenever a new question recurs.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..data import historical_stats as H
from ..strategy.bonus import build_bonus_questions
from ..strategy.expected_points import ev_components, expected_points

_LOW_CLUSTER_WARN = 0.70   # WARN if 1:0/0:1/1:1 exceed this share of recommended tips
_TITLE_SUM_TOL = 0.01
_QUALIFY_SUM_TOL = 0.05
_MATRIX_SUM_TOL = 1e-6


# --------------------------------------------------------------------------- helpers
def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - best-effort only
        return None


def _expected_total_goals(matrix: np.ndarray) -> tuple[float, float]:
    idx = np.arange(matrix.shape[0])
    e_home = float((matrix.sum(axis=1) * idx).sum())
    e_away = float((matrix.sum(axis=0) * idx).sum())
    return e_home, e_away


def _numeric_mean(dist: dict[str, float]) -> float | None:
    try:
        return sum(int(k) * v for k, v in dist.items())
    except ValueError:
        return None


def _mode(dist: dict[str, float]):
    return max(dist, key=dist.get) if dist else None


def _fixed_table(headers: list[str], rows: list[list]) -> str:
    str_rows = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in str_rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    def fmt(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
    lines = [fmt(headers), fmt(["-" * w for w in widths])]
    lines += [fmt(r) for r in str_rows]
    return "```\n" + "\n".join(lines) + "\n```"


# --------------------------------------------------------------------------- per-fixture
def _fixture_records(fixtures, teams, predictions, tipset) -> list[dict]:
    by_id = {m.match_id: m for m in fixtures}
    records = []
    for mid, pred in predictions.items():
        match = by_id.get(mid)
        tip = tipset.tips.get(mid)
        if match is None or tip is None:
            continue
        dist = pred.scoreline
        weight = match.stage.points_weight
        rec = (tip.tip_home, tip.tip_away)
        comps = ev_components(dist, rec[0], rec[1], weight)
        nh, na, np_ = dist.most_likely_scorelines(1)[0]
        e_home, e_away = _expected_total_goals(dist.matrix)
        records.append({
            "match_id": mid,
            "stage": match.stage.value,
            "weight": weight,
            "home": match.home.team_id,
            "away": match.away.team_id,
            "d_elo": teams[match.home.team_id].elo - teams[match.away.team_id].elo,
            "ldw": [dist.p_home_win(), dist.p_draw(), dist.p_away_win()],
            "top3": [(h, a, p) for h, a, p in dist.most_likely_scorelines(3)],
            "rec": rec,
            "rec_ev": tip.expected_points,
            "rec_components": comps,
            "rec_cell_prob": dist.cell(*rec),
            "naive": (nh, na),
            "naive_ev": expected_points(dist, nh, na, weight),
            "naive_cell_prob": np_,
            "exp_goals": e_home + e_away,
        })
    return records


def _predictor_behaviour(records: list[dict]) -> dict:
    n = len(records)
    freq = Counter(r["rec"] for r in records)
    tendency = {"home": 0, "draw": 0, "away": 0}
    for r in records:
        th, ta = r["rec"]
        tendency["home" if th > ta else "draw" if th == ta else "away"] += 1
    comp_keys = ["tendency", "home_goals", "away_goals", "goal_diff"]
    comp_mean = {k: (sum(r["rec_components"][k] for r in records) / n if n else 0.0)
                 for k in comp_keys}
    differs = [r for r in records if r["rec"] != r["naive"]]
    return {
        "n_fixtures": n,
        "tip_frequency": sorted(
            ({"score": f"{h}:{a}", "count": c, "share": c / n} for (h, a), c in freq.items()),
            key=lambda d: d["count"], reverse=True,
        ),
        "tendency_split": {k: {"count": v, "share": v / n if n else 0.0}
                           for k, v in tendency.items()},
        "mean_rec_total_goals": sum(sum(r["rec"]) for r in records) / n if n else 0.0,
        "mean_predicted_total_goals": sum(r["exp_goals"] for r in records) / n if n else 0.0,
        "mean_rec_cell_prob": sum(r["rec_cell_prob"] for r in records) / n if n else 0.0,
        "mean_naive_cell_prob": sum(r["naive_cell_prob"] for r in records) / n if n else 0.0,
        "optimal_differs_share": len(differs) / n if n else 0.0,
        "mean_ev_uplift": (sum(r["rec_ev"] - r["naive_ev"] for r in records) / n) if n else 0.0,
        "ev_component_mean": comp_mean,
    }


def _behaviour_notes(pb: dict) -> list[str]:
    notes = []
    by_score = {d["score"]: d["share"] for d in pb["tip_frequency"]}
    low = by_score.get("1:0", 0) + by_score.get("0:1", 0) + by_score.get("1:1", 0)
    if low > 0.5:
        notes.append(
            f"{low:.0%} of recommended tips are 1:0/0:1/1:1. This is expected under "
            f"EV-maximisation: mean predicted goals per match is "
            f"{pb['mean_predicted_total_goals']:.2f} (~{pb['mean_predicted_total_goals'] / 2:.2f}/side, "
            f"so the modal scoreline per side is ~1), and the SRF 5-point tendency term dominates "
            f"EV (mean contribution {pb['ev_component_mean']['tendency']:.2f} pts vs "
            f"{pb['ev_component_mean']['goal_diff']:.2f} for goal-diff). The optimiser therefore "
            f"picks the lowest-total scoreline that captures the dominant tendency: mean total goals "
            f"of recommended tips is only {pb['mean_rec_total_goals']:.2f} vs predicted "
            f"{pb['mean_predicted_total_goals']:.2f}."
        )
    draw_share = pb["tendency_split"]["draw"]["share"]
    if draw_share < 0.10:
        notes.append(
            f"Only {draw_share:.0%} of tips are draws: a draw's probability mass is split across "
            f"0:0/1:1/2:2…, so no single draw scoreline usually wins the exact + goal-marginal EV "
            f"terms even when the aggregate P(draw) is sizeable."
        )
    if pb["optimal_differs_share"] > 0:
        notes.append(
            f"{pb['optimal_differs_share']:.0%} of EV-optimal tips differ from the single "
            f"most-likely scoreline (mean EV uplift {pb['mean_ev_uplift']:.3f} pts): the optimiser "
            f"trades exact-hit probability for tendency / goal-difference points."
        )
    return notes


# --------------------------------------------------------------------------- simulation
def _simulation_section(outcome, teams, fixtures) -> dict | None:
    if outcome is None:
        return None
    # The first knockout round's reach metric (reach_r32 for WC2026, reach_qf for a 16-team
    # tournament) is the first reach_* key in the stage-ordered advancement dict.
    sample = next(iter(outcome.advancement.values()))
    first_reach = next((k for k in sample if k.startswith("reach_")), "wins_title")
    title_sum = sum(a.get("wins_title", 0.0) for a in outcome.advancement.values())
    qualify_sum = sum(a.get(first_reach, 0.0) for a in outcome.advancement.values())
    title_odds = sorted(
        ({"team": teams[t].name, "p": a.get("wins_title", 0.0)}
         for t, a in outcome.advancement.items() if a.get("wins_title", 0.0) > 0),
        key=lambda d: d["p"], reverse=True,
    )[:15]
    groups: dict[str, set] = {}
    for m in fixtures:
        if m.group:
            groups.setdefault(m.group, set()).update({m.home.team_id, m.away.team_id})
    group_qual = {}
    for letter in sorted(groups):
        rows = sorted(
            ({"team": teams[t].name, "p": outcome.advancement[t].get(first_reach, 0.0)}
             for t in groups[letter]),
            key=lambda d: d["p"], reverse=True,
        )
        group_qual[letter] = rows
    return {
        "iterations": outcome.mc_iterations,
        "seed": outcome.mc_seed,
        "max_standard_error": outcome.mc_standard_error,
        "first_reach_metric": first_reach,
        "title_prob_sum": title_sum,
        "qualify_prob_sum": qualify_sum,
        "title_odds": title_odds,
        "group_qualification": group_qual,
    }


# --------------------------------------------------------------------------- bonus
def _bonus_history(qid, dist, mean, mode) -> tuple[str, str]:
    if qid == "top_scorer_goals" and mean is not None:
        recent = H.recent_wc_top_scorer_mean()
        status = "PASS" if recent - 0.5 <= mean <= recent + 2.0 else "WARN"
        return status, f"prior mean {mean:.2f} vs recent-WC mean {recent:.2f}; mode {mode}"
    if qid == "zero_zero_count" and mean is not None:
        lo, hi = H.zero_zero_rate_band()
        rate = mean / 104
        status = "PASS" if lo * 0.9 <= rate <= hi * 1.1 else "WARN"
        return status, f"sim mean {mean:.2f} -> rate {rate:.3f} vs historical band [{lo:.3f}, {hi:.3f}]"
    if qid == "swiss_goals" and mean is not None:
        hist = [g for _, _, g in H.SWITZERLAND_RESULTS]
        status = "PASS" if min(hist) <= mean <= max(hist) + 3 else "WARN"
        return status, f"sim mean {mean:.2f} vs SUI recent totals {min(hist)}-{max(hist)}"
    if qid == "swiss_progress":
        hist_modal = Counter(s for _, s, _ in H.SWITZERLAND_RESULTS).most_common(1)[0][0]
        status = "PASS" if mode in {"Sechzehntelfinal", "Achtelfinal", "Viertelfinal"} else "WARN"
        return status, f"modal exit {mode}; SUI recent modal {hist_modal}"
    if qid == "champion" and dist:
        return "INFO", f"top title probability {max(dist.values()):.3f}"
    return "INFO", "no historical reference"


def _bonus_section(bundle, teams, outcome) -> list[dict]:
    out = []
    for q in build_bonus_questions(bundle.bonus_questions):
        dist = q.resolve(outcome) if outcome is not None or q.question_id == "top_scorer_goals" else {}
        if not dist:
            out.append({"id": q.question_id, "label": q.label, "available": False})
            continue
        mode = _mode(dist)
        mean = _numeric_mean(dist)
        status, detail = _bonus_history(q.question_id, dist, mean, mode)
        ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        out.append({
            "id": q.question_id,
            "label": q.label,
            "available": True,
            "mode": teams[mode].name if mode in teams else mode,
            "mean": mean,
            "distribution": [{"answer": teams[k].name if k in teams else k, "p": v}
                             for k, v in ranked],
            "status": status,
            "history": detail,
        })
    return out


# --------------------------------------------------------------------------- anomalies
def _anomaly_checks(predictions, records, pb, sim, bonus) -> list[dict]:
    checks = []

    bad_matrix = sum(1 for p in predictions.values()
                     if abs(float(p.scoreline.matrix.sum()) - 1.0) > _MATRIX_SUM_TOL)
    checks.append({"name": "scoreline matrices sum to 1",
                   "status": "PASS" if bad_matrix == 0 else "FAIL",
                   "detail": f"{bad_matrix} of {len(predictions)} off by > {_MATRIX_SUM_TOL}"})

    bad_ldw = sum(1 for r in records if abs(sum(r["ldw"]) - 1.0) > 1e-6)
    checks.append({"name": "L/D/W sums to 1",
                   "status": "PASS" if bad_ldw == 0 else "FAIL",
                   "detail": f"{bad_ldw} of {len(records)} off by > 1e-6"})

    bad_range = sum(1 for r in records if not all(0.0 <= x <= 1.0 for x in r["ldw"]))
    checks.append({"name": "probabilities in [0, 1]",
                   "status": "PASS" if bad_range == 0 else "FAIL",
                   "detail": f"{bad_range} fixtures with out-of-range L/D/W"})

    if sim is not None:
        ts_ok = abs(sim["title_prob_sum"] - 1.0) <= _TITLE_SUM_TOL
        checks.append({"name": "Sum(wins_title) ~ 1",
                       "status": "PASS" if ts_ok else "FAIL",
                       "detail": f"{sim['title_prob_sum']:.4f} (tol {_TITLE_SUM_TOL})"})
        # The number of teams reaching the first KO round is a fixed integer per iteration,
        # so its expected value must be (near) integer and positive (32 for WC2026, 8 here).
        qsum = sim["qualify_prob_sum"]
        qs_ok = qsum > 0 and abs(qsum - round(qsum)) <= _QUALIFY_SUM_TOL
        checks.append({"name": f"Sum({sim['first_reach_metric']}) is integer",
                       "status": "PASS" if qs_ok else "FAIL",
                       "detail": f"{qsum:.4f} ~ {round(qsum)} (tol {_QUALIFY_SUM_TOL})"})

    by_score = {d["score"]: d["share"] for d in pb["tip_frequency"]}
    low = by_score.get("1:0", 0) + by_score.get("0:1", 0) + by_score.get("1:1", 0)
    checks.append({"name": "low-scoreline clustering",
                   "status": "WARN" if low > _LOW_CLUSTER_WARN else "PASS",
                   "detail": f"1:0/0:1/1:1 share {low:.0%} (warn > {_LOW_CLUSTER_WARN:.0%}); "
                             f"expected behaviour, flagged so a regression is visible"})

    for b in bonus:
        if b.get("available") and b["status"] != "INFO":
            checks.append({"name": f"bonus calibration: {b['id']}",
                           "status": b["status"], "detail": b["history"]})
    return checks


# --------------------------------------------------------------------------- markdown
def _render_markdown(meta, pb, notes, records, sim, bonus, anomalies) -> str:
    L = []
    L.append("# Claude Diagnostic Report")
    L.append("")
    L.append(f"_Generated {meta['generated_at']}_  •  git `{meta['git_head'] or 'n/a'}`")
    L.append("")

    # 1. Run header
    L.append("## 1. Run header")
    L.append(_fixed_table(
        ["key", "value"],
        [["tournament", meta.get("tournament", "?")],
         ["predictor", meta["predictor_name"]],
         ["predictor_params", json.dumps(meta["predictor_params"])],
         ["strategy", meta["strategy_name"]],
         ["simulation", f"{meta['iterations']} iters, seed {meta['seed']}, "
                        f"penalty={meta['penalty_model']}" if meta["simulated"] else "skipped (--no-sim)"],
         ["data", f"{meta['n_teams']} teams, {meta['n_fixtures']} fixtures, "
                  f"{meta['n_results']} played, {pb['n_fixtures']} tippable"]],
    ))
    L.append("")

    # 2. Predictor behaviour
    L.append("## 2. Predictor behaviour")
    L.append("")
    L.append("### Recommended-tip frequency")
    L.append(_fixed_table(
        ["score", "count", "share"],
        [[d["score"], d["count"], f"{d['share']:.1%}"] for d in pb["tip_frequency"]],
    ))
    t = pb["tendency_split"]
    L.append("")
    L.append("### Tendency split & scoreline summary")
    L.append(_fixed_table(
        ["metric", "value"],
        [["home-win tips", f"{t['home']['count']} ({t['home']['share']:.0%})"],
         ["draw tips", f"{t['draw']['count']} ({t['draw']['share']:.0%})"],
         ["away-win tips", f"{t['away']['count']} ({t['away']['share']:.0%})"],
         ["mean rec total goals", f"{pb['mean_rec_total_goals']:.2f}"],
         ["mean predicted total goals", f"{pb['mean_predicted_total_goals']:.2f}"],
         ["mean P(rec exact cell)", f"{pb['mean_rec_cell_prob']:.3f}"],
         ["mean P(most-likely cell)", f"{pb['mean_naive_cell_prob']:.3f}"],
         ["EV-optimal != most-likely", f"{pb['optimal_differs_share']:.0%}"],
         ["mean EV uplift vs naive", f"{pb['mean_ev_uplift']:.3f} pts"]],
    ))
    c = pb["ev_component_mean"]
    L.append("")
    L.append("### Mean EV contribution per scoring term (recommended tips)")
    L.append(_fixed_table(
        ["term", "mean pts"],
        [["tendency (5)", f"{c['tendency']:.3f}"],
         ["home-goals (1)", f"{c['home_goals']:.3f}"],
         ["away-goals (1)", f"{c['away_goals']:.3f}"],
         ["goal-diff (3)", f"{c['goal_diff']:.3f}"]],
    ))
    if notes:
        L.append("")
        L.append("### Interpretation notes")
        for note in notes:
            L.append(f"- {note}")
    L.append("")

    # 3. Per-fixture detail
    L.append("## 3. Per-fixture detail")
    rows = []
    for r in records:
        ldw = "/".join(f"{x:.0%}".rstrip("%") for x in r["ldw"])
        top3 = " ".join(f"{h}:{a}={p:.2f}".replace("0.", ".") for h, a, p in r["top3"])
        cc = r["rec_components"]
        comps = f"{cc['tendency']:.1f}/{cc['home_goals']:.1f}/{cc['away_goals']:.1f}/{cc['goal_diff']:.1f}"
        rows.append([
            r["match_id"], f"{r['home']}-{r['away']}", f"{r['d_elo']:+.0f}", ldw, top3,
            f"{r['rec'][0]}:{r['rec'][1]}", f"{r['rec_ev']:.2f}", comps,
            f"{r['naive'][0]}:{r['naive'][1]}", f"{r['naive_ev']:.2f}",
        ])
    L.append("LDW = home/draw/away win %.  comps = EV split t/h/a/d (tendency/home-goals/away-goals/goal-diff).")
    L.append(_fixed_table(
        ["match", "tie", "dElo", "LDW", "top-3 cells", "rec", "recEV", "comps t/h/a/d", "naive", "naiveEV"],
        rows,
    ))
    L.append("")

    # 4. Simulation diagnostics
    L.append("## 4. Simulation diagnostics")
    if sim is None:
        L.append("_Simulation skipped (--no-sim); advancement/title/bonus-sim sections unavailable._")
    else:
        L.append(_fixed_table(
            ["invariant", "value", "status"],
            [["Sum(wins_title)", f"{sim['title_prob_sum']:.4f}",
              "PASS" if abs(sim['title_prob_sum'] - 1) <= _TITLE_SUM_TOL else "FAIL"],
             [f"Sum({sim['first_reach_metric']})", f"{sim['qualify_prob_sum']:.4f}",
              "PASS" if abs(sim['qualify_prob_sum'] - round(sim['qualify_prob_sum'])) <= _QUALIFY_SUM_TOL else "FAIL"],
             ["max standard error", f"{sim['max_standard_error']:.4f}", "-"]],
        ))
        L.append("")
        L.append("### Title odds (top 15)")
        L.append(_fixed_table(["team", "P(title)"],
                              [[d["team"], f"{d['p']:.1%}"] for d in sim["title_odds"]]))
        L.append("")
        L.append("### Group qualification P(reach R32)")
        for letter, grp in sim["group_qualification"].items():
            line = "  ".join(f"{d['team']} {d['p']:.0%}" for d in grp)
            L.append(f"- **{letter}**: {line}")
    L.append("")

    # 5. Bonus diagnostics
    L.append("## 5. Bonus-question diagnostics")
    for b in bonus:
        L.append("")
        L.append(f"### {b['label']} (`{b['id']}`)")
        if not b.get("available"):
            L.append("_unavailable (needs simulation)_")
            continue
        mean = f"{b['mean']:.2f}" if b["mean"] is not None else "n/a"
        L.append(f"recommended **{b['mode']}**  •  mean {mean}  •  calibration **{b['status']}** — {b['history']}")
        L.append(_fixed_table(
            ["answer", "p"],
            [[d["answer"], f"{d['p']:.3f}"] for d in b["distribution"][:10]],
        ))
    L.append("")

    # 6. Validation / anomaly summary
    L.append("## 6. Validation / anomaly summary")
    L.append(_fixed_table(
        ["check", "status", "detail"],
        [[a["name"], a["status"], a["detail"]] for a in anomalies],
    ))
    n_fail = sum(1 for a in anomalies if a["status"] == "FAIL")
    n_warn = sum(1 for a in anomalies if a["status"] == "WARN")
    L.append("")
    L.append(f"**{len(anomalies)} checks: {n_fail} FAIL, {n_warn} WARN, "
             f"{len(anomalies) - n_fail - n_warn} PASS/INFO.**")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- entrypoint
def build_diagnostics(cfg, bundle, teams, fixtures, results, predictions, tipset, outcome, predictor):
    """Return ``(markdown, data)`` for the Claude diagnostic report."""
    records = _fixture_records(fixtures, teams, predictions, tipset)
    pb = _predictor_behaviour(records)
    notes = _behaviour_notes(pb)
    sim = _simulation_section(outcome, teams, fixtures)
    bonus = _bonus_section(bundle, teams, outcome)
    anomalies = _anomaly_checks(predictions, records, pb, sim, bonus)
    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git_head": _git_head(),
        "tournament": bundle.display_name,
        "predictor_name": predictor.name,
        "predictor_params": getattr(predictor, "params", {}),
        "strategy_name": cfg.strategy.name,
        "simulated": outcome is not None,
        "iterations": cfg.simulation.iterations,
        "seed": cfg.simulation.seed,
        "penalty_model": cfg.simulation.penalty_model,
        "n_teams": len(teams),
        "n_fixtures": len(fixtures),
        "n_results": len(results),
    }
    pb_notes = dict(pb, notes=notes)
    data = {
        "meta": meta,
        "predictor_behaviour": pb_notes,
        "fixtures": records,
        "simulation": sim,
        "bonus": bonus,
        "anomalies": anomalies,
    }
    markdown = _render_markdown(meta, pb, notes, records, sim, bonus, anomalies)
    return markdown, data


class DiagnosticsWriter:
    def write(self, markdown: str, data: dict, output_dir: str | Path) -> dict[str, Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / "diagnostic.md"
        json_path = out_dir / "diagnostic.json"
        md_path.write_text(markdown, encoding="utf-8")
        json_path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
        return {"markdown": md_path, "json": json_path}


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON-serialisable: {type(o)}")
