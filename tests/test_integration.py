"""Full-pipeline integration test (spec §10): a well-formed, self-contained report."""

import dataclasses
import re
from pathlib import Path

import pytest

import tippspiel
from tippspiel.config import load_config, load_tournament
from tippspiel.model.types import Result
from tippspiel.pipeline import run_pipeline, write_report
from tippspiel.strategy.expected_points import score_tip

REPO = Path(tippspiel.__file__).parent.parent
BUNDLE = load_tournament(REPO / "config.yaml")


@pytest.fixture(scope="module")
def small_cfg():
    cfg = load_config(REPO / "config.yaml")
    sim = dataclasses.replace(cfg.simulation, iterations=400)
    return dataclasses.replace(cfg, simulation=sim)


def test_predict_only_pipeline(small_cfg):
    result = run_pipeline(small_cfg, BUNDLE, simulate=False)
    tips = result["tipset"].tips
    # All 72 group fixtures are tippable; no simulation-dependent bonus answers required.
    assert sum(1 for mid in tips if mid.startswith("G_")) == 72
    # Knockout fixtures the played results already decide are tipped too (groups A/B/C are
    # finished, so M73 = runner-up A vs runner-up B = RSA vs CAN is known).
    assert "M73" in tips
    assert result["outcome"] is None


def test_full_pipeline_self_contained_report(tmp_path, small_cfg):
    cfg = dataclasses.replace(
        small_cfg, report=dataclasses.replace(small_cfg.report, output_dir=str(tmp_path))
    )
    result = run_pipeline(cfg, BUNDLE, simulate=True)
    path = write_report(cfg, result["context"])
    html = Path(path).read_text()

    assert html.strip().endswith("</html>")
    # Figures ship as inert JSON payloads + the lazy-render runtime (eager rendering of the
    # ~200 embedded figures blocks page load for ~10s); nothing calls newPlot at parse time.
    assert "Plotly.newPlot" in html            # the runtime is present...
    assert html.count('class="lazy-plot"') > 100  # ...and the figures are lazy payloads
    assert '<script type="text/javascript">Plotly.newPlot' not in html
    # Self-contained: no external script/style/image loads.
    assert not re.search(r'<(script|link|img)[^>]*(src|href)=[\"\']https?://', html)
    # Champion recommendation present.
    assert result["tipset"].bonus_answers.get("champion")
    for section in ("Group-stage fixtures", "Group standings", "Group advancement", "Title odds",
                    "Bonus", "Model L/D/W", "Expected goals", "Top scorelines", "Why this tip"):
        assert section in html
    # Group standings: finished Group A shows Mexico through with 9 points.
    standings = {s["letter"]: s for s in result["context"]["group_standings"]}
    top = standings["A"]["rows"][0]
    assert top["team"] == "Mexico" and top["points"] == 9 and top["qualified"]
    assert "✓ through" in html
    # A fully-resolved knockout fixture shows both teams' names, not "None"
    # (a concrete side's TeamRef.placeholder is None — must fall back to the team name).
    ko = {b["match_id"]: b for b in result["context"]["knockout_fixtures"]}
    assert ko["M84"]["home"] == "Spain" and ko["M84"]["away"] == "Austria"
    assert ko["M75"]["away"] == "Morocco"
    # A later-round fixture with an unresolved leg shows its slot placeholder ("Winner of M…",
    # "Loser of M…", "Winner/Runner-up Group …", "3rd place (slot …)"). Asserted generically
    # rather than against a hardcoded still-pending match: those resolve as the tournament plays
    # out, so a hardcoded expectation would need re-editing every matchday. Any placeholder slot
    # present must be well-formed and never a stray "None"; when the bracket is fully decided
    # there are simply none (vacuously fine).
    _PLACEHOLDER_PREFIXES = ("Winner Group ", "Runner-up Group ", "3rd place (slot ",
                             "Winner of ", "Loser of ")
    placeholders = [v for b in result["context"]["knockout_fixtures"]
                    for v in (b["home"], b["away"]) if v.startswith(_PLACEHOLDER_PREFIXES)]
    assert all(p and p != "None" for p in placeholders)
    ko_html = html[html.find('id="knockout"'):html.find('id="title"')]
    assert "None" not in ko_html


def test_market_odds_tips_in_report(tmp_path, small_cfg):
    # An odds file makes the report show an extra "Market-odds tip" line per fixture that has
    # genuine odds — and only those fixtures. It renders regardless of the configured predictor
    # (here the default elo_poisson stays the recommended tip), so the market tip appears
    # alongside the Elo recommendation.
    odds_csv = tmp_path / "odds.csv"
    odds_csv.write_text(
        "match_id,odds_home,odds_draw,odds_away\n"
        "G_A_1,1.5,4.0,6.0\n"   # has odds -> market tip line rendered
        "G_A_2,2.1,3.3,3.4\n"
    )
    cfg = dataclasses.replace(
        small_cfg,
        report=dataclasses.replace(small_cfg.report, output_dir=str(tmp_path)),
    )
    bundle = dataclasses.replace(BUNDLE, odds_file=odds_csv)
    result = run_pipeline(cfg, bundle, simulate=False)
    path = write_report(cfg, result["context"])
    html = Path(path).read_text()
    # One odds line, gated to the two odds-backed fixtures only.
    assert html.count("Market-odds tip:") == 2
    # The blended de-vigged 1X2 row appears in the data table for exactly those two fixtures.
    assert html.count("Market — blended (de-vigged)") == 2
    # With no per-source sidecars present, only the blend renders (no ESPN/Polymarket rows).
    assert "Market — ESPN" not in html and "Market — Polymarket" not in html
    # The Elo recommended tip is unaffected: all 72 group fixtures still tipped.
    assert sum(1 for mid in result["tipset"].tips if mid.startswith("G_")) == 72


def test_market_source_rows_in_report(tmp_path, small_cfg):
    # When odds_espn.csv / odds_polymarket.csv sidecars sit beside odds.csv, the folded per-fixture
    # section shows each source's de-vigged 1X2 — gated per source to the fixtures it priced.
    (tmp_path / "odds.csv").write_text(
        "match_id,odds_home,odds_draw,odds_away\n"
        "G_A_1,1.5,4.0,6.0\nG_A_2,2.1,3.3,3.4\nG_A_3,1.8,3.5,4.5\n"
    )
    (tmp_path / "odds_espn.csv").write_text(
        "match_id,odds_home,odds_draw,odds_away\n"
        "G_A_1,1.5,4.0,6.0\nG_A_3,1.8,3.5,4.5\n"          # ESPN prices A_1 + A_3
    )
    (tmp_path / "odds_polymarket.csv").write_text(
        "match_id,odds_home,odds_draw,odds_away\n"
        "G_A_1,1.55,3.9,5.8\nG_A_2,2.05,3.35,3.45\n"      # Polymarket prices A_1 + A_2
    )
    cfg = dataclasses.replace(
        small_cfg, report=dataclasses.replace(small_cfg.report, output_dir=str(tmp_path)),
    )
    bundle = dataclasses.replace(BUNDLE, odds_file=tmp_path / "odds.csv")
    result = run_pipeline(cfg, bundle, simulate=False)
    html = Path(write_report(cfg, result["context"])).read_text()

    assert html.count("Market — ESPN") == 2          # A_1, A_3
    assert html.count("Market — Polymarket") == 2     # A_1, A_2
    assert html.count("Market — blended (de-vigged)") == 3  # all three odds-backed fixtures

    # The data block carries per-source triples, gated to where each source priced the fixture.
    blocks = {f["match_id"]: f["data"]["market_sources"]
              for f in result["context"]["group_fixtures"] if f["data"]}
    assert set(blocks["G_A_1"]) == {"espn", "poly"}
    assert set(blocks["G_A_2"]) == {"poly"}
    assert set(blocks["G_A_3"]) == {"espn"}
    espn = blocks["G_A_1"]["espn"]
    assert abs(espn["home"] + espn["draw"] + espn["away"] - 1.0) < 1e-9


def _with_alpha(cfg, alpha):
    """Set the off/def volume weight regardless of config shape: top-level params for
    elo_poisson, nested fallback_params for the market_odds wrapper."""
    p = dict(cfg.predictor.params)
    if "fallback_params" in p:
        p["fallback_params"] = {**p["fallback_params"], "alpha": alpha}
    else:
        p["alpha"] = alpha
    return dataclasses.replace(cfg, predictor=dataclasses.replace(cfg.predictor, params=p))


def _alpha_of(cfg) -> float:
    p = cfg.predictor.params
    return p.get("fallback_params", p).get("alpha", p.get("alpha", 0.0))


def test_offdef_display_gated_on_alpha(tmp_path, small_cfg):
    # The pool report surfaces each team's att/def + goal-volume effect when the predictor
    # actually uses them (alpha>0), and hides the row entirely when alpha=0.
    on = dataclasses.replace(
        small_cfg, report=dataclasses.replace(small_cfg.report, output_dir=str(tmp_path))
    )
    html_on = Path(write_report(on, run_pipeline(on, BUNDLE, simulate=False)["context"])).read_text()
    assert "Attack / defence" in html_on
    assert "goal-volume layer" in html_on  # the one-time legend

    off = _with_alpha(on, 0.0)
    ctx_off = run_pipeline(off, BUNDLE, simulate=False)["context"]
    html_off = Path(write_report(off, ctx_off)).read_text()
    assert "Attack / defence" not in html_off
    assert "goal-volume layer" not in html_off
    assert all(f["data"]["offdef"] is None for f in ctx_off["group_fixtures"] if f["data"])


def test_offdef_legend_hidden_when_ratings_all_zero(tmp_path, small_cfg):
    # alpha>0 but no team has fitted att/def (fit-offdef never run) must NOT show the legend —
    # otherwise it would promise a per-fixture att/def row that never renders.
    from tippspiel.data.file_provider import FileDataProvider

    orig = FileDataProvider.get_teams
    FileDataProvider.get_teams = lambda self: [
        dataclasses.replace(t, att_elo=0.0, def_elo=0.0) for t in orig(self)
    ]
    try:
        cfg = dataclasses.replace(
            small_cfg, report=dataclasses.replace(small_cfg.report, output_dir=str(tmp_path))
        )
        assert _alpha_of(cfg) > 0  # guard: the config does enable off/def
        ctx = run_pipeline(cfg, BUNDLE, simulate=False)["context"]
    finally:
        FileDataProvider.get_teams = orig
    assert ctx["header"]["uses_offdef"] is False
    html = Path(write_report(cfg, ctx)).read_text()
    assert "goal-volume layer" not in html
    assert "Attack / defence" not in html


def test_elo_history_section_in_report(tmp_path, small_cfg):
    # config.yaml is corpus-fitted (elo.source: corpus), so the report grows an
    # "Elo ratings over time" section: scalar Elo + att/def trajectories replayed from the
    # corpus with the same params/snapshot as fit-ratings.
    cfg = dataclasses.replace(
        small_cfg, report=dataclasses.replace(small_cfg.report, output_dir=str(tmp_path))
    )
    result = run_pipeline(cfg, BUNDLE, simulate=False)
    hist = result["context"]["elo_history"]
    assert hist is not None
    assert 1 <= len(hist["highlight"]) <= 8         # categorical-palette ceiling
    assert hist["window_start"] < hist["snapshot"]  # ISO strings compare chronologically
    assert hist["window_start"] == "2000-01-01"     # config.yaml report.elo_history_start
    for key in ("elo_chart", "att_chart", "def_chart"):
        assert 'class="lazy-plot"' in hist[key]
    html = Path(write_report(cfg, result["context"])).read_text()
    assert "Elo ratings over time" in html


def test_elo_history_skipped_for_external_elo(tmp_path):
    # An external-Elo tournament (the completed benchmarks, women's Euro) keeps a committed
    # ratings snapshot the corpus can't reproduce -> no history section.
    from tippspiel.pipeline import _elo_history_section

    cfg = load_config(REPO / "configs" / "wc2022.yaml")
    bundle = load_tournament(REPO / "configs" / "wc2022.yaml")
    from tippspiel.data.file_provider import FileDataProvider

    provider = FileDataProvider(bundle.teams_file, bundle.fixtures_file, bundle.results_file)
    teams = {t.team_id: t for t in provider.get_teams()}
    assert _elo_history_section(cfg, bundle, teams, provider.get_fixtures()) is None


def test_realism_tolerance_raises_both_score_share(small_cfg):
    # A realism tolerance on the EV optimiser lifts the share of tips where BOTH teams score
    # from the ~8% strict-EV rate toward a realistic ~50%.
    from tippspiel.config import StrategyConfig

    def both_score_share(tol):
        cfg = dataclasses.replace(small_cfg, strategy=StrategyConfig(realism_tolerance=tol))
        tips = run_pipeline(cfg, BUNDLE, simulate=False)["tipset"].tips
        return sum(t.tip_home > 0 and t.tip_away > 0 for t in tips.values()) / len(tips)

    legacy = both_score_share(0.0)
    realistic = both_score_share(0.15)
    assert realistic > 0.40             # tolerance reaches a realistic both-teams-score rate
    assert realistic - legacy >= 0.15   # and clearly lifts it above the strict-EV baseline


def test_strategy_config_parses_realism_tolerance(tmp_path):
    from tippspiel.config import load_config

    base = (
        "predictor: {name: elo_poisson, params: {}}\n"
        "simulation: {iterations: 10, seed: 1, penalty_model: coin_flip}\n"
        "report: {output_dir: out/, display_timezone: UTC}\n"
    )
    p = tmp_path / "c.yaml"
    p.write_text(base)
    assert load_config(p).strategy.realism_tolerance == 0.0  # omitted -> legacy default
    p.write_text(base + "strategy: {realism_tolerance: 0.2}\n")
    assert load_config(p).strategy.realism_tolerance == 0.2


def test_played_match_still_predicted_for_display(small_cfg):
    # A played match keeps its (pre-match) prediction and tip so the report can show the
    # forecast next to the real result; the fixture block carries BOTH result and tip.
    cfg = small_cfg
    from tippspiel.data.file_provider import FileDataProvider

    orig = FileDataProvider.get_results
    FileDataProvider.get_results = lambda self: [Result("G_A_1", 1, 0)]
    try:
        result = run_pipeline(cfg, BUNDLE, simulate=False)
    finally:
        FileDataProvider.get_results = orig
    # The played match is still tipped (for display), so no fixture drops out of the tipset.
    assert "G_A_1" in result["tipset"].tips
    assert len(result["tipset"].tips) == 72
    block = next(f for f in result["context"]["group_fixtures"] if f["match_id"] == "G_A_1")
    assert block["played"] is True
    assert block["result"] == {"home_goals": 1, "away_goals": 0}
    assert block["tip"] is not None and block["data"] is not None
    # ... plus the likelihood hint for the actual result and the tip's accuracy against it.
    assert block["actual"]["label"] in {"expected", "plausible", "surprising"}
    assert 0.0 <= block["actual"]["p_exact"] <= block["actual"]["p_tendency"] <= 1.0
    tip = result["tipset"].tips["G_A_1"]
    assert block["tip_outcome"]["points"] == score_tip(tip.tip_home, tip.tip_away, 1, 0, 1)
    assert block["tip_outcome"]["max"] == 10
    assert block["tip_outcome"]["cls"] in {"exact", "tendency", "miss"}


def test_fixture_block_carries_underlying_data(small_cfg):
    # Every tippable fixture exposes the underlying prediction numbers used in the report's
    # data table; the EV breakdown must reconstruct the recommended tip's EV exactly.
    result = run_pipeline(small_cfg, BUNDLE, simulate=False)
    fixtures = result["context"]["group_fixtures"]
    tipped = [f for f in fixtures if f["data"] and f["tip"]]
    assert tipped, "expected at least one tipped group fixture with data"
    for f in tipped:
        d = f["data"]
        assert {"ldw", "exp_goals", "top3", "elo", "offdef", "rec_components",
                "rec_cell_prob"} <= set(d)
        # L/D/W is a partition; EV components sum to the recommended (displayed) EV.
        assert d["ldw"]["home"] + d["ldw"]["draw"] + d["ldw"]["away"] == pytest.approx(1.0)
        assert d["rec_components"]["total"] == pytest.approx(f["tip"]["ev"])
        # Group fixtures have concrete teams -> Elo populated.
        assert d["elo"] is not None
        # config.yaml fits off/def + sets alpha>0, so att/def are surfaced per fixture.
        assert d["offdef"] is not None
        assert {"home_name", "att_home", "def_home", "att_away", "def_away", "pct"} <= set(
            d["offdef"]
        )
        # Market probs appear only for odds-backed fixtures; when present they are a de-vigged
        # 1X2 partition.
        if d["market_probs"] is not None:
            mp = d["market_probs"]
            assert mp["home"] + mp["draw"] + mp["away"] == pytest.approx(1.0)
        assert len(d["top3"]) == 3
        # Likelihood hint + accuracy tag exist exactly for played fixtures.
        if f["played"]:
            assert f["actual"] is not None and f["tip_outcome"] is not None
        else:
            assert f["actual"] is None and f["tip_outcome"] is None


def test_fixture_data_elo_omitted_for_placeholder():
    # A knockout fixture with a TBD side (placeholder ref) must omit Elo but still carry the
    # tip's EV breakdown — the Elo guard mirrors the team-name resolution guard.
    from types import SimpleNamespace

    import numpy as np

    from tippspiel.model.scoreline import ScorelineDistribution
    from tippspiel.pipeline import _fixture_data

    dist = ScorelineDistribution(np.ones((4, 4)) / 16)
    m = SimpleNamespace(
        match_id="K1",
        home=SimpleNamespace(is_concrete=True, team_id="ARG"),
        away=SimpleNamespace(is_concrete=False, team_id=None),
    )
    teams = {"ARG": SimpleNamespace(elo=1800.0)}
    data = _fixture_data(m, teams, dist, (1, 0), weight=2, market=None)
    assert data["elo"] is None
    assert data["rec_components"] is not None
    assert data["market_probs"] is None


def test_actual_likelihood_labels_and_out_of_grid():
    import numpy as np

    from tippspiel.model.scoreline import ScorelineDistribution
    from tippspiel.pipeline import _actual_likelihood

    mat = np.zeros((3, 3))
    mat[1, 0] = 0.60  # 1-0
    mat[0, 1] = 0.25  # 0-1
    mat[0, 0] = 0.15  # 0-0
    dist = ScorelineDistribution(mat)

    hint = _actual_likelihood(dist, 1, 0)
    assert hint == {"p_exact": pytest.approx(0.60), "p_tendency": pytest.approx(0.60),
                    "label": "expected"}
    # The label keys off the tendency probability, at the documented boundaries.
    assert _actual_likelihood(dist, 0, 1)["label"] == "plausible"   # 0.25, inclusive boundary
    assert _actual_likelihood(dist, 0, 0)["label"] == "surprising"  # 0.15
    # A result beyond the distribution grid: zero exact probability, no crash.
    freak = _actual_likelihood(dist, 7, 3)
    assert freak["p_exact"] == 0.0
    assert freak["p_tendency"] == pytest.approx(0.60)


def test_tip_outcome_tags():
    from tippspiel.pipeline import _tip_outcome

    assert _tip_outcome(2, 1, 2, 1, 1) == {"points": 10, "max": 10, "label": "exact hit",
                                           "cls": "exact"}
    t = _tip_outcome(2, 1, 3, 1, 1)  # right tendency (5) + away goals (1)
    assert (t["label"], t["cls"], t["points"], t["max"]) == ("correct tendency", "tendency", 6, 10)
    # KO weight doubles; a tipped win vs a 120-minute draw is a tendency miss
    # (still 2 pts for the matching home-goal count).
    assert _tip_outcome(1, 0, 1, 1, 2) == {"points": 2, "max": 20, "label": "miss",
                                           "cls": "miss"}
