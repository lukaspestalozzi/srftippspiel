"""Offline model-fitting tools (not part of the prediction/simulation hot path).

Currently houses the offensive/defensive Elo fitter (``offdef_elo``), which learns per-team
attack/defence log-rate deviations from historical match goals for ``Team.att_elo`` /
``Team.def_elo``. Run via ``tippspiel fit-ratings``.
"""
