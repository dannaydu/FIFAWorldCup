"""Assemble the modelling table: A-vs-B feature diffs + labels.

Result encoding (3-class): 0 = team_a win, 1 = draw, 2 = team_b win.
Also carries team_a_goals / team_b_goals for the Poisson/Dixon-Coles model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wcmodel import config
from wcmodel.features.confederations import team_conf_strength
from wcmodel.features.squad_features import SQUAD_FEATURE_COLS, load_squad_features
from wcmodel.features.team_features import build_team_features

RESULT_TO_CLASS = {"team_a_win": 0, "draw": 1, "team_b_win": 2}

# Form features that get differenced (A - B).
_FORM_DIFF_COLS = [
    "gf_l10", "ga_l10", "gd_l10", "winrate_l10", "sos_l10",
    "pts_vs_exp_l10", "gd_vs_exp_l10",
    "gd_decayed", "gf_decayed", "winrate_decayed",
]


def build_match_features(
    matches: pd.DataFrame,
    *,
    squad: pd.DataFrame | None = None,
    squad_panel: pd.DataFrame | None = None,
    conf_strength: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return one row per match with diff features and labels."""
    long = build_team_features(matches)
    conf_strength = conf_strength if conf_strength is not None else team_conf_strength(matches)

    base = matches.sort_values("date", kind="stable").reset_index(drop=True).copy()
    base["match_id"] = np.arange(len(base))

    # Split the long table back into A/B by joining on team identity per match.
    side_a = long.merge(
        base[["match_id", "team_a"]], left_on=["match_id", "team"],
        right_on=["match_id", "team_a"], how="inner",
    )
    side_b = long.merge(
        base[["match_id", "team_b"]], left_on=["match_id", "team"],
        right_on=["match_id", "team_b"], how="inner",
    )

    a = side_a.set_index("match_id")
    b = side_b.set_index("match_id")

    feat = pd.DataFrame(index=base["match_id"])
    feat["elo_diff"] = a["team_elo"] - b["team_elo"]
    feat["host_advantage"] = a["is_home"].astype(float)  # A listed as home & not neutral
    for c in _FORM_DIFF_COLS:
        feat[f"{c}_diff"] = a[c] - b[c]

    # Squad / roster diffs — era-correct via the panel if provided, else static.
    if squad_panel is not None:
        yr_min, yr_max = int(squad_panel["year"].min()), int(squad_panel["year"].max())
        match_year = pd.to_datetime(base["date"]).dt.year.clip(yr_min, yr_max)
        pidx = squad_panel.set_index(["team", "year"])
        sa = pidx.reindex(list(zip(base["team_a"], match_year)))
        sb = pidx.reindex(list(zip(base["team_b"], match_year)))
    else:
        squad = squad if squad is not None else load_squad_features()
        sq = squad.set_index("team")
        sa = sq.reindex(base["team_a"].to_numpy())
        sb = sq.reindex(base["team_b"].to_numpy())
    for c in SQUAD_FEATURE_COLS:
        col_a = pd.Series(sa[c].to_numpy(), index=base["match_id"]).fillna(0.0)
        col_b = pd.Series(sb[c].to_numpy(), index=base["match_id"]).fillna(0.0)
        feat[f"{c}_diff"] = col_a - col_b

    # Cross-confederation calibration diff.
    cs_a = base["team_a"].map(conf_strength).fillna(0.0)
    cs_b = base["team_b"].map(conf_strength).fillna(0.0)
    feat["conf_strength_diff"] = pd.Series(cs_a.to_numpy() - cs_b.to_numpy(),
                                           index=base["match_id"])

    feat = feat.reset_index()

    # Labels + meta.
    feat["date"] = base["date"]
    feat["team_a"] = base["team_a"]
    feat["team_b"] = base["team_b"]
    feat["importance"] = base["importance"]
    feat["neutral"] = base["neutral"]
    feat["team_a_goals"] = base["team_a_goals"]
    feat["team_b_goals"] = base["team_b_goals"]
    feat["y"] = base["result"].map(RESULT_TO_CLASS)

    return feat


def feature_columns(feat: pd.DataFrame) -> list[str]:
    """The model input columns (everything that is a diff / advantage term)."""
    meta = {
        "match_id", "date", "team_a", "team_b", "importance", "neutral",
        "team_a_goals", "team_b_goals", "y",
    }
    return [c for c in feat.columns if c not in meta]
