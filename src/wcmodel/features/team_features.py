"""Per-team, per-match form features — strictly leakage-free.

Every feature for a match uses only that team's *prior* matches (we shift by one
within each team before any rolling/EWM aggregation). Two flavours are produced:

* last-10 rolling means  -> "recent form"
* day-aware EWM means    -> "time-decayed 4-year form" (spec §4B), where older and
  lower-importance matches count less.

Output is a long table: one row per (match_id, team) with the team's pre-match
state. `match_features` pivots this into A-vs-B diffs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wcmodel import config
from wcmodel.ingest.elo import compute_elo, expected_score

# Importance weights used to scale how much a result informs "form".
_IMPORTANCE_FORM_WEIGHT = {
    "world_cup": 1.0,
    "continental": 0.9,
    "qualifier": 0.8,
    "minor_tournament": 0.6,
    "friendly": 0.4,
}


def with_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """Attach leakage-safe pre-match Elo (elo_a, elo_b) and a stable match_id."""
    df = matches.sort_values("date", kind="stable").reset_index(drop=True)
    history, _ = compute_elo(df)
    if len(history) != len(df):
        raise RuntimeError("Elo history misaligned with matches")
    df = df.copy()
    df["match_id"] = np.arange(len(df))
    df["elo_a"] = history["elo_a"].to_numpy()
    df["elo_b"] = history["elo_b"].to_numpy()
    return df


def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (match_id, team) from each side's perspective."""
    a = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "date": df["date"],
            "team": df["team_a"],
            "opp": df["team_b"],
            "team_elo": df["elo_a"],
            "opp_elo": df["elo_b"],
            "gf": df["team_a_goals"],
            "ga": df["team_b_goals"],
            "neutral": df["neutral"],
            "importance": df["importance"],
            "is_home": ~df["neutral"],
        }
    )
    b = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "date": df["date"],
            "team": df["team_b"],
            "opp": df["team_a"],
            "team_elo": df["elo_b"],
            "opp_elo": df["elo_a"],
            "gf": df["team_b_goals"],
            "ga": df["team_a_goals"],
            "neutral": df["neutral"],
            "importance": df["importance"],
            "is_home": False,  # away side never gets home advantage
        }
    )
    long = pd.concat([a, b], ignore_index=True)
    long["points"] = np.where(
        long["gf"] > long["ga"], config.POINTS_WIN,
        np.where(long["gf"] == long["ga"], config.POINTS_DRAW, config.POINTS_LOSS),
    )
    long["win"] = (long["gf"] > long["ga"]).astype(float)
    long["gd"] = long["gf"] - long["ga"]
    long["imp_w"] = long["importance"].map(_IMPORTANCE_FORM_WEIGHT).fillna(0.5)
    # Expected points from Elo alone (rough): 3 * P(win-ish).
    ha = np.where(long["is_home"], config.ELO_HOME_ADVANTAGE, 0.0)
    we = 1.0 / (1.0 + 10 ** (-(long["team_elo"] - long["opp_elo"] + ha) / 400.0))
    long["exp_points"] = 3.0 * we
    long["pts_minus_exp"] = long["points"] - long["exp_points"]
    long["gd_minus_exp"] = long["gd"] - (long["team_elo"] - long["opp_elo"] + ha) / 200.0
    return long


def build_team_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Return the long per-(match_id, team) feature table."""
    df = with_elo(matches)
    long = _to_long(df).sort_values(["team", "date", "match_id"], kind="stable")

    g = long.groupby("team", sort=False)

    def roll_mean(col: str, window: int = 10) -> pd.Series:
        # shift(1) excludes the current match -> no leakage.
        return g[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

    long["gf_l10"] = roll_mean("gf")
    long["ga_l10"] = roll_mean("ga")
    long["gd_l10"] = roll_mean("gd")
    long["winrate_l10"] = roll_mean("win")
    long["sos_l10"] = roll_mean("opp_elo")          # strength of schedule
    long["pts_vs_exp_l10"] = roll_mean("pts_minus_exp")
    long["gd_vs_exp_l10"] = roll_mean("gd_minus_exp")

    # Day-aware EWM, importance-weighted: the "4-year decayed form".
    hl = pd.Timedelta(days=config.FORM_HALFLIFE_DAYS)

    def ewm_decayed(value_col: str) -> pd.Series:
        def f(sub: pd.DataFrame) -> pd.Series:
            v = (sub[value_col] * sub["imp_w"]).shift(1)
            return v.ewm(halflife=hl, times=sub["date"]).mean()
        return g[[value_col, "imp_w", "date"]].apply(f).reset_index(level=0, drop=True)

    long["gd_decayed"] = ewm_decayed("gd")
    long["gf_decayed"] = ewm_decayed("gf")
    long["winrate_decayed"] = ewm_decayed("win")

    feature_cols = [
        "gf_l10", "ga_l10", "gd_l10", "winrate_l10", "sos_l10",
        "pts_vs_exp_l10", "gd_vs_exp_l10",
        "gd_decayed", "gf_decayed", "winrate_decayed",
    ]
    long[feature_cols] = long[feature_cols].fillna(0.0)

    keep = ["match_id", "team", "team_elo", "is_home"] + feature_cols
    return long[keep].sort_values("match_id", kind="stable").reset_index(drop=True)


_STATE_COLS = [
    "gf_l10", "ga_l10", "gd_l10", "winrate_l10", "sos_l10",
    "pts_vs_exp_l10", "gd_vs_exp_l10",
    "gd_decayed", "gf_decayed", "winrate_decayed",
]


def current_team_state(matches: pd.DataFrame, ref_date=None) -> pd.DataFrame:
    """One row per team: current Elo + current form, using ALL matches.

    Unlike `build_team_features` (which shifts to avoid leakage for *training*),
    this includes every match up to `ref_date` because we want each team's
    state *now* to feature upcoming fixtures. Returns columns:
        team, team_elo, <_STATE_COLS...>
    """
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    if ref_date is not None:
        df = df[df["date"] <= pd.Timestamp(ref_date)]

    df = with_elo(df)
    _, ratings = compute_elo(df.sort_values("date", kind="stable"))
    long = _to_long(df).sort_values(["team", "date"], kind="stable")

    g = long.groupby("team", sort=False)
    hl = pd.Timedelta(days=config.FORM_HALFLIFE_DAYS)

    rows = []
    for team, sub in g:
        last10 = sub.tail(10)
        rows.append(
            {
                "team": team,
                "team_elo": ratings.get(team, config.ELO_START),
                "gf_l10": last10["gf"].mean(),
                "ga_l10": last10["ga"].mean(),
                "gd_l10": last10["gd"].mean(),
                "winrate_l10": last10["win"].mean(),
                "sos_l10": last10["opp_elo"].mean(),
                "pts_vs_exp_l10": last10["pts_minus_exp"].mean(),
                "gd_vs_exp_l10": last10["gd_minus_exp"].mean(),
                "gd_decayed": (sub["gd"] * sub["imp_w"]).ewm(halflife=hl, times=sub["date"]).mean().iloc[-1],
                "gf_decayed": (sub["gf"] * sub["imp_w"]).ewm(halflife=hl, times=sub["date"]).mean().iloc[-1],
                "winrate_decayed": (sub["win"] * sub["imp_w"]).ewm(halflife=hl, times=sub["date"]).mean().iloc[-1],
            }
        )
    state = pd.DataFrame(rows)
    return state.fillna(0.0)
