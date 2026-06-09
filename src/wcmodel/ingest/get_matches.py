"""Load + normalize international match results into the canonical schema.

Canonical columns used everywhere downstream:
    date, team_a, team_b, team_a_goals, team_b_goals, tournament,
    neutral, importance, result

`importance` is one of config.ELO_K_BY_IMPORTANCE keys.

Primary source: the Kaggle "International football results from 1872" dataset
(martj42/international-football-results-from-1872-to-2017), whose `results.csv`
has columns: date, home_team, away_team, home_score, away_score, tournament,
city, country, neutral. Drop that file at data/raw/results.csv.
"""
from __future__ import annotations

import re

import pandas as pd

from wcmodel import config

# Map free-text `tournament` strings to Elo importance weight classes.
# Order matters: first regex that matches wins.
_IMPORTANCE_RULES: list[tuple[str, str]] = [
    (r"friendly", "friendly"),
    (r"world cup.*qualif", "qualifier"),
    (r"world cup", "world_cup"),
    (r"(uefa euro|copa am[eé]rica|african cup|afc asian cup|gold cup|confederations).*qualif", "qualifier"),
    (r"(uefa euro|copa am[eé]rica|african cup|afc asian cup|gold cup|confederations)", "continental"),
    (r"qualif", "qualifier"),
    (r"nations league", "minor_tournament"),
]


def tournament_to_importance(name: str) -> str:
    s = str(name).lower()
    for pattern, importance in _IMPORTANCE_RULES:
        if re.search(pattern, s):
            return importance
    return "minor_tournament"


def _result(a: int, b: int) -> str:
    return "team_a_win" if a > b else ("team_b_win" if a < b else "draw")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map a raw results frame onto the canonical schema."""
    colmap = {
        "home_team": "team_a",
        "away_team": "team_b",
        "home_score": "team_a_goals",
        "away_score": "team_b_goals",
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns}).copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["team_a_goals", "team_b_goals"])
    df["team_a_goals"] = df["team_a_goals"].astype(int)
    df["team_b_goals"] = df["team_b_goals"].astype(int)

    if "neutral" in df.columns:
        df["neutral"] = df["neutral"].astype(bool)
    else:
        df["neutral"] = False

    if "tournament" not in df.columns:
        df["tournament"] = "Friendly"
    df["importance"] = df["tournament"].map(tournament_to_importance)

    df["result"] = [
        _result(a, b) for a, b in zip(df["team_a_goals"], df["team_b_goals"])
    ]

    keep = [
        "date", "team_a", "team_b", "team_a_goals", "team_b_goals",
        "tournament", "neutral", "importance", "result",
    ]
    return df[keep].sort_values("date", kind="stable").reset_index(drop=True)


def load_matches(path=None) -> pd.DataFrame:
    """Load canonical matches; fall back to synthetic demo data if absent."""
    path = path or (config.RAW / "results.csv")
    if not path.exists():
        from wcmodel.ingest.synthetic import generate_matches

        return generate_matches()
    return normalize(pd.read_csv(path))
