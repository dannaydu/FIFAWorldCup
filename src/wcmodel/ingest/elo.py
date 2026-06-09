"""World Football Elo ratings computed from a match history.

Implements the eloratings.net update rule:

    R_new = R_old + K * G * (W - W_e)

where
    W_e = 1 / (1 + 10 ** (-dr / 400))           # expected score
    dr  = R_home - R_away + home_advantage      # rating diff (home POV)
    K   = importance weight (see config.ELO_K_BY_IMPORTANCE)
    G   = goal-difference multiplier            # see `goal_multiplier`
    W   = actual result in {1.0 win, 0.5 draw, 0.0 loss}

This is the single strongest non-market baseline feature in the whole project,
so it is implemented from scratch and tested rather than pulled from a library.
"""
from __future__ import annotations

import pandas as pd

from wcmodel import config


def expected_score(elo_a: float, elo_b: float, home_advantage: float = 0.0) -> float:
    """P(A scores the 'win') given ratings; `home_advantage` favours A."""
    dr = elo_a - elo_b + home_advantage
    return 1.0 / (1.0 + 10 ** (-dr / 400.0))


def goal_multiplier(goal_diff: int) -> float:
    """eloratings.net goal-difference weighting G."""
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    if gd == 3:
        return 1.75
    # 4+ : 1.75 + (gd - 3) / 8
    return 1.75 + (gd - 3) / 8.0


def _result_score(goals_a: int, goals_b: int) -> float:
    if goals_a > goals_b:
        return 1.0
    if goals_a < goals_b:
        return 0.0
    return 0.5


def compute_elo(
    matches: pd.DataFrame,
    *,
    start_rating: float = config.ELO_START,
    home_advantage: float = config.ELO_HOME_ADVANTAGE,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Walk the match history chronologically and update ratings.

    Parameters
    ----------
    matches : DataFrame with columns
        date, team_a, team_b, team_a_goals, team_b_goals, neutral, importance
        (`importance` must be a key of config.ELO_K_BY_IMPORTANCE; `neutral`
        is a bool — when True no home advantage is applied).

    Returns
    -------
    (history, ratings)
        history : the input rows plus pre-match `elo_a` / `elo_b` and the
                  post-match deltas (handy as leakage-free model features).
        ratings : final {team: rating} dict.
    """
    required = {"date", "team_a", "team_b", "team_a_goals", "team_b_goals"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches is missing columns: {sorted(missing)}")

    df = matches.sort_values("date", kind="stable").reset_index(drop=True)
    ratings: dict[str, float] = {}

    rows = []
    for r in df.itertuples(index=False):
        ra = ratings.get(r.team_a, start_rating)
        rb = ratings.get(r.team_b, start_rating)

        neutral = bool(getattr(r, "neutral", False))
        ha = 0.0 if neutral else home_advantage

        importance = getattr(r, "importance", config.ELO_DEFAULT_IMPORTANCE)
        k = config.ELO_K_BY_IMPORTANCE.get(importance, config.ELO_K_BY_IMPORTANCE[config.ELO_DEFAULT_IMPORTANCE])

        ga, gb = int(r.team_a_goals), int(r.team_b_goals)
        w = _result_score(ga, gb)
        we = expected_score(ra, rb, ha)
        g = goal_multiplier(ga - gb)

        delta = k * g * (w - we)
        ratings[r.team_a] = ra + delta
        ratings[r.team_b] = rb - delta

        rows.append(
            {
                "date": r.date,
                "team_a": r.team_a,
                "team_b": r.team_b,
                "elo_a": ra,          # pre-match -> safe as a feature
                "elo_b": rb,
                "elo_a_post": ra + delta,
                "elo_b_post": rb - delta,
                "elo_delta": delta,
            }
        )

    history = pd.DataFrame(rows)
    return history, ratings


def ratings_as_of(matches: pd.DataFrame, as_of, **kwargs) -> dict[str, float]:
    """Ratings using only matches strictly before `as_of` (leakage-safe)."""
    as_of = pd.Timestamp(as_of)
    past = matches[pd.to_datetime(matches["date"]) < as_of]
    _, ratings = compute_elo(past, **kwargs)
    return ratings
