"""Synthetic international-match generator.

Produces a canonical `matches` frame with real signal so the entire pipeline
(Elo -> features -> Poisson/GBM -> simulation -> markets) runs end-to-end before
any real data is downloaded. Goals are drawn from a Poisson model whose rates
depend on latent team strengths, so a well-built model *should* recover those
strengths — a useful sanity check on the modelling code itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wcmodel.teams2026 import ALL_TEAMS_2026, SEED_ELO

# A few non-2026 teams so the training pool isn't exactly the tournament field.
_EXTRA_TEAMS = {
    "Chile": 1800, "Scotland": 1780, "Hungary": 1770, "Wales": 1760,
    "Greece": 1750, "Mali": 1740, "Venezuela": 1730, "Bolivia": 1690,
    "Iraq": 1680, "South Africa": 1730, "Czechia": 1790, "Romania": 1770,
}

_IMPORTANCES = ["friendly", "qualifier", "continental", "world_cup", "minor_tournament"]
_IMPORTANCE_P = [0.42, 0.34, 0.10, 0.04, 0.10]


def _strength_to_attack(elo: float, base: float = 1900.0) -> float:
    """Map an Elo-like anchor to an attacking-rate offset (in log space)."""
    return (elo - base) / 220.0


def generate_matches(
    *,
    start: str = "2014-01-01",
    end: str = "2026-05-31",
    n_matches: int = 9000,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a canonical matches DataFrame."""
    rng = np.random.default_rng(seed)

    # Ensure every 2026 qualifier has synthetic history, even if not seeded.
    anchors = {**{t: 1700.0 for t in ALL_TEAMS_2026}, **SEED_ELO, **_EXTRA_TEAMS}
    teams = list(anchors)
    # Latent attack/defense per team (defense as a positive "concedes-less" term).
    attack = {t: _strength_to_attack(anchors[t]) for t in teams}
    defense = {t: _strength_to_attack(anchors[t]) * 0.9 for t in teams}

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    span_days = (end_ts - start_ts).days

    rows = []
    for _ in range(n_matches):
        a, b = rng.choice(teams, size=2, replace=False)
        day = start_ts + pd.Timedelta(days=int(rng.integers(0, span_days + 1)))
        importance = rng.choice(_IMPORTANCES, p=_IMPORTANCE_P)
        neutral = bool(rng.random() < 0.35)
        home_adv = 0.0 if neutral else 0.25  # log-rate home bump

        base = 0.2  # global log baseline (~1.2 goals/side)
        lam_a = np.exp(base + attack[a] - defense[b] + home_adv)
        lam_b = np.exp(base + attack[b] - defense[a])

        ga = int(rng.poisson(max(lam_a, 0.05)))
        gb = int(rng.poisson(max(lam_b, 0.05)))

        rows.append(
            {
                "date": day,
                "team_a": a,
                "team_b": b,
                "team_a_goals": ga,
                "team_b_goals": gb,
                "tournament": importance.replace("_", " ").title(),
                "neutral": neutral,
                "importance": importance,
                "result": "team_a_win" if ga > gb else ("team_b_win" if ga < gb else "draw"),
            }
        )

    df = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
    return df


def true_strengths() -> pd.DataFrame:
    """The latent anchors used to generate data — for validating recovery."""
    anchors = {**SEED_ELO, **_EXTRA_TEAMS}
    return (
        pd.DataFrame({"team": list(anchors), "true_anchor": list(anchors.values())})
        .sort_values("true_anchor", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    m = generate_matches()
    print(m.head())
    print(f"\n{len(m):,} synthetic matches, {m['team_a'].nunique()} teams, "
          f"{m['date'].min().date()}..{m['date'].max().date()}")
