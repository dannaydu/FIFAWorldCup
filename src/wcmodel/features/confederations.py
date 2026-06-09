"""Confederation membership + empirical cross-confederation strength offsets.

National teams mostly play within their own confederation, so Elo can drift out
of calibration *between* confederations (a side that farms weak regional
opponents looks stronger than it is at a World Cup). This module measures that
directly: over inter-confederation matches only, how much each confederation
over/under-performs its Elo expectation. The per-team offset becomes a feature
(`conf_strength`) the models can use to discount inflated ratings.

Offset sign: positive => that confederation OUTperforms its Elo cross-confed
(Elo underrates it); negative => it underperforms (Elo overrates it).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from wcmodel import config
from wcmodel.ingest.elo import compute_elo

CONFEDERATION: dict[str, str] = {}


def _add(conf: str, teams: list[str]) -> None:
    for t in teams:
        CONFEDERATION[t] = conf


_add("UEFA", [
    "Spain", "France", "England", "Germany", "Netherlands", "Belgium", "Portugal",
    "Italy", "Croatia", "Switzerland", "Austria", "Norway", "Sweden", "Denmark",
    "Poland", "Serbia", "Scotland", "Czech Republic", "Turkey", "Ukraine", "Wales",
    "Hungary", "Greece", "Romania", "Bosnia and Herzegovina", "Republic of Ireland",
    "Slovakia", "Slovenia", "Iceland", "Finland", "Russia", "North Macedonia",
    "Albania", "Northern Ireland", "Bulgaria", "Montenegro", "Georgia", "Israel",
    "Kosovo", "Luxembourg", "Cyprus", "Estonia", "Latvia", "Lithuania", "Armenia",
    "Azerbaijan", "Belarus", "Kazakhstan", "Moldova", "Malta", "Andorra",
])
_add("CONMEBOL", [
    "Brazil", "Argentina", "Uruguay", "Colombia", "Chile", "Peru", "Ecuador",
    "Paraguay", "Venezuela", "Bolivia",
])
_add("CONCACAF", [
    "Mexico", "United States", "Canada", "Costa Rica", "Panama", "Jamaica",
    "Honduras", "Haiti", "Curaçao", "El Salvador", "Trinidad and Tobago",
    "Guatemala", "Suriname", "Nicaragua",
])
_add("CAF", [
    "Morocco", "Senegal", "Egypt", "Nigeria", "Algeria", "Tunisia", "Ghana",
    "Cameroon", "Ivory Coast", "Mali", "South Africa", "DR Congo", "Cape Verde",
    "Burkina Faso", "Guinea", "Zambia", "Angola", "Gabon", "Equatorial Guinea",
    "Mozambique", "Uganda", "Kenya", "Benin", "Madagascar",
])
_add("AFC", [
    "Japan", "South Korea", "Iran", "Saudi Arabia", "Australia", "Qatar", "Iraq",
    "Uzbekistan", "Jordan", "United Arab Emirates", "China PR", "Oman", "Bahrain",
    "Syria", "Vietnam", "Thailand", "Lebanon", "India", "Kuwait", "Palestine",
])
_add("OFC", [
    "New Zealand", "Fiji", "New Caledonia", "Tahiti", "Solomon Islands",
    "Papua New Guinea", "Vanuatu",
])


def confederation_offsets(matches: pd.DataFrame) -> dict[str, float]:
    """Mean Elo-expectation residual per confederation over inter-confed matches."""
    df = matches.sort_values("date", kind="stable").reset_index(drop=True)
    hist, _ = compute_elo(df)
    df = df.copy()
    df["elo_a"] = hist["elo_a"].to_numpy()
    df["elo_b"] = hist["elo_b"].to_numpy()
    df["conf_a"] = df["team_a"].map(CONFEDERATION)
    df["conf_b"] = df["team_b"].map(CONFEDERATION)

    inter = df[df["conf_a"].notna() & df["conf_b"].notna() & (df["conf_a"] != df["conf_b"])]
    if inter.empty:
        return {}

    ha = np.where(inter["neutral"].astype(bool), 0.0, config.ELO_HOME_ADVANTAGE)
    exp_a = 1.0 / (1.0 + 10 ** (-(inter["elo_a"] - inter["elo_b"] + ha) / 400.0))
    act_a = np.where(inter["team_a_goals"] > inter["team_b_goals"], 1.0,
                     np.where(inter["team_a_goals"] == inter["team_b_goals"], 0.5, 0.0))
    res_a = act_a - exp_a.to_numpy()  # team B's residual is exactly -res_a

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for c, r in zip(inter["conf_a"], res_a):
        sums[c] += r; counts[c] += 1
    for c, r in zip(inter["conf_b"], res_a):
        sums[c] += -r; counts[c] += 1

    offsets = {c: sums[c] / counts[c] for c in sums if counts[c] >= 30}
    mean = float(np.mean(list(offsets.values()))) if offsets else 0.0
    return {c: v - mean for c, v in offsets.items()}


def team_conf_strength(matches: pd.DataFrame) -> dict[str, float]:
    """Map each team -> its confederation offset (0 if confederation unknown)."""
    offsets = confederation_offsets(matches)
    return {t: offsets.get(CONFEDERATION.get(t), 0.0)
            for t in pd.concat([matches["team_a"], matches["team_b"]]).unique()}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from wcmodel.ingest.get_matches import load_matches
    off = confederation_offsets(load_matches())
    for c, v in sorted(off.items(), key=lambda kv: -kv[1]):
        print(f"{c:10} {v:+.4f}")
