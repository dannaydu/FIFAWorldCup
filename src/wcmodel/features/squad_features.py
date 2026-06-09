"""Squad value + roster-continuity features (spec §4D, §4E, §6).

Source priority (first available wins):
  1. data/raw/players.csv  — the Transfermarkt "player-scores" dataset (real
     market values; auto-downloadable with no auth, see scripts/fetch_squads.py).
     Squad value per nation = its top players by market value (a talent-pool
     proxy that is arguably better than the literal 26-man list).
  2. data/raw/squads.csv   — a hand-provided squad snapshot (schema below).
  3. synthetic             — values inferred from Elo so the pipeline still runs.

squads.csv schema:
    snapshot_date, team, player, position, club, age, caps, goals,
    market_value, minutes_last_365, injury_status, is_projected_starter
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wcmodel.teams2026 import SEED_ELO, canonicalize

# Transfermarkt citizenship spelling -> our canonical names.
_TM_TO_CANONICAL = {
    "Korea, South": "South Korea",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Curacao": "Curaçao",
    "Cote d'Ivoire": "Ivory Coast",
}

# Columns every consumer can rely on existing.
SQUAD_FEATURE_COLS = [
    "squad_value",        # total squad market value (proxy, €m)
    "top11_value",        # starting-XI value (proxy, €m)
    "avg_age",
    "avg_caps",
    "roster_continuity",  # share of projected XI that were 2022 regulars
    "injury_score",       # availability penalty (0 = full strength)
]


def _synthetic_team_row(team: str, elo: float, rng: np.random.Generator) -> dict:
    # Higher Elo -> exponentially higher market value, with noise.
    strength = (elo - 1650) / 100.0
    squad_value = float(np.clip(np.exp(strength) * 60 + rng.normal(0, 40), 20, 1400))
    top11_value = squad_value * float(rng.uniform(0.55, 0.72))
    return {
        "team": team,
        "squad_value": round(squad_value, 1),
        "top11_value": round(top11_value, 1),
        "avg_age": round(float(rng.uniform(25.5, 28.5)), 1),
        "avg_caps": round(float(np.clip(strength * 9 + rng.uniform(20, 45), 10, 90)), 1),
        "roster_continuity": round(float(np.clip(rng.normal(0.6, 0.15), 0.1, 0.95)), 2),
        "injury_score": round(float(np.clip(rng.exponential(0.08), 0, 0.6)), 3),
    }


def synthetic_squad_features(seed: int = 11,
                             team_elo: dict[str, float] | None = None) -> pd.DataFrame:
    """Synthesise team-level squad features from Elo anchors.

    `team_elo` (e.g. computed Elo for every team in the data) is preferred so
    that all real qualifiers get a row; falls back to SEED_ELO for the pure demo.
    """
    rng = np.random.default_rng(seed)
    anchors = team_elo or SEED_ELO
    rows = [_synthetic_team_row(t, float(e), rng) for t, e in anchors.items()]
    return pd.DataFrame(rows)


def transfermarkt_squad_features(players_csv, *, squad_n: int = 26,
                                 ref_date: str = "2026-06-01",
                                 min_last_season: int = 2023) -> pd.DataFrame:
    """Build team-level squad features from the Transfermarkt players.csv.

    For each nation we take its `squad_n` most valuable active players (by
    citizenship) as a squad proxy and aggregate value / age / caps.
    """
    p = pd.read_csv(players_csv)
    p = p[p["market_value_in_eur"].notna()]
    if "last_season" in p.columns:
        p = p[p["last_season"].fillna(0) >= min_last_season]
    p["team"] = p["country_of_citizenship"].map(
        lambda c: canonicalize(_TM_TO_CANONICAL.get(c, c)) if pd.notna(c) else c
    )
    ref = pd.Timestamp(ref_date)
    p["age"] = (ref - pd.to_datetime(p["date_of_birth"], errors="coerce")).dt.days / 365.25
    p["caps"] = pd.to_numeric(p.get("international_caps"), errors="coerce").fillna(0)
    p["mv_m"] = p["market_value_in_eur"] / 1e6

    rows = []
    for team, g in p.groupby("team"):
        g = g.sort_values("market_value_in_eur", ascending=False)
        squad = g.head(squad_n)
        top11 = g.head(11)
        rows.append({
            "team": team,
            "squad_value": round(float(squad["mv_m"].sum()), 1),
            "top11_value": round(float(top11["mv_m"].sum()), 1),
            "avg_age": round(float(squad["age"].mean()), 1),
            "avg_caps": round(float(squad["caps"].mean()), 1),
            "roster_continuity": 0.6,   # needs a prior-cycle snapshot to compute
            "injury_score": 0.0,        # needs a live injury feed
        })
    return pd.DataFrame(rows)[["team"] + SQUAD_FEATURE_COLS]


def _canon_citizenship(c):
    return canonicalize(_TM_TO_CANONICAL.get(c, c)) if pd.notna(c) else c


def build_squad_panel(valuations_csv, players_csv, current_squad: pd.DataFrame,
                      *, years=range(2005, 2027), squad_n: int = 26) -> pd.DataFrame:
    """Per-(team, year) squad value/age, so historical matches get era-correct
    valuations instead of a single 2026 snapshot.

    caps / roster_continuity / injury_score are not historically available, so
    they are broadcast from `current_squad` (the latest snapshot).
    """
    v = pd.read_csv(valuations_csv, usecols=["player_id", "date", "market_value_in_eur"])
    v = v.dropna(subset=["market_value_in_eur"])
    v["date"] = pd.to_datetime(v["date"], errors="coerce")
    v = v.sort_values("date")

    p = pd.read_csv(players_csv, usecols=["player_id", "country_of_citizenship", "date_of_birth"])
    p["team"] = p["country_of_citizenship"].map(_canon_citizenship)
    p["dob"] = pd.to_datetime(p["date_of_birth"], errors="coerce")
    pmap = p.set_index("player_id")[["team", "dob"]]

    rows = []
    for year in years:
        ref = pd.Timestamp(f"{year}-06-01")
        latest = v[v["date"] <= ref].groupby("player_id", sort=False).tail(1)
        latest = latest.join(pmap, on="player_id").dropna(subset=["team"])
        latest["age"] = (ref - latest["dob"]).dt.days / 365.25
        latest["mv_m"] = latest["market_value_in_eur"] / 1e6
        for team, g in latest.groupby("team"):
            g = g.sort_values("market_value_in_eur", ascending=False)
            squad, top11 = g.head(squad_n), g.head(11)
            rows.append({
                "team": team, "year": year,
                "squad_value": round(float(squad["mv_m"].sum()), 1),
                "top11_value": round(float(top11["mv_m"].sum()), 1),
                "avg_age": round(float(squad["age"].mean()), 1),
            })
    panel = pd.DataFrame(rows)

    # broadcast the non-time-varying columns from the current snapshot
    cur = current_squad.set_index("team")[["avg_caps", "roster_continuity", "injury_score"]]
    panel = panel.join(cur, on="team")
    panel[["avg_caps", "roster_continuity", "injury_score"]] = \
        panel[["avg_caps", "roster_continuity", "injury_score"]].fillna(0.0)
    return panel[["team", "year"] + SQUAD_FEATURE_COLS]


def load_squad_panel(team_elo: dict[str, float] | None = None) -> pd.DataFrame | None:
    """Time-varying squad panel if the Transfermarkt files are present, else None."""
    from wcmodel import config

    val = config.RAW / "player_valuations.csv"
    pl = config.RAW / "players.csv"
    if not (val.exists() and pl.exists()):
        return None
    current = load_squad_features(team_elo=team_elo)
    return build_squad_panel(val, pl, current)


def load_squad_features(path=None, team_elo: dict[str, float] | None = None) -> pd.DataFrame:
    """Real Transfermarkt squad features if available, else a snapshot, else synth.

    Synthetic values backfill any team missing from the real source so every
    qualifier has a row.
    """
    from wcmodel import config

    tm_path = config.RAW / "players.csv"
    if tm_path.exists():
        real = transfermarkt_squad_features(tm_path)
        if team_elo:  # backfill missing teams with synthetic-from-Elo
            have = set(real["team"])
            missing = {t: e for t, e in team_elo.items() if t not in have}
            if missing:
                real = pd.concat([real, synthetic_squad_features(team_elo=missing)],
                                 ignore_index=True)
        return real

    path = path or (config.RAW / "squads.csv")
    if not path.exists():
        return synthetic_squad_features(team_elo=team_elo)

    s = pd.read_csv(path)
    s["market_value"] = pd.to_numeric(s.get("market_value"), errors="coerce")
    starters = s[s.get("is_projected_starter", False).astype(bool)]
    grp = s.groupby("team")
    out = pd.DataFrame(
        {
            "squad_value": grp["market_value"].sum() / 1e6,
            "avg_age": grp["age"].mean(),
            "avg_caps": grp["caps"].mean(),
        }
    )
    out["top11_value"] = (
        starters.groupby("team")["market_value"].sum().reindex(out.index).fillna(0) / 1e6
    )
    out["roster_continuity"] = 0.6  # requires a 2022 snapshot to compute for real
    if "injury_status" in s.columns:
        inj = s.assign(out_flag=(s["injury_status"].astype(str) != "fit"))
        out["injury_score"] = inj.groupby("team")["out_flag"].mean()
    else:
        out["injury_score"] = 0.0
    return out.reset_index()[["team"] + SQUAD_FEATURE_COLS]
