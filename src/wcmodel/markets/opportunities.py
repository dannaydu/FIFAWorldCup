"""Assemble model-vs-market betting opportunities across tournament market types.

Joins simulated tournament probabilities to live market contracts, de-vigs the
market (mutually-exclusive sets normalized; binaries passed through), shrinks the
model toward the market prior, and returns filtered, sided opportunities ready
for the paper-trading ledger.

Supported market types: group_winner, champion, reach_round.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wcmodel import config
from wcmodel.ingest.get_market_prices import _kalshi_round
from wcmodel.markets.shrink import blend_to_market

# reach-round label -> simulated probability column
_ROUND_TO_COL = {
    "round_32": "p_round_32", "round_16": "p_round_16",
    "quarterfinal": "p_quarterfinal", "semifinal": "p_semifinal",
    "final": "p_final", "champion": "p_champion",
}


def _devig(market: pd.DataFrame) -> pd.DataFrame:
    """Add a `market_prob` column: normalize mutually-exclusive sets, else midpoint."""
    m = market.copy()
    m["market_prob"] = m["midpoint"]
    gw = m["market_type"] == "group_winner"
    if gw.any() and "group" in m.columns:
        m.loc[gw, "market_prob"] = (
            m.loc[gw].groupby("group")["midpoint"].transform(lambda s: s / s.sum())
        )
    ch = m["market_type"] == "champion"
    if ch.any():
        tot = m.loc[ch, "midpoint"].sum()
        if tot > 0:
            m.loc[ch, "market_prob"] = m.loc[ch, "midpoint"] / tot
    return m


def _model_prob(simi: pd.DataFrame, row) -> float | None:
    team, mt = row.team, row.market_type
    if team not in simi.index:
        return None
    if mt == "group_winner":
        return float(simi.loc[team, "p_group_winner"])
    if mt == "champion":
        return float(simi.loc[team, "p_champion"])
    if mt == "reach_round":
        rnd = _kalshi_round(str(row.market_id))
        col = _ROUND_TO_COL.get(rnd)
        return float(simi.loc[team, col]) if col else None
    return None


def build_opportunities(
    sim: pd.DataFrame,
    market: pd.DataFrame,
    *,
    lam: float = config.MARKET_SHRINKAGE_LAMBDA,
    min_edge: float = config.MIN_EDGE,
    max_spread: float = config.MAX_SPREAD,
    min_liquidity: float = config.MIN_LIQUIDITY,
) -> pd.DataFrame:
    """Return filtered, sided opportunities with shrunk edges."""
    if market.empty:
        return pd.DataFrame()
    m = _devig(market.dropna(subset=["team", "midpoint"]).copy())
    simi = sim.set_index("team")

    rows = []
    for r in m.itertuples(index=False):
        mp = _model_prob(simi, r)
        if mp is None or not (0 <= r.market_prob <= 1):
            continue
        blended = float(blend_to_market(mp, r.market_prob, lam))
        edge_yes = blended - r.market_prob
        side = "YES" if edge_yes >= 0 else "NO"
        entry_price = r.market_prob if side == "YES" else 1.0 - r.market_prob
        model_side = blended if side == "YES" else 1.0 - blended
        rows.append({
            "platform": r.platform,
            "market_id": r.market_id,
            "contract": r.contract,
            "team": r.team,
            "market_type": r.market_type,
            "round": _kalshi_round(str(r.market_id)) if r.market_type == "reach_round" else None,
            "side": side,
            "model_prob": round(mp, 4),
            "market_prob": round(float(r.market_prob), 4),
            "blended": round(blended, 4),
            "edge": round(abs(blended - r.market_prob), 4),
            "entry_price": round(float(entry_price), 4),
            "model_prob_side": round(float(model_side), 4),
            "midpoint": round(float(r.midpoint), 4),
            "spread": None if pd.isna(r.spread) else round(float(r.spread), 4),
            "liquidity": None if pd.isna(r.liquidity) else float(r.liquidity),
        })
    opps = pd.DataFrame(rows)
    if opps.empty:
        return opps

    keep = opps["edge"] >= min_edge
    keep &= opps["liquidity"].fillna(np.inf) >= min_liquidity
    keep &= opps["spread"].fillna(0.0) <= max_spread
    return opps[keep].sort_values("edge", ascending=False).reset_index(drop=True)
