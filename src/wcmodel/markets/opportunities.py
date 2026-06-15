"""Assemble model-vs-market betting opportunities across tournament market types.

Joins model probabilities to live market contracts, de-vigs the market
(mutually-exclusive sets normalized; binaries passed through), shrinks the model
toward the market prior, and returns filtered, sided opportunities for the ledger.

Market types:
  group_winner / champion / reach_round  -> from the tournament simulation `sim`
  match (KXWCGAME 3-way W/D/L)            -> from `predictor.predict_wdl` per fixture
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from wcmodel import config
from wcmodel.ingest.get_market_prices import _kalshi_round
from wcmodel.markets.shrink import blend_to_market
from wcmodel.teams2026 import HOST_NATIONS, canonicalize

# reach-round label -> simulated probability column
_ROUND_TO_COL = {
    "round_32": "p_round_32", "round_16": "p_round_16",
    "quarterfinal": "p_quarterfinal", "semifinal": "p_semifinal",
    "final": "p_final", "champion": "p_champion",
}

# "Qatar vs Switzerland Winner?" -> ("Qatar", "Switzerland")
_FIXTURE_RE = re.compile(r"^(.+?)\s+vs\.?\s+(.+?)\s+Winner", re.I)


def _parse_fixture(contract: str) -> tuple[str | None, str | None]:
    m = _FIXTURE_RE.match(str(contract))
    if not m:
        return None, None
    return canonicalize(m.group(1).strip()), canonicalize(m.group(2).strip())


def _fixture_key(market_id: str) -> str:
    # KXWCGAME-26JUN13QATSUI-SUI -> KXWCGAME-26JUN13QATSUI
    return str(market_id).rsplit("-", 1)[0]


def _devig(market: pd.DataFrame) -> pd.DataFrame:
    """Add `market_prob`: normalize mutually-exclusive sets, else pass midpoint."""
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

    # match: 3-way (team A / Tie / team B) -> normalize within each fixture
    mt = m["market_type"] == "match"
    if mt.any():
        fk = m.loc[mt, "market_id"].map(_fixture_key)
        m.loc[mt, "market_prob"] = (
            m.loc[mt].assign(_fk=fk).groupby("_fk")["midpoint"].transform(lambda s: s / s.sum())
        )
    return m


def _model_prob(simi, row, predictor, cache) -> float | None:
    mt = row.market_type
    if mt == "group_winner":
        return float(simi.loc[row.team, "p_group_winner"]) if row.team in simi.index else None
    if mt == "champion":
        return float(simi.loc[row.team, "p_champion"]) if row.team in simi.index else None
    if mt == "reach_round":
        col = _ROUND_TO_COL.get(_kalshi_round(str(row.market_id)))
        if col and row.team in simi.index:
            return float(simi.loc[row.team, col])
        return None
    if mt == "match":
        if predictor is None:
            return None
        a, b = _parse_fixture(row.contract)
        if not a or not b:
            return None
        wdl = cache.get((a, b))
        if wdl is None:
            wdl = predictor.predict_wdl(a, b, neutral=True, host_a=a in HOST_NATIONS)
            cache[(a, b)] = wdl
        yt = row.team
        if yt in ("Tie", "Draw"):
            return float(wdl[1])
        if yt == a:
            return float(wdl[0])
        if yt == b:
            return float(wdl[2])
        return None
    return None


def build_opportunities(
    sim: pd.DataFrame,
    market: pd.DataFrame,
    *,
    predictor=None,
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
    cache: dict = {}

    rows = []
    for r in m.itertuples(index=False):
        mp = _model_prob(simi, r, predictor, cache)
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
