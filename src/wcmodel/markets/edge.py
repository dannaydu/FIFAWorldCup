"""Market-edge engine (spec §10).

Compares model probabilities to market prices and surfaces only contracts where
the edge clears the configured thresholds on edge size, spread, and liquidity.
Picks the side (YES if model > market, NO otherwise) and reports the directional
edge for that side.
"""
from __future__ import annotations

import pandas as pd

from wcmodel import config
from wcmodel.markets.implied import implied_from_book


def find_edges(
    model_probs: pd.DataFrame,   # columns: contract, model_prob  (+ optional team, market_type)
    market: pd.DataFrame,        # columns: contract, midpoint/bid/ask, spread, liquidity
    *,
    min_edge: float = config.MIN_EDGE,
    max_spread: float = config.MAX_SPREAD,
    min_liquidity: float = config.MIN_LIQUIDITY,
) -> pd.DataFrame:
    """Return ranked, filtered betting opportunities."""
    mk = implied_from_book(market)
    df = model_probs.merge(mk, on="contract", how="inner", suffixes=("", "_mkt"))

    if "liquidity" not in df.columns:
        df["liquidity"] = float("inf")
    if "spread" not in df.columns:
        df["spread"] = 0.0

    # Directional edge: back YES if undervalued, NO if overvalued.
    df["yes_edge"] = df["model_prob"] - df["market_prob"]
    df["side"] = df["yes_edge"].apply(lambda e: "YES" if e >= 0 else "NO")
    df["edge"] = df["yes_edge"].abs()
    # Price you would pay for the chosen side.
    df["entry_price"] = df.apply(
        lambda r: r["market_prob"] if r["side"] == "YES" else 1.0 - r["market_prob"],
        axis=1,
    )
    # Model prob of the chosen side winning.
    df["model_prob_side"] = df.apply(
        lambda r: r["model_prob"] if r["side"] == "YES" else 1.0 - r["model_prob"],
        axis=1,
    )

    keep = (
        (df["edge"] >= min_edge)
        & (df["spread"] <= max_spread)
        & (df["liquidity"] >= min_liquidity)
    )
    cols = [
        "contract", "side", "model_prob", "market_prob", "edge",
        "entry_price", "model_prob_side", "spread", "liquidity",
    ]
    extra = [c for c in ("team", "market_type", "platform") if c in df.columns]
    out = df.loc[keep, extra + cols].sort_values("edge", ascending=False)
    return out.reset_index(drop=True)
