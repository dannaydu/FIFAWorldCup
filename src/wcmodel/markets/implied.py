"""Convert market prices to (de-vigged) implied probabilities (spec §5, Model 0).

Kalshi/Polymarket binary contracts trade in [0, 1] (cents/100), where the price
is itself the market-implied probability of YES. For a *set* of mutually
exclusive outcomes (e.g. "World Cup winner"), the prices sum to > 1 because of
the overround / market margin; normalizing removes most of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def price_to_prob(price: float) -> float:
    """A single binary YES price is already an implied probability."""
    return float(np.clip(price, 0.0, 1.0))


def devig_mutually_exclusive(prices: pd.Series) -> pd.Series:
    """Normalize a set of mutually exclusive YES prices to sum to 1."""
    p = prices.clip(lower=1e-9)
    total = p.sum()
    if total <= 0:
        return p
    return p / total


def devig_two_sided(yes_price: float, no_price: float) -> float:
    """De-vig a single binary market given both YES and NO prices.

    Returns the fair YES probability. If yes+no > 1 (the usual book margin),
    this splits the overround proportionally.
    """
    yes_price = max(yes_price, 1e-9)
    no_price = max(no_price, 1e-9)
    return yes_price / (yes_price + no_price)


def implied_from_book(df: pd.DataFrame) -> pd.DataFrame:
    """Add `market_prob` and `spread` columns to a market-prices frame.

    Expects columns: at least one of {midpoint} or {bid, ask}. Optional `yes`,
    `no` for two-sided de-vig.
    """
    out = df.copy()
    if "midpoint" not in out.columns and {"bid", "ask"}.issubset(out.columns):
        out["midpoint"] = (out["bid"] + out["ask"]) / 2.0
    if "spread" not in out.columns and {"bid", "ask"}.issubset(out.columns):
        out["spread"] = (out["ask"] - out["bid"]).abs()
    out["market_prob"] = out["midpoint"].clip(0, 1)
    return out
