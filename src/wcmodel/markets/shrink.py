"""Market-prior shrinkage (spec §5, Model 0 as a prior).

The model is weak on exactly the things liquid markets price well (squad depth,
cross-confederation strength, knockout pedigree). So our best probability
estimate is a blend of the two, and we should only bet on the *residual*
disagreement after acknowledging the market:

    blended = (1 - lambda) * model + lambda * market
    edge    = blended - market = (1 - lambda) * (model - market)

`lambda` encodes how much we trust the market over our model. Calibrate it by
maximizing closing-line value / ROI in paper trading, not by taste.
"""
from __future__ import annotations

import numpy as np

from wcmodel import config


def blend_to_market(model_prob, market_prob, lam: float = config.MARKET_SHRINKAGE_LAMBDA):
    """Shrink model probabilities toward the market prior by fraction `lam`."""
    model_prob = np.asarray(model_prob, dtype=float)
    market_prob = np.asarray(market_prob, dtype=float)
    return (1.0 - lam) * model_prob + lam * market_prob


def shrunk_edge(model_prob, market_prob, lam: float = config.MARKET_SHRINKAGE_LAMBDA):
    """Edge after shrinkage = blended - market = (1 - lam) * (model - market)."""
    return blend_to_market(model_prob, market_prob, lam) - np.asarray(market_prob, dtype=float)
