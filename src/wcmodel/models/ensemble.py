"""Weighted probability ensemble (spec §5, Model 5).

Combines per-model [P(A win), P(draw), P(B win)] arrays with configurable
weights and renormalizes. Missing models are simply dropped and the remaining
weights renormalized, so the ensemble degrades gracefully.
"""
from __future__ import annotations

import numpy as np

from wcmodel import config


def combine(prob_by_model: dict[str, np.ndarray],
            weights: dict[str, float] | None = None) -> np.ndarray:
    """Weighted-average probability arrays.

    Each value in `prob_by_model` is an (n, 3) array (or shape (3,)). Returns the
    same shape, renormalized to sum to 1 along the last axis.
    """
    weights = weights or config.ENSEMBLE_WEIGHTS
    present = {k: v for k, v in prob_by_model.items() if v is not None and k in weights}
    if not present:
        raise ValueError("no models with matching weights to combine")

    wsum = sum(weights[k] for k in present)
    stacked = None
    for k, v in present.items():
        arr = np.asarray(v, dtype=float)
        contrib = (weights[k] / wsum) * arr
        stacked = contrib if stacked is None else stacked + contrib

    stacked = np.clip(stacked, 1e-12, None)
    stacked = stacked / stacked.sum(axis=-1, keepdims=True)
    return stacked
