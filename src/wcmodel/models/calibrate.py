"""Probability calibration + reliability diagnostics (spec §8).

* `fit_temperature` / `apply_temperature`: single-parameter temperature scaling
  for multiclass probabilities (cheap, robust, preserves argmax).
* `reliability_table`: binned predicted-vs-observed frequencies for calibration
  plots and "when the model says 20%, does it happen ~20%?" checks.
* metrics: multiclass log-loss, Brier score, and Ranked Probability Score (RPS).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def _onehot(y: np.ndarray, n_classes: int = 3) -> np.ndarray:
    oh = np.zeros((len(y), n_classes))
    oh[np.arange(len(y)), y.astype(int)] = 1.0
    return oh


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, None)) / T
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """Find T>0 minimizing log-loss of temperature-scaled probabilities."""
    y = np.asarray(y)

    def loss(T):
        p = apply_temperature(probs, max(T, 1e-3))
        return log_loss(y, p)

    res = minimize_scalar(loss, bounds=(0.3, 5.0), method="bounded")
    return float(res.x)


def log_loss(y: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y)), y.astype(int)])))


def brier_score(y: np.ndarray, probs: np.ndarray) -> float:
    oh = _onehot(np.asarray(y), probs.shape[1])
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def ranked_probability_score(y: np.ndarray, probs: np.ndarray) -> float:
    """RPS for ordered outcomes (A win < draw < B win). Lower is better."""
    oh = _onehot(np.asarray(y), probs.shape[1])
    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(oh, axis=1)
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1)) / (probs.shape[1] - 1))


def reliability_table(y: np.ndarray, probs: np.ndarray, *,
                      class_idx: int = 0, n_bins: int = 10) -> pd.DataFrame:
    """Binned predicted vs. observed frequency for one class."""
    p = probs[:, class_idx]
    hit = (np.asarray(y) == class_idx).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = which == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                "predicted_mean": p[m].mean(),
                "observed_freq": hit[m].mean(),
                "count": int(m.sum()),
            }
        )
    return pd.DataFrame(rows)


def all_metrics(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": log_loss(y, probs),
        "brier": brier_score(y, probs),
        "rps": ranked_probability_score(y, probs),
    }
