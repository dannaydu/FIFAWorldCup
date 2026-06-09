"""Elo logistic baseline (spec §5, Model 1).

Multinomial logistic regression on a handful of interpretable inputs predicting
3-class W/D/L. Simple, fast, and a genuinely strong baseline — hard to beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_INPUTS = ["elo_diff", "host_advantage"]


class EloLogistic:
    def __init__(self, inputs: list[str] | None = None):
        self.inputs = inputs or _INPUTS
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=1000, C=1.0)
        self.classes_ = np.array([0, 1, 2])

    def fit(self, feat: pd.DataFrame) -> "EloLogistic":
        X = self.scaler.fit_transform(feat[self.inputs].to_numpy())
        self.clf.fit(X, feat["y"].to_numpy())
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, feat: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(feat[self.inputs].to_numpy())
        return self._align(self.clf.predict_proba(X))

    def _align(self, proba: np.ndarray) -> np.ndarray:
        """Ensure column order is [P(A win), P(draw), P(B win)] = classes 0,1,2."""
        out = np.zeros((proba.shape[0], 3))
        for j, cls in enumerate(self.classes_):
            out[:, int(cls)] = proba[:, j]
        return out
