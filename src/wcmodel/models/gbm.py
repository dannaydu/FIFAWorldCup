"""Gradient-boosted 3-class match model (spec §5, Model 3).

Uses LightGBM when available; otherwise falls back to scikit-learn's
HistGradientBoostingClassifier so the pipeline never hard-fails on a missing
optional dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # optional dependency
    import lightgbm as lgb

    _HAVE_LGB = True
except Exception:  # pragma: no cover - depends on environment
    _HAVE_LGB = False

from sklearn.ensemble import HistGradientBoostingClassifier


class GBMatchModel:
    def __init__(self, features: list[str], **kwargs):
        self.features = features
        self.backend = "lightgbm" if _HAVE_LGB else "sklearn-hist"
        self.classes_ = np.array([0, 1, 2])
        if _HAVE_LGB:
            self.model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=kwargs.get("n_estimators", 400),
                learning_rate=kwargs.get("learning_rate", 0.03),
                num_leaves=kwargs.get("num_leaves", 31),
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=kwargs.get("min_child_samples", 40),
                reg_lambda=1.0,
                verbose=-1,
            )
        else:
            self.model = HistGradientBoostingClassifier(
                learning_rate=kwargs.get("learning_rate", 0.05),
                max_iter=kwargs.get("n_estimators", 400),
                l2_regularization=1.0,
            )

    def fit(self, feat: pd.DataFrame) -> "GBMatchModel":
        X = feat[self.features].to_numpy()
        y = feat["y"].to_numpy()
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, feat: pd.DataFrame) -> np.ndarray:
        X = feat[self.features].to_numpy()
        proba = self.model.predict_proba(X)
        out = np.zeros((proba.shape[0], 3))
        for j, cls in enumerate(self.classes_):
            out[:, int(cls)] = proba[:, j]
        return out

    def feature_importance(self) -> pd.Series:
        if _HAVE_LGB:
            imp = self.model.feature_importances_
        else:
            # HistGBM has no native importances; use permutation-free proxy of 0s.
            imp = np.zeros(len(self.features))
        return pd.Series(imp, index=self.features).sort_values(ascending=False)
