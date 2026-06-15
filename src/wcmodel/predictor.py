"""MatchPredictor — the trained bundle the simulator and markets consume.

Wraps the fitted Elo-logistic, Dixon-Coles, and GBM models plus the current
team state, and exposes fixture-level predictions for *any* matchup:

    pred = train_predictor(matches)
    pred.predict_wdl("Argentina", "France", neutral=True)   # [pA, pDraw, pB]
    pred.score_matrix("Argentina", "France")                # full score grid
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wcmodel import config
from wcmodel.features.confederations import team_conf_strength
from wcmodel.features.match_features import build_match_features, feature_columns
from wcmodel.features.squad_features import (
    SQUAD_FEATURE_COLS,
    load_squad_features,
    load_squad_panel,
)
from wcmodel.features.team_features import current_team_state
from wcmodel.models.elo_logistic import EloLogistic
from wcmodel.models.ensemble import combine
from wcmodel.models.gbm import GBMatchModel
from wcmodel.models.poisson import DixonColesModel

_FORM_DIFF_COLS = [
    "gf_l10", "ga_l10", "gd_l10", "winrate_l10", "sos_l10",
    "pts_vs_exp_l10", "gd_vs_exp_l10",
    "gd_decayed", "gf_decayed", "winrate_decayed",
]

_IMPORTANCE_FEATURES = [
    "friendly", "minor_tournament", "qualifier", "continental", "world_cup",
]


class MatchPredictor:
    def __init__(self, elo_model, gbm_model, dc_model, team_state, squad,
                 feature_cols, weights=None, conf_strength=None):
        self.elo = elo_model
        self.gbm = gbm_model
        self.dc = dc_model
        self.state = team_state.set_index("team")
        self.squad = squad.set_index("team")
        self.feature_cols = feature_cols
        self.weights = weights or config.ENSEMBLE_WEIGHTS
        self.conf_strength = conf_strength or {}

    # ------------------------------------------------- fixture featurizer --- #
    def _team_row(self, team: str) -> pd.Series:
        if team in self.state.index:
            return self.state.loc[team]
        return pd.Series({c: 0.0 for c in self.state.columns}).replace(
            {"team_elo": config.ELO_START}
        )

    def _feature_dict(self, team_a: str, team_b: str, *, neutral: bool = True,
                      host_a: bool = False, importance: str = "world_cup") -> dict[str, float]:
        a, b = self._team_row(team_a), self._team_row(team_b)
        row: dict[str, float] = {}
        row["elo_diff"] = float(a.get("team_elo", config.ELO_START) -
                                b.get("team_elo", config.ELO_START))
        row["abs_elo_diff"] = abs(row["elo_diff"])
        row["host_advantage"] = 1.0 if (host_a or not neutral) else 0.0
        for match_type in _IMPORTANCE_FEATURES:
            row[f"is_{match_type}"] = 1.0 if importance == match_type else 0.0
        for c in _FORM_DIFF_COLS:
            row[f"{c}_diff"] = float(a.get(c, 0.0)) - float(b.get(c, 0.0))
        for c in SQUAD_FEATURE_COLS:
            va = float(self.squad.loc[team_a, c]) if team_a in self.squad.index else 0.0
            vb = float(self.squad.loc[team_b, c]) if team_b in self.squad.index else 0.0
            row[f"{c}_diff"] = va - vb
        row["conf_strength_diff"] = (self.conf_strength.get(team_a, 0.0)
                                     - self.conf_strength.get(team_b, 0.0))
        return row

    def build_features(self, team_a: str, team_b: str, *, neutral: bool = True,
                       host_a: bool = False, importance: str = "world_cup") -> pd.DataFrame:
        row = self._feature_dict(
            team_a, team_b, neutral=neutral, host_a=host_a, importance=importance,
        )
        return pd.DataFrame([row]).reindex(columns=self.feature_cols, fill_value=0.0)

    def fixture_feature_frame(self, pairs, host_set=frozenset(),
                              importance: str = "world_cup") -> pd.DataFrame:
        """Batched feature table for many (team_a, team_b) fixtures (neutral)."""
        rows = [
            self._feature_dict(
                a, b, neutral=True, host_a=a in host_set, importance=importance,
            )
            for a, b in pairs
        ]
        return pd.DataFrame(rows).reindex(columns=self.feature_cols, fill_value=0.0)

    # --------------------------------------------------------- prediction --- #
    def model_probs(self, team_a: str, team_b: str, *, neutral: bool = True,
                    host_a: bool = False, importance: str = "world_cup") -> dict[str, np.ndarray]:
        feat = self.build_features(
            team_a, team_b, neutral=neutral, host_a=host_a, importance=importance,
        )
        return {
            "elo": self.elo.predict_proba(feat)[0],
            "gbm": self.gbm.predict_proba(feat)[0],
            "poisson": self.dc.match_probs(team_a, team_b, neutral=neutral, host_a=host_a),
        }

    def predict_wdl(self, team_a: str, team_b: str, *, neutral: bool = True,
                    host_a: bool = False, importance: str = "world_cup") -> np.ndarray:
        probs = self.model_probs(
            team_a, team_b, neutral=neutral, host_a=host_a, importance=importance,
        )
        return combine(probs, self.weights)

    def score_matrix(self, team_a: str, team_b: str, *, neutral: bool = True,
                     host_a: bool = False) -> np.ndarray:
        return self.dc.score_matrix(team_a, team_b, neutral=neutral, host_a=host_a)


def train_predictor(matches: pd.DataFrame, *, squad: pd.DataFrame | None = None,
                    weights: dict | None = None, ref_date=None) -> MatchPredictor:
    """Fit all sub-models and assemble a MatchPredictor."""
    state = current_team_state(matches, ref_date=ref_date)
    team_elo = dict(zip(state["team"], state["team_elo"]))
    if squad is None:
        squad = load_squad_features(team_elo=team_elo)
    # Era-correct squad values for *training* (inference still uses `squad`).
    panel = load_squad_panel(team_elo=team_elo)

    conf = team_conf_strength(matches)
    feat = build_match_features(matches, squad=squad, squad_panel=panel, conf_strength=conf)
    cols = feature_columns(feat)

    elo = EloLogistic().fit(feat)
    gbm = GBMatchModel(features=cols).fit(feat)
    dc = DixonColesModel().fit(matches, ref_date=ref_date)

    return MatchPredictor(elo, gbm, dc, state, squad, cols,
                          weights=weights, conf_strength=conf)
