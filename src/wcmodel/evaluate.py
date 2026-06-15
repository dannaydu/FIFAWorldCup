"""Time-split backtest + probabilistic evaluation (spec §7, §8).

Trains on matches before a cutoff, tests after — never a random split (that
leaks the future). Reports log-loss / Brier / RPS for each model and the
ensemble, plus a calibration table so you can check "when it says 20%, does it
happen ~20%?".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wcmodel.features.match_features import build_match_features, feature_columns
from wcmodel.features.confederations import team_conf_strength
from wcmodel.features.squad_features import load_squad_features, load_squad_panel
from wcmodel.models import calibrate
from wcmodel.models.elo_logistic import EloLogistic
from wcmodel.models.ensemble import combine
from wcmodel.models.gbm import GBMatchModel
from wcmodel.models.poisson import DixonColesModel


def run_backtest(matches: pd.DataFrame, cutoff: str = "2023-01-01") -> dict:
    cut = pd.Timestamp(cutoff)
    match_dates = pd.to_datetime(matches["date"])
    train_matches = matches[match_dates < cut]
    squad = load_squad_features()
    panel = load_squad_panel()
    # Confederation offsets are target-derived, so estimate them from the
    # training side only. Computing them on the full frame leaks test outcomes.
    conf_strength = team_conf_strength(train_matches)
    feat = build_match_features(
        matches, squad=squad, squad_panel=panel, conf_strength=conf_strength,
    )
    cols = feature_columns(feat)

    feat["date"] = pd.to_datetime(feat["date"])
    train = feat[feat["date"] < cut]
    test = feat[feat["date"] >= cut].reset_index(drop=True)
    if len(test) < 50:
        raise ValueError(f"too few test matches ({len(test)}) after {cutoff}")

    # Fit on train only.
    elo = EloLogistic().fit(train)
    gbm = GBMatchModel(features=cols).fit(train)
    dc = DixonColesModel().fit(
        train_matches, ref_date=cut
    )

    # Predict on test.
    p_elo = elo.predict_proba(test)
    p_gbm = gbm.predict_proba(test)
    p_dc = np.vstack([
        dc.match_probs(r.team_a, r.team_b, neutral=bool(r.neutral))
        for r in test.itertuples(index=False)
    ])
    p_ens = combine({"elo": p_elo, "gbm": p_gbm, "poisson": p_dc})

    y = test["y"].to_numpy()
    results = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "metrics": {
            "elo_logistic": calibrate.all_metrics(y, p_elo),
            "poisson_dc": calibrate.all_metrics(y, p_dc),
            "gbm": calibrate.all_metrics(y, p_gbm),
            "ensemble": calibrate.all_metrics(y, p_ens),
        },
        "calibration_ensemble_home_win": calibrate.reliability_table(y, p_ens, class_idx=0),
    }
    return results


def metrics_frame(results: dict) -> pd.DataFrame:
    return (pd.DataFrame(results["metrics"]).T
            .sort_values("log_loss"))
