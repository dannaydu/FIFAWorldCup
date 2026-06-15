import numpy as np

from wcmodel.features.match_features import build_match_features


def test_draw_and_importance_features_are_known_before_kickoff(matches):
    feat = build_match_features(matches)

    assert np.allclose(feat["abs_elo_diff"], feat["elo_diff"].abs())
    importance_cols = [
        "is_friendly", "is_minor_tournament", "is_qualifier",
        "is_continental", "is_world_cup",
    ]
    assert (feat[importance_cols].sum(axis=1) == 1.0).all()


def test_predictor_defaults_fixture_to_world_cup(predictor):
    feat = predictor.build_features("Argentina", "France")

    assert feat.loc[0, "is_world_cup"] == 1.0
    assert feat.loc[0, "is_friendly"] == 0.0
