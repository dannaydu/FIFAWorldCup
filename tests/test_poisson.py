import numpy as np


def test_score_matrix_normalized(predictor):
    mat = predictor.dc.score_matrix("Brazil", "Qatar", neutral=True)
    assert np.isclose(mat.sum(), 1.0, atol=1e-9)
    assert (mat >= 0).all()


def test_match_probs_sum_to_one(predictor):
    p = predictor.dc.match_probs("Spain", "Japan", neutral=True)
    assert np.isclose(p.sum(), 1.0, atol=1e-9)
    assert (p >= 0).all()


def test_stronger_team_favored(predictor):
    # Strong vs weak: P(strong win) should dominate.
    p = predictor.dc.match_probs("Argentina", "Qatar", neutral=True)
    assert p[0] > p[2]


def test_home_advantage_increases_rate(predictor):
    lam_home_a, _ = predictor.dc.rates("USA", "Mexico", neutral=False, host_a=True)
    lam_neutral_a, _ = predictor.dc.rates("USA", "Mexico", neutral=True)
    assert lam_home_a >= lam_neutral_a
