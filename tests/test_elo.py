import numpy as np

from wcmodel.ingest.elo import compute_elo, expected_score, goal_multiplier
from wcmodel.ingest.synthetic import generate_matches


def test_expected_score_symmetry():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1900, 1500) > 0.5
    # home advantage helps the home side
    assert expected_score(1500, 1500, 100) > 0.5


def test_goal_multiplier_monotone():
    assert goal_multiplier(0) == 1.0
    assert goal_multiplier(1) == 1.0
    assert goal_multiplier(2) == 1.5
    assert goal_multiplier(3) == 1.75
    assert goal_multiplier(5) > goal_multiplier(4) > goal_multiplier(3)


def test_elo_is_zero_sum():
    m = generate_matches(n_matches=800, seed=1)
    _, ratings = compute_elo(m)
    total = sum(ratings.values())
    # every team starts at ELO_START; updates are zero-sum, so the mean is preserved
    n = len(ratings)
    assert np.isclose(total / n, 1500.0, atol=1e-6)


def test_stronger_team_rates_higher():
    m = generate_matches(n_matches=4000, seed=2)
    _, ratings = compute_elo(m)
    # Argentina/France are top seeds in the generator; minnows should trail.
    assert ratings["Argentina"] > ratings["Qatar"]
    assert ratings["France"] > ratings["New Zealand"]
