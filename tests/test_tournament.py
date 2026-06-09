import numpy as np

from wcmodel import config
from wcmodel.simulate.tournament import _bracket_seed_order, simulate_tournament
from wcmodel.teams2026 import GROUPS_2026


def test_bracket_seed_order_is_permutation():
    order = _bracket_seed_order(32)
    assert sorted(order) == list(range(1, 33))
    # classic property: seeds 1 and 2 land in opposite halves
    assert order[0] == 1
    assert 2 in order[16:]


def test_tournament_invariants(predictor):
    sim = simulate_tournament(predictor, GROUPS_2026, n_sims=400, seed=0)

    # one row per team, all probs in [0, 1]
    assert len(sim) == config.N_TEAMS
    prob_cols = [c for c in sim.columns if c.startswith("p_")]
    assert ((sim[prob_cols] >= 0) & (sim[prob_cols] <= 1)).all().all()

    # exactly one champion, 32 qualifiers, 12 group winners per tournament
    assert np.isclose(sim["p_champion"].sum(), 1.0, atol=1e-9)
    assert np.isclose(sim["p_round_32"].sum(), config.KNOCKOUT_SIZE, atol=1e-9)
    assert np.isclose(sim["p_group_winner"].sum(), config.N_GROUPS, atol=1e-9)


def test_reach_probs_are_monotone(predictor):
    sim = simulate_tournament(predictor, GROUPS_2026, n_sims=400, seed=5)
    chain = ["p_round_32", "p_round_16", "p_quarterfinal",
             "p_semifinal", "p_final", "p_champion"]
    for earlier, later in zip(chain, chain[1:]):
        assert (sim[earlier] >= sim[later] - 1e-9).all()


def test_stronger_team_more_likely_to_win(predictor):
    sim = simulate_tournament(predictor, GROUPS_2026, n_sims=600, seed=7)
    by_team = sim.set_index("team")["p_champion"]
    assert by_team["Argentina"] > by_team["Qatar"]
