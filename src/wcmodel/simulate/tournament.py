"""Monte-Carlo simulation of the 48-team / 12-group 2026 World Cup (spec §9).

Pipeline per simulated tournament:
    1. simulate all group matches (round robin, 6 per group)
    2. rank each group with FIFA tiebreakers (pts -> GD -> GF -> head-to-head)
    3. advance top 2 of every group + the 8 best third-placed teams (32 total)
    4. seed + build the Round-of-32 bracket
    5. simulate single elimination R32 -> R16 -> QF -> SF -> Final
    6. record each team's furthest stage + group-winner flag

Aggregated over many sims this yields p(win group), p(reach each stage), and
p(champion) — the inputs the market-edge engine prices against.

NOTE on the bracket: qualifiers are seeded by (placement, strength) into a
standard serpentine 32-team bracket (keeps strong sides apart). This is a clean,
transparent approximation of FIFA's fixed third-place allocation table; swap
`build_bracket` for the official slot table when exact bracket geometry matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wcmodel import config
from wcmodel.models.ensemble import combine
from wcmodel.teams2026 import HOST_NATIONS as HOSTS

# Group round-robin fixture index pairs for a group of 4.
_GROUP_FIXTURES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# Stage ordering for cumulative "reached" probabilities.
_STAGE_IDX = {s: i for i, s in enumerate(config.STAGES)}


# --------------------------------------------------------------------------- #
# Pre-computed per-fixture sampling distributions (built once per predictor)
# --------------------------------------------------------------------------- #
@dataclass
class FixtureDists:
    mg: int
    cdf: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)   # group sampling
    p_a_ko: dict[tuple[str, str], float] = field(default_factory=dict)     # knockout A-win prob
    elo: dict[str, float] = field(default_factory=dict)


def _reweight_to_marginals(mat: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Scale a score matrix so its (A win, draw, B win) marginals match `target`.

    Keeps the Dixon-Coles *shape* of the scoreline distribution but bends the
    win/draw/loss split to the ensemble's outcome probabilities. This is how
    Elo + GBM signal enters the simulator (the raw matrix is DC-only).
    """
    xx, yy = np.indices(mat.shape)
    lower, diag, upper = xx > yy, xx == yy, xx < yy
    m = np.array([mat[lower].sum(), mat[diag].sum(), mat[upper].sum()])
    out = mat.copy()
    for mask, mi, ti in zip((lower, diag, upper), m, target):
        if mi > 1e-12:
            out[mask] *= ti / mi
    s = out.sum()
    return out / s if s > 0 else mat


def precompute_dists(predictor, teams: list[str], *, use_ensemble: bool = True) -> FixtureDists:
    """Score CDFs (for groups) + decisive A-win probs (for knockouts).

    With `use_ensemble`, each fixture's W/D/L is the full ensemble (Elo + DC +
    GBM) and the DC score matrix is reshaped to those marginals; otherwise the
    raw DC matrix is used.
    """
    mg = predictor.dc.max_goals
    fd = FixtureDists(mg=mg)
    for t in teams:
        fd.elo[t] = float(predictor.state.loc[t, "team_elo"]) if t in predictor.state.index \
            else config.ELO_START

    pairs = [(a, b) for a in teams for b in teams if a != b]

    # Batched ensemble W/D/L for every ordered pair.
    targets: dict[tuple[str, str], np.ndarray] = {}
    if use_ensemble:
        X = predictor.fixture_feature_frame(pairs, host_set=HOSTS)
        p_elo = predictor.elo.predict_proba(X)
        p_gbm = predictor.gbm.predict_proba(X)

    for i, (a, b) in enumerate(pairs):
        mat = predictor.dc.score_matrix(a, b, neutral=True, host_a=a in HOSTS, host_b=b in HOSTS)
        p_dc = np.array([np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()])
        if use_ensemble:
            target = combine({"elo": p_elo[i], "gbm": p_gbm[i], "poisson": p_dc},
                             predictor.weights)
            mat = _reweight_to_marginals(mat, target)
        else:
            target = p_dc

        fd.cdf[(a, b)] = np.cumsum(mat.ravel())
        p_a, p_draw, p_b = float(target[0]), float(target[1]), float(target[2])
        p_a_dec = p_a / (p_a + p_b) if (p_a + p_b) > 0 else 0.5
        fd.p_a_ko[(a, b)] = p_a + p_draw * p_a_dec
    return fd


def _sample_score(fd: FixtureDists, a: str, b: str, rng) -> tuple[int, int]:
    idx = int(np.searchsorted(fd.cdf[(a, b)], rng.random()))
    idx = min(idx, (fd.mg + 1) ** 2 - 1)
    return divmod(idx, fd.mg + 1)


# --------------------------------------------------------------------------- #
# Group stage
# --------------------------------------------------------------------------- #
def _rank_group(teams: list[str], stats: dict, results: dict, rng) -> list[str]:
    """Apply FIFA tiebreakers: pts -> GD -> GF -> head-to-head -> random."""
    def primary_key(t):
        s = stats[t]
        return (s["pts"], s["gd"], s["gf"])

    ordered = sorted(teams, key=primary_key, reverse=True)

    # Break exact (pts, gd, gf) ties with a head-to-head mini-league, then random.
    final: list[str] = []
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and primary_key(ordered[j + 1]) == primary_key(ordered[i]):
            j += 1
        block = ordered[i:j + 1]
        if len(block) > 1:
            block = _head_to_head(block, results, rng)
        final.extend(block)
        i = j + 1
    return final


def _head_to_head(block: list[str], results: dict, rng) -> list[str]:
    h = {t: {"pts": 0, "gf": 0, "ga": 0} for t in block}
    bset = set(block)
    for (x, y), (gx, gy) in results.items():
        if x in bset and y in bset:
            for t, gf, ga in ((x, gx, gy), (y, gy, gx)):
                h[t]["gf"] += gf
                h[t]["ga"] += ga
                h[t]["pts"] += 3 if gf > ga else (1 if gf == ga else 0)
    return sorted(
        block,
        key=lambda t: (h[t]["pts"], h[t]["gf"] - h[t]["ga"], h[t]["gf"], rng.random()),
        reverse=True,
    )


def _simulate_group(group_teams: list[str], fd: FixtureDists, rng):
    stats = {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0} for t in group_teams}
    results: dict[tuple[str, str], tuple[int, int]] = {}
    for i, j in _GROUP_FIXTURES:
        a, b = group_teams[i], group_teams[j]
        ga, gb = _sample_score(fd, a, b, rng)
        results[(a, b)] = (ga, gb)
        stats[a]["gf"] += ga; stats[a]["ga"] += gb
        stats[b]["gf"] += gb; stats[b]["ga"] += ga
        if ga > gb:
            stats[a]["pts"] += 3
        elif ga < gb:
            stats[b]["pts"] += 3
        else:
            stats[a]["pts"] += 1; stats[b]["pts"] += 1
    for t in group_teams:
        stats[t]["gd"] = stats[t]["gf"] - stats[t]["ga"]
    ranked = _rank_group(group_teams, stats, results, rng)
    return ranked, stats


# --------------------------------------------------------------------------- #
# Bracket
# --------------------------------------------------------------------------- #
def _bracket_seed_order(n: int) -> list[int]:
    """Standard bracket slot order of seeds (1-indexed) for power-of-two n."""
    order = [1, 2]
    while len(order) < n:
        size = len(order)
        new = []
        for s in order:
            new.append(s)
            new.append(2 * size + 1 - s)
        order = new
    return order


def build_bracket(winners: list[str], runners: list[str], thirds: list[str],
                  fd: FixtureDists) -> list[str]:
    """Seed 32 qualifiers (winners > runners > thirds, by Elo) into bracket slots."""
    def by_elo(ts):
        return sorted(ts, key=lambda t: fd.elo.get(t, config.ELO_START), reverse=True)

    ranked = by_elo(winners) + by_elo(runners) + by_elo(thirds)  # seed 1..32
    order = _bracket_seed_order(len(ranked))
    return [ranked[s - 1] for s in order]


def _simulate_knockout(bracket: list[str], fd: FixtureDists, rng) -> dict[str, str]:
    """Single elimination; returns {team: furthest stage reached}."""
    # entering-stage label by number of survivors.
    entering = {32: "round_32", 16: "round_16", 8: "quarterfinal",
                4: "semifinal", 2: "final"}
    stage = {t: "round_32" for t in bracket}
    survivors = list(bracket)
    while len(survivors) > 1:
        cur = entering[len(survivors)]
        nxt_idx = _STAGE_IDX[cur] + 1
        nxt = config.STAGES[nxt_idx]
        winners = []
        for k in range(0, len(survivors), 2):
            a, b = survivors[k], survivors[k + 1]
            a_wins = rng.random() < fd.p_a_ko[(a, b)]
            w, loser = (a, b) if a_wins else (b, a)
            stage[loser] = cur          # loser reached current round
            stage[w] = nxt              # winner advances
            winners.append(w)
        survivors = winners
    # the lone survivor is champion (already set to "champion" as nxt of final)
    return stage


# --------------------------------------------------------------------------- #
# Full tournament
# --------------------------------------------------------------------------- #
def _simulate_once(groups: dict[str, list[str]], fd: FixtureDists, rng):
    winners, runners = [], []
    thirds = []  # (team, pts, gd, gf)
    group_winner_flag = {}
    stage = {}

    for label, teams in groups.items():
        ranked, stats = _simulate_group(teams, fd, rng)
        winners.append(ranked[0])
        runners.append(ranked[1])
        third = ranked[2]
        thirds.append((third, stats[third]["pts"], stats[third]["gd"], stats[third]["gf"]))
        group_winner_flag[ranked[0]] = True
        # everyone starts "group"; knockout overwrites for qualifiers
        for t in teams:
            stage[t] = "group"

    # 8 best third-placed teams
    thirds_sorted = sorted(thirds, key=lambda r: (r[1], r[2], r[3], rng.random()),
                           reverse=True)
    best_thirds = [r[0] for r in thirds_sorted[:config.N_BEST_THIRDS]]

    bracket = build_bracket(winners, runners, best_thirds, fd)
    ko_stage = _simulate_knockout(bracket, fd, rng)
    stage.update(ko_stage)
    return stage, group_winner_flag


def simulate_tournament(predictor, groups: dict[str, list[str]] | None = None, *,
                        n_sims: int = config.N_SIMULATIONS, seed: int = 0) -> pd.DataFrame:
    """Run `n_sims` tournaments; return per-team stage/championship probabilities."""
    from wcmodel.teams2026 import GROUPS_2026

    groups = groups or GROUPS_2026
    teams = [t for ts in groups.values() for t in ts]
    fd = precompute_dists(predictor, teams)
    rng = np.random.default_rng(seed)

    reach = {t: np.zeros(len(config.STAGES)) for t in teams}  # counts by stage idx
    group_win = {t: 0 for t in teams}

    for _ in range(n_sims):
        stage, gwf = _simulate_once(groups, fd, rng)
        for t, s in stage.items():
            reach[t][_STAGE_IDX[s]] += 1
        for t in gwf:
            group_win[t] += 1

    rows = []
    for t in teams:
        counts = reach[t]
        cum = np.cumsum(counts[::-1])[::-1] / n_sims  # P(stage >= idx)
        rows.append(
            {
                "team": t,
                "p_group_winner": group_win[t] / n_sims,
                "p_round_32": cum[_STAGE_IDX["round_32"]],
                "p_round_16": cum[_STAGE_IDX["round_16"]],
                "p_quarterfinal": cum[_STAGE_IDX["quarterfinal"]],
                "p_semifinal": cum[_STAGE_IDX["semifinal"]],
                "p_final": cum[_STAGE_IDX["final"]],
                "p_champion": counts[_STAGE_IDX["champion"]] / n_sims,
            }
        )
    return (pd.DataFrame(rows).sort_values("p_champion", ascending=False)
            .reset_index(drop=True))
