"""Dixon-Coles bivariate-Poisson goal model (spec §5, Model 2).

Goals:
    lambda_home = exp(mu + attack[home] - defense[away] + gamma * home_ind)
    lambda_away = exp(mu + attack[away] - defense[home])

with the Dixon-Coles low-score dependence correction tau(x, y; lh, la, rho) and
match weights that decay with age (time) and scale with match importance.

Fitted by maximum weighted likelihood (L-BFGS-B). Produces a full score-
probability matrix per fixture, which both prices score markets and drives the
Monte-Carlo tournament simulator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson as poisson_dist

from wcmodel import config

_IMPORTANCE_FIT_WEIGHT = {
    "world_cup": 1.0,
    "continental": 0.9,
    "qualifier": 0.8,
    "minor_tournament": 0.6,
    "friendly": 0.5,
}


def _tau(x, y, lh, la, rho):
    """Dixon-Coles correction for the four low-score cells."""
    out = np.ones_like(lh, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    out[m00] = 1.0 - lh[m00] * la[m00] * rho
    out[m01] = 1.0 + lh[m01] * rho
    out[m10] = 1.0 + la[m10] * rho
    out[m11] = 1.0 - rho
    return out


class DixonColesModel:
    def __init__(self, time_decay: float = config.DC_TIME_DECAY, max_goals: int = 12,
                 fit_max_age_years: float | None = config.DC_FIT_MAX_AGE_YEARS):
        self.time_decay = time_decay
        self.max_goals = max_goals
        self.fit_max_age_years = fit_max_age_years
        self.teams: list[str] = []
        self.team_idx: dict[str, int] = {}
        self.attack = None
        self.defense = None
        self.mu = 0.0
        self.gamma = 0.0
        self.rho = 0.0

    # ----------------------------------------------------------------- fit --- #
    def fit(self, matches: pd.DataFrame, ref_date=None) -> "DixonColesModel":
        df = matches.dropna(subset=["team_a_goals", "team_b_goals"]).copy()
        df["date"] = pd.to_datetime(df["date"])
        ref = pd.Timestamp(ref_date) if ref_date is not None else df["date"].max()

        # The MLE only needs recent matches (older ones carry ~0 decay weight).
        if self.fit_max_age_years is not None:
            cutoff = ref - pd.Timedelta(days=self.fit_max_age_years * 365.25)
            df = df[df["date"] >= cutoff].reset_index(drop=True)

        self.teams = sorted(set(df["team_a"]) | set(df["team_b"]))
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        ih = df["team_a"].map(self.team_idx).to_numpy()
        ia = df["team_b"].map(self.team_idx).to_numpy()
        gh = df["team_a_goals"].to_numpy().astype(int)
        ga = df["team_b_goals"].to_numpy().astype(int)
        home_ind = (~df["neutral"].astype(bool)).to_numpy().astype(float)

        age_days = (ref - df["date"]).dt.days.clip(lower=0).to_numpy()
        imp_w = df["importance"].map(_IMPORTANCE_FIT_WEIGHT).fillna(0.5).to_numpy()
        w = np.exp(-self.time_decay * age_days) * imp_w

        # Param layout: [attack(n), defense(n), mu, gamma, rho]
        def unpack(p):
            attack = p[:n]
            defense = p[n:2 * n]
            mu, gamma, rho = p[2 * n], p[2 * n + 1], p[2 * n + 2]
            return attack, defense, mu, gamma, rho

        def neg_ll(p):
            attack, defense, mu, gamma, rho = unpack(p)
            lh = np.exp(mu + attack[ih] - defense[ia] + gamma * home_ind)
            la = np.exp(mu + attack[ia] - defense[ih])
            tau = _tau(gh, ga, lh, la, rho)
            tau = np.clip(tau, 1e-10, None)
            ll = (
                np.log(tau)
                + gh * np.log(lh) - lh
                + ga * np.log(la) - la
            )
            # soft sum-to-zero on attack & defense for identifiability + L2.
            penalty = 50.0 * (attack.sum() ** 2 + defense.sum() ** 2)
            penalty += 1e-3 * (attack @ attack + defense @ defense)
            return -np.sum(w * ll) + penalty

        x0 = np.zeros(2 * n + 3)
        x0[2 * n] = np.log(max(np.average(np.r_[gh, ga]), 0.3))  # mu init
        x0[2 * n + 1] = 0.2  # gamma init (home edge)
        x0[2 * n + 2] = 0.0  # rho init

        bounds = [(-3, 3)] * (2 * n) + [(-2, 2), (-1, 1), (-0.3, 0.3)]
        res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})

        self.attack, self.defense, self.mu, self.gamma, self.rho = unpack(res.x)
        return self

    # ------------------------------------------------------------- predict --- #
    def _ad(self, team: str) -> tuple[float, float]:
        i = self.team_idx.get(team)
        if i is None:
            return 0.0, 0.0  # unseen team -> average
        return float(self.attack[i]), float(self.defense[i])

    def rates(self, team_a: str, team_b: str, *, neutral: bool = True,
              host_a: bool = False, host_b: bool = False) -> tuple[float, float]:
        """Expected goals (lambda_a, lambda_b)."""
        aa, da = self._ad(team_a)
        ab, db = self._ad(team_b)
        home_a = self.gamma if (host_a or not neutral) else 0.0
        home_b = self.gamma if host_b else 0.0
        lam_a = np.exp(self.mu + aa - db + home_a)
        lam_b = np.exp(self.mu + ab - da + home_b)
        return float(lam_a), float(lam_b)

    def score_matrix(self, team_a: str, team_b: str, **kw) -> np.ndarray:
        """(max_goals+1) x (max_goals+1) matrix P[x, y] of A scoring x, B scoring y."""
        lam_a, lam_b = self.rates(team_a, team_b, **kw)
        mg = self.max_goals
        gx = np.arange(mg + 1)
        px = poisson_dist.pmf(gx, lam_a)
        py = poisson_dist.pmf(gx, lam_b)
        mat = np.outer(px, py)

        # apply DC correction to the four low cells
        rho = self.rho
        mat[0, 0] *= 1.0 - lam_a * lam_b * rho
        mat[0, 1] *= 1.0 + lam_a * rho
        mat[1, 0] *= 1.0 + lam_b * rho
        mat[1, 1] *= 1.0 - rho
        mat = np.clip(mat, 0.0, None)
        mat /= mat.sum()
        return mat

    def match_probs(self, team_a: str, team_b: str, **kw) -> np.ndarray:
        """[P(A win), P(draw), P(B win)] from the score matrix."""
        mat = self.score_matrix(team_a, team_b, **kw)
        p_a = np.tril(mat, -1).sum()   # x > y
        p_draw = np.trace(mat)         # x == y
        p_b = np.triu(mat, 1).sum()    # x < y
        return np.array([p_a, p_draw, p_b])

    def strengths(self) -> pd.DataFrame:
        """Per-team attack/defense ratings — useful for inspection."""
        return (
            pd.DataFrame(
                {"team": self.teams, "attack": self.attack, "defense": self.defense}
            )
            .sort_values("attack", ascending=False)
            .reset_index(drop=True)
        )
