"""Streamlit dashboard (spec §11).

Run with:
    pip install streamlit
    streamlit run src/wcmodel/app/streamlit_dashboard.py

Pages: Team rankings | Match predictor | Tournament simulation |
       Market edge board | Calibration.
Everything trains once and is cached for the session.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import streamlit as st  # noqa: E402

from wcmodel.evaluate import metrics_frame, run_backtest          # noqa: E402
from wcmodel.ingest.get_matches import load_matches                # noqa: E402
from wcmodel.markets.edge import find_edges                        # noqa: E402
from wcmodel.predictor import train_predictor                      # noqa: E402
from wcmodel.simulate.match import outcome_probs, top_scorelines   # noqa: E402
from wcmodel.simulate.tournament import simulate_tournament        # noqa: E402
from wcmodel.teams2026 import ALL_TEAMS_2026, GROUPS_2026          # noqa: E402

st.set_page_config(page_title="World Cup 2026 Model", layout="wide")


@st.cache_resource(show_spinner="Loading data + training models…")
def _load():
    matches = load_matches()
    pred = train_predictor(matches)
    return matches, pred


@st.cache_data(show_spinner="Simulating tournament…")
def _simulate(n_sims: int):
    _, pred = _load()
    return simulate_tournament(pred, GROUPS_2026, n_sims=n_sims, seed=1)


matches, pred = _load()
page = st.sidebar.radio(
    "Page",
    ["Team rankings", "Match predictor", "Tournament simulation",
     "Market edge board", "Calibration"],
)

st.sidebar.caption(
    "⚠️ Demo runs on synthetic data + an EXAMPLE 2026 draw unless you drop real "
    "files into data/raw/ and set teams2026.GROUPS_2026."
)

# --------------------------------------------------------------------------- #
if page == "Team rankings":
    st.title("Team ratings")
    strengths = pred.dc.strengths()
    state = pred.state.reset_index()[["team", "team_elo"]]
    merged = strengths.merge(state, on="team").sort_values("team_elo", ascending=False)
    st.dataframe(merged.round(3), use_container_width=True, height=600)

# --------------------------------------------------------------------------- #
elif page == "Match predictor":
    st.title("Match predictor")
    c1, c2, c3 = st.columns(3)
    teams = sorted(set(ALL_TEAMS_2026) | set(pred.dc.teams))
    a = c1.selectbox("Team A", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    b = c2.selectbox("Team B", teams, index=teams.index("France") if "France" in teams else 1)
    neutral = c3.checkbox("Neutral venue", value=True)

    wdl = pred.predict_wdl(a, b, neutral=neutral)
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{a} win", f"{wdl[0]:.1%}")
    m2.metric("Draw", f"{wdl[1]:.1%}")
    m3.metric(f"{b} win", f"{wdl[2]:.1%}")

    mat = pred.score_matrix(a, b, neutral=neutral)
    st.subheader("Most likely scorelines")
    st.dataframe(top_scorelines(mat, a, b, n=12), use_container_width=True)

    st.subheader("Per-model breakdown")
    mp = pred.model_probs(a, b, neutral=neutral)
    st.dataframe(pd.DataFrame(mp, index=[f"{a} win", "draw", f"{b} win"]).T.round(3))

# --------------------------------------------------------------------------- #
elif page == "Tournament simulation":
    st.title("Tournament simulation (2026 format)")
    n_sims = st.slider("Simulations", 2000, 50000, 10000, step=2000)
    sim = _simulate(n_sims)
    pct = sim.copy()
    for c in [c for c in pct.columns if c.startswith("p_")]:
        pct[c] = pct[c].map("{:.1%}".format)
    st.dataframe(pct, use_container_width=True, height=600)
    st.bar_chart(sim.head(15).set_index("team")["p_champion"])

# --------------------------------------------------------------------------- #
elif page == "Market edge board":
    st.title("Market edge board")
    st.caption("Synthetic book here; wire ingest.get_market_prices for live Kalshi/Polymarket.")
    n_sims = 10000
    sim = _simulate(n_sims)
    rng = np.random.default_rng(7)
    champ = sim[["team", "p_champion"]]
    mid = np.clip(champ["p_champion"].to_numpy() + rng.normal(0, 0.02, len(champ)), 0.002, 0.6)
    mid = mid / mid.sum() * 1.06
    spread = rng.uniform(0.005, 0.04, len(champ))
    market = pd.DataFrame({
        "contract": champ["team"] + " — Win World Cup",
        "midpoint": mid, "spread": spread,
        "liquidity": rng.uniform(200, 5000, len(champ)).round(0),
    })
    model_probs = pd.DataFrame({
        "contract": champ["team"] + " — Win World Cup",
        "team": champ["team"], "model_prob": champ["p_champion"],
    })
    edges = find_edges(model_probs, market)
    st.dataframe(edges.round(4), use_container_width=True)

# --------------------------------------------------------------------------- #
elif page == "Calibration":
    st.title("Backtest + calibration")
    try:
        res = run_backtest(matches, cutoff="2023-01-01")
        st.subheader("Probabilistic metrics (lower is better)")
        st.dataframe(metrics_frame(res).round(4))
        st.subheader("Ensemble reliability — P(home win)")
        rel = res["calibration_ensemble_home_win"]
        st.line_chart(rel.set_index("predicted_mean")[["observed_freq"]])
        st.dataframe(rel.round(3))
    except ValueError as e:
        st.warning(str(e))
