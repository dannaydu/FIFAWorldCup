"""End-to-end demo: data -> Elo -> models -> backtest -> simulation -> edges.

Runs entirely on synthetic data if no real files are present, so:

    python scripts/run_demo.py

works out of the box. Drop real data into data/raw/ (see README) and rerun to
use it instead — no code changes needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wcmodel.evaluate import metrics_frame, run_backtest          # noqa: E402
from wcmodel.ingest.get_matches import load_matches                # noqa: E402
from wcmodel.markets.edge import find_edges                        # noqa: E402
from wcmodel.markets.paper_trading import PaperTradingLog          # noqa: E402
from wcmodel.predictor import train_predictor                      # noqa: E402
from wcmodel.simulate.match import outcome_probs, top_scorelines   # noqa: E402
from wcmodel.simulate.tournament import simulate_tournament        # noqa: E402
from wcmodel.teams2026 import GROUPS_2026                          # noqa: E402

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

N_SIMS = 10_000  # bump to 50_000 for production-grade probabilities


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    banner("1. LOAD MATCHES")
    matches = load_matches()
    src = "real (data/raw/results.csv)" if (Path("data/raw/results.csv")).exists() else "synthetic"
    print(f"source: {src}")
    print(f"{len(matches):,} matches | {matches['team_a'].nunique()} teams | "
          f"{matches['date'].min().date()} .. {matches['date'].max().date()}")

    banner("2. BACKTEST (time split, train < 2023-01-01 < test)")
    try:
        results = run_backtest(matches, cutoff="2023-01-01")
        print(f"train={results['n_train']:,}  test={results['n_test']:,}\n")
        print("Probabilistic metrics (lower is better):")
        print(metrics_frame(results).round(4))
        print("\nEnsemble calibration — P(home win) bins:")
        print(results["calibration_ensemble_home_win"].round(3).to_string(index=False))
    except ValueError as e:
        print(f"skipped backtest: {e}")

    banner("3. TRAIN FULL PREDICTOR")
    pred = train_predictor(matches)
    print(f"Dixon-Coles backend fitted on {len(pred.dc.teams)} teams; "
          f"GBM backend = {pred.gbm.backend}")
    print("\nTop attack ratings (Dixon-Coles):")
    print(pred.dc.strengths().head(8).round(3).to_string(index=False))

    banner("4. SINGLE-MATCH PREDICTION  (example: Argentina vs France, neutral)")
    a, b = "Argentina", "France"
    wdl = pred.predict_wdl(a, b, neutral=True)
    print(f"P({a} win)={wdl[0]:.1%}   P(draw)={wdl[1]:.1%}   P({b} win)={wdl[2]:.1%}")
    mat = pred.score_matrix(a, b, neutral=True)
    print("\nMost likely scorelines:")
    print(top_scorelines(mat, a, b, n=8).assign(prob=lambda d: d["prob"].map("{:.1%}".format))
          .to_string(index=False))

    banner(f"5. TOURNAMENT SIMULATION  ({N_SIMS:,} runs, EXAMPLE 2026 draw)")
    sim = simulate_tournament(pred, GROUPS_2026, n_sims=N_SIMS, seed=1)
    show = sim.head(12).copy()
    for c in ["p_group_winner", "p_round_16", "p_quarterfinal", "p_semifinal",
              "p_final", "p_champion"]:
        show[c] = show[c].map("{:.1%}".format)
    print(show[["team", "p_group_winner", "p_quarterfinal", "p_semifinal",
                "p_final", "p_champion"]].to_string(index=False))

    banner("6. MARKET EDGE + PAPER TRADING  (synthetic 'to win World Cup' book)")
    rng = np.random.default_rng(123)
    champ = sim[["team", "p_champion"]].copy()
    # Build a noisy bookmaker: market disagrees with the model + adds overround.
    noise = rng.normal(0, 0.02, len(champ))
    mid = np.clip(champ["p_champion"].to_numpy() + noise, 0.002, 0.6)
    mid = mid / mid.sum() * 1.06  # ~6% overround
    spread = rng.uniform(0.005, 0.04, len(champ))
    market = pd.DataFrame({
        "contract": champ["team"] + " — Win World Cup",
        "midpoint": mid,
        "bid": np.clip(mid - spread / 2, 0, 1),
        "ask": np.clip(mid + spread / 2, 0, 1),
        "spread": spread,
        "liquidity": rng.uniform(200, 5000, len(champ)).round(0),
    })
    model_probs = pd.DataFrame({
        "contract": champ["team"] + " — Win World Cup",
        "team": champ["team"],
        "model_prob": champ["p_champion"],
    })
    edges = find_edges(model_probs, market)
    print(f"{len(edges)} contracts clear the edge/spread/liquidity filters "
          f"(>= 4pp edge, <= 3c spread, >= $500 liq):\n")
    if not edges.empty:
        disp = edges.head(10).copy()
        for c in ["model_prob", "market_prob", "edge", "entry_price"]:
            disp[c] = disp[c].map("{:.1%}".format)
        print(disp[["contract", "side", "model_prob", "market_prob", "edge",
                    "entry_price", "liquidity"]].to_string(index=False))

        # Paper trade them, then settle against one simulated champion.
        log = PaperTradingLog(bankroll=10_000)
        champion = rng.choice(champ["team"], p=(champ["p_champion"] /
                                                champ["p_champion"].sum()))
        for r in edges.itertuples(index=False):
            bet = log.place(r.contract, r.side, r.entry_price, r.model_prob_side, r.edge)
            if bet is None:
                continue
            team = r.team if hasattr(r, "team") else r.contract.split(" — ")[0]
            won = (team == champion) if r.side == "YES" else (team != champion)
            # closing price ~ drifts toward the model (illustrative CLV)
            closing = float(np.clip(r.entry_price + (r.model_prob_side - r.entry_price) * 0.5,
                                    0, 1))
            log.settle(bet, won=won, closing_price=closing)

        print(f"\nSimulated champion this settlement: {champion}")
        print("Paper-trading summary:")
        for k, v in log.summary().items():
            print(f"  {k:>14}: {v}")
    else:
        print("(no qualifying edges this run — try rerunning or widening filters)")

    print("\nDone. This is a MODELLING SCAFFOLD on SYNTHETIC data + an EXAMPLE draw.")
    print("Replace data/raw/* and teams2026.GROUPS_2026 with real inputs before "
          "trusting any number.")


if __name__ == "__main__":
    main()
