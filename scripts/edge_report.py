"""Live edge report: model (real data + real draw) vs live market prices.

    python scripts/edge_report.py

Trains the ensemble on real international results, simulates the real 2026 draw,
pulls LIVE Polymarket group-winner prices (+ Kalshi reach-round), de-vigs them,
and prints contracts where the model edge clears the filters. Paper-trade only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wcmodel import config                                                        # noqa: E402
from wcmodel.ingest.get_market_prices import kalshi_series, polymarket_world_cup  # noqa: E402
from wcmodel.ingest.get_matches import load_matches                               # noqa: E402
from wcmodel.markets.shrink import blend_to_market                                # noqa: E402
from wcmodel.predictor import train_predictor                                     # noqa: E402
from wcmodel.simulate.tournament import simulate_tournament                       # noqa: E402
from wcmodel.teams2026 import GROUPS_2026                                         # noqa: E402

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 30)

N_SIMS = 20_000
MIN_EDGE = 0.04
MIN_LIQUIDITY = 500.0
LAMBDA = config.MARKET_SHRINKAGE_LAMBDA


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    banner("1. TRAIN ON REAL RESULTS + SIMULATE REAL 2026 DRAW")
    matches = load_matches(refresh=True)
    # Use FULL history: Elo + form stay cross-confederation calibrated; the DC
    # MLE applies its own recency window internally (config.DC_FIT_MAX_AGE_YEARS).
    print(f"{len(matches):,} matches (full history) | "
          f"{matches['team_a'].nunique()} teams")
    pred = train_predictor(matches)
    sim = simulate_tournament(pred, GROUPS_2026, n_sims=N_SIMS, seed=1)
    print("\nModel championship favorites:")
    top = sim.head(8).copy()
    for c in ["p_group_winner", "p_quarterfinal", "p_semifinal", "p_champion"]:
        top[c] = top[c].map("{:.1%}".format)
    print(top[["team", "p_group_winner", "p_quarterfinal", "p_semifinal",
               "p_champion"]].to_string(index=False))

    banner("2. LIVE POLYMARKET — GROUP-WINNER EDGES (de-vigged per group)")
    pm = polymarket_world_cup()
    gw = pm[(pm["market_type"] == "group_winner")].dropna(subset=["team", "midpoint"]).copy()
    if gw.empty:
        print("No live Polymarket group-winner markets returned (offline?).")
    else:
        # de-vig: normalize the YES prices within each group to sum to 1
        gw["market_prob"] = gw.groupby("group")["midpoint"].transform(lambda s: s / s.sum())
        model = sim.set_index("team")["p_group_winner"]
        gw["model_prob"] = gw["team"].map(model)
        gw = gw.dropna(subset=["model_prob"])
        gw["raw_edge"] = gw["model_prob"] - gw["market_prob"]
        # shrink model toward the market prior, then bet only the residual.
        gw["blended"] = blend_to_market(gw["model_prob"], gw["market_prob"], LAMBDA)
        gw["edge"] = gw["blended"] - gw["market_prob"]
        gw["abs_edge"] = gw["edge"].abs()
        gw["side"] = np.where(gw["edge"] >= 0, "BUY", "FADE")

        filt = gw[(gw["abs_edge"] >= MIN_EDGE)
                  & (gw["liquidity"].fillna(0) >= MIN_LIQUIDITY)]
        print(f"After shrinkage toward market (lambda={LAMBDA}): {len(gw)} contracts "
              f"joined; {len(filt)} clear |edge|>={MIN_EDGE:.0%} & liq>=${MIN_LIQUIDITY:.0f}.\n")
        show = filt.sort_values("abs_edge", ascending=False).head(15).copy()
        for c in ["midpoint", "market_prob", "model_prob", "blended", "raw_edge", "edge"]:
            fmt = "{:+.1%}" if c in ("raw_edge", "edge") else "{:.1%}"
            show[c] = show[c].map(fmt.format)
        print(show[["team", "group", "side", "model_prob", "market_prob",
                    "blended", "raw_edge", "edge", "liquidity"]].to_string(index=False))

    banner("3. LIVE KALSHI — REACH-ROUND MARKETS (model vs market)")
    kr = kalshi_series("KXWCROUND", market_type="reach_round")
    priced = kr.dropna(subset=["midpoint"]) if not kr.empty else kr
    if kr.empty:
        print("No Kalshi reach-round markets returned (offline?).")
    else:
        print(f"{len(kr)} reach-round markets; {len(priced)} currently priced "
              f"(many are unquoted pre-tournament).")
        if not priced.empty:
            # map round label -> model column
            col = {"round_16": "p_round_16", "quarterfinal": "p_quarterfinal",
                   "semifinal": "p_semifinal", "final": "p_final",
                   "champion": "p_champion"}
            from wcmodel.ingest.get_market_prices import _kalshi_round
            priced = priced.copy()
            priced["round"] = priced["market_id"].map(_kalshi_round)
            simidx = sim.set_index("team")
            def model_for(r):
                c = col.get(r["round"])
                if c is None or r["team"] not in simidx.index:
                    return np.nan
                return simidx.loc[r["team"], c]
            priced["model_prob"] = priced.apply(model_for, axis=1)
            priced = priced.dropna(subset=["model_prob"])
            priced["edge"] = priced["model_prob"] - priced["midpoint"]
            show = priced.sort_values("edge", key=lambda s: s.abs(), ascending=False).head(15).copy()
            for c in ["midpoint", "model_prob", "edge"]:
                show[c] = show[c].map("{:.1%}".format)
            print(show[["team", "round", "model_prob", "midpoint", "edge"]].to_string(index=False))

    print("\nReal data + real draw + live prices. Edges are pre-cost & pre-execution; "
          "paper-trade and track closing-line value before risking money.")


if __name__ == "__main__":
    main()
