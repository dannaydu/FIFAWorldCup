"""Paper-trading harness — run periodically through the tournament.

    python scripts/paper_trade.py                 # scan: mark-to-market + open new
    python scripts/paper_trade.py --report         # just print the ledger summary
    python scripts/paper_trade.py --settle res.json  # book P/L from a results file

Each scan trains the model on real results, simulates the real 2026 draw, pulls
LIVE Kalshi/Polymarket prices, finds shrunk edges, and updates the persistent
ledger at artifacts/paper_ledger.json. CLV accrues from day one; settle when
results land. Paper money only.

Results file for --settle: {"<bet_id>": true|false, ...} where the bool is
whether the bet's chosen side won (see bet_id in the ledger / `--report`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wcmodel import config                                              # noqa: E402
from wcmodel.ingest.get_market_prices import (                          # noqa: E402
    all_resolutions, kalshi_series, polymarket_world_cup, union_markets,
)
from wcmodel.ingest.get_matches import load_matches                     # noqa: E402
from wcmodel.markets.opportunities import build_opportunities           # noqa: E402
from wcmodel.markets.paper_ledger import PaperLedger                    # noqa: E402
from wcmodel.predictor import train_predictor                          # noqa: E402
from wcmodel.simulate.tournament import simulate_tournament            # noqa: E402
from wcmodel.teams2026 import GROUPS_2026                              # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def fetch_markets() -> pd.DataFrame:
    return union_markets([
        polymarket_world_cup(),
        kalshi_series("KXWCROUND", market_type="reach_round"),
        kalshi_series("KXWCGAME", market_type="match"),
    ])


def print_summary(ledger: PaperLedger) -> None:
    s = ledger.summary()
    print("\nLedger summary:")
    for k, v in s.items():
        print(f"  {k:>16}: {v}")
    df = ledger.to_frame()
    if not df.empty:
        cols = ["ts", "contract", "side", "entry_price", "last_price", "clv",
                "edge", "stake", "status"]
        openpos = df[df["status"] == "open"].sort_values("edge", ascending=False)
        if not openpos.empty:
            print("\nOpen positions (top 15 by edge):")
            print(openpos[cols].head(15).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print summary only")
    ap.add_argument("--settle", metavar="RESULTS.json", help="settle from results file")
    ap.add_argument("--auto-settle", action="store_true",
                    help="settle from live market resolution status")
    ap.add_argument("--bankroll", type=float, default=10_000.0)
    ap.add_argument("--lambda", dest="lam", type=float, default=config.MARKET_SHRINKAGE_LAMBDA)
    ap.add_argument("--n-sims", type=int, default=20_000)
    args = ap.parse_args()

    ledger = PaperLedger(start_bankroll=args.bankroll)

    if args.report:
        print_summary(ledger)
        return

    if args.settle:
        won = json.loads(Path(args.settle).read_text())
        n = ledger.settle(won)
        ledger.save()
        print(f"Settled {n} bets from {args.settle}.")
        print_summary(ledger)
        return

    if args.auto_settle:
        res = all_resolutions()
        n = ledger.auto_settle(res)
        ledger.save()
        print(f"Auto-settled {n} bets from {len(res)} resolved markets.")
        print_summary(ledger)
        return

    # scan
    print("Training model on real results + simulating real 2026 draw…")
    pred = train_predictor(load_matches())
    sim = simulate_tournament(pred, GROUPS_2026, n_sims=args.n_sims, seed=1)

    print("Fetching live markets (Kalshi + Polymarket)…")
    market = fetch_markets()
    print(f"  {len(market)} live contracts; "
          f"{market['midpoint'].notna().sum() if not market.empty else 0} priced.")

    opps = build_opportunities(sim, market, predictor=pred, lam=args.lam)
    print(f"  {len(opps)} opportunities clear filters "
          f"(edge>={config.MIN_EDGE:.0%}, spread<={config.MAX_SPREAD:.0%}, "
          f"liq>=${config.MIN_LIQUIDITY:.0f}).")

    settled = ledger.auto_settle(all_resolutions())
    if settled:
        print(f"  auto-settled {settled} resolved bets.")
    res = ledger.scan(opps, market)
    ledger.save()
    print(f"\nScan: marked {res['marked']} open to market, opened {res['opened']} "
          f"new -> {res['open_total']} open positions total.")
    print(f"Ledger: {ledger.path}")
    print_summary(ledger)


if __name__ == "__main__":
    main()
