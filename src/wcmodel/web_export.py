"""Build the JSON snapshots the web UI consumes (Firestore docs / static JSON).

`build_snapshots()` runs the full real-data pipeline once and returns four
plain-dict documents: tournament, edges, ledger, meta. Both the static export
(scripts/export_web.py) and the Firestore publisher (scripts/publish_firestore.py)
call this so the two stay in sync.
"""
from __future__ import annotations

import time

import pandas as pd

from wcmodel import config
from wcmodel.ingest.get_market_prices import (
    all_resolutions, kalshi_series, polymarket_world_cup, union_markets,
)
from wcmodel.ingest.get_matches import load_matches
from wcmodel.markets.opportunities import build_opportunities
from wcmodel.markets.paper_ledger import PaperLedger
from wcmodel.predictor import train_predictor
from wcmodel.simulate.tournament import simulate_tournament
from wcmodel.teams2026 import GROUPS_2026


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fetch_markets() -> pd.DataFrame:
    return union_markets([
        polymarket_world_cup(),
        kalshi_series("KXWCROUND", market_type="reach_round"),
    ])


def build_snapshots(*, n_sims: int = 20_000, lam: float = config.MARKET_SHRINKAGE_LAMBDA,
                    update_ledger: bool = True) -> dict[str, dict]:
    generated_at = _now()
    matches = load_matches()
    pred = train_predictor(matches)
    sim = simulate_tournament(pred, GROUPS_2026, n_sims=n_sims, seed=1)

    group_of = {t: g for g, ts in GROUPS_2026.items() for t in ts}
    tournament = {
        "generated_at": generated_at,
        "teams": [
            {
                "team": r.team, "group": group_of.get(r.team),
                "p_group_winner": round(float(r.p_group_winner), 4),
                "p_round_16": round(float(r.p_round_16), 4),
                "p_quarterfinal": round(float(r.p_quarterfinal), 4),
                "p_semifinal": round(float(r.p_semifinal), 4),
                "p_final": round(float(r.p_final), 4),
                "p_champion": round(float(r.p_champion), 4),
            }
            for r in sim.itertuples(index=False)
        ],
    }

    market = _fetch_markets()
    opps = build_opportunities(sim, market, lam=lam)
    edges = {
        "generated_at": generated_at,
        "lambda": lam,
        "opportunities": opps.to_dict(orient="records") if not opps.empty else [],
    }

    ledger = PaperLedger()
    if update_ledger and not market.empty:
        ledger.auto_settle(all_resolutions())
        ledger.scan(opps, market)
        ledger.save()
    ledger_snap = {
        "generated_at": generated_at,
        "summary": ledger.summary(),
        "bets": ledger.bets,
    }

    meta = {
        "generated_at": generated_at,
        "n_matches": int(len(matches)),
        "n_sims": n_sims,
        "lambda": lam,
        "gbm_backend": pred.gbm.backend,
        "n_live_contracts": int(len(market)),
        "n_opportunities": int(len(opps)),
        "note": "Probabilistic model output. Paper-trading only, pre-cost edges.",
    }

    return {"tournament": tournament, "edges": edges,
            "ledger": ledger_snap, "meta": meta}
