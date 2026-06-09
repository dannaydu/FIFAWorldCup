"""Persistent paper-trading ledger with mark-to-market + closing-line value.

A tournament runs for weeks, so the ledger is a JSON file you append to across
many runs. Each `scan`:

  * marks every OPEN position to the current market (running CLV = last - entry),
  * opens a new paper position for any fresh opportunity (one per market_id+side),
  * never re-bets a contract it already holds.

When results arrive, `settle` books P/L. Closing-line value — the last price
before a market resolved vs. your entry — is the signal that the model is finding
value even before any bet settles, so it is tracked from day one.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from wcmodel import config
from wcmodel.markets.paper_trading import position_size


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class PaperLedger:
    def __init__(self, path=None, start_bankroll: float = 10_000.0):
        self.path = Path(path or (config.ARTIFACTS / "paper_ledger.json"))
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {"start_bankroll": start_bankroll, "bets": []}

    # ------------------------------------------------------------------ io --- #
    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))

    @property
    def bets(self) -> list[dict]:
        return self.data["bets"]

    def _by_id(self) -> dict[str, dict]:
        return {b["bet_id"]: b for b in self.bets}

    @staticmethod
    def _bet_id(platform, market_id, side) -> str:
        return f"{platform}:{market_id}:{side}"

    # ---------------------------------------------------------------- scan --- #
    def scan(self, opportunities: pd.DataFrame, market: pd.DataFrame,
             *, ts: str | None = None) -> dict:
        """Mark open bets to market, then open new positions. Returns counts."""
        ts = ts or _now()
        existing = self._by_id()

        # 1) mark-to-market every open position using the full current feed
        yes_price = {row.market_id: row.midpoint for row in market.itertuples(index=False)
                     if pd.notna(getattr(row, "midpoint", None))}
        marked = 0
        for b in self.bets:
            if b["status"] != "open" or b["market_id"] not in yes_price:
                continue
            yp = float(yes_price[b["market_id"]])
            last = yp if b["side"] == "YES" else 1.0 - yp
            b["last_price"] = round(last, 4)
            b["last_ts"] = ts
            b["clv"] = round(last - b["entry_price"], 4)
            marked += 1

        # 2) open new positions for fresh opportunities
        open_markets = {b["market_id"] for b in self.bets if b["status"] == "open"}
        opened = 0
        for o in opportunities.itertuples(index=False):
            bid = self._bet_id(o.platform, o.market_id, o.side)
            # one position per market: never re-enter or take a contradictory side
            if bid in existing or o.market_id in open_markets:
                continue
            stake = position_size(o.model_prob_side, o.entry_price,
                                  self.data["start_bankroll"])
            if stake <= 0:
                continue
            self.bets.append({
                "bet_id": bid,
                "ts": ts,
                "platform": o.platform,
                "market_id": o.market_id,
                "contract": o.contract,
                "team": o.team,
                "market_type": o.market_type,
                "round": o.round,
                "side": o.side,
                "entry_price": float(o.entry_price),
                "model_prob": float(o.model_prob_side),
                "edge": float(o.edge),
                "stake": stake,
                "shares": round(stake / o.entry_price, 2),
                "status": "open",
                "last_price": float(o.entry_price),
                "last_ts": ts,
                "clv": 0.0,
                "closing_price": None,
                "outcome": None,
                "pnl": None,
            })
            opened += 1

        return {"marked": marked, "opened": opened, "open_total": len(self.open_bets())}

    # -------------------------------------------------------------- settle --- #
    def open_bets(self) -> list[dict]:
        return [b for b in self.bets if b["status"] == "open"]

    def auto_settle(self, yes_won_by_market: dict[str, bool]) -> int:
        """Settle open bets from resolved markets {market_id: yes_won}.

        Converts a market's YES outcome into the chosen side's outcome.
        """
        won_by_id = {}
        for b in self.open_bets():
            if b["market_id"] in yes_won_by_market:
                yw = bool(yes_won_by_market[b["market_id"]])
                won_by_id[b["bet_id"]] = yw if b["side"] == "YES" else (not yw)
        return self.settle(won_by_id)

    def settle(self, won_by_id: dict[str, bool]) -> int:
        """Settle bets given {bet_id: chosen_side_won}. Returns count settled."""
        n = 0
        for b in self.bets:
            if b["status"] != "open" or b["bet_id"] not in won_by_id:
                continue
            won = bool(won_by_id[b["bet_id"]])
            b["outcome"] = int(won)
            b["closing_price"] = b["last_price"]
            payoff = b["shares"] * (1.0 if won else 0.0)
            b["pnl"] = round(payoff - b["stake"], 2)
            b["status"] = "settled"
            n += 1
        return n

    # -------------------------------------------------------------- report --- #
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.bets)

    def summary(self) -> dict:
        df = self.to_frame()
        if df.empty:
            return {"n_bets": 0}
        settled = df[df["status"] == "settled"]
        staked = settled["stake"].sum() if not settled.empty else 0.0
        pnl = settled["pnl"].sum() if not settled.empty else 0.0
        open_df = df[df["status"] == "open"]
        return {
            "n_bets": len(df),
            "open": len(open_df),
            "settled": len(settled),
            "bankroll": round(self.data["start_bankroll"] + float(pnl), 2),
            "open_staked": round(float(open_df["stake"].sum()), 2) if not open_df.empty else 0.0,
            "avg_open_clv": round(float(open_df["clv"].mean()), 4) if not open_df.empty else None,
            "settled_pnl": round(float(pnl), 2),
            "roi": round(float(pnl / staked), 4) if staked else None,
            "hit_rate": round(float(settled["outcome"].mean()), 4) if not settled.empty else None,
            "avg_settled_clv": round(float(settled["clv"].mean()), 4) if not settled.empty else None,
        }
