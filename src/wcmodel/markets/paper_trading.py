"""Paper-trading ledger with fractional-Kelly sizing + CLV (spec §10).

Always paper-trade first. This logs hypothetical bets, sizes them with a capped
fractional Kelly, and settles them against the outcome and the closing price so
you can measure both realized ROI and closing-line value (CLV) — the latter is
the cleaner signal that the model is actually finding value.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from wcmodel import config


def kelly_fraction(model_prob: float, price: float) -> float:
    """Full-Kelly stake fraction for a binary contract bought at `price`.

    Contract pays $1 if it resolves YES. With $1 you buy 1/price shares, so net
    odds b = (1 - price)/price and full Kelly is (q - price)/(1 - price).
    Negative -> no bet.
    """
    if price <= 0 or price >= 1:
        return 0.0
    f = (model_prob - price) / (1.0 - price)
    return max(f, 0.0)


def position_size(model_prob: float, price: float, bankroll: float, *,
                  kelly: float = config.KELLY_FRACTION,
                  cap: float = config.MAX_POSITION_FRACTION) -> float:
    """Dollar stake: fractional Kelly, hard-capped at `cap` of bankroll."""
    f = kelly * kelly_fraction(model_prob, price)
    f = min(f, cap)
    return round(f * bankroll, 2)


@dataclass
class Bet:
    timestamp: str
    contract: str
    side: str            # YES / NO
    entry_price: float
    model_prob: float
    edge: float
    stake: float
    shares: float
    # filled at settlement
    closing_price: float | None = None
    outcome: int | None = None   # 1 if the chosen side won, else 0
    pnl: float | None = None
    clv: float | None = None


class PaperTradingLog:
    def __init__(self, bankroll: float = 10_000.0):
        self.bankroll = bankroll
        self.start_bankroll = bankroll
        self.bets: list[Bet] = []

    def place(self, contract: str, side: str, entry_price: float, model_prob: float,
              edge: float, *, timestamp: str = "") -> Bet | None:
        stake = position_size(model_prob, entry_price, self.bankroll)
        if stake <= 0:
            return None
        shares = stake / entry_price
        bet = Bet(timestamp=timestamp, contract=contract, side=side,
                  entry_price=entry_price, model_prob=model_prob, edge=edge,
                  stake=stake, shares=round(shares, 2))
        self.bets.append(bet)
        return bet

    def settle(self, bet: Bet, *, won: bool, closing_price: float | None = None) -> None:
        bet.outcome = int(won)
        # payoff: each share returns $1 if won, else $0; stake was already spent.
        payoff = bet.shares * (1.0 if won else 0.0)
        bet.pnl = round(payoff - bet.stake, 2)
        self.bankroll = round(self.bankroll + bet.pnl, 2)
        if closing_price is not None:
            bet.closing_price = closing_price
            # CLV from the chosen side's perspective.
            bet.clv = round(closing_price - bet.entry_price, 4)

    # ------------------------------------------------------------- reports --- #
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(b) for b in self.bets])

    def summary(self) -> dict:
        df = self.to_frame()
        settled = df[df["pnl"].notna()] if not df.empty else df
        total_stake = settled["stake"].sum() if not settled.empty else 0.0
        total_pnl = settled["pnl"].sum() if not settled.empty else 0.0
        return {
            "n_bets": len(df),
            "n_settled": len(settled),
            "bankroll": self.bankroll,
            "total_staked": round(float(total_stake), 2),
            "total_pnl": round(float(total_pnl), 2),
            "roi": round(float(total_pnl / total_stake), 4) if total_stake else 0.0,
            "avg_clv": round(float(settled["clv"].dropna().mean()), 4)
            if not settled.empty and settled["clv"].notna().any() else None,
            "hit_rate": round(float(settled["outcome"].mean()), 4)
            if not settled.empty else None,
        }
