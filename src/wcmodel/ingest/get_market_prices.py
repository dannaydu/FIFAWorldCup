"""Prediction-market price ingestion: Kalshi + Polymarket (spec §2, §11).

Thin, dependency-light clients over the documented PUBLIC endpoints (no auth for
read). Returns rows normalized to:

    platform, market_id, contract, team, group, market_type,
    midpoint, spread, liquidity, volume, timestamp

`market_type` is one of: group_winner, champion, reach_round, match, other.
Team names are canonicalized to the dataset spelling so contracts join to model
teams. Everything degrades to an empty frame without network access.

Verified live (June 2026): Polymarket serves "World Cup Group X Winner" and
tournament markets under tag_slug=world-cup; Kalshi serves reach-round markets
under series KXWCROUND (team in `yes_sub_title`, round in the ticker) and match
markets under KXWCGAME.

Docs: https://docs.kalshi.com  |  https://docs.polymarket.com
"""
from __future__ import annotations

import json
import re
import time
import warnings

import pandas as pd

from wcmodel.teams2026 import canonicalize

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
POLYMARKET_CLOB = "https://clob.polymarket.com"
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"

_COLUMNS = [
    "platform", "market_id", "contract", "team", "group", "market_type",
    "midpoint", "spread", "liquidity", "volume", "timestamp",
]

_ROUND_FROM_TICKER = {
    "R32": "round_32", "R16": "round_16", "QUARTER": "quarterfinal",
    "QTR": "quarterfinal", "SEMI": "semifinal", "FINAL": "final",
    "CHAMP": "champion", "WIN": "champion",
}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLUMNS)


def union_markets(frames) -> pd.DataFrame:
    """Concatenate market frames, dropping empties (quietly re: all-NA columns)."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return _empty()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return pd.concat(frames, ignore_index=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _maybe_list(val):
    if val is None or isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except Exception:
        return None


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Polymarket
# --------------------------------------------------------------------------- #
def _classify_polymarket(title: str) -> tuple[str, str | None]:
    """Return (market_type, group_letter) from an event title."""
    m = re.search(r"Group ([A-L])\b.*Winner", title, re.I)
    if m:
        return "group_winner", m.group(1).upper()
    if re.search(r"world cup winner", title, re.I) or re.search(r"win the .*world cup", title, re.I):
        return "champion", None
    if re.search(r"reach|advance|make the", title, re.I):
        return "reach_round", None
    return "other", None


def _team_from_question(q: str) -> str | None:
    m = re.match(r"Will (.+?) (?:win|reach|advance|make|qualify)\b", q, re.I)
    if not m:
        return None
    team = m.group(1).strip()
    if team.lower().startswith(("another", "any other", "the field")):
        return None
    return canonicalize(team)


def polymarket_world_cup(*, tag_slug: str = "world-cup", limit: int = 200,
                         timeout: float = 15.0) -> pd.DataFrame:
    """Normalized World Cup markets from Polymarket (Gamma + outcome prices)."""
    if requests is None:
        return _empty()
    try:
        r = requests.get(
            f"{POLYMARKET_GAMMA}/events",
            params={"closed": "false", "limit": limit, "tag_slug": tag_slug},
            timeout=timeout,
        )
        r.raise_for_status()
        events = r.json()
    except Exception:
        return _empty()

    rows = []
    for ev in events:
        title = ev.get("title", "")
        mtype, group = _classify_polymarket(title)
        for mk in ev.get("markets", []):
            q = mk.get("question", "")
            team = _team_from_question(q)
            prices = _maybe_list(mk.get("outcomePrices"))
            outcomes = _maybe_list(mk.get("outcomes")) or ["Yes", "No"]
            if not prices:
                continue
            yes_i = outcomes.index("Yes") if "Yes" in outcomes else 0
            mid = _to_float(prices[yes_i])
            if mid is None:
                continue
            rows.append({
                "platform": "polymarket",
                "market_id": (_maybe_list(mk.get("clobTokenIds")) or [mk.get("id")])[0],
                "contract": f"{title} — {q}",
                "team": team,
                "group": group,
                "market_type": mtype,
                "midpoint": mid,
                "spread": _to_float(mk.get("spread")),
                "liquidity": _to_float(mk.get("liquidity") or mk.get("liquidityNum")),
                "volume": _to_float(mk.get("volume") or mk.get("volumeNum")),
                "timestamp": _now(),
            })
    return pd.DataFrame(rows, columns=_COLUMNS)


# --------------------------------------------------------------------------- #
# Kalshi
# --------------------------------------------------------------------------- #
def _kalshi_round(ticker: str) -> str | None:
    # tickers look like KXWCROUND-26SEMI-USA -> middle segment carries the round
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    seg = re.sub(r"^\d+", "", parts[1]).upper()  # strip leading year, e.g. 26SEMI -> SEMI
    return _ROUND_FROM_TICKER.get(seg)


def kalshi_series(series_ticker: str, *, market_type: str = "reach_round",
                  limit: int = 500, timeout: float = 15.0) -> pd.DataFrame:
    """Normalized markets for a Kalshi series (e.g. KXWCROUND, KXWCGAME)."""
    if requests is None:
        return _empty()
    rows, cursor = [], None
    try:
        while True:
            params = {"series_ticker": series_ticker, "status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            r = requests.get(f"{KALSHI_BASE}/markets", params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            ms = data.get("markets", [])
            for m in ms:
                yb, ya = m.get("yes_bid"), m.get("yes_ask")
                last = m.get("last_price")
                # cents -> probability; prefer mid of bid/ask, fall back to last
                if yb is not None and ya is not None and (yb or ya):
                    mid = (yb + ya) / 200.0
                    spread = abs(ya - yb) / 100.0
                elif last is not None:
                    mid, spread = last / 100.0, None
                else:
                    mid, spread = None, None
                rows.append({
                    "platform": "kalshi",
                    "market_id": m.get("ticker"),
                    "contract": m.get("title") or m.get("ticker"),
                    "team": canonicalize(m.get("yes_sub_title") or ""),
                    "group": None,
                    "market_type": market_type,
                    "midpoint": mid,
                    "spread": spread,
                    "liquidity": (m.get("liquidity") or 0) / 100.0 if m.get("liquidity") else None,
                    "volume": m.get("volume"),
                    "timestamp": _now(),
                })
            cursor = data.get("cursor")
            if not cursor or len(rows) >= limit:
                break
    except Exception:
        return pd.DataFrame(rows, columns=_COLUMNS)
    return pd.DataFrame(rows, columns=_COLUMNS)


def polymarket_resolutions(*, tag_slug: str = "world-cup", limit: int = 400,
                           timeout: float = 15.0) -> dict[str, bool]:
    """Resolved Polymarket WC markets -> {clobTokenId: yes_won}.

    A market is resolved once closed with outcome prices collapsed to ~[1,0]/[0,1].
    """
    if requests is None:
        return {}
    out: dict[str, bool] = {}
    try:
        r = requests.get(
            f"{POLYMARKET_GAMMA}/events",
            params={"closed": "true", "limit": limit, "tag_slug": tag_slug},
            timeout=timeout,
        )
        r.raise_for_status()
        events = r.json()
    except Exception:
        return {}
    for ev in events:
        for mk in ev.get("markets", []):
            prices = _maybe_list(mk.get("outcomePrices"))
            outcomes = _maybe_list(mk.get("outcomes")) or ["Yes", "No"]
            tokens = _maybe_list(mk.get("clobTokenIds"))
            if not prices or not tokens:
                continue
            yes_i = outcomes.index("Yes") if "Yes" in outcomes else 0
            yp = _to_float(prices[yes_i])
            if yp is None or 0.01 < yp < 0.99:   # not cleanly resolved
                continue
            out[tokens[0]] = yp >= 0.99
    return out


def kalshi_resolutions(series_tickers=("KXWCROUND", "KXWCGAME"),
                       timeout: float = 15.0) -> dict[str, bool]:
    """Settled Kalshi markets -> {ticker: yes_won} (uses the market `result`)."""
    if requests is None:
        return {}
    out: dict[str, bool] = {}
    for series in series_tickers:
        cursor = None
        try:
            while True:
                params = {"series_ticker": series, "status": "settled", "limit": 200}
                if cursor:
                    params["cursor"] = cursor
                r = requests.get(f"{KALSHI_BASE}/markets", params=params, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                for m in data.get("markets", []):
                    res = (m.get("result") or "").lower()
                    if res in ("yes", "no"):
                        out[m.get("ticker")] = res == "yes"
                cursor = data.get("cursor")
                if not cursor:
                    break
        except Exception:
            continue
    return out


def all_resolutions() -> dict[str, bool]:
    """Union of resolved markets across venues, keyed by market_id (token/ticker)."""
    res = {}
    res.update(polymarket_resolutions())
    res.update(kalshi_resolutions())
    return res


def load_market_prices(*, polymarket: bool = True, kalshi: bool = True) -> pd.DataFrame:
    """Best-effort union of WC markets across both venues."""
    frames = []
    if polymarket:
        frames.append(polymarket_world_cup())
    if kalshi:
        frames.append(kalshi_series("KXWCROUND", market_type="reach_round"))
        frames.append(kalshi_series("KXWCGAME", market_type="match"))
    return union_markets(frames)
