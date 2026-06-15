"""Build the JSON snapshots the web UI consumes (Firestore docs / static JSON).

`build_snapshots()` runs the full real-data pipeline once and returns four
plain-dict documents: tournament, edges, ledger, meta. Both the static export
(scripts/export_web.py) and the Firestore publisher (scripts/publish_firestore.py)
call this so the two stay in sync.
"""
from __future__ import annotations

import json
import math
import time

import pandas as pd

from wcmodel import config

# Persistent state: model predictions locked at first sighting (pre-kickoff),
# so the scorecard stays a fair out-of-sample test even if results later land in
# the training mirror.
_LOCKED_PATH = config.DATA / "locked_predictions.json"
from wcmodel.ingest.get_market_prices import (
    all_resolutions, kalshi_resolutions, kalshi_series, polymarket_results,
    polymarket_world_cup, union_markets,
)
from wcmodel.ingest.get_matches import load_matches
from wcmodel.markets.opportunities import _fixture_key, _parse_fixture, build_opportunities
from wcmodel.markets.paper_ledger import PaperLedger
from wcmodel.predictor import train_predictor
from wcmodel.simulate.tournament import simulate_tournament
from wcmodel.teams2026 import GROUPS_2026, HOST_NATIONS

_MONTHS = {"JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05",
           "JUN": "06", "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10",
           "NOV": "11", "DEC": "12"}


def _ticker_date(market_id: str) -> str | None:
    """KXWCGAME-26JUN13QATSUI-SUI -> 2026-06-13."""
    try:
        tok = str(market_id).split("-")[1]   # 26JUN13QATSUI
        yy, mon, dd = tok[0:2], tok[2:5], tok[5:7]
        return f"20{yy}-{_MONTHS[mon]}-{dd}"
    except Exception:
        return None


def _match_results() -> dict[str, str]:
    """Settled Kalshi match markets -> {"m:<fixture_key>": "A"|"D"|"B"}.

    Parsed straight from tickers: KXWCGAME-26JUN13QATSUI-SUI -> fixture
    KXWCGAME-26JUN13QATSUI, code QATSUI (QAT=A, SUI=B), outcome SUI -> "B".
    """
    out: dict[str, str] = {}
    for ticker, won in kalshi_resolutions(("KXWCGAME",)).items():
        if not won:
            continue
        parts = str(ticker).split("-")
        if len(parts) < 3:
            continue
        code6, out_code = parts[1][7:], parts[2]   # strip 7-char date prefix
        if len(code6) < 6:
            continue
        a3, b3 = code6[:3], code6[3:6]
        side = ("A" if out_code == a3 else "B" if out_code == b3
                else "D" if out_code in ("TIE", "DRW", "DRAW") else None)
        if side:
            out["m:" + "-".join(parts[:2])] = side
    return out


def game_results() -> dict[str, str]:
    """Game-ready results map for the fantasy settler (matches + futures)."""
    res = _match_results()
    res.update(polymarket_results())   # {"grp:<L>": team, "champion": team}
    return res


def _lock_predictions(fixtures: list[dict], generated_at: str) -> dict:
    """Lock each fixture's first-seen model prediction; returns the full store."""
    try:
        locked = json.loads(_LOCKED_PATH.read_text()) if _LOCKED_PATH.exists() else {}
    except Exception:
        locked = {}
    changed = False
    for f in fixtures:
        if f["key"] not in locked:
            locked[f["key"]] = {
                "team_a": f["team_a"], "team_b": f["team_b"],
                "p_a": f["p_a"], "p_draw": f["p_draw"], "p_b": f["p_b"],
                "locked_at": generated_at,
            }
            changed = True
    if changed:
        _LOCKED_PATH.write_text(json.dumps(locked, indent=2))
    return locked


def _scorecard(locked: dict, results: dict, generated_at: str) -> dict:
    """Score locked pre-match predictions against actual results (completed games)."""
    games = []
    for mid, res in results.items():
        if not mid.startswith("m:"):
            continue
        lp = locked.get(mid[2:])
        if not lp or res not in ("A", "D", "B"):
            continue
        probs = {"A": lp["p_a"], "D": lp["p_draw"], "B": lp["p_b"]}
        pick = max(probs, key=probs.get)
        p_res = min(max(probs[res], 1e-9), 1.0)
        brier = sum((probs[k] - (1.0 if k == res else 0.0)) ** 2 for k in probs)
        label = {"A": lp["team_a"], "B": lp["team_b"], "D": "Draw"}
        games.append({
            "team_a": lp["team_a"], "team_b": lp["team_b"],
            "p_a": lp["p_a"], "p_draw": lp["p_draw"], "p_b": lp["p_b"],
            "result": res, "result_label": label[res],
            "model_pick": pick, "model_pick_label": label[pick],
            "model_pick_prob": round(probs[pick], 4),
            "correct": pick == res,
            "brier": round(brier, 4), "logloss": round(-math.log(p_res), 4),
        })
    games.sort(key=lambda g: (g["team_a"], g["team_b"]))
    n = len(games)
    summary = {
        "n": n,
        "accuracy": round(sum(g["correct"] for g in games) / n, 4) if n else None,
        "avg_brier": round(sum(g["brier"] for g in games) / n, 4) if n else None,
        "avg_logloss": round(sum(g["logloss"] for g in games) / n, 4) if n else None,
        "random_logloss": round(-math.log(1 / 3), 4),
    }
    return {"generated_at": generated_at, "games": games, "summary": summary}


def _build_fixtures(market: pd.DataFrame, predictor) -> list[dict]:
    """Upcoming fixtures (real teams only) with the model's W/D/L per match."""
    if market.empty:
        return []
    mm = market[market["market_type"] == "match"]
    out: dict[str, dict] = {}
    for r in mm.itertuples(index=False):
        a, b = _parse_fixture(r.contract)
        if not a or not b:
            continue
        if a not in predictor.dc.team_idx or b not in predictor.dc.team_idx:
            continue  # skip knockout placeholders (TBD slots)
        key = _fixture_key(r.market_id)
        if key in out:
            continue
        wdl = predictor.predict_wdl(a, b, neutral=True, host_a=a in HOST_NATIONS)
        out[key] = {
            "key": key, "date": _ticker_date(r.market_id),
            "team_a": a, "team_b": b,
            "p_a": round(float(wdl[0]), 4),
            "p_draw": round(float(wdl[1]), 4),
            "p_b": round(float(wdl[2]), 4),
            "result": None,
        }
    return sorted(out.values(), key=lambda f: (f["date"] or "9999", f["team_a"]))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fetch_markets() -> pd.DataFrame:
    return union_markets([
        polymarket_world_cup(),
        kalshi_series("KXWCROUND", market_type="reach_round"),
        kalshi_series("KXWCGAME", market_type="match"),
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
    opps = build_opportunities(sim, market, predictor=pred, lam=lam)
    edges = {
        "generated_at": generated_at,
        "lambda": lam,
        "opportunities": opps.to_dict(orient="records") if not opps.empty else [],
    }

    # Fixtures for the game/scorecard include settled matches too (completed
    # games), so the model's locked prediction can be scored against results.
    settled_matches = kalshi_series("KXWCGAME", market_type="match", status="settled")
    fixtures = _build_fixtures(union_markets([market, settled_matches]), pred)
    matches = {"generated_at": generated_at, "fixtures": fixtures}
    results_map = game_results()
    results = {"generated_at": generated_at, "markets": results_map}
    locked = _lock_predictions(fixtures, generated_at)
    scorecard = _scorecard(locked, results_map, generated_at)

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
        "n_fixtures": len(matches["fixtures"]),
        "n_results": len(results["markets"]),
        "scorecard": scorecard["summary"],
        "note": "Probabilistic model output. Paper-trading only, pre-cost edges.",
    }

    return {"tournament": tournament, "edges": edges, "matches": matches,
            "results": results, "scorecard": scorecard, "ledger": ledger_snap,
            "meta": meta}
