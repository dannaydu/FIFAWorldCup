# Probabilistic 2026 FIFA World Cup Forecasting + Prediction-Market Edge Detection

A probabilistic forecasting system — **not** a "who wins" predictor. It estimates
match-score probabilities, simulates the full 48-team tournament with Monte Carlo,
and compares its probabilities to Kalshi / Polymarket prices to flag contracts
where the estimated edge survives fees and spread. **Paper-trade first.**

> The whole point: the model is only useful if it beats the *market* probability,
> not if it merely says good teams are good.

Built around an ensemble (Elo logistic + Dixon–Coles Poisson + gradient boosting +
market baseline), recency- and importance-weighted form, squad value & roster
continuity, and a tournament simulator that implements the real 2026 format
(12 groups of 4, top two + the **eight best third-placed teams** → Round of 32).

> **Data is not committed.** Large/licensed datasets are gitignored. On a fresh
> clone, run `pip install -r requirements.txt` then `python scripts/fetch_squads.py`;
> match results auto-download on first run. **No API keys required** — every data
> source (Kalshi, Polymarket, Kaggle, GitHub) is read keyless.

---

## Quickstart

```bash
pip install -r requirements.txt          # numpy/pandas/scipy/scikit-learn (lightgbm optional)
python scripts/fetch_squads.py           # real Transfermarkt squad values (no Kaggle auth)
python scripts/run_demo.py               # full pipeline on local data (synthetic if none)
python scripts/edge_report.py            # REAL data + real draw + LIVE Kalshi/Polymarket edges
python scripts/paper_trade.py            # log live edges to a persistent ledger + track CLV
pytest -q                                # 12 invariant/sanity tests
```

Match results are auto-pulled from a public GitHub mirror on first run; squad
values come from `fetch_squads.py` (optional — falls back to an Elo proxy).

`edge_report.py` pulls real international results, simulates the real 2026 draw,
fetches live Polymarket/Kalshi prices (no API key), de-vigs them, shrinks the
model toward the market prior, and prints residual edges — all in ~17s.

`run_demo.py` exercises the entire pipeline: load → Elo → features → backtest
(with calibration) → train ensemble → single-match prediction → tournament
simulation → market-edge detection → paper trade. It works **out of the box** on
synthetic data, then transparently uses real data once you drop it in.

Dashboard:

```bash
pip install streamlit
streamlit run src/wcmodel/app/streamlit_dashboard.py
```

---

## What it outputs

| Output | Where |
|---|---|
| Match outcome P(win/draw/loss) | `predictor.predict_wdl(a, b)` |
| Exact-score probabilities | `predictor.score_matrix(a, b)` → `simulate.match.top_scorelines` |
| Tournament probabilities (group win, reach R32…final, champion) | `simulate.tournament.simulate_tournament` |
| Market edge (model − market, filtered) | `markets.edge.find_edges` |
| Paper-trading ROI + closing-line value | `markets.paper_trading.PaperTradingLog` |

---

## Plugging in real data

The loaders fall back to synthetic data when a file is missing, so you can adopt
sources incrementally. Drop files into `data/raw/`:

| File | Source | Used for |
|---|---|---|
| `results.csv` | [Kaggle: international results 1872–2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | match history → Elo, form, Dixon–Coles |
| `players.csv` | `scripts/fetch_squads.py` (Kaggle player-scores, **no auth**) | squad market value, age, caps |
| `squads.csv` | [FIFA squad announcements](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/all-world-cup-squad-announcements) / Reuters trackers | roster continuity, availability (optional override) |
| market prices | Kalshi / Polymarket APIs (live) | edge detection |

Then **edit `src/wcmodel/teams2026.py`** → `GROUPS_2026` with the official draw
(it currently holds a clearly-marked EXAMPLE draw). The simulator depends only on
that one dict.

`results.csv` is auto-normalized (`ingest/get_matches.py`); team-strength Elo is
computed from it (`ingest/elo.py`, eloratings.net conventions). Live prices come
from `ingest/get_market_prices.py` (Kalshi + Polymarket public endpoints).

---

## Architecture

```
src/wcmodel/
  config.py                 # all knobs: Elo K-factors, format, weights, thresholds
  teams2026.py              # 2026 field, seed Elo, EXAMPLE group draw  <-- edit me
  ingest/
    get_matches.py          # load + normalize match results
    elo.py                  # World Football Elo from scratch
    synthetic.py            # demo-data generator (so everything runs offline)
    get_market_prices.py    # Kalshi + Polymarket clients
  features/
    team_features.py        # leakage-free form: last-10 + day-decayed EWM, SoS
    squad_features.py       # squad value + roster continuity + injury score
    confederations.py       # cross-confederation Elo-calibration offsets
    match_features.py       # A-vs-B diff table + labels
  models/
    elo_logistic.py         # baseline multinomial W/D/L
    poisson.py              # Dixon–Coles MLE -> score matrices (recency-windowed)
    gbm.py                  # LightGBM, falls back to sklearn HistGBM
    ensemble.py             # weighted probability blend
    calibrate.py            # temperature scaling, reliability, log-loss/Brier/RPS
  predictor.py              # MatchPredictor: trains all models, fixture featurizer
  simulate/
    match.py                # scorelines, outcome probs, xG from a score matrix
    tournament.py           # 2026 Monte Carlo; ensemble-reweighted score matrices
  markets/
    implied.py              # price -> (de-vigged) implied probability
    edge.py                 # edge detector with edge/spread/liquidity filters
    shrink.py               # market-prior shrinkage (blend model toward market)
    opportunities.py        # model-vs-market opps across group/champion/reach-round
    paper_trading.py        # fractional-Kelly sizing helpers
    paper_ledger.py         # persistent ledger: mark-to-market + CLV + settle
  evaluate.py               # time-split backtest + metrics
  web_export.py             # build the UI's snapshot documents
  app/streamlit_dashboard.py
scripts/run_demo.py         # local/synthetic end-to-end demo
scripts/edge_report.py      # real data + live market edges
scripts/paper_trade.py      # persistent paper-trading harness (scan/report/settle)
scripts/fetch_squads.py     # no-auth Transfermarkt data download
scripts/export_web.py       # write web/public/data/*.json
scripts/publish_firestore.py # push snapshots to Firestore (live mode)
web/public/                 # Firebase Hosting static UI (HTML/CSS/JS, no build)
firebase.json .firebaserc firestore.rules   # Firebase config
tests/
```

---

## Modelling notes

- **Elo** (`ingest/elo.py`): faithful eloratings.net update — K by match
  importance, goal-difference multiplier, 100-pt home advantage, zero-sum updates
  (tested). Computed on the **full history** so ratings stay cross-confederation
  calibrated (truncating to recent years deflates traditional powers ~100 Elo and
  inflates weak-confederation sides — a real bias we measured and fixed).
- **Dixon–Coles** (`models/poisson.py`): bivariate-Poisson with the low-score
  correction, fit by **weighted MLE** (age decay + importance). The MLE uses a
  recency window (`config.DC_FIT_MAX_AGE_YEARS`) for speed; Elo/form still use all
  history. Produces the score matrix that prices score markets.
- **Gradient boosting** (`models/gbm.py`): LightGBM on the diff features; falls
  back to scikit-learn's `HistGradientBoostingClassifier` if LightGBM is absent.
  Training is deterministic, and the fallback is regularized against the
  overfitting seen in chronological backtests.
- **Draw-aware context**: the W/D/L models get the absolute Elo gap explicitly,
  so an evenly matched fixture is distinguishable from merely knowing which
  side is stronger. Known pre-match competition type (friendly, qualifier,
  continental tournament, World Cup) is also included because selection and
  draw dynamics differ.
- **Confederation calibration** (`features/confederations.py`): measures, over
  inter-confederation matches only, how much each confederation over/under-performs
  its Elo (AFC −0.04, OFC −0.13, UEFA +0.09, CONMEBOL +0.08). Feeds the models a
  `conf_strength_diff` feature to discount inflated ratings.
- **Ensemble-driven simulation**: the simulator no longer samples from Dixon–Coles
  alone — each fixture's W/D/L is the full ensemble (Elo + DC + GBM) and the DC
  score matrix is **reshaped to those marginals**, so Elo/GBM signal actually
  reaches the tournament probabilities.
- **Real, era-correct squad values** (`features/squad_features.py`): per-nation
  market value / age / caps from the Transfermarkt player-scores dataset
  (no-auth download via `scripts/fetch_squads.py`). Training uses a per-year
  panel (`player_valuations.csv`) so a 2014 match gets 2014 valuations, not 2026;
  inference uses the current snapshot.
- **Tuned ensemble weights**: a real-data 2023+ grid search showed GBM strongest
  and Dixon–Coles weakest for W/D/L, so weights are GBM-dominant
  (`config.ENSEMBLE_WEIGHTS`); DC keeps a small weight because it supplies the
  score-matrix shape the simulator needs.
- **Market-prior shrinkage** (`markets/shrink.py`): blends model probabilities
  toward the de-vigged market (the market embeds info the model is weak on) and
  bets only the residual. `config.MARKET_SHRINKAGE_LAMBDA`; tune on closing-line
  value.
- **Leakage control**: every form feature is shifted to use only prior matches;
  backtests use **time-based** splits (`evaluate.run_backtest`), never random.
- **Fresh results on production runs**: the web exporter, edge report, and paper
  trader refresh the public results mirror before training, then fall back to
  the existing local file if the network/source is unavailable.
- **Calibration first**: evaluation reports log-loss, Brier, and Ranked
  Probability Score plus a reliability table — "when it says 20%, does it happen
  ~20%?" matters more than accuracy.

## Market edge + risk

`markets/edge.py` keeps a contract only if **edge ≥ 4pp**, **spread ≤ 3¢**, and
**liquidity ≥ $500** (all in `config.py`). Sizing is **fractional Kelly (0.25)**
hard-capped at **1% of bankroll** per position (`markets/paper_trading.py`).

`scripts/paper_trade.py` is the harness you run periodically through the
tournament. Each run trains on real results, simulates the real draw, pulls live
prices, and updates a persistent ledger (`artifacts/paper_ledger.json`):

* **scan** (default): marks every open position to the current market (running
  **closing-line value** = last − entry), and opens one paper position per fresh
  opportunity — never re-betting or taking a contradictory side on a market it
  already holds.
* `--report`: print the ledger summary + open positions.
* `--settle results.json`: book P/L once outcomes are known.

CLV is the cleaner signal that the model is finding value even before any bet
settles, so it accrues from day one. Use it to tune `MARKET_SHRINKAGE_LAMBDA`.

**Auto-settlement**: `paper_trade.py --auto-settle` (and every scan) reads live
market *resolution* status from both venues — Polymarket outcome prices collapsing
to 0/1, Kalshi's settled `result` — and books P/L automatically, so the loop
closes itself as group winners and eliminations resolve.

## Hosting the dashboard (Firebase)

A dependency-free web UI lives in `web/public/`, served by **Firebase Hosting**.
The heavy Python ML never runs on serverless — it computes snapshots the static
UI reads. Tabs:

* **🔮 Play — "Beat the Oracle"**: a fantasy game where each visitor gets a
  virtual 🪙1,000, stakes it on match / group / champion markets at the model's
  implied odds, and tries to beat the model (fade the 🔮 where they disagree).
  State is per-browser (localStorage, no login); cards are shareable via URL.
  Picks **auto-settle** as results land — the exporter writes a `results` snapshot
  (match outcomes parsed from settled Kalshi tickers, group/champion winners from
  Polymarket resolutions) and the game books each player's bankroll on next load.
* **📊 Forecast**: championship + advancement probabilities.
* **✅ Results**: scores the model's **pre-match** predictions (locked before
  kickoff) against actual results — accuracy, Brier, log-loss vs a random guess.
* **💸 Market edges**: live model-vs-market edges, **filterable by venue**
  (Polymarket / Kalshi) with per-row platform badges.
* **🤖 Model bets**: the model's own paper-trading ledger with CLV.

Optional **accounts**: fill in `firebase-config.js` and enable Authentication
(Google) + Firestore, and players can sign in to sync their fantasy card across
devices (rules already scope each user to their own doc). Without it, the game
runs fine on per-browser localStorage.

**Auto-refresh (cron):** `.github/workflows/refresh.yml` re-runs the model on live
data and redeploys every 3 hours during the tournament. One-time setup: add a
`FIREBASE_SERVICE_ACCOUNT` repo secret (Firebase console → Project settings →
Service accounts → Generate key). Match results + squad data download with no
keys; the workflow commits refreshed snapshots (incl. locked predictions) back.

```bash
python scripts/export_web.py             # write web/public/data/*.json
npm i -g firebase-tools && firebase login
# put your project id in .firebaserc, then:
firebase deploy
```

That alone gives a live, hosted dashboard (the UI reads the bundled JSON). For
**live-updating** data without redeploying, switch to Firestore:

```bash
pip install firebase-admin
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/serviceAccount.json
python scripts/publish_firestore.py      # push snapshots -> Firestore
# paste your web config into web/public/firebase-config.js, redeploy once
```

The UI auto-detects: real `firebase-config.js` → reads Firestore live; otherwise
→ bundled JSON. Run `publish_firestore.py` (or `export_web.py` + `firebase
deploy`) on a cron / GitHub Action during the tournament to keep it fresh.

---

## Status & caveats

**Done:** real results (martj42 GitHub mirror) + full-history Elo +
recency/importance-weighted form + confederation calibration + squad/roster
features → Elo-logistic + Dixon–Coles + GBM → **ensemble-driven** 2026
Monte-Carlo simulation (real draw) → de-vig + market-prior shrinkage → live
Kalshi/Polymarket edge detection → paper trading. `scripts/edge_report.py` runs
it on live data.

**Caveats — read before trusting any number:**
- Squad values are **real and era-correct** (Transfermarkt, per-year panel), but
  **roster-continuity and injuries still default to neutral** — they need a
  prior-cycle squad snapshot and a live injury feed. Squad value is a talent-pool
  proxy (top-26 by citizenship), not the literal 26-man call-up.
- Live Kalshi reach-round / match markets exist but are largely **unpriced
  pre-tournament**; Polymarket group markets are the liquid ones today.
- Edges are **pre-cost and pre-execution**. Paper-trade and track closing-line
  value before risking money; `MARKET_SHRINKAGE_LAMBDA` should be tuned on CLV.
- The knockout bracket seeds qualifiers into a standard serpentine 32-team bracket
  — a transparent approximation of FIFA's fixed third-place allocation table. Swap
  `simulate.tournament.build_bracket` for the official slot table for exact geometry.

**Advanced roadmap (spec §13):** player club minutes / xG-xA, projected starting
XI, injury updates, weather & rest/travel, market-movement features, and an LLM
that *extracts structured* injury/lineup updates (never predicts results directly)
to feed the model.
