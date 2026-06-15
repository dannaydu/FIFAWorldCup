"""Central configuration: paths, Elo constants, and the 2026 tournament format.

Everything that is "a knob" lives here so the rest of the code reads cleanly.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ARTIFACTS = ROOT / "artifacts"

for _p in (RAW, PROCESSED, ARTIFACTS):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Elo (World Football Elo Ratings conventions — eloratings.net)
# --------------------------------------------------------------------------- #
ELO_START = 1500.0          # rating for a team with no history
ELO_HOME_ADVANTAGE = 100.0  # added to the home side's rating in the expectation

# K-factor by match importance (eloratings.net weight classes).
ELO_K_BY_IMPORTANCE = {
    "world_cup": 60.0,        # World Cup finals
    "continental": 50.0,      # continental championship finals / major intercontinental
    "qualifier": 40.0,        # World Cup & continental qualifiers + major tournaments
    "minor_tournament": 30.0, # other tournaments (Nations League group, Gold Cup, etc.)
    "friendly": 20.0,         # friendlies
}
ELO_DEFAULT_IMPORTANCE = "friendly"


# --------------------------------------------------------------------------- #
# 2026 tournament format (FIFA: 48 teams, 12 groups of 4, 32-team knockout)
# --------------------------------------------------------------------------- #
N_TEAMS = 48
N_GROUPS = 12
TEAMS_PER_GROUP = 4
QUALIFY_PER_GROUP = 2        # top two of every group
N_BEST_THIRDS = 8           # plus the eight best third-placed teams
KNOCKOUT_SIZE = 32          # -> Round of 32

GROUP_LABELS = [chr(ord("A") + i) for i in range(N_GROUPS)]  # A..L

POINTS_WIN = 3
POINTS_DRAW = 1
POINTS_LOSS = 0

# Monte-Carlo tournament simulation default.
N_SIMULATIONS = 50_000

# Stage labels tracked for every team during simulation.
STAGES = [
    "group",        # eliminated in group stage
    "round_32",
    "round_16",
    "quarterfinal",
    "semifinal",
    "final",        # lost the final (runner-up)
    "champion",
]


# --------------------------------------------------------------------------- #
# Feature / model defaults
# --------------------------------------------------------------------------- #
# Recency weighting for "last 4 years of form" (see spec §4B). Half-life in days.
FORM_HALFLIFE_DAYS = 365.0
FORM_LOOKBACK_DAYS = 4 * 365

# Dixon-Coles time-decay (per day). Smaller = longer memory.
DC_TIME_DECAY = 0.0018       # ~ half-life of ~1 year
# Drop matches older than this from the DC MLE only (they carry ~0 weight under
# the decay above). Elo + form features still use the FULL history so ratings
# stay cross-confederation calibrated.
DC_FIT_MAX_AGE_YEARS = 14.0

# Ensemble weights (spec §5), selected on 2022-23 and checked on a held-out
# 2024+ time split. GBM is strongest for W/D/L; Dixon-Coles still supplies the
# score-matrix shape in the simulator, so it keeps a small non-zero weight.
# `market` only applies where a match market exists; the simulator renormalizes
# over the other three.
ENSEMBLE_WEIGHTS = {
    "elo": 0.12,
    "poisson": 0.04,
    "gbm": 0.64,
    "market": 0.20,
}


# --------------------------------------------------------------------------- #
# Market edge engine thresholds (spec §10)
# --------------------------------------------------------------------------- #
MIN_EDGE = 0.04             # 4 percentage points
MAX_SPREAD = 0.03           # 3 cents
MIN_LIQUIDITY = 500.0       # dollars
KELLY_FRACTION = 0.25       # fractional Kelly
MAX_POSITION_FRACTION = 0.01  # hard cap: 1% of bankroll per position

# Market-prior shrinkage: blend model probs toward the (de-vigged) market before
# computing edges. The market embeds squad/cross-confederation info the model is
# weak on, so trust it partly. 0 = pure model, 1 = pure market. Bet only on the
# residual disagreement. Tune on closing-line value.
MARKET_SHRINKAGE_LAMBDA = 0.5
