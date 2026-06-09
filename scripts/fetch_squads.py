"""Download real Transfermarkt squad/market-value data — no Kaggle auth needed.

The Kaggle "player-scores" dataset (davidcariboo) is Transfermarkt-derived and
auto-refreshed. Kaggle's public download endpoint 302-redirects to a *pre-signed*
Google Cloud Storage URL, so it can be fetched with no credentials:

    python scripts/fetch_squads.py

Writes data/raw/players.csv (+ national_teams.csv, countries.csv). After this,
features/squad_features.py uses real market values automatically. The signed URL
expires after ~3 days, so just rerun this if the download 403s.

(Transfermarkt has no official API and actively blocks scrapers; community
wrappers and Sofascore were both unreachable. This published dataset is the
practical, license-clean route — Transfermarkt data via Kaggle's CC distribution.)
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
URL = "https://www.kaggle.com/api/v1/datasets/download/davidcariboo/player-scores"
WANT = ["players.csv", "player_valuations.csv", "national_teams.csv", "countries.csv"]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading player-scores archive from Kaggle (no auth)…")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
    except Exception as e:
        sys.exit(f"download failed ({e}); the signed URL may have expired — rerun.")

    print(f"  got {len(data) / 1e6:.0f} MB; extracting {WANT}…")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        members = set(z.namelist())
        for name in WANT:
            if name in members:
                z.extract(name, RAW)
                print(f"  -> {RAW / name}")
            else:
                print(f"  (skip, not in archive: {name})")
    print("Done. Re-run scripts/edge_report.py to use real squad values.")


if __name__ == "__main__":
    main()
