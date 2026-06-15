"""Write the web UI's data snapshots as static JSON.

    python scripts/export_web.py

Outputs web/public/data/{tournament,edges,ledger,meta}.json. These are bundled
with Firebase Hosting so the UI works immediately; the publisher
(publish_firestore.py) pushes the same data to Firestore for live updates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wcmodel.web_export import build_snapshots  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "web" / "public" / "data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building snapshots (train + simulate + live markets)…")
    snaps = build_snapshots()
    for name, doc in snaps.items():
        (OUT / f"{name}.json").write_text(json.dumps(doc, indent=2))
        print(f"  wrote {OUT / f'{name}.json'}")

    # Append a compact, time-stamped history row — saved on every refresh.
    hist_path = OUT / "history.json"
    try:
        hist = json.loads(hist_path.read_text())
    except Exception:
        hist = []
    champ = sorted(snaps["tournament"]["teams"], key=lambda t: -t["p_champion"])[:5]
    hist.append({
        "t": snaps["meta"]["generated_at"],
        "champion_top": [{"team": c["team"], "p": c["p_champion"]} for c in champ],
        "n_opportunities": snaps["meta"]["n_opportunities"],
        "scorecard": snaps["scorecard"]["summary"],
    })
    hist = hist[-300:]
    hist_path.write_text(json.dumps(hist, indent=2))
    print(f"  appended history.json ({len(hist)} snapshots)")
    print("Done. Run `firebase deploy` (or open web/public/index.html) to view.")


if __name__ == "__main__":
    main()
