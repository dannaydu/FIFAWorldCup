"""Publish the model snapshots to Cloud Firestore for live UI updates.

    # one-time: create a Firebase project + a service account key, then:
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/serviceAccount.json
    pip install firebase-admin
    python scripts/publish_firestore.py

Writes documents tournament / edges / ledger / meta into the `snapshots`
collection. The web app reads them live. Run on a schedule (cron / GitHub
Actions) during the tournament to keep the UI fresh without redeploying.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wcmodel.web_export import build_snapshots  # noqa: E402

COLLECTION = "snapshots"


def main() -> None:
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError:
        sys.exit("firebase-admin not installed. Run: pip install firebase-admin")

    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path or not Path(cred_path).exists():
        sys.exit("Set GOOGLE_APPLICATION_CREDENTIALS to your service-account JSON.")

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(cred_path))
    db = firestore.client()

    print("Building snapshots…")
    snaps = build_snapshots()
    for name, doc in snaps.items():
        db.collection(COLLECTION).document(name).set(doc)
        print(f"  published snapshots/{name}")
    print("Done. The hosted UI will pick these up live.")


if __name__ == "__main__":
    main()
