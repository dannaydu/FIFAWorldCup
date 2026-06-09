import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wcmodel.ingest.synthetic import generate_matches  # noqa: E402
from wcmodel.predictor import train_predictor           # noqa: E402


@pytest.fixture(scope="session")
def matches():
    # Smaller than the demo for speed, still enough signal.
    return generate_matches(n_matches=2500, seed=3)


@pytest.fixture(scope="session")
def predictor(matches):
    return train_predictor(matches)
