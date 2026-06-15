import numpy as np

from wcmodel.models.calibrate import all_metrics


def test_all_metrics_reports_top_pick_accuracy():
    y = np.array([0, 1, 2])
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.2, 0.6, 0.2],
        [0.5, 0.2, 0.3],
    ])

    assert all_metrics(y, probs)["accuracy"] == 2 / 3
