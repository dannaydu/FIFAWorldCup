import json

import numpy as np

from wcmodel.json_utils import json_safe


def test_json_safe_replaces_non_finite_and_numpy_values():
    value = {
        "round": np.nan,
        "score": np.float64(0.75),
        "items": [np.int64(2), float("inf"), float("-inf")],
    }

    safe = json_safe(value)

    assert safe == {"round": None, "score": 0.75, "items": [2, None, None]}
    assert json.loads(json.dumps(safe, allow_nan=False)) == safe
