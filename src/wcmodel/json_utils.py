"""Helpers for producing standards-compliant JSON / Firestore values."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def json_safe(value):
    """Recursively replace NumPy scalars and non-finite floats with JSON values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value
