from __future__ import annotations

import math
from typing import Any

import numpy as np


def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize an object so it is valid JSON.

    Flask's JSON encoder may emit bare NaN/Infinity (JavaScript identifiers),
    which is invalid JSON and crashes JSON.parse() in browsers.

    Policy:
      - NaN / +/-Inf -> None
      - numpy scalars -> python scalars
      - numpy arrays -> lists
    """

    # numpy scalars
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(x) for x in obj.tolist()]

    # floats
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # containers
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]

    return obj


def assert_json_strict(obj: Any) -> None:
    """Fail fast if a response still contains NaN/Inf after sanitization.

    This prevents the frontend from crashing with JSON.parse errors and
    surfaces the underlying bug server-side.
    """

    # numpy scalars
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()

    if isinstance(obj, float):
        assert not math.isnan(obj), "JSON contains NaN"
        assert not math.isinf(obj), "JSON contains Infinity"
        return

    if isinstance(obj, np.ndarray):
        for x in obj.tolist():
            assert_json_strict(x)
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            assert k is not None
            assert_json_strict(v)
        return

    if isinstance(obj, (list, tuple)):
        for x in obj:
            assert_json_strict(x)
        return

    # everything else is JSON-safe by construction (None/bool/int/str)
