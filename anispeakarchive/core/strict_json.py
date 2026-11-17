from __future__ import annotations

import json
from typing import Any

from flask.json.provider import DefaultJSONProvider


class StrictJSONProvider(DefaultJSONProvider):
    """JSON provider that forbids NaN/Infinity.

    The JSON spec does not allow NaN/Infinity. Some Python encoders emit them
    as bare identifiers, which breaks JSON.parse() in browsers.
    """

    def dumps(self, obj: Any, **kwargs: Any) -> str:  # type: ignore[override]
        # Enforce strict JSON.
        kwargs.setdefault("allow_nan", False)
        return json.dumps(obj, **kwargs)
