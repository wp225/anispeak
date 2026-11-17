from __future__ import annotations

import os


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader without external dependency.

    Does not overwrite already-exported environment variables.
    Supports KEY=VALUE with optional single/double quotes.
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
                v = v[1:-1]
            os.environ.setdefault(k, v)
