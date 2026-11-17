from __future__ import annotations

from typing import Any


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def require_str(v: Any, name: str) -> str:
    require(isinstance(v, str), f"{name} must be str, got {type(v)}")
    s = v.strip()
    require(bool(s), f"{name} must be non-empty")
    return s
