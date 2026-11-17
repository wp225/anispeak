from __future__ import annotations

import hashlib
import os
import random
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class OtpConfig:
    secret_key: str
    ttl_sec: int


def generate_code() -> str:
    # 6-digit numeric OTP
    return f"{random.randint(0, 999999):06d}"


def _salted(secret_key: str) -> bytes:
    assert isinstance(secret_key, str) and secret_key, "secret_key must be a non-empty string"
    return secret_key.encode("utf-8")


def hash_code(email: str, code: str, secret_key: str) -> str:
    assert isinstance(email, str) and email
    assert isinstance(code, str) and code
    h = hashlib.sha256()
    h.update(_salted(secret_key))
    h.update(b"\n")
    h.update(email.strip().lower().encode("utf-8"))
    h.update(b"\n")
    h.update(code.encode("utf-8"))
    return h.hexdigest()


def now_ts() -> int:
    return int(time.time())


def is_valid_email(email: str) -> bool:
    if not isinstance(email, str):
        return False
    e = email.strip()
    if "@" not in e:
        return False
    if len(e) < 6:
        return False
    # Minimal sanity: local@domain.tld
    local, _, domain = e.partition("@")
    return bool(local) and "." in domain


def domain_of(email: str) -> str:
    assert is_valid_email(email), "email must be valid"
    return email.strip().lower().split("@", 1)[1]
