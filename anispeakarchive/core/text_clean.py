from __future__ import annotations

import re
from html import unescape
from typing import Any, Optional


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(text: Any) -> Optional[str]:
    """Remove HTML tags and unescape entities.

    Returns None if input is None.
    """
    if text is None:
        return None
    s = str(text)
    if not s:
        return ""
    s = unescape(s)
    return _TAG_RE.sub("", s)


def normalize_retrieval_text(text: Any) -> str:
    if text is None:
        return ""
    s = strip_html(text)
    if s is None:
        return ""
    s = s.replace("\x00", " ")
    s = _WS_RE.sub(" ", s).strip()
    return s
