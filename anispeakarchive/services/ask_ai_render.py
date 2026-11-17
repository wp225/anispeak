from __future__ import annotations

import re

from markdown_it import MarkdownIt


_MD = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": True})
_RX_CITATION = re.compile(r"\[(\d+)\]")


def render_answer_markdown(markdown_text: str) -> str:
    text = str(markdown_text or "").strip()
    if not text:
        return ""
    html = _MD.render(text)
    html = _RX_CITATION.sub(r'<span class="ask-citation">[\1]</span>', html)
    return html
