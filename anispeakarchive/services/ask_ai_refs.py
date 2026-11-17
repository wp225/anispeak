from __future__ import annotations

import re
import unicodedata
from typing import Any


_RX_NON_WORD = re.compile(r"[^A-Za-z0-9]+")
_RX_LASTNAME_SPLIT = re.compile(r"\s+")


def doi_to_url(doi: str | None) -> str | None:
    if doi is None:
        return None
    value = str(doi).strip()
    if not value:
        return None
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.lower().startswith(prefix):
            value = value[len(prefix) :]
            break
    value = value.strip()
    if not value:
        return None
    return f"https://doi.org/{value}"


def normalize_author_name(author: str) -> str:
    return unicodedata.normalize("NFKC", str(author or "")).strip()


def first_author_last_name(authors: list[str]) -> str:
    if not authors:
        return "Unknown"
    first = normalize_author_name(authors[0])
    if not first:
        return "Unknown"
    if "," in first:
        return _RX_NON_WORD.sub("", first.split(",", 1)[0]) or "Unknown"
    parts = [p for p in _RX_LASTNAME_SPLIT.split(first) if p]
    return _RX_NON_WORD.sub("", parts[-1]) or "Unknown"


def _short_title_token(title: str) -> str:
    words = [w for w in _RX_NON_WORD.split(unicodedata.normalize("NFKC", str(title or ""))) if w]
    if not words:
        return "Paper"
    for word in words:
        if len(word) >= 4:
            return word[:20]
    return words[0][:20]


def escape_bibtex_value(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{").replace("}", "\\}")
    text = text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    return text


def make_bibtex_key(record: dict[str, Any], used_keys: set[str]) -> str:
    authors = record.get("authors") or []
    year = record.get("year")
    title = record.get("title") or ""
    base = f"{first_author_last_name(authors)}{year if year else 'NoYear'}{_short_title_token(str(title))}"
    key = _RX_NON_WORD.sub("", base) or "UnknownNoYearPaper"
    if key not in used_keys:
        used_keys.add(key)
        return key
    suffix_ord = ord("b")
    while True:
        candidate = f"{key}{chr(suffix_ord)}"
        if candidate not in used_keys:
            used_keys.add(candidate)
            return candidate
        suffix_ord += 1


def _bibtex_entry_type(record: dict[str, Any]) -> str:
    paper_type = str(record.get("paper_type") or "").lower()
    if "conference" in paper_type or "proceeding" in paper_type:
        return "inproceedings"
    if "book" in paper_type:
        return "book"
    if "preprint" in paper_type or "manuscript" in paper_type:
        return "misc"
    return "article"


def _author_field(authors: list[str]) -> str | None:
    cleaned = [normalize_author_name(a) for a in authors if normalize_author_name(a)]
    if not cleaned:
        return None
    return " and ".join(escape_bibtex_value(a) for a in cleaned)


def make_bibtex(record: dict[str, Any], citation_key: str) -> str:
    entry_type = _bibtex_entry_type(record)
    fields: list[tuple[str, str]] = []
    title = str(record.get("title") or "").strip()
    if title:
        fields.append(("title", escape_bibtex_value(title)))
    author_field = _author_field(record.get("authors") or [])
    if author_field:
        fields.append(("author", author_field))
    year = record.get("year")
    if year:
        fields.append(("year", escape_bibtex_value(str(year))))
    venue = str(record.get("venue") or "").strip()
    if venue:
        fields.append(("booktitle" if entry_type == "inproceedings" else "journal", escape_bibtex_value(venue)))
    doi = str(record.get("doi") or "").strip()
    if doi:
        fields.append(("doi", escape_bibtex_value(doi)))
    url = str(record.get("url") or "").strip()
    if url:
        fields.append(("url", escape_bibtex_value(url)))

    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)
    return f"@{entry_type}{{{citation_key},\n{body}\n}}"


def reference_url(record: dict[str, Any]) -> str | None:
    return (
        doi_to_url(record.get("doi"))
        or (str(record.get("landing_page_url")).strip() if record.get("landing_page_url") else None)
        or (str(record.get("oa_url")).strip() if record.get("oa_url") else None)
        or (str(record.get("pdf_url")).strip() if record.get("pdf_url") else None)
    )
