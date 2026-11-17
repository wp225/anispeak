from __future__ import annotations

import re

from flask import Blueprint, current_app, make_response, request

from anispeakarchive.core.json_sanitize import sanitize_for_json


bp = Blueprint("citation", __name__, url_prefix="/api/citation")


def _svc():
    return current_app.config["SEARCH_SERVICE"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _bibtex_key(authors: list[str], year: int | None, title: str) -> str:
    base = "paper"
    if authors:
        last = authors[0].split()[-1]
        base = re.sub(r"[^A-Za-z0-9]+", "", last) or base
    y = str(year) if year else "n.d."
    t0 = re.sub(r"[^A-Za-z0-9]+", "", (title.split()[:1] or [""])[0])
    t0 = t0[:12] if t0 else "work"
    return f"{base}{y}{t0}"


def _bibtex(p: dict) -> str:
    title = _norm(p.get("title") or p.get("display_name") or "")
    year = p.get("year")
    try:
        year_i = int(year) if year is not None else None
    except Exception:
        year_i = None
    authors = p.get("authors") or []
    if not isinstance(authors, list):
        authors = []
    authors_s = [str(a).strip() for a in authors if str(a).strip()]
    venue = _norm(p.get("venue") or p.get("host_venue") or "")
    doi = str(p.get("doi") or "").replace("https://doi.org/", "").strip()
    url = str(p.get("landing_page_url") or "").strip()
    key = _bibtex_key(authors_s, year_i, title)

    fields = {
        "title": title,
        "author": " and ".join(authors_s) if authors_s else None,
        "year": str(year_i) if year_i else None,
        "journal": venue or None,
        "doi": doi or None,
        "url": url or None,
    }
    body = "\n".join([f"  {k} = {{{v}}}," for k, v in fields.items() if v])
    return f"@article{{{key},\n{body}\n}}\n"


def _apa(p: dict) -> str:
    title = _norm(p.get("title") or p.get("display_name") or "")
    year = p.get("year")
    year_s = str(year) if year else "n.d."
    authors = p.get("authors") or []
    if not isinstance(authors, list):
        authors = []
    authors_s = [str(a).strip() for a in authors if str(a).strip()]
    venue = _norm(p.get("venue") or p.get("host_venue") or "")
    doi = str(p.get("doi") or "").replace("https://doi.org/", "").strip()
    url = str(p.get("landing_page_url") or "").strip()
    link = f"https://doi.org/{doi}" if doi else (url or "")
    a = ", ".join(authors_s) if authors_s else ""
    parts = [x for x in [a, f"({year_s}).", f"{title}.", venue + "." if venue else None, link] if x]
    return " ".join(parts).strip() + "\n"


@bp.get("/<doc_id>")
def citation(doc_id: str):
    fmt = str(request.args.get("format", "bibtex")).strip().lower()
    if fmt not in {"bibtex", "apa"}:
        fmt = "bibtex"

    p = _svc().get_paper(doc_id)
    if not isinstance(p, dict):
        return make_response("Not found", 404)

    text = _bibtex(p) if fmt == "bibtex" else _apa(p)
    resp = make_response(text, 200)
    resp.headers["Content-Type"] = "text/plain; charset=utf-8"
    resp.headers["Content-Disposition"] = f"attachment; filename=\"{doc_id}.{ 'bib' if fmt=='bibtex' else 'txt'}\""
    return resp
