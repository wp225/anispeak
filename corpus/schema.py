from __future__ import annotations

import re
from typing import Any, Dict, Optional
from corpus.validation import require, require_str


_ID_RE = re.compile(r"(W\d+)$")


def _normalize_oa_id(oa_id_or_url: str) -> str:
    s = require_str(oa_id_or_url, "id")
    m = _ID_RE.search(s)
    require(m is not None, f"Could not parse OpenAlex work id from: {s}")
    return m.group(1)


def _doi_to_str(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    d = doi.strip()
    d = d.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return d if d else None


def _abstract_from_inverted_index(inv: Optional[dict]) -> Optional[str]:
    if not inv or not isinstance(inv, dict):
        return None
    pos_to_tok: dict[int, str] = {}
    for tok, poses in inv.items():
        if not isinstance(tok, str) or not isinstance(poses, list):
            continue
        for p in poses:
            if isinstance(p, int):
                pos_to_tok[p] = tok
    if not pos_to_tok:
        return None
    max_pos = max(pos_to_tok.keys())
    words = [pos_to_tok[i] for i in range(max_pos + 1) if i in pos_to_tok]
    text = " ".join(words).strip()
    return text if text else None


def _extract_authors_affils_insts(authorships: Any) -> tuple[list[str], list[str], list[str]]:
    authors: list[str] = []
    affils: list[str] = []
    insts: list[str] = []

    if not isinstance(authorships, list):
        return authors, affils, insts

    for a in authorships:
        if not isinstance(a, dict):
            continue

        author = a.get("author")
        if isinstance(author, dict):
            name = author.get("display_name")
            if isinstance(name, str) and name.strip():
                authors.append(name.strip())

        ras = a.get("raw_affiliation_strings")
        if isinstance(ras, list):
            for s in ras:
                if isinstance(s, str) and s.strip():
                    affils.append(s.strip())

        institutions = a.get("institutions")
        if isinstance(institutions, list):
            for inst in institutions:
                if isinstance(inst, dict):
                    dn = inst.get("display_name")
                    if isinstance(dn, str) and dn.strip():
                        insts.append(dn.strip())

    def uniq(xs: list[str]) -> list[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return uniq(authors), uniq(affils), uniq(insts)


def _facet_list(raw: Dict[str, Any], field_name: str, key: str) -> list[str]:
    """Normalize a facet field into a de-duplicated list[str], checking both root and ai_metadata."""
    assert isinstance(raw, dict), "Input raw must be a dictionary"

    items = raw.get(field_name)
    if not items:
        items = (raw.get("ai_metadata") or {}).get(field_name)

    if not isinstance(items, list):
        return []

    out: list[str] = []
    for it in items:
        if isinstance(it, str):
            s = it.strip()
            if s:
                out.append(s)
            continue
        if isinstance(it, dict):
            v = it.get(key)
            if isinstance(v, str) and v.strip():
                out.append(v.strip())

    seen: set[str] = set()
    uniq: list[str] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)

    assert isinstance(uniq, list), "Facet list must return a list of strings"
    return uniq


def to_web_doc(raw: Dict[str, Any], include_abstract: bool, include_affiliations: bool) -> Dict[str, Any]:
    assert isinstance(raw, dict), "Input must be a dictionary"
    doc_id = _normalize_oa_id(raw.get("id") or raw.get("oa_id") or "")
    title = require_str(raw.get("title") or raw.get("display_name") or "", "title")

    doi = _doi_to_str(raw.get("doi"))
    year = raw.get("publication_year") or raw.get("year")
    if year is not None:
        require(isinstance(year, int), f"publication_year must be int or None: {year}")

    ai_summary = raw.get("summary") or raw.get("ai_summary") or (raw.get("ai_metadata") or {}).get("summary") or ""
    if not isinstance(ai_summary, str):
        ai_summary = ""

    abstract_text = _abstract_from_inverted_index(raw.get("abstract_inverted_index")) if include_abstract else None

    authors, affiliations, institutions = _extract_authors_affils_insts(raw.get("authorships"))
    if not authors and isinstance(raw.get("authors"), list):
        authors = [a for a in raw.get("authors") if isinstance(a, str)]

    species = _facet_list(raw, "species", "name")
    species_group = _facet_list(raw, "species", "category")
    stages = _facet_list(raw, "stages", "stage")
    features = _facet_list(raw, "features", "feature")
    modalities = _facet_list(raw, "modalities", "modality")
    domains = _facet_list(raw, "domains", "domain")
    contexts = _facet_list(raw, "contexts", "context")

    research_tradition = raw.get("research_tradition") or (raw.get("ai_metadata") or {}).get("research_tradition")
    paper_type = raw.get("paper_type") or (raw.get("ai_metadata") or {}).get("paper_type")
    social_scale = raw.get("social_scale") or (raw.get("ai_metadata") or {}).get("social_scale")

    tokens: list[str] = []
    for s in species:
        tokens.append(f"SPECIES:{s}")
    for g in species_group:
        tokens.append(f"SPECIES_GROUP:{g}")
    for s in stages:
        tokens.append(f"STAGE:{s}")
    for f in features:
        tokens.append(f"FEATURE:{f}")
    for m in modalities:
        tokens.append(f"MODALITY:{m}")
    for d in domains:
        tokens.append(f"DOMAIN:{d}")
    for c in contexts:
        tokens.append(f"CONTEXT:{c}")
    if isinstance(research_tradition, str) and research_tradition.strip():
        tokens.append(f"TRADITION:{research_tradition.strip()}")
    if isinstance(paper_type, str) and paper_type.strip():
        tokens.append(f"PAPER_TYPE:{paper_type.strip()}")
    if isinstance(social_scale, str) and social_scale.strip():
        tokens.append(f"SOCIAL_SCALE:{social_scale.strip()}")

    parts = [title]
    if abstract_text:
        parts.append(abstract_text)
    if ai_summary.strip():
        parts.append(ai_summary.strip())
    if authors:
        parts.append("AUTHORS: " + ", ".join(authors))
    if include_affiliations and institutions:
        parts.append("INSTITUTIONS: " + ", ".join(institutions))
    if include_affiliations and affiliations:
        parts.append("AFFILIATIONS: " + " | ".join(affiliations[:20]))
    if tokens:
        parts.append(" ".join(tokens))

    text = "\n\n".join(parts).strip()
    require_str(text, "text")

    best_loc = raw.get("best_oa_location") or raw.get("primary_location") or {}
    if not isinstance(best_loc, dict):
        best_loc = {}

    venue = None
    host_venue = raw.get("host_venue")
    if isinstance(host_venue, dict):
        vn = host_venue.get("display_name")
        if isinstance(vn, str) and vn.strip():
            venue = vn.strip()
    if venue is None:
        src = best_loc.get("source")
        if isinstance(src, dict):
            vn = src.get("display_name")
            if isinstance(vn, str) and vn.strip():
                venue = vn.strip()
    if venue is None:
        vn = raw.get("venue")
        if isinstance(vn, str) and vn.strip():
            venue = vn.strip()

    ai_meta = raw.get("ai_metadata") or {}
    has_dataset = bool(raw.get("has_dataset") or ai_meta.get("has_dataset"))
    has_code = bool(raw.get("has_code") or ai_meta.get("has_code"))
    has_benchmark = bool(raw.get("has_benchmark") or ai_meta.get("has_benchmark"))

    web = {
        "doc_id": doc_id,
        "oa_url": f"https://openalex.org/{doc_id}",
        "doi": doi,
        "title": title,
        "year": year,
        "publication_date": raw.get("publication_date"),
        "type": raw.get("type"),
        "cited_by_count": raw.get("cited_by_count"),
        "authors": authors,
        "institutions": institutions,
        "affiliations": affiliations,
        "landing_page_url": best_loc.get("landing_page_url"),
        "pdf_url": best_loc.get("pdf_url"),
        "venue": venue,
        "relevance_score": raw.get("relevance_score") or ai_meta.get("relevance_score"),
        "summary": ai_summary.strip(),
        "abstract_text": abstract_text,
        "species": species,
        "species_group": species_group,
        "stages": stages,
        "features": features,
        "modalities": modalities,
        "domains": domains,
        "contexts": contexts,
        "research_tradition": research_tradition,
        "paper_type": paper_type,
        "social_scale": social_scale,
        "has_dataset": has_dataset,
        "has_code": has_code,
        "has_benchmark": has_benchmark,
        "text": text,
    }

    assert "summary" in web, "Payload must contain a summary field"
    return web
