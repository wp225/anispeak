from __future__ import annotations

import re
from typing import Any, Dict, Optional
from corpus.validation import require, require_str
from anispeakarchive.core.text_clean import normalize_retrieval_text


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


def _fmt_csv(values: list[str]) -> str:
    cleaned = [normalize_retrieval_text(v) for v in values if normalize_retrieval_text(v)]
    return ", ".join(cleaned)


def _build_retrieval_text(
    *,
    title: str,
    abstract_text: Optional[str],
    ai_summary: str,
    authors: list[str],
    institutions: list[str],
    affiliations: list[str],
    venue: Optional[str],
    year: Optional[int],
    publication_date: Any,
    species: list[str],
    species_group: list[str],
    stages: list[str],
    features: list[str],
    modalities: list[str],
    domains: list[str],
    contexts: list[str],
    research_tradition: Any,
    paper_type: Any,
    social_scale: Any,
    has_dataset: bool,
    has_code: bool,
    has_benchmark: bool,
    include_affiliations: bool,
) -> str:
    parts: list[str] = []

    def add(label: str, value: Any) -> None:
        text = normalize_retrieval_text(value)
        if text:
            parts.append(f"{label}: {text}")

    add("Title", title)
    add("Summary", ai_summary)
    add("Abstract", abstract_text)
    add("Authors", _fmt_csv(authors))
    add("Institutions", _fmt_csv(institutions))
    if include_affiliations:
        add("Affiliations", " | ".join(normalize_retrieval_text(x) for x in affiliations[:20] if normalize_retrieval_text(x)))
    add("Venue", venue)
    if year is not None:
        add("Year", str(year))
    if publication_date is not None:
        add("Publication date", str(publication_date))

    resources: list[str] = []
    if has_dataset:
        resources.append("has dataset")
    if has_code:
        resources.append("has code")
    if has_benchmark:
        resources.append("has benchmark")
    if resources:
        add("Resources", ", ".join(resources))

    add("Species", _fmt_csv(species))
    add("Species group", _fmt_csv(species_group))
    add("Computational stages", _fmt_csv(stages))
    add("Linguistic features", _fmt_csv(features))
    add("Modalities", _fmt_csv(modalities))
    add("Research domains", _fmt_csv(domains))
    add("Research contexts", _fmt_csv(contexts))
    add("Research tradition", research_tradition)
    add("Paper type", paper_type)
    add("Social scale", social_scale)

    token_lines: list[str] = []
    for s in species:
        token_lines.append(f"SPECIES:{normalize_retrieval_text(s)}")
    for g in species_group:
        token_lines.append(f"SPECIES_GROUP:{normalize_retrieval_text(g)}")
    for s in stages:
        token_lines.append(f"STAGE:{normalize_retrieval_text(s)}")
    for f in features:
        token_lines.append(f"FEATURE:{normalize_retrieval_text(f)}")
    for m in modalities:
        token_lines.append(f"MODALITY:{normalize_retrieval_text(m)}")
    for d in domains:
        token_lines.append(f"DOMAIN:{normalize_retrieval_text(d)}")
    for c in contexts:
        token_lines.append(f"CONTEXT:{normalize_retrieval_text(c)}")
    if isinstance(research_tradition, str) and research_tradition.strip():
        token_lines.append(f"TRADITION:{normalize_retrieval_text(research_tradition)}")
    if isinstance(paper_type, str) and paper_type.strip():
        token_lines.append(f"PAPER_TYPE:{normalize_retrieval_text(paper_type)}")
    if isinstance(social_scale, str) and social_scale.strip():
        token_lines.append(f"SOCIAL_SCALE:{normalize_retrieval_text(social_scale)}")
    if resources:
        for item in resources:
            token_lines.append(f"RESOURCE:{normalize_retrieval_text(item)}")
    if venue:
        token_lines.append(f"VENUE:{normalize_retrieval_text(venue)}")
    if year is not None:
        token_lines.append(f"YEAR:{int(year)}")
    for author in authors:
        norm_author = normalize_retrieval_text(author)
        if norm_author:
            token_lines.append(f"AUTHOR:{norm_author}")
    for institution in institutions:
        norm_inst = normalize_retrieval_text(institution)
        if norm_inst:
            token_lines.append(f"INSTITUTION:{norm_inst}")

    if token_lines:
        parts.append("Tags: " + " ".join(t for t in token_lines if t))

    text = "\n".join(parts).strip()
    return normalize_retrieval_text(text)


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
    text = _build_retrieval_text(
        title=title,
        abstract_text=abstract_text,
        ai_summary=ai_summary.strip(),
        authors=authors,
        institutions=institutions,
        affiliations=affiliations,
        venue=venue,
        year=year,
        publication_date=raw.get("publication_date"),
        species=species,
        species_group=species_group,
        stages=stages,
        features=features,
        modalities=modalities,
        domains=domains,
        contexts=contexts,
        research_tradition=research_tradition,
        paper_type=paper_type,
        social_scale=social_scale,
        has_dataset=has_dataset,
        has_code=has_code,
        has_benchmark=has_benchmark,
        include_affiliations=include_affiliations,
    )
    require_str(text, "text")

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
