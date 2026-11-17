from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from typing import List, Tuple

from flask import Blueprint, current_app, jsonify, request
from anispeakarchive.core.json_sanitize import sanitize_for_json

bp = Blueprint("facets", __name__)

# Hard guardrail: returning huge facet lists can choke the browser.
MAX_LIMIT = 20000

# Defaults when ?limit is not provided
DEFAULT_LIMITS = {
    "species": 5000,
    "authors": 8000,
    "institutions": 8000,
    "venue": 5000,
    "venues": 5000,
    "domains": 2000,
    "contexts": 2000,
    "features": 2000,
    "modalities": 2000,
    "stages": 2000,
    "paper_type": 2000,
    "type": 2000,
    "tradition": 2000,
    "research_tradition": 2000,
    "year": 300,
}

# For typeahead we want a cached pool that's big enough to be meaningful.
# NOTE: high-cardinality fields (authors/institutions) must not be truncated too aggressively,
# otherwise many values become unsearchable.
TYPEAHEAD_CACHE_LIMIT_DEFAULT = 20000
TYPEAHEAD_CACHE_LIMIT_BY_FIELD = {
    "authors": 200000,
    "institutions": 200000,
}
_FUZZY_CANDIDATE_CAP_DEFAULT = 20000
_FUZZY_CANDIDATE_CAP_BY_FIELD = {
    "authors": 40000,
    "institutions": 40000,
}

# Prefer rapidfuzz if available (much better than SequenceMatcher).
try:
    from rapidfuzz import fuzz, process  # type: ignore

    _HAS_RAPIDFUZZ = True
except Exception:
    fuzz = None
    process = None
    _HAS_RAPIDFUZZ = False

_RX_NON_ALNUM = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
_RX_WS = re.compile(r"\s+")


def _parse_limit(field: str, default: int | None) -> int | None:
    limit_raw = (request.args.get("limit") or "").strip()
    lim: int | None
    if limit_raw:
        try:
            lim = int(limit_raw)
        except ValueError:
            lim = None
    else:
        lim = default

    if lim is None or lim <= 0:
        return None
    return min(lim, MAX_LIMIT)


def _typeahead_cache_limit(field: str) -> int:
    assert isinstance(field, str) and field, "field must be non-empty"
    v = TYPEAHEAD_CACHE_LIMIT_BY_FIELD.get(field, TYPEAHEAD_CACHE_LIMIT_DEFAULT)
    assert isinstance(v, int) and v > 0, "typeahead cache limit must be positive"
    return v


def _fuzzy_candidate_cap(field: str) -> int:
    assert isinstance(field, str) and field, "field must be non-empty"
    v = _FUZZY_CANDIDATE_CAP_BY_FIELD.get(field, _FUZZY_CANDIDATE_CAP_DEFAULT)
    assert isinstance(v, int) and v > 0, "fuzzy candidate cap must be positive"
    return v


def _norm_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = _RX_NON_ALNUM.sub(" ", s)
    s = _RX_WS.sub(" ", s).strip()
    return s


def _norm_person_name(s: str) -> str:
    # Make "kenny zhu" match "Kenny Q. Zhu" by dropping single-letter tokens.
    s = _norm_text(s)
    if not s:
        return ""
    toks = [t for t in s.split() if len(t) > 1]
    return " ".join(toks)


def _normalize_by_field(field: str, s: str) -> str:
    if field == "authors":
        return _norm_person_name(s)
    # institutions, venue, species, etc.
    return _norm_text(s)


def _fuzzy_score_fallback(q: str, v: str) -> float:
    # q and v are already normalized.
    return SequenceMatcher(None, q, v).ratio()


def _dedupe_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _rapidfuzz_scoring_params(field: str) -> tuple[int, int]:
    """
    Returns (limit_multiplier, threshold) for rapidfuzz.
    We keep thresholds conservative to avoid junk suggestions.
    """
    assert isinstance(field, str) and field, "field must be non-empty"
    # Names are short; token_set_ratio works well but needs a slightly lower threshold.
    if field == "authors":
        return (10, 74)
    # Institutions/venues are longer; keep threshold higher.
    if field in {"institutions", "venue", "venues"}:
        return (8, 78)
    return (6, 80)


@bp.get("/api/facets")
def facets():
    t0 = time.time()
    field = (request.args.get("field") or "").strip()
    if not field:
        return jsonify({"error": "missing_field"}), 400

    limit = _parse_limit(field, DEFAULT_LIMITS.get(field))

    svc = current_app.config.get("SEARCH_SERVICE")
    assert svc is not None, "SEARCH_SERVICE not initialized"

    try:
        current_app.logger.info("API /api/facets start field=%s limit=%s", field, limit)
        values = svc.facet_values(field, limit=limit)
    except KeyError:
        return jsonify({"error": "unknown_field", "field": field}), 400
    except Exception:
        current_app.logger.exception("Facet computation failed for field=%s", field)
        return jsonify({"error": "facet_failed", "field": field}), 500

    current_app.logger.info(
        "API /api/facets done field=%s values=%s latency_ms=%s",
        field,
        len(values),
        int((time.time() - t0) * 1000),
    )
    return jsonify(sanitize_for_json({"field": field, "values": values, "limit": limit}))


@bp.get("/api/facets/search")
def facets_search():
    """Server-side typeahead over facet values.

    Example:
      /api/facets/search?field=authors&q=kenny%20zhu&limit=50
    """

    t0 = time.time()
    field = (request.args.get("field") or "").strip()
    q = (request.args.get("q") or "").strip()
    if not field:
        return jsonify({"error": "missing_field"}), 400

    limit = _parse_limit(field, 50) or 50
    limit = max(1, min(int(limit), 200))

    # Guardrails (prevents pathological input)
    if len(q) > 120:
        q = q[:120]

    svc = current_app.config.get("SEARCH_SERVICE")
    assert svc is not None, "SEARCH_SERVICE not initialized"

    try:
        current_app.logger.info("API /api/facets/search start field=%s q=%r limit=%s", field, q, limit)
        pool = svc.facet_values(field, limit=_typeahead_cache_limit(field))

        # If query empty, return the head of pool (display strings)
        if not q:
            out = {"field": field, "q": q, "values": pool[:limit], "limit": limit}
            current_app.logger.info(
                "API /api/facets/search done field=%s q=%r values=%s latency_ms=%s",
                field,
                q,
                len(out["values"]),
                int((time.time() - t0) * 1000),
            )
            return jsonify(sanitize_for_json(out))

        qn = _normalize_by_field(field, q)
        if not qn:
            out = {"field": field, "q": q, "values": [], "limit": limit}
            current_app.logger.info(
                "API /api/facets/search done field=%s q=%r values=0 latency_ms=%s",
                field,
                q,
                int((time.time() - t0) * 1000),
            )
            return jsonify(sanitize_for_json(out))

        # Build [(display, normalized)] once per request.
        # We match on normalized, but return display.
        pairs: List[Tuple[str, str]] = []
        for v in pool:
            if not isinstance(v, str):
                continue
            vn = _normalize_by_field(field, v)
            if vn:
                pairs.append((v, vn))

        assert len(pairs) <= _typeahead_cache_limit(field), "pairs grew beyond typeahead pool unexpectedly"

        # Stage 1: prefix
        prefix = [v for (v, vn) in pairs if vn.startswith(qn)]
        if len(prefix) >= limit:
                out = {"field": field, "q": q, "values": prefix[:limit], "limit": limit}
                current_app.logger.info(
                    "API /api/facets/search done field=%s q=%r values=%s latency_ms=%s",
                    field,
                    q,
                    len(out["values"]),
                    int((time.time() - t0) * 1000),
                )
                return jsonify(sanitize_for_json(out))

        # Stage 2: substring
        sub = [v for (v, vn) in pairs if (qn in vn) and (not vn.startswith(qn))]
        out = _dedupe_keep_order(prefix + sub)
        if len(out) >= limit:
            out_payload = {"field": field, "q": q, "values": out[:limit], "limit": limit}
            current_app.logger.info(
                "API /api/facets/search done field=%s q=%r values=%s latency_ms=%s",
                field,
                q,
                len(out_payload["values"]),
                int((time.time() - t0) * 1000),
            )
            return jsonify(sanitize_for_json(out_payload))

        # Stage 3: fuzzy
        if len(qn) >= 2:
            used = set(out)

            # cap candidates scanned for speed
            candidates = [v for (v, _) in pairs if v not in used]
            cap = _fuzzy_candidate_cap(field)
            if len(candidates) > cap:
                candidates = candidates[:cap]

            if _HAS_RAPIDFUZZ and process is not None and fuzz is not None:
                mult, thresh = _rapidfuzz_scoring_params(field)
                scored = process.extract(
                    qn,
                    candidates,
                    processor=lambda s: _normalize_by_field(field, s),
                    scorer=fuzz.token_set_ratio,
                    limit=limit * mult,
                )
                # rapidfuzz returns list[(choice, score, index)]
                fuzzy_hits = [v for (v, score, _idx) in scored if score >= thresh]
                out = _dedupe_keep_order(out + fuzzy_hits)
            else:
                # fallback: SequenceMatcher on normalized strings
                scored2: List[Tuple[float, str]] = []
                for v in candidates:
                    vn = _normalize_by_field(field, v)
                    # small prefilter
                    if qn[:2] not in vn and (qn[0] not in vn):
                        continue
                    s = _fuzzy_score_fallback(qn, vn)
                    if s >= 0.55:
                        scored2.append((s, v))
                scored2.sort(key=lambda x: x[0], reverse=True)
                out = _dedupe_keep_order(out + [v for _s, v in scored2])

        out_payload = {"field": field, "q": q, "values": out[:limit], "limit": limit}
        current_app.logger.info(
            "API /api/facets/search done field=%s q=%r values=%s latency_ms=%s",
            field,
            q,
            len(out_payload["values"]),
            int((time.time() - t0) * 1000),
        )
        return jsonify(sanitize_for_json(out_payload))

    except KeyError:
        return jsonify({"error": "unknown_field", "field": field}), 400
    except Exception:
        current_app.logger.exception("Facet search failed for field=%s q=%s", field, q)
        return jsonify({"error": "facet_search_failed", "field": field}), 500


@bp.get("/api/analytics/summary")
def analytics_summary():
    svc = current_app.config.get("SEARCH_SERVICE")
    assert svc is not None, "SEARCH_SERVICE not initialized"
    return jsonify(sanitize_for_json(svc.analytics_summary()))
