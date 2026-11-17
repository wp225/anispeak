from __future__ import annotations
import time
from flask import Blueprint, jsonify, request, current_app
from anispeakarchive.api.schemas import SearchRequest
from anispeakarchive.index.hybrid import SearchFilters
from anispeakarchive.core.json_sanitize import sanitize_for_json, assert_json_strict

bp = Blueprint("search", __name__)

@bp.get("/api/search")
def search():
    t0 = time.time()
    current_app.logger.info("API /api/search start q=%r page=%s", request.args.get("q", ""), request.args.get("page", "1"))
    data = dict(request.args)

    for key in [
        "species",
        "modalities",
        "features",
        "stages",
        "domains",
        "contexts",
        "institutions",
        "authors",
        "venues",
    ]:
        vals = request.args.getlist(key)
        if vals:
            data[key] = vals


    # Boolean filters: accept "true"/"false"/"1"/"0"
    for bkey in ["has_code", "has_dataset", "has_benchmark"]:
        if bkey in request.args:
            raw = request.args.get(bkey, "").strip().lower()
            if raw in ["1", "true", "yes", "y", "t"]:
                data[bkey] = True
            elif raw in ["0", "false", "no", "n", "f"]:
                data[bkey] = False

    req = SearchRequest(**data)
    svc = current_app.config["SEARCH_SERVICE"]
    assert svc is not None, "SEARCH_SERVICE not initialized"

    filters = SearchFilters(
        year_min=req.year_min,
        year_max=req.year_max,
        species=req.species,
        modalities=req.modalities,
        features=req.features,
        stages=req.stages,
        domains=req.domains,
        contexts=req.contexts,
        institutions=req.institutions,
        authors=req.authors,
        venues=req.venues,
        paper_type=req.paper_type,
        tradition=req.tradition,
        has_code=req.has_code,
        has_dataset=req.has_dataset,
        has_benchmark=req.has_benchmark,
    )
    out = svc.search(req.q, filters=filters, page=req.page, page_size=req.page_size)
    payload = sanitize_for_json(out)
    # If anything slips through, fail server-side instead of emitting invalid JSON.
    assert_json_strict(payload)
    current_app.logger.info(
        "API /api/search done q=%r results=%s latency_ms=%s",
        req.q,
        len(payload.get("results", [])) if isinstance(payload, dict) else -1,
        int((time.time() - t0) * 1000),
    )
    return jsonify(payload)
