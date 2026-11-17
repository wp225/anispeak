from __future__ import annotations
import time
from flask import Blueprint, jsonify, current_app, Response
from anispeakarchive.core.json_sanitize import sanitize_for_json

bp = Blueprint("paper", __name__)

@bp.get("/api/paper/<doc_id>")
def paper(doc_id: str):
    t0 = time.time()
    svc = current_app.config["SEARCH_SERVICE"]
    assert svc is not None, "SEARCH_SERVICE not initialized"
    try:
        out = sanitize_for_json(svc.get_paper(doc_id))
        current_app.logger.info("API /api/paper/%s done latency_ms=%s", doc_id, int((time.time() - t0) * 1000))
        return jsonify(out)
    except KeyError:
        return jsonify({"error": "not_found", "doc_id": doc_id}), 404

@bp.get("/api/paper/<doc_id>/similar")
def paper_similar(doc_id: str):
    t0 = time.time()
    svc = current_app.config["SEARCH_SERVICE"]
    assert svc is not None, "SEARCH_SERVICE not initialized"
    try:
        results = svc.similar_papers(doc_id, k=8)
        current_app.logger.info("API /api/paper/%s/similar done results=%s latency_ms=%s", doc_id, len(results), int((time.time() - t0) * 1000))
        return jsonify({"doc_id": doc_id, "results": results})
    except KeyError:
        return jsonify({"error": "not_found", "doc_id": doc_id}), 404

@bp.get("/api/paper/<doc_id>/ris")
def paper_ris(doc_id: str):
    t0 = time.time()
    svc = current_app.config["SEARCH_SERVICE"]
    assert svc is not None, "SEARCH_SERVICE not initialized"
    try:
        ris = svc.export_ris(doc_id)
    except KeyError:
        return jsonify({"error": "not_found", "doc_id": doc_id}), 404
    current_app.logger.info("API /api/paper/%s/ris done latency_ms=%s", doc_id, int((time.time() - t0) * 1000))
    return Response(ris, mimetype="application/x-research-info-systems")
