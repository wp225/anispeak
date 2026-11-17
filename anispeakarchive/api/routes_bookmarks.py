from __future__ import annotations
import time

from flask import Blueprint, current_app, jsonify, request, session

from anispeakarchive.auth.db import exec_one, exec_all, init_db
from anispeakarchive.core.json_sanitize import sanitize_for_json


bp = Blueprint("bookmarks", __name__, url_prefix="/api/bookmarks")


def _cfg():
    return current_app.config.get("AUTH_CFG", {})


def _db_path() -> str:
    p = str(_cfg().get("db_path", "cache/auth.sqlite"))
    assert p, "auth db_path must be set"
    return p


def _user_id() -> int | None:
    uid = session.get("user_id")
    if uid is None:
        return None
    try:
        return int(uid)
    except Exception:
        return None


@bp.before_app_request
def _ensure_db():
    if not current_app.config.get("_AUTH_DB_INIT", False):
        t0 = time.time()
        current_app.logger.info("Auth DB init start path=%s", _db_path())
        init_db(_db_path())
        current_app.config["_AUTH_DB_INIT"] = True
        current_app.logger.info("Auth DB init done latency_ms=%s", int((time.time() - t0) * 1000))


@bp.get("/me")
def me():
    t0 = time.time()
    uid = _user_id()
    email = session.get("user_email")
    out = {"logged_in": bool(uid and email), "email": email if uid else None}
    current_app.logger.info("API /api/bookmarks/me done logged_in=%s latency_ms=%s", out["logged_in"], int((time.time() - t0) * 1000))
    return jsonify(sanitize_for_json(out))


@bp.post("/toggle")
def toggle():
    uid = _user_id()
    if uid is None:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401

    doc_id = (request.json or {}).get("doc_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        return jsonify({"ok": False, "error": "bad_doc_id"}), 400
    doc_id = doc_id.strip()

    # Back-compat endpoint: maps bookmarks -> saved_papers (source='paper').
    existing = exec_one(_db_path(), "SELECT 1 FROM saved_papers WHERE user_id = ? AND doc_id = ?", (uid, doc_id))
    if existing is None:
        # SQLite doesn't evaluate strftime in parameter positions; do it directly.
        exec_one(
            _db_path(),
            "INSERT OR REPLACE INTO saved_papers(user_id, doc_id, saved_at, source, collection_id, note, updated_at) "
            "VALUES(?, ?, strftime('%s','now'), 'paper', NULL, NULL, strftime('%s','now'))",
            (uid, doc_id),
        )
        return jsonify({"ok": True, "bookmarked": True})
    exec_one(_db_path(), "DELETE FROM saved_papers WHERE user_id = ? AND doc_id = ?", (uid, doc_id))
    return jsonify({"ok": True, "bookmarked": False})


@bp.get("/list")
def list_():
    uid = _user_id()
    if uid is None:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401
    rows = exec_all(_db_path(), "SELECT doc_id, saved_at AS created_at FROM saved_papers WHERE user_id = ? ORDER BY saved_at DESC", (uid,))
    out = {"ok": True, "items": [{"doc_id": str(r["doc_id"]), "created_at": int(r["created_at"])} for r in rows]}
    return jsonify(sanitize_for_json(out))


@bp.get("/has")
def has():
    uid = _user_id()
    if uid is None:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401
    doc_id = str(request.args.get("doc_id", "")).strip()
    if not doc_id:
        return jsonify({"ok": False, "error": "bad_doc_id"}), 400
    existing = exec_one(_db_path(), "SELECT 1 FROM saved_papers WHERE user_id = ? AND doc_id = ?", (uid, doc_id))
    return jsonify({"ok": True, "bookmarked": existing is not None})
