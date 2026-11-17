from __future__ import annotations

import json

from flask import Blueprint, current_app, jsonify, request, session

from anispeakarchive.auth.db import exec_one, exec_all, init_db
from anispeakarchive.core.json_sanitize import sanitize_for_json
from anispeakarchive.auth.otp import now_ts


bp = Blueprint("library", __name__, url_prefix="/api/library")


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
        init_db(_db_path())
        current_app.config["_AUTH_DB_INIT"] = True


def _require_user() -> int:
    uid = _user_id()
    if uid is None:
        raise PermissionError("not_logged_in")
    return uid


@bp.get("/collections")
def list_collections():
    try:
        uid = _require_user()
    except PermissionError:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401

    rows = exec_all(_db_path(), "SELECT id, name FROM collections WHERE user_id = ? ORDER BY name ASC", (uid,))
    return jsonify(sanitize_for_json({"ok": True, "items": [{"id": int(r["id"]), "name": str(r["name"])} for r in rows]}))


@bp.post("/collections")
def create_collection():
    try:
        uid = _require_user()
    except PermissionError:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401

    name = str((request.json or {}).get("name", "")).strip()
    if not name or len(name) > 60:
        return jsonify({"ok": False, "error": "bad_name"}), 400

    now = now_ts()
    try:
        exec_one(_db_path(), "INSERT INTO collections(user_id, name, created_at) VALUES(?,?,?)", (uid, name, now))
    except Exception:
        return jsonify({"ok": False, "error": "name_taken"}), 409
    row = exec_one(_db_path(), "SELECT id, name FROM collections WHERE user_id = ? AND name = ?", (uid, name))
    assert row is not None
    return jsonify({"ok": True, "item": {"id": int(row["id"]), "name": str(row["name"])}})


@bp.delete("/collections/<int:cid>")
def delete_collection(cid: int):
    try:
        uid = _require_user()
    except PermissionError:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401
    exec_one(_db_path(), "DELETE FROM collections WHERE id = ? AND user_id = ?", (cid, uid))
    return jsonify({"ok": True})


@bp.get("/papers")
def list_saved_papers():
    try:
        uid = _require_user()
    except PermissionError:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401

    rows = exec_all(
        _db_path(),
        """
        SELECT sp.doc_id, sp.saved_at, sp.source, sp.collection_id, c.name AS collection_name, sp.note, sp.updated_at
        FROM saved_papers sp
        LEFT JOIN collections c ON c.id = sp.collection_id
        WHERE sp.user_id = ?
        ORDER BY sp.saved_at DESC
        """,
        (uid,),
    )
    items = []
    for r in rows:
        items.append(
            {
                "doc_id": str(r["doc_id"]),
                "saved_at": int(r["saved_at"]),
                "source": str(r["source"]),
                "collection_id": int(r["collection_id"]) if r["collection_id"] is not None else None,
                "collection_name": str(r["collection_name"]) if r["collection_name"] is not None else None,
                "note": str(r["note"]) if r["note"] is not None else "",
                "updated_at": int(r["updated_at"]),
            }
        )
    return jsonify(sanitize_for_json({"ok": True, "items": items}))


@bp.post("/papers/toggle")
def toggle_saved_paper():
    try:
        uid = _require_user()
    except PermissionError:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401

    payload = request.json or {}
    doc_id = str(payload.get("doc_id", "")).strip()
    source = str(payload.get("source", "paper")).strip().lower()
    if not doc_id:
        return jsonify({"ok": False, "error": "bad_doc_id"}), 400
    if source not in {"paper", "search"}:
        return jsonify({"ok": False, "error": "bad_source"}), 400

    existing = exec_one(_db_path(), "SELECT 1 FROM saved_papers WHERE user_id = ? AND doc_id = ?", (uid, doc_id))
    if existing is None:
        now = now_ts()
        exec_one(
            _db_path(),
            "INSERT INTO saved_papers(user_id, doc_id, saved_at, source, collection_id, note, updated_at) VALUES(?,?,?,?,?,?,?)",
            (uid, doc_id, now, source, None, None, now),
        )
        return jsonify({"ok": True, "saved": True})
    exec_one(_db_path(), "DELETE FROM saved_papers WHERE user_id = ? AND doc_id = ?", (uid, doc_id))
    return jsonify({"ok": True, "saved": False})


@bp.post("/papers/update")
def update_saved_paper():
    try:
        uid = _require_user()
    except PermissionError:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401

    payload = request.json or {}
    doc_id = str(payload.get("doc_id", "")).strip()
    if not doc_id:
        return jsonify({"ok": False, "error": "bad_doc_id"}), 400

    note = payload.get("note")
    if note is not None:
        note = str(note)
        if len(note) > 4000:
            return jsonify({"ok": False, "error": "note_too_long"}), 400

    collection_id = payload.get("collection_id")
    if collection_id is not None:
        try:
            collection_id = int(collection_id)
        except Exception:
            return jsonify({"ok": False, "error": "bad_collection_id"}), 400
        # Ensure collection belongs to user
        ok = exec_one(_db_path(), "SELECT 1 FROM collections WHERE id = ? AND user_id = ?", (collection_id, uid))
        if ok is None:
            return jsonify({"ok": False, "error": "collection_not_found"}), 404

    now = now_ts()
    # Only update existing saved paper.
    row = exec_one(_db_path(), "SELECT 1 FROM saved_papers WHERE user_id = ? AND doc_id = ?", (uid, doc_id))
    if row is None:
        return jsonify({"ok": False, "error": "not_saved"}), 404

    fields = []
    params = []
    if note is not None:
        fields.append("note = ?")
        params.append(note)
    if payload.get("collection_id", None) is None and "collection_id" in payload:
        # explicit null clears
        fields.append("collection_id = NULL")
    elif collection_id is not None:
        fields.append("collection_id = ?")
        params.append(collection_id)

    fields.append("updated_at = ?")
    params.append(now)
    params.extend([uid, doc_id])
    sql = "UPDATE saved_papers SET " + ", ".join(fields) + " WHERE user_id = ? AND doc_id = ?"
    exec_one(_db_path(), sql, tuple(params))
    return jsonify({"ok": True})


@bp.get("/papers/has")
def has_saved_paper():
    try:
        uid = _require_user()
    except PermissionError:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401
    doc_id = str(request.args.get("doc_id", "")).strip()
    if not doc_id:
        return jsonify({"ok": False, "error": "bad_doc_id"}), 400
    row = exec_one(_db_path(), "SELECT 1 FROM saved_papers WHERE user_id = ? AND doc_id = ?", (uid, doc_id))
    return jsonify({"ok": True, "saved": row is not None})


@bp.post("/papers/has_batch")
def has_saved_papers_batch():
    """Batch check whether the current user has saved multiple papers.

    Motivation: the UI previously issued one request per result card, which can
    trigger reverse-proxy / WAF rate limits.

    Request JSON:
        {"doc_ids": ["W...", ...]}

    Response JSON:
        {"ok": true, "saved": {"W...": true/false, ...}}
    """
    try:
        uid = _require_user()
    except PermissionError:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401

    payload = request.json or {}
    doc_ids = payload.get("doc_ids", [])
    if not isinstance(doc_ids, list):
        return jsonify({"ok": False, "error": "bad_doc_ids"}), 400

    # Defensive limits: keep payloads bounded.
    max_ids = 200
    cleaned: list[str] = []
    seen = set()
    for x in doc_ids[:max_ids]:
        s = str(x).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        cleaned.append(s)

    if not cleaned:
        return jsonify({"ok": True, "saved": {}})

    placeholders = ",".join(["?"] * len(cleaned))
    sql = f"SELECT doc_id FROM saved_papers WHERE user_id = ? AND doc_id IN ({placeholders})"
    rows = exec_all(_db_path(), sql, tuple([uid] + cleaned))
    saved_set = {str(r["doc_id"]) for r in rows}

    out = {doc_id: (doc_id in saved_set) for doc_id in cleaned}
    assert isinstance(out, dict)
    assert all(isinstance(k, str) and isinstance(v, bool) for k, v in out.items())
    return jsonify({"ok": True, "saved": out})


@bp.get("/saved_searches")
def list_saved_searches():
    try:
        uid = _require_user()
    except PermissionError:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401

    rows = exec_all(
        _db_path(),
        "SELECT id, name, cadence, enabled, created_at, last_sent_at, query_json FROM saved_searches WHERE user_id = ? ORDER BY created_at DESC",
        (uid,),
    )
    items = []
    for r in rows:
        try:
            q = json.loads(str(r["query_json"]))
        except Exception:
            q = {}
        items.append(
            {
                "id": int(r["id"]),
                "name": str(r["name"]),
                "cadence": str(r["cadence"]),
                "enabled": bool(int(r["enabled"])),
                "created_at": int(r["created_at"]),
                "last_sent_at": int(r["last_sent_at"]) if r["last_sent_at"] is not None else None,
                "query": q,
            }
        )
    return jsonify(sanitize_for_json({"ok": True, "items": items}))


@bp.post("/saved_searches")
def create_saved_search():
    try:
        uid = _require_user()
    except PermissionError:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401

    payload = request.json or {}
    name = str(payload.get("name", "")).strip()[:80]
    cadence = str(payload.get("cadence", "weekly")).strip().lower()
    query = payload.get("query")
    if not name:
        return jsonify({"ok": False, "error": "bad_name"}), 400
    if cadence not in {"daily", "weekly"}:
        return jsonify({"ok": False, "error": "bad_cadence"}), 400
    if not isinstance(query, dict):
        return jsonify({"ok": False, "error": "bad_query"}), 400

    now = now_ts()
    exec_one(
        _db_path(),
        "INSERT INTO saved_searches(user_id, name, query_json, cadence, enabled, created_at) VALUES(?,?,?,?,1,?)",
        (uid, name, json.dumps(query, ensure_ascii=False), cadence, now),
    )
    row = exec_one(_db_path(), "SELECT id FROM saved_searches WHERE user_id = ? AND name = ? ORDER BY id DESC", (uid, name))
    assert row is not None
    return jsonify({"ok": True, "id": int(row["id"])})


@bp.post("/saved_searches/<int:sid>")
def update_saved_search(sid: int):
    try:
        uid = _require_user()
    except PermissionError:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401
    payload = request.json or {}
    enabled = payload.get("enabled")
    cadence = payload.get("cadence")
    name = payload.get("name")
    fields = []
    params = []
    if enabled is not None:
        fields.append("enabled = ?")
        params.append(1 if bool(enabled) else 0)
    if cadence is not None:
        cadence = str(cadence).strip().lower()
        if cadence not in {"daily", "weekly"}:
            return jsonify({"ok": False, "error": "bad_cadence"}), 400
        fields.append("cadence = ?")
        params.append(cadence)
    if name is not None:
        name = str(name).strip()[:80]
        if not name:
            return jsonify({"ok": False, "error": "bad_name"}), 400
        fields.append("name = ?")
        params.append(name)
    if not fields:
        return jsonify({"ok": True})
    params.extend([sid, uid])
    exec_one(_db_path(), "UPDATE saved_searches SET " + ", ".join(fields) + " WHERE id = ? AND user_id = ?", tuple(params))
    return jsonify({"ok": True})


@bp.delete("/saved_searches/<int:sid>")
def delete_saved_search(sid: int):
    try:
        uid = _require_user()
    except PermissionError:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401
    exec_one(_db_path(), "DELETE FROM saved_searches WHERE id = ? AND user_id = ?", (sid, uid))
    return jsonify({"ok": True})
