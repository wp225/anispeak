from __future__ import annotations

import json
import time
import uuid
from typing import Any

from flask import Blueprint, Response, current_app, jsonify, request, session, stream_with_context

from anispeakarchive.auth.db import exec_all, exec_one
from anispeakarchive.services.ask_ai_client import AskAIResponseError, AskAIUnavailableError


bp = Blueprint("ask", __name__)


def _db_path() -> str:
    return str(current_app.config.get("AUTH_CFG", {}).get("db_path", "cache/auth.sqlite"))


def _sid() -> str:
    if "ask_sid" not in session:
        session["ask_sid"] = str(uuid.uuid4())
    return session["ask_sid"]


def _user_id() -> int | None:
    uid = session.get("user_id")
    if uid is None:
        return None
    try:
        return int(uid)
    except Exception:
        return None


def _ask_cfg() -> Any:
    return current_app.config.get("ASK_AI_CFG")


def _ask_service() -> Any:
    svc = current_app.config.get("ASK_AI_SERVICE")
    if svc is None:
        raise AskAIUnavailableError("Ask AI model server is not available.")
    return svc


def _now() -> int:
    return int(time.time())


def _conversation_title(message: str) -> str:
    text = str(message or "").strip()
    return text[:80] + ("…" if len(text) > 80 else "")


def _require_conv_owner(conv_id: str, sid: str) -> None:
    row = exec_one(
        _db_path(),
        "SELECT id FROM ask_conversations WHERE id = ? AND session_id = ?",
        (conv_id, sid),
    )
    if row is None:
        raise KeyError(conv_id)


def _load_recent_messages(conv_id: str) -> list[dict[str, Any]]:
    rows = exec_all(
        _db_path(),
        "SELECT role, content, retrieval_json, response_json, created_at FROM ask_messages WHERE conversation_id = ? ORDER BY id",
        (conv_id,),
    )
    return [dict(r) for r in rows]


def _get_or_create_conversation(conv_id: str | None, message: str, sid: str, user_id: int | None) -> tuple[str, str, int]:
    now = _now()
    if not conv_id:
        new_id = str(uuid.uuid4())
        title = _conversation_title(message)
        exec_one(
            _db_path(),
            "INSERT INTO ask_conversations(id, session_id, user_id, title, created_at, updated_at) VALUES(?,?,?,?,?,?)",
            (new_id, sid, user_id, title, now, now),
        )
        return new_id, title, now

    _require_conv_owner(conv_id, sid)
    exec_one(
        _db_path(),
        "UPDATE ask_conversations SET updated_at = ? WHERE id = ?",
        (now, conv_id),
    )
    row = exec_one(_db_path(), "SELECT title, created_at FROM ask_conversations WHERE id = ?", (conv_id,))
    assert row is not None
    return conv_id, str(row["title"]), int(row["created_at"])


def _save_message(
    conv_id: str,
    role: str,
    content: str,
    retrieval_json: str | None = None,
    response_json: str | None = None,
) -> None:
    exec_one(
        _db_path(),
        "INSERT INTO ask_messages(conversation_id, role, content, created_at, retrieval_json, response_json) VALUES(?,?,?,?,?,?)",
        (conv_id, role, content, _now(), retrieval_json, response_json),
    )


def _validate_payload(data: dict[str, Any]) -> tuple[str, str | None]:
    cfg = _ask_cfg()
    svc = _ask_service()
    raw_message = data.get("message", data.get("question", ""))
    if not isinstance(raw_message, str):
        raise ValueError("message must be a string")
    message = svc.normalize_message(raw_message)
    if not message:
        raise ValueError("message must be non-empty")
    if len(message) > int(cfg.max_question_chars):
        raise ValueError(f"message exceeds max_question_chars={cfg.max_question_chars}")
    conv_raw = data.get("conversation_id")
    if conv_raw is None:
        return message, None
    conv_id = str(conv_raw).strip()
    return message, (conv_id or None)


def _build_success_payload(conv_id: str, title: str, created_at: int, ask_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "conversation_id": conv_id,
        "answer": ask_result["answer"],
        "answer_html": ask_result.get("answer_html"),
        "model_output": ask_result["model_output"],
        "references": ask_result["references"],
        "retrieval": ask_result["retrieval"],
        "title": title,
        "created_at": created_at,
    }


def _stream_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@bp.get("/api/ask/conversations")
def list_conversations():
    sid = _sid()
    rows = exec_all(
        _db_path(),
        "SELECT id, title, created_at, updated_at FROM ask_conversations WHERE session_id = ? ORDER BY updated_at DESC LIMIT 100",
        (sid,),
    )
    return jsonify({"conversations": [dict(r) for r in rows]})


@bp.delete("/api/ask/conversations")
def clear_conversations():
    sid = _sid()
    exec_one(_db_path(), "DELETE FROM ask_conversations WHERE session_id = ?", (sid,))
    return jsonify({"ok": True})


@bp.delete("/api/ask/conversations/<conv_id>")
def delete_conversation(conv_id: str):
    sid = _sid()
    exec_one(_db_path(), "DELETE FROM ask_conversations WHERE id = ? AND session_id = ?", (conv_id, sid))
    return jsonify({"ok": True})


@bp.get("/api/ask/conversations/<conv_id>/messages")
def get_messages(conv_id: str):
    sid = _sid()
    try:
        _require_conv_owner(conv_id, sid)
    except KeyError:
        return jsonify({"error": "not_found"}), 404
    msgs = exec_all(
        _db_path(),
        "SELECT role, content, retrieval_json, response_json FROM ask_messages WHERE conversation_id = ? ORDER BY id",
        (conv_id,),
    )
    return jsonify({"messages": [dict(m) for m in msgs]})


@bp.post("/api/ask")
def ask():
    if session.get("user_id") is None:
        return jsonify({"error": "login_required"}), 401
    try:
        current_app.logger.info("API /api/ask start")
        message, conv_id = _validate_payload(request.get_json(silent=True) or {})
        sid = _sid()
        uid = _user_id()
        conv_id, title, created_at = _get_or_create_conversation(conv_id, message, sid, uid)
        recent = _load_recent_messages(conv_id)
        _save_message(conv_id, "user", message)
        current_app.logger.info("API /api/ask retrieval+generation conversation_id=%s", conv_id)
        ask_result = _ask_service().answer_question(message, recent)
        payload = _build_success_payload(conv_id, title, created_at, ask_result)
        _save_message(
            conv_id,
            "assistant",
            ask_result["answer"],
            ask_result["retrieval_json"],
            json.dumps(payload, ensure_ascii=False),
        )
        current_app.logger.info(
            "API /api/ask done conversation_id=%s num_context_records=%s num_references=%s",
            conv_id,
            payload["retrieval"]["num_context_records"],
            len(payload["references"]),
        )
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except KeyError:
        return jsonify({"error": "not_found"}), 404
    except AskAIUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503
    except AskAIResponseError as exc:
        return jsonify({"error": str(exc)}), 502


@bp.post("/api/ask/stream")
def ask_stream():
    if session.get("user_id") is None:
        return jsonify({"error": "login_required"}), 401

    try:
        current_app.logger.info("API /api/ask/stream start")
        payload = request.get_json(silent=True) or {}
        message, conv_id = _validate_payload(payload)
        sid = _sid()
        uid = _user_id()
        conv_id, title, created_at = _get_or_create_conversation(conv_id, message, sid, uid)
        recent = _load_recent_messages(conv_id)
        _save_message(conv_id, "user", message)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except KeyError:
        return jsonify({"error": "not_found"}), 404

    @stream_with_context
    def _gen():
        yield _stream_event("conversation", {"conversation_id": conv_id, "title": title, "created_at": created_at})
        yield _stream_event("status", {"stage": "retrieval"})
        try:
            current_app.logger.info("API /api/ask/stream retrieval conversation_id=%s", conv_id)
            yield _stream_event("status", {"stage": "generation"})
            for event in _ask_service().answer_question_stream_events(message, recent):
                if event["type"] == "token":
                    yield _stream_event("token", {"text": event["text"]})
                    continue
                ask_result = event["payload"]
                payload = _build_success_payload(conv_id, title, created_at, ask_result)
                yield _stream_event("status", {"stage": "generation_complete"})
                _save_message(
                    conv_id,
                    "assistant",
                    ask_result["answer"],
                    ask_result["retrieval_json"],
                    json.dumps(payload, ensure_ascii=False),
                )
                current_app.logger.info(
                    "API /api/ask/stream done conversation_id=%s num_context_records=%s num_references=%s",
                    conv_id,
                    payload["retrieval"]["num_context_records"],
                    len(payload["references"]),
                )
                yield _stream_event("final", payload)
        except AskAIUnavailableError as exc:
            yield _stream_event("error", {"status": 503, "error": str(exc)})
        except AskAIResponseError as exc:
            yield _stream_event("error", {"status": 502, "error": str(exc)})
        except Exception as exc:
            current_app.logger.exception("Ask stream failed")
            yield _stream_event("error", {"status": 500, "error": str(exc)})

    return Response(_gen(), mimetype="text/event-stream")
