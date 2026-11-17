from __future__ import annotations

from typing import Any, Dict, Optional

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException


def _is_api_request() -> bool:
    try:
        return request.path.startswith("/api/")
    except Exception:
        return False


def _json_error(
    *,
    status: int,
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Any:
    payload: Dict[str, Any] = {
        "ok": False,
        "error": code,
        "message": message,
    }
    if details:
        payload["details"] = details
    return jsonify(payload), status


def register_api_error_handlers(app: Flask) -> None:
    """Return JSON for exceptions on /api/* routes.

    Without this, Flask serves HTML error pages which cause frontend
    JSON parsing to fail with: Unexpected token '<'.
    """

    @app.errorhandler(HTTPException)
    def _http_exc(e: HTTPException):  # type: ignore[override]
        if not _is_api_request():
            return e
        status = int(getattr(e, "code", 500) or 500)
        name = getattr(e, "name", "HTTPException")
        desc = getattr(e, "description", "Request failed")
        return _json_error(status=status, code="http_error", message=f"{name}: {desc}")

    # Pydantic v2 ValidationError
    try:
        from pydantic import ValidationError  # type: ignore

        @app.errorhandler(ValidationError)
        def _pydantic_exc(e: ValidationError):  # type: ignore[override]
            if not _is_api_request():
                raise e
            # Keep it compact, but structured.
            return _json_error(
                status=400,
                code="validation_error",
                message="Invalid request parameters",
                details={"errors": e.errors()},
            )

    except Exception:
        pass

    @app.errorhandler(Exception)
    def _any_exc(e: Exception):  # type: ignore[override]
        if not _is_api_request():
            raise e
        # Avoid leaking internals in production; frontend can show this message.
        app.logger.exception("Unhandled API exception")
        return _json_error(status=500, code="internal_error", message="Internal server error")
