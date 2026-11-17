from __future__ import annotations

import os
import time
from datetime import timedelta

from flask import Flask, g, request
from werkzeug.middleware.proxy_fix import ProxyFix
from anispeakarchive.core.prefix_middleware import PrefixMiddleware

from anispeakarchive.core.config import load_settings
from anispeakarchive.core.env import load_dotenv
from anispeakarchive.core.logging import setup_logging
from anispeakarchive.core.strict_json import StrictJSONProvider
from anispeakarchive.api.error_handlers import register_api_error_handlers

from anispeakarchive.index.dense import Embedder
from anispeakarchive.index.rerank import Reranker
from anispeakarchive.index.hybrid import SearchService
from anispeakarchive.services.ask_ai_client import AskAIClient
from anispeakarchive.services.ask_ai_service import AskAIService

from anispeakarchive.web.routes_pages import bp as pages_bp
from anispeakarchive.web.routes_auth import bp as auth_bp

from anispeakarchive.api.routes_search import bp as search_bp
from anispeakarchive.api.routes_paper import bp as paper_bp
from anispeakarchive.api.routes_facets import bp as facets_bp
from anispeakarchive.api.routes_analytics import bp as analytics_bp
from anispeakarchive.api.routes_library import bp as library_bp
from anispeakarchive.api.routes_bookmarks import bp as bookmarks_bp
from anispeakarchive.api.routes_citation import bp as citation_bp
from anispeakarchive.api.routes_ask import bp as ask_bp


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    v = str(raw).strip().lower()
    if v in {"1", "true", "yes", "y", "t", "on"}:
        return True
    if v in {"0", "false", "no", "n", "f", "off"}:
        return False
    return bool(default)


def create_app() -> Flask:
    setup_logging()
    load_dotenv(".env")
    settings = load_settings("configs/default.yaml")

    app = Flask(
        __name__,
        template_folder="anispeakarchive/web/templates",
        static_folder="anispeakarchive/web/static",
    )

    # Reverse proxy support (Apache/Nginx). Trust one proxy hop.
    # This fixes request.scheme/host for redirects + URL generation behind TLS termination.
    app.wsgi_app = ProxyFix(
        app.wsgi_app,
        x_for=1,
        x_proto=1,
        x_host=1,
        x_port=1,
    )

    # Support deployment under a subpath (e.g., /anispeakarchive) via Apache/Nginx.
    # The reverse proxy should set: X-Forwarded-Prefix: /anispeakarchive
    app.wsgi_app = PrefixMiddleware(app.wsgi_app)

    # Ensure we never emit invalid JSON (NaN/Infinity) to the browser.
    app.json = StrictJSONProvider(app)
    register_api_error_handlers(app)

    # Core security config.
    assert isinstance(settings.app.secret_key, str) and len(settings.app.secret_key.strip()) >= 16, (
        "configs/default.yaml: app.secret_key must be a non-empty secret string (>= 16 chars)."
    )
    app.config["SECRET_KEY"] = settings.app.secret_key

    # Auth / OTP / SMTP configuration is env-driven to avoid committing secrets.
    allow_edu_only = _truthy_env("ASA_ALLOW_EDU_ONLY", True)
    allowed_domains_raw = str(os.environ.get("ASA_ALLOWED_DOMAINS", "")).strip()
    allowed_domains = [d.strip().lower() for d in allowed_domains_raw.split(",") if d.strip()] if allowed_domains_raw else []

    app.config["AUTH_CFG"] = {
        "db_path": os.environ.get("ASA_AUTH_DB_PATH", "cache/auth.sqlite"),
        "allow_edu_only": allow_edu_only,
        "allowed_domains": allowed_domains,
        "otp_ttl_sec": int(os.environ.get("ASA_OTP_TTL_SEC", "600")),
        "otp_max_per_window": int(os.environ.get("ASA_OTP_MAX_PER_WINDOW", "5")),
        "otp_window_sec": int(os.environ.get("ASA_OTP_WINDOW_SEC", "3600")),
        "smtp_host": os.environ.get("ASA_SMTP_HOST"),
        "smtp_port": int(os.environ.get("ASA_SMTP_PORT", "587")),
        "smtp_username": os.environ.get("ASA_SMTP_USERNAME"),
        "smtp_password": os.environ.get("ASA_SMTP_PASSWORD"),
        "smtp_from": os.environ.get("ASA_SMTP_FROM"),
        "smtp_use_tls": _truthy_env("ASA_SMTP_USE_TLS", True),
        "dev_otp_console": _truthy_env("ASA_DEV_OTP_TO_CONSOLE", False),
        "cookie_consent_enabled": _truthy_env("ASA_COOKIE_CONSENT_ENABLED", True),
        "session_days": int(os.environ.get("ASA_SESSION_DAYS", "30")),
    }

    # Session persistence:
    # - If user accepts cookies: session.permanent=True, cookie expires in session_days.
    # - If user rejects: session.permanent=False, cookie expires at browser close.
    sd = int(app.config["AUTH_CFG"].get("session_days", 30))
    assert 1 <= sd <= 365, "AUTH_CFG.session_days must be in [1, 365]"
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=sd)
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    app.logger.info("App startup: initializing embedder")

    embedder = Embedder.create(
        model_name=settings.embedding.model_name,
        provider=settings.embedding.provider,
        use_gpu=settings.embedding.use_gpu,
        batch_size=settings.embedding.batch_size,
        normalize=settings.embedding.normalize,
        max_length=settings.embedding.max_length,
        trust_remote_code=settings.embedding.trust_remote_code,
        device=settings.embedding.device,
        dtype=settings.embedding.dtype,
        device_map=settings.embedding.device_map,
    )

    reranker = None
    if settings.rerank.enabled:
        app.logger.info("App startup: initializing reranker")
        reranker = Reranker.create(
            model_name=settings.rerank.model_name,
            provider=settings.rerank.provider,
            use_gpu=settings.rerank.use_gpu,
            batch_size=settings.rerank.batch_size,
            max_length=settings.rerank.max_length,
            trust_remote_code=settings.rerank.trust_remote_code,
            device=settings.rerank.device,
            dtype=settings.rerank.dtype,
            device_map=settings.rerank.device_map,
        )

    cfg = {
        "search": settings.search.__dict__,
        "rerank": settings.rerank.__dict__,
        "ui": settings.ui.__dict__,
    }

    assert isinstance(settings.paths.lancedb_dir, str) and settings.paths.lancedb_dir.strip(), (
        "configs/default.yaml: paths.lancedb_dir must be a non-empty string."
    )
    assert isinstance(settings.index.table_name, str) and settings.index.table_name.strip(), (
        "configs/default.yaml: index.table_name must be a non-empty string."
    )

    svc = SearchService(
        lancedb_dir=settings.paths.lancedb_dir,
        table_name=settings.index.table_name,
        embedder=embedder,
        reranker=reranker,
        cfg=cfg,
    )
    app.logger.info("App startup: search service initialized table=%s", settings.index.table_name)
    app.config["SEARCH_SERVICE"] = svc
    svc.prime_small_facet_caches()
    app.logger.info("App startup: small facet caches primed")

    app.config["ASK_AI_CFG"] = settings.ask_ai
    app.config["ASK_AI_SERVICE"] = None
    if settings.ask_ai.enabled:
        app.logger.info("App startup: Ask AI enabled provider=%s base_url=%s", settings.ask_ai.provider, settings.ask_ai.base_url)
        if settings.ask_ai.provider == "vllm_openai":
            ask_client = AskAIClient(settings.ask_ai)
            app.config["ASK_AI_SERVICE"] = AskAIService(settings.ask_ai, svc, ask_client)

    @app.before_request
    def _log_request_start() -> None:
        g._req_t0 = time.time()
        app.logger.info("HTTP start method=%s path=%s query=%s", request.method, request.path, request.query_string.decode("utf-8", errors="ignore"))

    @app.after_request
    def _log_request_done(response):
        t0 = getattr(g, "_req_t0", None)
        if t0 is not None:
            app.logger.info(
                "HTTP done method=%s path=%s status=%s latency_ms=%s",
                request.method,
                request.path,
                response.status_code,
                int((time.time() - t0) * 1000),
            )
        return response

    # Web pages + auth
    app.register_blueprint(pages_bp)
    app.register_blueprint(auth_bp)

    # Core APIs
    app.register_blueprint(search_bp)
    app.register_blueprint(paper_bp)
    app.register_blueprint(facets_bp)
    app.register_blueprint(analytics_bp)

    # Library / bookmarks / citation export APIs
    app.register_blueprint(library_bp)
    app.register_blueprint(bookmarks_bp)
    app.register_blueprint(citation_bp)
    app.register_blueprint(ask_bp)

    return app


if __name__ == "__main__":
    s = load_settings("configs/default.yaml")
    app = create_app()

    assert isinstance(s.app.host, str) and s.app.host.strip(), "configs/default.yaml: app.host must be set."
    assert isinstance(s.app.port, int) and 1 <= int(s.app.port) <= 65535, "configs/default.yaml: app.port must be 1..65535."
    assert isinstance(s.app.debug, bool), "configs/default.yaml: app.debug must be boolean."

    app.run(host=s.app.host, port=s.app.port, debug=s.app.debug, use_reloader=False)
