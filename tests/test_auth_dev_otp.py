from __future__ import annotations

from flask import Flask, session

from anispeakarchive.auth.db import exec_one, init_db
from anispeakarchive.auth.otp import now_ts
from anispeakarchive.web.routes_auth import bp as auth_bp


def test_dev_otp_fallback_allows_verify_flow(tmp_path):
    db_path = str(tmp_path / "auth.sqlite")
    init_db(db_path)

    app = Flask(__name__)
    app.secret_key = "test-secret-key"
    app.debug = True
    app.register_blueprint(auth_bp)
    app.config["AUTH_CFG"] = {
        "db_path": db_path,
        "allow_edu_only": True,
        "allowed_domains": [],
        "otp_ttl_sec": 600,
        "otp_max_per_window": 5,
        "otp_window_sec": 3600,
        "smtp_host": None,
        "smtp_port": 587,
        "smtp_username": None,
        "smtp_password": None,
        "smtp_from": None,
        "smtp_use_tls": True,
        "dev_otp_console": False,
        "cookie_consent_enabled": False,
        "session_days": 30,
    }

    client = app.test_client()
    res = client.post("/login", data={"email": "user@school.edu"}, follow_redirects=False)
    assert res.status_code == 302
    assert res.headers["Location"].endswith("/verify")

    with client.session_transaction() as sess:
        assert sess["pending_email"] == "user@school.edu"


def test_no_dev_fallback_rejects_missing_smtp(tmp_path):
    db_path = str(tmp_path / "auth.sqlite")
    init_db(db_path)

    app = Flask(__name__)
    app.secret_key = "test-secret-key"
    app.debug = False
    app.register_blueprint(auth_bp)

    @app.get("/whoami")
    def whoami():
        return {"user_id": session.get("user_id"), "user_email": session.get("user_email")}

    app.config["AUTH_CFG"] = {
        "db_path": db_path,
        "allow_edu_only": True,
        "allowed_domains": [],
        "otp_ttl_sec": 600,
        "otp_max_per_window": 5,
        "otp_window_sec": 3600,
        "smtp_host": None,
        "smtp_port": 587,
        "smtp_username": None,
        "smtp_password": None,
        "smtp_from": None,
        "smtp_use_tls": True,
        "dev_otp_console": False,
        "cookie_consent_enabled": False,
        "session_days": 30,
    }

    client = app.test_client()
    res = client.post("/login", data={"email": "user@school.edu"}, follow_redirects=False)
    assert res.status_code == 302
    assert res.headers["Location"].endswith("/login")


def test_signed_auth_cookie_restores_session(tmp_path):
    db_path = str(tmp_path / "auth.sqlite")
    init_db(db_path)

    app = Flask(__name__)
    app.secret_key = "test-secret-key"
    app.debug = False
    app.register_blueprint(auth_bp)

    @app.get("/whoami")
    def whoami():
        return {"user_id": session.get("user_id"), "user_email": session.get("user_email")}

    app.config["AUTH_CFG"] = {
        "db_path": db_path,
        "allow_edu_only": True,
        "allowed_domains": [],
        "otp_ttl_sec": 600,
        "otp_max_per_window": 5,
        "otp_window_sec": 3600,
        "smtp_host": None,
        "smtp_port": 587,
        "smtp_username": None,
        "smtp_password": None,
        "smtp_from": None,
        "smtp_use_tls": True,
        "dev_otp_console": False,
        "cookie_consent_enabled": False,
        "session_days": 30,
    }

    now = now_ts()
    exec_one(db_path, "INSERT INTO users(email, created_at) VALUES(?,?)", ("user@school.edu", now))

    client = app.test_client()
    with client.session_transaction() as sess:
        sess["pending_email"] = "user@school.edu"
    from anispeakarchive.auth.otp import hash_code

    app.config["SECRET_KEY"] = app.secret_key
    exec_one(
        db_path,
        "INSERT INTO otps(email, otp_hash, expires_at, created_at) VALUES(?,?,?,?)",
        ("user@school.edu", hash_code("user@school.edu", "123456", app.secret_key), now + 600, now),
    )
    res = client.post("/verify", data={"code": "123456"}, follow_redirects=False)
    assert res.status_code == 302
    set_cookies = res.headers.getlist("Set-Cookie")
    assert any("asa_auth=" in header for header in set_cookies)

    client2 = app.test_client()
    auth_header = next(header for header in set_cookies if "asa_auth=" in header)
    cookie_value = auth_header.split("asa_auth=", 1)[1].split(";", 1)[0]
    client2.set_cookie("asa_auth", cookie_value)
    restored = client2.get("/whoami", follow_redirects=False)
    assert restored.status_code == 200
    payload = restored.get_json()
    assert payload["user_id"] == 1
    assert payload["user_email"] == "user@school.edu"
