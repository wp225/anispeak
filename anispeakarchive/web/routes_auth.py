from __future__ import annotations

import time
from flask import Blueprint, current_app, render_template, request, redirect, url_for, session, flash, make_response
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from anispeakarchive.auth.db import init_db, exec_one, exec_all
from anispeakarchive.auth.otp import generate_code, hash_code, is_valid_email, domain_of, now_ts
from anispeakarchive.auth.emailer import SmtpCfg, can_send, send_otp


bp = Blueprint("auth", __name__)


def _auth_cfg():
    return current_app.config.get("AUTH_CFG", {})


def _db_path() -> str:
    p = str(_auth_cfg().get("db_path", "cache/auth.sqlite"))
    assert p, "auth db_path must be set"
    return p


def _smtp_cfg() -> SmtpCfg:
    c = _auth_cfg()
    return SmtpCfg(
        host=c.get("smtp_host"),
        port=int(c.get("smtp_port", 587)),
        username=c.get("smtp_username"),
        password=c.get("smtp_password"),
        from_addr=c.get("smtp_from"),
        use_tls=bool(c.get("smtp_use_tls", True)),
    )


def _allowed_email(email: str) -> bool:
    c = _auth_cfg()
    if not is_valid_email(email):
        return False
    dom = domain_of(email)
    allowed = c.get("allowed_domains")
    if isinstance(allowed, list) and allowed:
        return dom in {str(x).lower() for x in allowed}
    if bool(c.get("allow_edu_only", True)):
        return dom.endswith(".edu")
    return True


def _dev_otp_console_enabled() -> bool:
    cfg = _auth_cfg()
    if bool(cfg.get("dev_otp_console", False)):
        return True
    return bool(current_app.debug)


def _auth_cookie_name() -> str:
    return "asa_auth"


def _auth_cookie_days() -> int:
    return int(_auth_cfg().get("session_days", 30))


def _auth_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"], salt="asa-auth-cookie-v1")


def _auth_cookie_value(user_id: int, email: str) -> str:
    return _auth_serializer().dumps({"user_id": int(user_id), "email": str(email)})


def _set_auth_cookie(resp, user_id: int, email: str) -> None:
    max_age = _auth_cookie_days() * 24 * 3600
    resp.set_cookie(
        _auth_cookie_name(),
        _auth_cookie_value(user_id, email),
        max_age=max_age,
        httponly=True,
        samesite="Lax",
        secure=bool(request.is_secure),
    )


def _clear_auth_cookie(resp) -> None:
    resp.delete_cookie(_auth_cookie_name(), samesite="Lax")


def _restore_auth_from_cookie() -> None:
    if session.get("user_id") is not None and session.get("user_email") is not None:
        return
    raw = request.cookies.get(_auth_cookie_name())
    if not raw:
        return
    try:
        payload = _auth_serializer().loads(raw, max_age=_auth_cookie_days() * 24 * 3600)
    except (BadSignature, SignatureExpired):
        return
    try:
        user_id = int(payload.get("user_id"))
        email = str(payload.get("email") or "").strip().lower()
    except Exception:
        return
    if user_id <= 0 or not email:
        return
    session["user_id"] = user_id
    session["user_email"] = email
    session.permanent = True
    session.modified = True


def current_user() -> dict | None:
    uid = session.get("user_id")
    em = session.get("user_email")
    if uid is None or em is None:
        return None
    return {"id": int(uid), "email": str(em)}


@bp.before_app_request
def _ensure_db():
    # Lazy init once per process.
    if not current_app.config.get("_AUTH_DB_INIT", False):
        init_db(_db_path())
        current_app.config["_AUTH_DB_INIT"] = True
    _restore_auth_from_cookie()


@bp.app_context_processor
def _inject_user():
    c = _auth_cfg()
    return {
        "current_user": current_user(),
        "cookie_consent_enabled": bool(c.get("cookie_consent_enabled", True)),
        "cookie_session_days": int(c.get("session_days", 30)),
    }


@bp.get("/login")
def login_page():
    if current_user() is not None:
        return redirect(url_for("auth.profile_page"))
    return render_template("pages/login.html")


@bp.post("/login")
def login_send_otp():
    email = str(request.form.get("email", "")).strip().lower()
    if not _allowed_email(email):
        flash("Email not allowed. Use an approved academic domain.", "error")
        return redirect(url_for("auth.login_page"))

    code = generate_code()
    otp_hash = hash_code(email, code, current_app.config["SECRET_KEY"])
    ttl = int(_auth_cfg().get("otp_ttl_sec", 600))
    now = now_ts()
    exp = now + ttl

    # Store OTP
    exec_one(_db_path(), "DELETE FROM otps WHERE email = ?", (email,))
    exec_one(
        _db_path(),
        "INSERT INTO otps(email, otp_hash, expires_at, created_at) VALUES(?,?,?,?)",
        (email, otp_hash, exp, now),
    )

    # Rate limit: max N OTPs per rolling window.
    # Stored in DB so it works across workers.
    c = _auth_cfg()
    max_per_window = int(c.get("otp_max_per_window", 5))
    window_sec = int(c.get("otp_window_sec", 3600))
    wrow = exec_one(_db_path(), "SELECT window_start, count FROM otp_rate WHERE email = ?", (email,))
    now_rl = now_ts()
    if wrow is None or (now_rl - int(wrow["window_start"])) > window_sec:
        exec_one(_db_path(), "INSERT OR REPLACE INTO otp_rate(email, window_start, count) VALUES(?,?,?)", (email, now_rl, 1))
    else:
        cnt = int(wrow["count"]) + 1
        if cnt > max_per_window:
            flash("Too many verification attempts. Please wait and try again.", "error")
            return redirect(url_for("auth.login_page"))
        exec_one(_db_path(), "UPDATE otp_rate SET count = ? WHERE email = ?", (cnt, email))

    # Send OTP. In development, fall back to logging the code if SMTP is missing/broken.
    smtp = _smtp_cfg()
    try:
        if can_send(smtp):
            send_otp(smtp, email, code)
            current_app.logger.info("[AUTH] OTP sent email=%s", email)
            flash("Verification code sent. Check your email.", "ok")
        elif _dev_otp_console_enabled():
            current_app.logger.warning("[AUTH][DEV OTP] email=%s code=%s", email, code)
            flash("Development mode: OTP email unavailable. Code printed to server logs.", "ok")
        else:
            raise RuntimeError("SMTP not configured")
    except Exception as e:
        if _dev_otp_console_enabled():
            current_app.logger.warning("[AUTH][DEV OTP FALLBACK] email=%s code=%s err=%s", email, code, type(e).__name__)
            flash("Development mode: OTP email failed. Code printed to server logs.", "ok")
        else:
            current_app.logger.error("[AUTH] SMTP send failed for %s: %s", email, type(e).__name__)
            exec_one(_db_path(), "DELETE FROM otps WHERE email = ?", (email,))
            flash("Email delivery is not available right now. Please try again later.", "error")
            return redirect(url_for("auth.login_page"))

    session["pending_email"] = email
    next_url = request.args.get("next", "")
    if next_url and next_url.startswith("/"):
        session["login_next"] = next_url
    return redirect(url_for("auth.verify_page"))


@bp.get("/verify")
def verify_page():
    if current_user() is not None:
        return redirect(url_for("auth.profile_page"))
    email = session.get("pending_email")
    if not email:
        return redirect(url_for("auth.login_page"))
    return render_template("pages/verify.html", email=email)


@bp.post("/verify")
def verify_submit():
    email = str(session.get("pending_email") or "").strip().lower()
    if not email:
        return redirect(url_for("auth.login_page"))

    code = str(request.form.get("code", "")).strip()
    if not code.isdigit() or len(code) != 6:
        flash("Invalid code format.", "error")
        return redirect(url_for("auth.verify_page"))

    row = exec_one(_db_path(), "SELECT otp_hash, expires_at FROM otps WHERE email = ?", (email,))
    if row is None:
        flash("No active code. Please request a new one.", "error")
        return redirect(url_for("auth.login_page"))
    if int(row["expires_at"]) < now_ts():
        flash("Code expired. Please request a new one.", "error")
        exec_one(_db_path(), "DELETE FROM otps WHERE email = ?", (email,))
        return redirect(url_for("auth.login_page"))

    expected = str(row["otp_hash"])
    got = hash_code(email, code, current_app.config["SECRET_KEY"])
    if got != expected:
        flash("Incorrect code.", "error")
        return redirect(url_for("auth.verify_page"))

    # Upsert user
    now = now_ts()
    u = exec_one(_db_path(), "SELECT id FROM users WHERE email = ?", (email,))
    if u is None:
        exec_one(_db_path(), "INSERT INTO users(email, created_at) VALUES(?,?)", (email, now))
        u = exec_one(_db_path(), "SELECT id FROM users WHERE email = ?", (email,))
    assert u is not None

    # Consume OTP
    exec_one(_db_path(), "DELETE FROM otps WHERE email = ?", (email,))
    session.pop("pending_email", None)

    session["user_id"] = int(u["id"])
    session["user_email"] = email
    current_app.logger.info("[AUTH] Verified login email=%s user_id=%s", email, int(u["id"]))

    # Apply consent policy
    consent = request.cookies.get("asa_consent")
    if consent == "accept":
        session.permanent = True
    else:
        session.permanent = False

    flash("Signed in.", "ok")
    next_url = session.pop("login_next", None)
    dest = next_url if (next_url and next_url.startswith("/")) else url_for("auth.profile_page")
    resp = make_response(redirect(dest))
    _set_auth_cookie(resp, int(u["id"]), email)
    return resp


@bp.get("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("user_email", None)
    session.pop("pending_email", None)
    flash("Signed out.", "ok")
    resp = make_response(redirect(url_for("pages.search_page")))
    _clear_auth_cookie(resp)
    return resp


@bp.get("/profile")
def profile_page():
    u = current_user()
    if u is None:
        return redirect(url_for("auth.login_page"))
    return render_template("pages/profile.html", user=u)


@bp.get("/consent")
def consent_set():
    choice = str(request.args.get("choice", "")).strip().lower()
    if choice not in {"accept", "reject"}:
        return redirect(url_for("pages.search_page"))
    days = int(_auth_cfg().get("session_days", 30))
    max_age = days * 24 * 3600
    resp = make_response(redirect(request.referrer or url_for("pages.search_page")))
    if choice == "accept":
        resp.set_cookie("asa_consent", "accept", max_age=max_age, samesite="Lax")
        # Make sessions persistent for this browser.
        session.permanent = True
        session.modified = True
    else:
        # keep a short-lived reject cookie to avoid re-asking every page load
        resp.set_cookie("asa_consent", "reject", max_age=7 * 24 * 3600, samesite="Lax")
        # also ensure session isn't persistent
        session.permanent = False
        session.modified = True
    return resp
