from __future__ import annotations

import smtplib
from dataclasses import dataclass
from email.message import EmailMessage


@dataclass(frozen=True)
class SmtpCfg:
    host: str | None
    port: int
    username: str | None
    password: str | None
    from_addr: str | None
    use_tls: bool = True


def can_send(cfg: SmtpCfg) -> bool:
    return bool(cfg.host and cfg.from_addr)


def send_otp(cfg: SmtpCfg, to_email: str, code: str) -> None:
    assert isinstance(to_email, str) and to_email
    assert isinstance(code, str) and code

    if not can_send(cfg):
        # No SMTP configured: caller should log OTP.
        raise RuntimeError("SMTP not configured")

    msg = EmailMessage()
    msg["Subject"] = "Your AniSpeakArchive verification code"
    msg["From"] = cfg.from_addr
    msg["To"] = to_email
    msg.set_content(
        "Your verification code is: " + code + "\n\n" +
        "This code expires soon. If you did not request it, you can ignore this email."
    )

    _send(cfg, msg)


def send_text_email(cfg: SmtpCfg, to_email: str, subject: str, body: str) -> None:
    assert isinstance(to_email, str) and to_email
    assert isinstance(subject, str) and subject
    assert isinstance(body, str) and body
    if not can_send(cfg):
        raise RuntimeError("SMTP not configured")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.from_addr
    msg["To"] = to_email
    msg.set_content(body)
    _send(cfg, msg)


def _send(cfg: SmtpCfg, msg: EmailMessage) -> None:
    assert can_send(cfg)

    if cfg.use_tls:
        with smtplib.SMTP(cfg.host, cfg.port, timeout=15) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            if cfg.username and cfg.password:
                s.login(cfg.username, cfg.password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(cfg.host, cfg.port, timeout=15) as s:
            if cfg.username and cfg.password:
                s.login(cfg.username, cfg.password)
            s.send_message(msg)
