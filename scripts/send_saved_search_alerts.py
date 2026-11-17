from __future__ import annotations

import json
import os
from typing import Any

from app import create_app

from anispeakarchive.auth.db import exec_all, exec_one, init_db
from anispeakarchive.auth.emailer import SmtpCfg, send_text_email
from anispeakarchive.index.hybrid import SearchFilters
from anispeakarchive.auth.otp import now_ts


def _smtp_from_cfg(auth_cfg: dict) -> SmtpCfg:
    return SmtpCfg(
        host=auth_cfg.get("smtp_host"),
        port=int(auth_cfg.get("smtp_port", 587)),
        username=auth_cfg.get("smtp_username"),
        password=auth_cfg.get("smtp_password"),
        from_addr=auth_cfg.get("smtp_from"),
        use_tls=bool(auth_cfg.get("smtp_use_tls", True)),
    )


def _db_path(auth_cfg: dict) -> str:
    p = str(auth_cfg.get("db_path", "cache/auth.sqlite"))
    assert p
    return p


def _due(cadence: str, last_sent_at: int | None, now: int) -> bool:
    if last_sent_at is None:
        return True
    delta = now - int(last_sent_at)
    if cadence == "daily":
        return delta >= 23 * 3600
    if cadence == "weekly":
        return delta >= 6 * 24 * 3600
    return False


def _filters_from_query(q: dict[str, Any]) -> tuple[str, SearchFilters, int]:
    query_text = str(q.get("q", "")).strip()
    assert query_text, "saved search requires non-empty q"

    def _lst(key: str):
        v = q.get(key)
        if v is None:
            return None
        if isinstance(v, list):
            return [str(x) for x in v if str(x).strip()]
        s = str(v).strip()
        return [s] if s else None

    page_size = int(q.get("page_size", 20))
    page_size = max(1, min(50, page_size))

    f = SearchFilters(
        year_min=int(q["year_min"]) if q.get("year_min") not in (None, "") else None,
        year_max=int(q["year_max"]) if q.get("year_max") not in (None, "") else None,
        species=_lst("species"),
        modalities=_lst("modalities"),
        features=_lst("features"),
        stages=_lst("stages"),
        domains=_lst("domains"),
        contexts=_lst("contexts"),
        institutions=_lst("institutions"),
        authors=_lst("authors"),
        venues=_lst("venues"),
        paper_type=str(q.get("paper_type")) if q.get("paper_type") else None,
        tradition=str(q.get("tradition")) if q.get("tradition") else None,
        has_code=bool(q.get("has_code")) if q.get("has_code") is not None else None,
        has_dataset=bool(q.get("has_dataset")) if q.get("has_dataset") is not None else None,
        has_benchmark=bool(q.get("has_benchmark")) if q.get("has_benchmark") is not None else None,
    )
    return query_text, f, page_size


def main() -> None:
    app = create_app()
    auth_cfg = app.config.get("AUTH_CFG", {})
    if not bool(auth_cfg.get("enabled", False)):
        raise SystemExit("Auth is disabled; saved search alerts are unavailable.")

    db_path = _db_path(auth_cfg)
    init_db(db_path)

    smtp = _smtp_from_cfg(auth_cfg)
    # Force early failure if SMTP is missing.
    if not (smtp.host and smtp.from_addr):
        raise SystemExit("SMTP not configured. Set auth.smtp_* in configs/default.yaml (or load from env).")

    base_url = str(auth_cfg.get("public_base_url") or "").rstrip("/")
    if not base_url:
        raise SystemExit("Set auth.public_base_url (e.g., https://anispeakarchive.org) for email links.")

    now = now_ts()

    with app.app_context():
        svc = app.config["SEARCH_SERVICE"]

        searches = exec_all(
            db_path,
            """
            SELECT ss.id, ss.user_id, ss.name, ss.query_json, ss.cadence, ss.last_sent_at, u.email
            FROM saved_searches ss
            JOIN users u ON u.id = ss.user_id
            WHERE ss.enabled = 1
            """,
        )

        for s in searches:
            sid = int(s["id"])
            cadence = str(s["cadence"])
            last_sent = int(s["last_sent_at"]) if s["last_sent_at"] is not None else None
            if not _due(cadence, last_sent, now):
                continue

            try:
                q = json.loads(str(s["query_json"]))
            except Exception:
                continue

            try:
                query_text, filters, page_size = _filters_from_query(q)
            except Exception:
                continue

            res = svc.search(query_text, filters, page=1, page_size=page_size)
            candidates = res.get("results", [])
            new_items = []

            for p in candidates:
                doc_id = str(p.get("doc_id") or "").strip()
                if not doc_id:
                    continue
                seen = exec_one(db_path, "SELECT 1 FROM saved_search_hits WHERE search_id = ? AND doc_id = ?", (sid, doc_id))
                if seen is not None:
                    continue
                exec_one(db_path, "INSERT OR IGNORE INTO saved_search_hits(search_id, doc_id, first_seen_at) VALUES(?,?,?)", (sid, doc_id, now))
                new_items.append(p)
                if len(new_items) >= 10:
                    break

            # Update last_sent_at even if no new items, to avoid spamming.
            exec_one(db_path, "UPDATE saved_searches SET last_sent_at = ? WHERE id = ?", (now, sid))

            if not new_items:
                continue

            lines = []
            lines.append(f"AniSpeakArchive alert: {s['name']}")
            lines.append("")
            for p in new_items:
                title = str(p.get("title") or p.get("display_name") or "").strip()
                year = str(p.get("year") or "").strip()
                link = f"{base_url}/paper/{p.get('doc_id')}"
                lines.append(f"- {title} ({year})")
                lines.append(f"  {link}")
            lines.append("")
            lines.append("You can manage alerts in your profile: " + f"{base_url}/profile")

            subject = f"AniSpeakArchive saved search: {s['name']}"
            send_text_email(smtp, str(s["email"]), subject, "\n".join(lines))


if __name__ == "__main__":
    main()
