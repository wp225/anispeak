from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class DbPaths:
    db_path: str


def _connect(path: str) -> sqlite3.Connection:
    assert isinstance(path, str) and path, "db path must be a non-empty string"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _table_columns(cur: sqlite3.Cursor, table_name: str) -> set[str]:
    cur.execute(f"PRAGMA table_info({table_name})")
    return {str(row[1]) for row in cur.fetchall()}


def init_db(path: str) -> None:
    con = _connect(path)
    try:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              email TEXT NOT NULL UNIQUE,
              created_at INTEGER NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS otps (
              email TEXT NOT NULL,
              otp_hash TEXT NOT NULL,
              expires_at INTEGER NOT NULL,
              created_at INTEGER NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_otps_email ON otps(email)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_otps_exp ON otps(expires_at)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bookmarks (
              user_id INTEGER NOT NULL,
              doc_id TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              PRIMARY KEY (user_id, doc_id),
              FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        # --- v2 schema: user library (collections, notes, saved searches) ---
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS collections (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              name TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              UNIQUE(user_id, name),
              FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_collections_user ON collections(user_id)")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_papers (
              user_id INTEGER NOT NULL,
              doc_id TEXT NOT NULL,
              saved_at INTEGER NOT NULL,
              source TEXT NOT NULL, -- 'paper' | 'search'
              collection_id INTEGER, -- nullable
              note TEXT,
              updated_at INTEGER NOT NULL,
              PRIMARY KEY (user_id, doc_id),
              FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
              FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE SET NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_saved_papers_user ON saved_papers(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_saved_papers_saved_at ON saved_papers(saved_at)")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_searches (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              name TEXT NOT NULL,
              query_json TEXT NOT NULL,
              cadence TEXT NOT NULL, -- 'daily'|'weekly'
              enabled INTEGER NOT NULL DEFAULT 1,
              created_at INTEGER NOT NULL,
              last_sent_at INTEGER,
              FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_saved_searches_user ON saved_searches(user_id)")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_search_hits (
              search_id INTEGER NOT NULL,
              doc_id TEXT NOT NULL,
              first_seen_at INTEGER NOT NULL,
              PRIMARY KEY (search_id, doc_id),
              FOREIGN KEY (search_id) REFERENCES saved_searches(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_saved_search_hits_search ON saved_search_hits(search_id)")

        # Ask AI: conversation history
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ask_conversations (
              id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              user_id INTEGER,
              title TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL,
              FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ask_messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              conversation_id TEXT NOT NULL REFERENCES ask_conversations(id) ON DELETE CASCADE,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              retrieval_json TEXT,
              response_json TEXT
            )
            """
        )

        ask_conv_cols = _table_columns(cur, "ask_conversations")
        if "user_id" not in ask_conv_cols:
            cur.execute("ALTER TABLE ask_conversations ADD COLUMN user_id INTEGER")

        ask_msg_cols = _table_columns(cur, "ask_messages")
        if "retrieval_json" not in ask_msg_cols:
            cur.execute("ALTER TABLE ask_messages ADD COLUMN retrieval_json TEXT")
        if "response_json" not in ask_msg_cols:
            cur.execute("ALTER TABLE ask_messages ADD COLUMN response_json TEXT")

        cur.execute("CREATE INDEX IF NOT EXISTS idx_ask_conv_session ON ask_conversations(session_id, updated_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ask_conv_user ON ask_conversations(user_id, updated_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ask_msg_conv ON ask_messages(conversation_id, id)")

        # OTP request throttling
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS otp_rate (
              email TEXT NOT NULL,
              window_start INTEGER NOT NULL,
              count INTEGER NOT NULL,
              PRIMARY KEY (email)
            )
            """
        )

        # One-time migration: move old bookmarks into saved_papers
        # (idempotent thanks to PRIMARY KEY on (user_id, doc_id)).
        cur.execute(
            """
            INSERT OR IGNORE INTO saved_papers(user_id, doc_id, saved_at, source, collection_id, note, updated_at)
            SELECT user_id, doc_id, created_at, 'paper', NULL, NULL, created_at FROM bookmarks
            """
        )
        con.commit()
    finally:
        con.close()


def exec_one(path: str, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
    con = _connect(path)
    try:
        cur = con.cursor()
        cur.execute(sql, tuple(params))
        row = cur.fetchone()
        con.commit()
        return row
    finally:
        con.close()


def exec_all(path: str, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    con = _connect(path)
    try:
        cur = con.cursor()
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        con.commit()
        return list(rows)
    finally:
        con.close()
