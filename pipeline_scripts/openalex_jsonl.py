#!/usr/bin/env python3
"""
local_openalex_jsonl.py
=======================
Simulates the OpenAlex /works API by streaming a pre-filtered JSONL.

The cursor encodes a byte offset into the file → zero-overhead seeking,
no DB needed, no RAM index.

Set in fetch_papers.py:
    OPENALEX_BASE = "http://127.0.0.1:5001"
Everything else in fetch_papers.py stays unchanged.

Usage:
    python local_openalex_jsonl.py \
        --jsonl ./filtered_works.jsonl \
        --port 5001
"""

import argparse
import base64
import json
import os
import time
from datetime import datetime
from threading import RLock

from flask import Flask, request, jsonify

app         = Flask(__name__)
JSONL_PATH  = None
TOTAL_COUNT = 0
_lock       = RLock()


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Cursor = base64(byte_offset) ─────────────────────────────────────────────

def encode_cursor(offset: int) -> str:
    return base64.b64encode(str(offset).encode()).decode()


def decode_cursor(cursor: str) -> int:
    if not cursor or cursor == "*":
        return 0
    try:
        return int(base64.b64decode(cursor.encode()).decode())
    except Exception:
        return 0


# ── Count lines once at startup ───────────────────────────────────────────────

def count_lines(path: str) -> int:
    log(f"Counting lines in {path} ...")
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    log(f"  Total records: {n:,}")
    return n


# ── /works endpoint ───────────────────────────────────────────────────────────

@app.route("/works", methods=["GET"])
def works():
    t0       = time.time()
    per_page = min(int(request.args.get("per-page", 50)), 200)
    cursor   = request.args.get("cursor", "*")

    # accept but ignore these — dataset is pre-filtered offline
    # request.args.get("filter", "")
    # request.args.get("search", "")
    # request.args.get("mailto", "")
    # request.args.get("api_key", "")

    offset  = decode_cursor(cursor)
    results = []

    with _lock:
        with open(JSONL_PATH, "rb") as f:
            f.seek(offset)
            for _ in range(per_page):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line.decode("utf-8")))
                except Exception:
                    continue
            next_off = f.tell()
            # peek to check if we're really at EOF
            at_eof = f.read(1) == b""

    next_cursor = None if (at_eof or not results) else encode_cursor(next_off)

    return jsonify({
        "meta": {
            "count":               TOTAL_COUNT,
            "db_response_time_ms": int((time.time() - t0) * 1000),
            "page":                1,
            "per_page":            per_page,
            "next_cursor":         next_cursor,
        },
        "results": results,
    })


# ── /works/<id> endpoint ──────────────────────────────────────────────────────

@app.route("/works/<work_id>", methods=["GET"])
def get_work(work_id):
    with _lock:
        with open(JSONL_PATH, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if work_id in str(obj.get("id", "")):
                    return jsonify(obj)
    return jsonify({"error": "Not found"}), 404


# ── / health check ────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "description": "Local OpenAlex /works emulator — pre-filtered JSONL backend",
        "jsonl_path":  JSONL_PATH,
        "works_count": TOTAL_COUNT,
        "endpoints":   ["/works?cursor=*&per-page=50", "/works/<id>"],
        "note":        "filter= and search= params are accepted but ignored (pre-filtered)",
    })


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global JSONL_PATH, TOTAL_COUNT

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Filtered JSONL path")
    ap.add_argument("--host",  default="127.0.0.1")
    ap.add_argument("--port",  type=int, default=5001)
    args = ap.parse_args()

    JSONL_PATH = args.jsonl
    if not os.path.exists(JSONL_PATH):
        raise SystemExit(f"JSONL not found: {JSONL_PATH}")

    TOTAL_COUNT = count_lines(JSONL_PATH)

    log(f"Starting local OpenAlex emulator on http://{args.host}:{args.port}")
    log(f"  JSONL     : {JSONL_PATH}")
    log(f"  Records   : {TOTAL_COUNT:,}")
    log(f"  In fetch_papers.py set: OPENALEX_BASE = 'http://{args.host}:{args.port}'")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
