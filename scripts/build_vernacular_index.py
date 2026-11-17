#!/usr/bin/env python3
"""
Build a local vernacular -> GBIF usageKey index (SQLite) and test lookups.

Usage:
  # Build
  python scripts/build_vernacular_index.py build \
    --vernacular-tsv data/gbif_vernacular.tsv \
    --out-sqlite data/vernacular_index.sqlite \
    --gbif-base http://127.0.0.1:5643/v2 \
    --commit-every 50000

  # Lookup
  python scripts/build_vernacular_index.py lookup \
    --sqlite data/vernacular_index.sqlite \
    --k 10 \
    --two-pass \
    "eastern black rail" "bottlenose dolphins"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import requests

# Fix: _csv.Error: field larger than field limit (131072)
csv.field_size_limit(sys.maxsize)

STOPWORDS = {
    # life stage / sex / common modifiers
    "male", "female", "juvenile", "adult", "adults", "immature",
    "pup", "pups", "calf", "calves", "hatchling", "hatchlings",
    "larva", "larvae", "nymph", "nymphs",
    "nestling", "nestlings", "chick", "chicks",
    "young", "offspring",
    # behavioral modifiers that often appear in abstracts
    "foraging", "calling", "vocal", "roaring", "singing",
    # vague group words
    "species", "sp", "spp", "taxa", "taxon",
    # common ecology adjectives (optional, conservative set)
    "wild", "feral", "domestic",
}

GEO_STOPWORDS = {
    "eastern", "western", "northern", "southern",
    "pacific", "atlantic", "arctic", "antarctic",
    "eurasian", "african", "asian", "american", "european",
}

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Safety caps
MAX_RAW_LEN = 4096          # skip raw phrases longer than this
MAX_NORM_LEN = 256          # cap normalized key length
MAX_USAGEKEY_LEN = 64       # conservative


def _assert(cond: bool, msg: str) -> None:
    assert cond, msg


def singularize_token(tok: str) -> str:
    _assert(tok and isinstance(tok, str), "singularize_token expects non-empty str")
    if tok.endswith("ies") and len(tok) > 3:
        return tok[:-3] + "y"   # butterflies -> butterfly
    if tok.endswith("ves") and len(tok) > 3:
        return tok[:-3] + "f"   # wolves -> wolf (heuristic)
    if tok.endswith("ses") and len(tok) > 3:
        return tok[:-2]         # buses -> bus (heuristic)
    if tok.endswith("s") and len(tok) > 1 and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def normalize_vernacular(s: str, drop_geo: bool) -> str:
    """
    Normalize a vernacular string into a canonical lookup key.

    - lowercases
    - replaces hyphens/dashes with spaces
    - tokenizes alnum words
    - drops STOPWORDS
    - optionally drops GEO_STOPWORDS
    - naive singularization of final token
    """
    _assert(isinstance(s, str), f"normalize_vernacular expects str, got {type(s)}")
    s0 = s.strip().lower()
    if not s0:
        return ""

    # normalize separators
    s0 = s0.replace("-", " ").replace("–", " ").replace("—", " ")

    toks = _WORD_RE.findall(s0)
    _assert(isinstance(toks, list), "tokenization failed unexpectedly")

    out: List[str] = []
    for t in toks:
        if t in STOPWORDS:
            continue
        if drop_geo and t in GEO_STOPWORDS:
            continue
        out.append(t)

    if not out:
        return ""

    out[-1] = singularize_token(out[-1])
    norm = " ".join(out).strip()

    # Hard cap (prevents pathological phrases)
    if len(norm) > MAX_NORM_LEN:
        return ""
    return norm


def detect_columns(fieldnames: Sequence[str]) -> Tuple[str, str, Optional[str], Optional[str]]:
    lower = {f.lower(): f for f in fieldnames}

    key_candidates = ["taxonkey", "usagekey", "key", "taxonid", "id"]
    name_candidates = ["vernacularname", "vernacular", "name", "commonname"]
    lang_candidates = ["language", "lang"]
    pref_candidates = ["ispreferredname", "preferred", "preferredname", "status"]

    key_col = next((lower[c] for c in key_candidates if c in lower), None)
    name_col = next((lower[c] for c in name_candidates if c in lower), None)
    lang_col = next((lower[c] for c in lang_candidates if c in lower), None)
    pref_col = next((lower[c] for c in pref_candidates if c in lower), None)

    _assert(key_col is not None, f"Could not find a key column. Have: {fieldnames}")
    _assert(name_col is not None, f"Could not find a name column. Have: {fieldnames}")
    return key_col, name_col, lang_col, pref_col


def check_gbif_api(gbif_base: str, timeout_s: float = 2.0) -> None:
    _assert(gbif_base.startswith("http://") or gbif_base.startswith("https://"),
            f"gbif_base must be a URL, got: {gbif_base}")

    url = gbif_base.rstrip("/") + "/species/match"
    params = {"vernacularName": "lion", "kingdom": "Animalia", "verbose": "false"}
    try:
        r = requests.get(url, params=params, timeout=timeout_s)
    except Exception as e:
        raise AssertionError(f"GBIF API not reachable at {gbif_base}: {e}") from e

    _assert(r.status_code == 200, f"GBIF API check failed: {r.status_code} {r.text[:200]}")
    try:
        j = r.json()
    except Exception as e:
        raise AssertionError(f"GBIF API returned non-JSON: {r.text[:200]}") from e

    _assert(isinstance(j, dict), "GBIF API JSON was not an object")
    _assert(("usage" in j) or ("matchType" in j) or ("usageKey" in j),
            f"Unexpected GBIF payload keys: {list(j.keys())[:20]}")


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vernacular_index (
            norm TEXT NOT NULL,
            usage_key TEXT NOT NULL,
            raw TEXT NOT NULL,
            lang TEXT,
            score INTEGER NOT NULL,
            PRIMARY KEY (norm, usage_key, raw)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_norm ON vernacular_index(norm);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage ON vernacular_index(usage_key);")
    conn.commit()


def score_row(lang: Optional[str], preferred: Optional[str]) -> int:
    s = 0
    if lang:
        l = lang.strip().lower()
        if l in {"en", "eng", "english"}:
            s += 10
    if preferred:
        p = preferred.strip().lower()
        if p in {"true", "1", "yes", "preferred", "accepted"}:
            s += 5
    return s


def _open_tsv(path: str) -> Tuple[Iterable[dict], Sequence[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.excel_tab
        if "\t" not in sample and "," in sample:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        _assert(reader.fieldnames is not None and len(reader.fieldnames) > 0, "TSV has no header row")
        # Return a generator that keeps file alive by re-opening in build_index. We just use this for detection.
        fieldnames = list(reader.fieldnames)
    # Re-open is handled in build_index
    return [], fieldnames


def build_index(
    vernacular_tsv: str,
    out_sqlite: str,
    gbif_base: str,
    drop_geo: bool,
    limit: Optional[int],
    commit_every: int,
    raw_max_len: int,
) -> None:
    _assert(commit_every > 0, "commit_every must be > 0")
    _assert(raw_max_len > 16, "raw_max_len too small")

    check_gbif_api(gbif_base)

    # Detect columns using a short open
    _, fieldnames = _open_tsv(vernacular_tsv)

    # Now do the real read
    with open(vernacular_tsv, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.excel_tab
        if "\t" not in sample and "," in sample:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        _assert(reader.fieldnames is not None, "missing header row unexpectedly")
        key_col, name_col, lang_col, pref_col = detect_columns(reader.fieldnames)

        conn = sqlite3.connect(out_sqlite)
        try:
            ensure_schema(conn)
            cur = conn.cursor()

            inserted = 0
            seen = 0
            skip_long_raw = 0
            skip_empty_norm = 0
            skip_long_norm = 0
            skip_bad_key = 0

            t0 = time.time()

            for row in reader:
                seen += 1
                if limit is not None and seen > limit:
                    break

                raw_name = (row.get(name_col) or "")
                key = (row.get(key_col) or "")

                raw_name = raw_name.strip()
                key = key.strip()

                if not raw_name or not key:
                    continue

                if len(raw_name) > raw_max_len:
                    skip_long_raw += 1
                    continue

                if len(key) > MAX_USAGEKEY_LEN:
                    skip_bad_key += 1
                    continue

                norm = normalize_vernacular(raw_name, drop_geo=drop_geo)
                if not norm:
                    skip_empty_norm += 1
                    continue
                if len(norm) > MAX_NORM_LEN:
                    skip_long_norm += 1
                    continue

                lang = (row.get(lang_col) or "").strip() if lang_col else None
                pref = (row.get(pref_col) or "").strip() if pref_col else None
                sc = score_row(lang, pref)

                cur.execute(
                    """
                    INSERT OR IGNORE INTO vernacular_index (norm, usage_key, raw, lang, score)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (norm, key, raw_name, lang if lang else None, sc),
                )
                inserted += cur.rowcount

                if seen % commit_every == 0:
                    conn.commit()
                    dt = time.time() - t0
                    rate = seen / max(dt, 1e-9)
                    print(
                        f"[build] seen={seen} inserted={inserted} rate={rate:.1f}/s "
                        f"skip_long_raw={skip_long_raw} skip_bad_key={skip_bad_key} "
                        f"skip_empty_norm={skip_empty_norm} skip_long_norm={skip_long_norm}",
                        file=sys.stderr,
                    )

            conn.commit()
            dt = time.time() - t0

            _assert(seen > 0, "No rows read from TSV")
            _assert(inserted > 0, "Inserted 0 rows. Probably wrong columns or empty TSV.")
            print(
                f"[build] DONE seen={seen} inserted={inserted} seconds={dt:.1f} "
                f"skip_long_raw={skip_long_raw} skip_bad_key={skip_bad_key} "
                f"skip_empty_norm={skip_empty_norm} skip_long_norm={skip_long_norm}",
                file=sys.stderr,
            )
        finally:
            conn.close()


def _lookup_once(conn: sqlite3.Connection, norm: str, k: int) -> List[Tuple[str, str, Optional[str], int]]:
    rows = conn.execute(
        """
        SELECT usage_key, raw, lang, score
        FROM vernacular_index
        WHERE norm = ?
        ORDER BY score DESC, lang DESC, raw ASC
        LIMIT ?;
        """,
        (norm, k),
    ).fetchall()
    # rows: List[Tuple[str, str, Optional[str], int]]
    return rows


def lookup(
    sqlite_path: str,
    queries: Sequence[str],
    k: int,
    drop_geo: bool,
    two_pass: bool,
) -> None:
    _assert(k > 0, "k must be > 0")
    _assert(len(queries) > 0, "need at least one query")

    conn = sqlite3.connect(sqlite_path)
    try:
        for q in queries:
            _assert(isinstance(q, str) and q.strip(), "query must be non-empty string")
            q = q.strip()

            # pass 1
            norm1 = normalize_vernacular(q, drop_geo=drop_geo)
            rows1: List[Tuple[str, str, Optional[str], int]] = []
            if norm1:
                rows1 = _lookup_once(conn, norm1, k)

            rows_out = rows1
            norm_out = norm1

            # pass 2 fallback: keep-geo then drop-geo
            if two_pass and not rows_out:
                norm2 = normalize_vernacular(q, drop_geo=True)
                if norm2 and norm2 != norm1:
                    rows2 = _lookup_once(conn, norm2, k)
                    if rows2:
                        rows_out = rows2
                        norm_out = norm2

            print(json.dumps({"query": q, "norm": norm_out}, ensure_ascii=False))
            print(json.dumps({"candidates": rows_out}, ensure_ascii=False))
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Build vernacular sqlite index from a TSV")
    pb.add_argument("--vernacular-tsv", required=True)
    pb.add_argument("--out-sqlite", required=True)
    pb.add_argument("--gbif-base", required=True)
    pb.add_argument("--drop-geo", action="store_true", default=False,
                    help="Drop geographic adjectives (pacific/eastern/etc). Off by default for safety.")
    pb.add_argument("--limit", type=int, default=None)
    pb.add_argument("--commit-every", type=int, default=50_000)
    pb.add_argument("--raw-max-len", type=int, default=MAX_RAW_LEN)

    pl = sub.add_parser("lookup", help="Lookup queries in the sqlite index")
    pl.add_argument("--sqlite", required=True)
    pl.add_argument("--k", type=int, default=5)
    pl.add_argument("--drop-geo", action="store_true", default=False,
                    help="If set, normalize queries with drop_geo=True.")
    pl.add_argument("--two-pass", action="store_true", default=False,
                    help="If set, try (drop_geo as requested) then fallback to drop_geo=True if no candidates.")
    pl.add_argument("queries", nargs="+")

    args = p.parse_args()

    if args.cmd == "build":
        build_index(
            vernacular_tsv=args.vernacular_tsv,
            out_sqlite=args.out_sqlite,
            gbif_base=args.gbif_base,
            drop_geo=args.drop_geo,
            limit=args.limit,
            commit_every=args.commit_every,
            raw_max_len=args.raw_max_len,
        )
        return

    if args.cmd == "lookup":
        lookup(
            sqlite_path=args.sqlite,
            queries=args.queries,
            k=args.k,
            drop_geo=args.drop_geo,
            two_pass=args.two_pass,
        )
        return

    raise AssertionError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
