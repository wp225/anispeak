#!/usr/bin/env python3
# scripts/build_index.py
#
# RUN INSTRUCTIONS:
# To run this script for one-shot indexing, execute the following from your project root:
#   python -m scripts.build_index
#
# If you are integrating this into your development server routine, you can alternatively run:
#   python -m scripts.serve_dev
#
# Prerequisites:
#   IMPORTANT: Remove your old cache to force the new cleaning rules to apply!
#   rm -f data/species_cache.json
#   rm -rf cache/lancedb
#   docker compose up -d
#
# Description:
# Builds web_corpus.jsonl, extracts/cleans species, splits into scientific/vernacular,
# canonicalizes via GNparser, matches via GBIF, resolves vernaculars locally,
# embeds documents into LanceDB, and strictly singularizes/filters taxonomic entities.

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import lancedb
import pandas as pd
import pyarrow as pa
import requests
import yaml
from rapidfuzz.fuzz import token_set_ratio
from tqdm import tqdm

from anispeakarchive.corpus.loader import build_web_corpus, iter_jsonl
from anispeakarchive.index.dense import Embedder
from anispeakarchive.index.metadata import (
    attach_metadata_to_arrow_table,
    build_index_metadata,
    write_metadata_sidecar,
)


# -------------------------
# Hard assert helper
# -------------------------

def _assert(cond: bool, msg: str) -> None:
    assert cond, msg


# -------------------------
# YAML helpers
# -------------------------

def _get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, None)
            continue
        if hasattr(cur, part):
            cur = getattr(cur, part)
            continue
        try:
            cur = cur[part]  # type: ignore[index]
            continue
        except Exception:
            return default
    return default if cur is None else cur


def load_raw_yaml(path: str = "configs/default.yaml") -> Dict[str, Any]:
    p = Path(path)
    _assert(p.exists(), f"Config not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _assert(isinstance(data, dict), "configs/default.yaml must parse to a dict at top-level")
    return data


def cfg_species_from_yaml(y: Dict[str, Any]) -> Dict[str, Any]:
    sp = y.get("species_pipeline", {}) if isinstance(y.get("species_pipeline", {}), dict) else {}
    coarse = sp.get("coarse", {}) if isinstance(sp.get("coarse", {}), dict) else {}

    gbif_port = int(sp.get("gbif_port", 5643))
    gn_port = int(sp.get("gnparser_port", 5644))
    gbif_timeout = int(sp.get("gbif_timeout", 15))
    gn_timeout = int(sp.get("gnparser_timeout", 20))
    gbif_conc = int(sp.get("gbif_concurrency", 24))
    gn_batch = int(sp.get("gnparser_batch", 200))

    accept_matchtypes = tuple(str(x).upper() for x in (sp.get("gbif_accept_matchtypes", ["EXACT", "FUZZY", "HIGHERRANK"]) or []))
    accept_ranks = tuple(str(x).upper() for x in (sp.get("gbif_accept_ranks", ["SPECIES", "SUBSPECIES", "GENUS"]) or []))
    min_conf = int(sp.get("gbif_min_conf", 0))
    dedup_threshold = int(sp.get("dedup_threshold", 98))

    coarse_scientific_to_genus = bool(coarse.get("scientific_to_genus", True))
    coarse_common_collapse = bool(coarse.get("commonname_collapse", True))

    cache_file = str(sp.get("cache_file", "data/species_cache.json"))
    vernacular_sqlite = str(sp.get("vernacular_sqlite", "data/vernacular_index.sqlite"))

    gbif_port = int(os.environ.get("GBIF_PORT", str(gbif_port)))
    gn_port = int(os.environ.get("GNPARSER_PORT", str(gn_port)))

    return {
        "cache_file": os.environ.get("SPECIES_CACHE_FILE", cache_file),
        "vernacular_sqlite": os.environ.get("VERNACULAR_SQLITE", vernacular_sqlite),
        "gbif_port": gbif_port,
        "gn_port": gn_port,
        "gbif_timeout": gbif_timeout,
        "gn_timeout": gn_timeout,
        "gbif_conc": gbif_conc,
        "gn_batch": gn_batch,
        "gbif_accept_matchtypes": accept_matchtypes,
        "gbif_accept_ranks": accept_ranks,
        "gbif_min_conf": min_conf,
        "dedup_threshold": dedup_threshold,
        "coarse_scientific_to_genus": coarse_scientific_to_genus,
        "coarse_common_collapse": coarse_common_collapse,
    }


# -------------------------
# Cache I/O
# -------------------------

def load_species_cache(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        _assert(isinstance(j, dict), "species_cache.json must be a dict")
        return j
    return {}


def save_species_cache(path: str, cache: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True, ensure_ascii=False)


# -------------------------
# Best-effort docker compose up
# -------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_services_running() -> None:
    root = _repo_root()
    compose_yml = root / "docker-compose.yml"
    compose_yaml = root / "docker-compose.yaml"
    if not compose_yml.exists() and not compose_yaml.exists():
        return
    try:
        subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=str(root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


# -------------------------
# Service health + strict self-tests
# -------------------------

def gnparser_health(gn_port: int) -> bool:
    try:
        url = f"http://127.0.0.1:{gn_port}/api/v1/Bubo+bubo"
        return requests.get(url, timeout=2.5).status_code == 200
    except Exception:
        return False


def gbif_health(gbif_port: int) -> bool:
    try:
        url = f"http://127.0.0.1:{gbif_port}/v2/species/match?scientificName=Bubo%20bubo&verbose=true"
        r = requests.get(url, timeout=2.5)
        return r.status_code == 200
    except Exception:
        return False


def gbif_self_test(gbif_port: int) -> None:
    base = f"http://127.0.0.1:{gbif_port}/v2/species/match"

    r1 = requests.get(base, params={"scientificName": "Bubo bubo", "verbose": "true"}, timeout=5)
    _assert(r1.status_code == 200, f"GBIF match scientificName failed: {r1.status_code} {r1.text[:200]}")
    j1 = r1.json()
    _assert(isinstance(j1, dict), "GBIF match scientificName payload not dict")
    diag1 = j1.get("diagnostics")
    _assert(isinstance(diag1, dict), f"diagnostics missing/invalid: keys={list(j1.keys())[:20]}")
    note1 = str(diag1.get("note") or "").lower()
    _assert("no name given" not in note1, f"GBIF thinks no name given even with scientificName: note={diag1.get('note')}")
    usage1 = j1.get("usage")
    _assert(isinstance(usage1, dict) and usage1.get("key"), f"usage.key missing: keys={list(j1.keys())[:20]}")

    r2 = requests.get(base, params={"name": "Osprey", "verbose": "true"}, timeout=5)
    _assert(r2.status_code == 200, f"GBIF match name check failed: {r2.status_code} {r2.text[:200]}")
    j2 = r2.json()
    _assert(isinstance(j2, dict), "GBIF match name payload not dict")
    diag2 = j2.get("diagnostics")
    _assert(isinstance(diag2, dict), "GBIF match name diagnostics missing")
    note2 = str(diag2.get("note") or "").lower()
    _assert("no name given" in note2, "Expected 'name' param to be ignored; service behavior changed")

    key = str(usage1.get("key"))
    r3 = requests.get(f"http://127.0.0.1:{gbif_port}/v2/species/{urllib.parse.quote(key)}", timeout=5)
    _assert(r3.status_code == 404, f"Expected /v2/species/{{key}} to be 404 in this service, got {r3.status_code}")


# -------------------------
# Species cleaning + heuristics
# -------------------------

# Extracts exactly genus species (and optionally subspecies) components
_STRICT_BINOMIAL_RE = re.compile(r"^([A-Z][a-z\-]+)\s+([a-z][a-z\-]+)(?:\s+[a-z][a-z\-]+)?$")
_BINOMIAL_RE = re.compile(r"\b([A-Z][a-z\-]+)\s+([a-z][a-z\-]+)\b")

_STRAINISH_RE = re.compile(
    r"(c57bl/6|balb/c|129[sx]?\d*|fvb|nude|scid|rag1|rag2|cd-?1|b6|f1|dba/2|sv129|"
    r"btbr|sprague[- ]dawley|wistar|"
    r"\bstrain\b|\bline\b|\bknockout\b|\btransgenic\b|\bmutant\b)",
    re.IGNORECASE,
)

DESCRIPTORS_RE = re.compile(
    r"\b(adults?|juveniles?|females?|males?|young|infants?|newborns?|normal|wild-?type|"
    r"live|strains?|day-?old|captive|domestic|feral|laboratory|lab|calling|endangered|at-?risk|"
    r"egg-?laying|sample|specimens?|hybrid|hybrids|complex|group|sensu lato)\b",
    re.IGNORECASE,
)

_TAXONOMIC_FILLER_RE = re.compile(
    r"\b(sp\.?\s*nov\.?|n\.?\s*sp\.?|sp\.?|spp\.?|cf\.?|aff\.?|var\.?)\b",
    re.IGNORECASE
)

_COUNT_FILLER_RE = re.compile(
    r"^\d+\s+(species|ecologically|different|other|European|Neotropical|Swiss|African|Asian)\b",
    re.IGNORECASE
)

GENERIC_DROP = {
    "species", "animal", "animals", "insect", "insects",
    "mammal", "mammals", "cetacean", "cetaceans",
    "actinopterygii", "vertebrate", "vertebrates",
    "organism", "organisms", "wildlife", "fauna"
}


def _is_asciiish(s: str) -> bool:
    _assert(isinstance(s, str), "Input must be a string.")
    return not bool(re.search(r"[^\x00-\x7F\u0080-\u00FF\u2018\u2019\u201C\u201D]", s))


def looks_strainish(s: str) -> bool:
    _assert(isinstance(s, str), "Input must be a string.")
    return bool(_STRAINISH_RE.search(s or ""))


def is_strict_scientific(s: str) -> bool:
    if not s:
        return False
    return bool(_STRICT_BINOMIAL_RE.fullmatch(s.strip()))


def to_scientific_case(s: str) -> str:
    _assert(isinstance(s, str) or s is None, "Input must be a string or None.")
    s = (s or "").strip()
    parts = s.split()
    if len(parts) >= 2:
        g = parts[0]
        parts[0] = (g[:1].upper() + g[1:].lower()) if g else g
        parts[1] = parts[1].lower()
        return " ".join(parts)
    if parts:
        p = parts[0]
        if re.fullmatch(r"[A-Za-z][A-Za-z\-]+", p):
            return p[:1].upper() + p[1:].lower()
    return s


def smart_clean_mention(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()

    if s.isdigit():
        return ""

    if _COUNT_FILLER_RE.search(s):
        return ""

    if not s or not _is_asciiish(s):
        return ""

    s = re.sub(r"\([^)]*\)", " ", s)
    s = DESCRIPTORS_RE.sub(" ", s)
    s = _TAXONOMIC_FILLER_RE.sub(" ", s)

    s = re.sub(r"^[-\s]+", "", s)
    s = re.sub(r"^(a|and)\s+", "", s, flags=re.IGNORECASE)

    s = re.sub(r"['\u2018\u2019]", "", s)
    s = re.sub(r"[^A-Za-z0-9\s\-\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return ""

    if len(s) > 100 or len(s.split()) > 6:
        return ""

    if s.lower() in GENERIC_DROP:
        return ""

    return s


def normalize_for_dedupe(s: str) -> str:
    _assert(isinstance(s, str), "Input must be a string")
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = []
    for t in s.split():
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        toks.append(t)
    return " ".join(toks)


def genus_from_scientific(s: str) -> Optional[str]:
    s = (s or "").strip()
    m = _BINOMIAL_RE.search(s)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Z][a-z\-]{2,}", s):
        return s
    return None


IRREGULAR_PLURALS = {
    "mice": "mouse",
    "geese": "goose",
    "calves": "calf",
    "wolves": "wolf",
    "men": "man",
    "women": "woman",
    "teeth": "tooth",
    "feet": "foot"
}

def singularize_token(tok: str) -> str:
    _assert(tok and isinstance(tok, str), "singularize_token expects non-empty str")
    tok_lower = tok.lower()

    if tok_lower in IRREGULAR_PLURALS:
        res = IRREGULAR_PLURALS[tok_lower]
        return res.capitalize() if tok.istitle() else res

    if tok.endswith("ies") and len(tok) > 3:
        return tok[:-3] + "y"
    if tok.endswith("ves") and len(tok) > 3:
        return tok[:-3] + "f"
    if tok.endswith("ses") and len(tok) > 3:
        return tok[:-2]
    if tok.endswith("ches") and len(tok) > 4:
        return tok[:-2]
    if tok.endswith("shes") and len(tok) > 4:
        return tok[:-2]
    if tok.endswith("oes") and len(tok) > 3:
        return tok[:-2]
    if tok.endswith("s") and len(tok) > 1 and not tok.endswith("ss") and not tok.endswith("us") and not tok.endswith("is"):
        return tok[:-1]
    return tok


def coarse_common_head(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    t = re.sub(r"\([^)]*\)", " ", s)
    t = re.sub(r"[^A-Za-z\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    toks = [x for x in t.split() if x]
    if not toks:
        return None
    head = toks[-1].lower()
    if len(head) <= 1:
        return None
    return singularize_token(head)


def to_coarse_label(s: str, scientific_to_genus: bool, common_collapse: bool, is_scientific: bool) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None

    # Strictly respect the source of resolution to avoid mangling common names
    if is_scientific and scientific_to_genus:
        g = genus_from_scientific(s)
        if g:
            return g
    elif common_collapse:
        head = coarse_common_head(s)
        if head:
            return head
    return s


# -------------------------
# GNparser (POST then GET fallback)
# -------------------------

def gnparser_parse_batch(names: List[str], gn_port: int, timeout: int, session: requests.Session) -> List[Dict[str, Any]]:
    base = f"http://127.0.0.1:{gn_port}/api/v1"
    try:
        r = session.post(base, json=names, timeout=timeout)
        if r.status_code == 405:
            raise requests.HTTPError("405", response=r)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) == len(names):
            return data
        return [{} for _ in names]
    except Exception:
        pass

    joined = "|".join(urllib.parse.quote_plus(n) for n in names)
    url = f"{base}/{joined}"
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and len(data) == len(names):
        return data
    return [{} for _ in names]


def extract_gnparser_canonical(parsed: Dict[str, Any]) -> Optional[str]:
    def _validate_gn(val: Optional[str]) -> Optional[str]:
        if not val or not isinstance(val, str):
            return None
        val = val.strip()
        # Reject single-word results that are English stopwords
        _GN_REJECT = re.compile(r"^(trick|treat|and|the|a|an|of|in|for|with|by)$", re.IGNORECASE)
        if _GN_REJECT.fullmatch(val):
            return None
        return val

    if not isinstance(parsed, dict):
        return None
    for k in ("canonicalSimple", "CanonicalSimple", "canonicalName", "CanonicalName", "canonicalFull", "CanonicalFull"):
        v = parsed.get(k)
        if isinstance(v, str) and v.strip():
            return _validate_gn(v)
    for ck in ("canonical", "Canonical"):
        c = parsed.get(ck)
        if isinstance(c, dict):
            for sk in ("simple", "Simple", "full", "Full"):
                v = c.get(sk)
                if isinstance(v, str) and v.strip():
                    return _validate_gn(v)
    for outer in ("parsed", "Parsed", "name", "Name"):
        o = parsed.get(outer)
        if isinstance(o, dict):
            got = extract_gnparser_canonical(o)
            if got:
                return _validate_gn(got)
    for k in ("normalized", "Normalized", "scientificName", "ScientificName"):
        v = parsed.get(k)
        if isinstance(v, str) and v.strip():
            return _validate_gn(v)
    return None


def gnparser_canonicalize(names: List[str], gn_port: int, timeout: int, batch: int) -> Dict[str, Optional[str]]:
    if not names:
        return {}
    if not gnparser_health(gn_port):
        return {n: None for n in names}

    out: Dict[str, Optional[str]] = {}
    batches = [names[i:i + batch] for i in range(0, len(names), batch)]
    with requests.Session() as sess:
        for b in tqdm(batches, desc="GNparser"):
            data = gnparser_parse_batch(b, gn_port=gn_port, timeout=timeout, session=sess)
            for raw, parsed in zip(b, data):
                canon = extract_gnparser_canonical(parsed) if isinstance(parsed, dict) else None
                out[raw] = canon
    return out


# -------------------------
# GBIF matching-ws (scientificName only)
# -------------------------

def gbif_match_one(scientific_name: str, gbif_port: int, timeout: int, session: requests.Session) -> Dict[str, Any]:
    _assert(isinstance(scientific_name, str) and bool(scientific_name.strip()), "gbif_match_one expects non-empty scientific_name")
    url = f"http://127.0.0.1:{gbif_port}/v2/species/match"
    r = session.get(url, params={"scientificName": scientific_name, "verbose": "true"}, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    _assert(isinstance(j, dict), f"GBIF match payload not dict: {type(j)}")
    return j


def gbif_extract_usage(j: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    u = j.get("usage")
    return u if isinstance(u, dict) else None


def gbif_extract_key(j: Dict[str, Any]) -> Optional[str]:
    u = gbif_extract_usage(j)
    if not u:
        return None
    k = u.get("key")
    if isinstance(k, str) and k.strip():
        return k.strip()
    return None


def gbif_extract_rank(j: Dict[str, Any]) -> Optional[str]:
    u = gbif_extract_usage(j)
    if not u:
        return None
    r = u.get("rank")
    return str(r) if r is not None else None


def gbif_extract_canonical(j: Dict[str, Any]) -> Optional[str]:
    u = gbif_extract_usage(j)
    if not u:
        return None
    for k in ("canonicalName", "name"):
        v = u.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def gbif_extract_diag(j: Dict[str, Any]) -> Dict[str, Any]:
    d = j.get("diagnostics")
    return d if isinstance(d, dict) else {}


def gbif_extract_matchtype(j: Dict[str, Any]) -> Optional[str]:
    d = gbif_extract_diag(j)
    mt = d.get("matchType")
    return str(mt) if mt is not None else None


def gbif_extract_confidence(j: Dict[str, Any]) -> Optional[int]:
    d = gbif_extract_diag(j)
    c = d.get("confidence")
    if isinstance(c, (int, float)):
        return int(c)
    return None


def gbif_accept(j: Dict[str, Any], accept_matchtypes: Tuple[str, ...], accept_ranks: Tuple[str, ...], min_conf: int) -> bool:
    mt = gbif_extract_matchtype(j)
    rk = gbif_extract_rank(j)
    conf = gbif_extract_confidence(j)

    if mt is None:
        return False
    mtU = mt.upper()
    rkU = (rk or "").upper()

    if accept_matchtypes and mtU not in accept_matchtypes:
        return False
    if accept_ranks and rkU and rkU not in accept_ranks:
        return False
    if min_conf > 0 and conf is not None and conf < min_conf:
        return False

    return gbif_extract_key(j) is not None


def gbif_parallel_scientific(
    sci_names: List[str],
    gbif_port: int,
    timeout: int,
    concurrency: int,
    accept_matchtypes: Tuple[str, ...],
    accept_ranks: Tuple[str, ...],
    min_conf: int,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    def worker(nm: str) -> Tuple[str, Dict[str, Any]]:
        with requests.Session() as sess:
            try:
                j = gbif_match_one(nm, gbif_port=gbif_port, timeout=timeout, session=sess)
                if gbif_accept(j, accept_matchtypes, accept_ranks, min_conf):
                    return nm, {
                        "canonical": gbif_extract_canonical(j) or nm,
                        "usageKey": gbif_extract_key(j),
                        "rank": gbif_extract_rank(j),
                        "confidence": gbif_extract_confidence(j),
                        "matchType": gbif_extract_matchtype(j),
                        "source": "gbif-local",
                    }
                return nm, {
                    "canonical": nm,
                    "usageKey": None,
                    "rank": gbif_extract_rank(j),
                    "confidence": gbif_extract_confidence(j),
                    "matchType": f"REJECTED_{(gbif_extract_matchtype(j) or 'NONE').upper()}",
                    "source": "gbif-local",
                }
            except Exception:
                return nm, {
                    "canonical": nm,
                    "usageKey": None,
                    "rank": None,
                    "confidence": None,
                    "matchType": "FAILED",
                    "source": "gbif-local",
                }

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = [ex.submit(worker, n) for n in sci_names]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="GBIF match"):
            nm, entry = fut.result()
            out[nm] = entry
    return out


# -------------------------
# Vernacular sqlite resolver (no GBIF validation)
# -------------------------

STOPWORDS = {
    "male", "female", "juvenile", "adult", "adults", "immature",
    "pup", "pups", "calf", "calves", "hatchling", "hatchlings",
    "larva", "larvae", "nymph", "nymphs",
    "nestling", "nestlings", "chick", "chicks",
    "young", "offspring",
    "foraging", "calling", "vocal", "roaring", "singing",
    "species", "sp", "spp", "taxa", "taxon",
    "wild", "feral", "domestic",
}

GEO_STOPWORDS = {
    "eastern", "western", "northern", "southern",
    "pacific", "atlantic", "arctic", "antarctic",
    "eurasian", "african", "asian", "american", "european",
}

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

def normalize_vernacular(s: str, drop_geo: bool) -> str:
    _assert(isinstance(s, str), f"normalize_vernacular expects str, got {type(s)}")
    s0 = s.strip().lower()
    if not s0:
        return ""
    s0 = s0.replace("-", " ").replace("–", " ").replace("—", " ")
    toks = _WORD_RE.findall(s0)
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
    return " ".join(out).strip()


@dataclass
class VernacularResolver:
    sqlite_path: str
    drop_geo: bool = False
    two_pass: bool = True
    k: int = 5

    def _connect(self) -> sqlite3.Connection:
        _assert(os.path.exists(self.sqlite_path), f"vernacular sqlite missing: {self.sqlite_path}")
        conn = sqlite3.connect(self.sqlite_path)
        return conn

    def lookup(self, conn: sqlite3.Connection, query: str, k: int) -> List[Tuple[str, str, Optional[str], int]]:
        norm = normalize_vernacular(query, drop_geo=self.drop_geo)
        if not norm:
            return []
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
        out: List[Tuple[str, str, Optional[str], int]] = []
        for usage_key, raw, lang, score in rows:
            out.append((str(usage_key), str(raw), (str(lang) if lang is not None else None), int(score)))
        return out

    def resolve_many(self, queries: List[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        with self._connect() as conn:
            for q in tqdm(queries, desc="Vernacular resolve"):
                q0 = q
                best: Optional[Tuple[str, str, Optional[str], int]] = None

                rows = self.lookup(conn, q0, self.k)
                if rows:
                    best = rows[0]

                if best is None and self.two_pass:
                    norm = normalize_vernacular(q0, drop_geo=True)
                    if norm:
                        old = self.drop_geo
                        self.drop_geo = True
                        rows2 = self.lookup(conn, q0, self.k)
                        self.drop_geo = old
                        if rows2:
                            best = rows2[0]

                if best is None:
                    out[q0] = {
                        "canonical": q0,
                        "usageKey": None,
                        "rank": None,
                        "confidence": None,
                        "matchType": "VERNACULAR_NOT_FOUND",
                        "source": "vernacular-sqlite",
                    }
                    continue

                usage_key, raw, lang, score = best
                out[q0] = {
                    "canonical": q0,
                    "usageKey": usage_key,
                    "rank": None,
                    "confidence": None,
                    "matchType": "VERNACULAR_SQLITE",
                    "source": "vernacular-sqlite",
                    "raw": raw,
                    "lang": lang,
                    "score": score,
                }
        return out


# -------------------------
# Dedupe for unresolved only
# -------------------------

def lexical_dedupe_unresolved(all_mentions: List[str], cache: Dict[str, Any], dedup_threshold: int) -> None:
    candidates = [m for m in all_mentions if cache.get(m, {}).get("usageKey") is None]
    norms = [(m, normalize_for_dedupe(m)) for m in candidates]
    rep: Dict[str, str] = {}

    for i, (a_raw, a_norm) in enumerate(norms):
        if not a_norm or a_norm in rep:
            continue
        rep[a_norm] = a_raw
        for j in range(i + 1, len(norms)):
            b_raw, b_norm = norms[j]
            if not b_norm:
                continue
            if token_set_ratio(a_norm, b_norm) >= dedup_threshold:
                rep[b_norm] = a_raw

    for m in candidates:
        nm = normalize_for_dedupe(m)
        canon = rep.get(nm, m)
        cache[m]["canonical"] = canon

        # Inherit usageKey from the representative if it exists
        canon_entry = cache.get(canon)
        if canon_entry and canon_entry.get("usageKey") is not None:
            _assert(isinstance(canon_entry["usageKey"], str), "usageKey being inherited must be a string.")
            cache[m]["usageKey"] = canon_entry["usageKey"]
            cache[m]["rank"] = canon_entry.get("rank")
            cache[m]["matchType"] = "DEDUP_INHERITED"
            cache[m]["source"] = canon_entry.get("source", "dedup")
        else:
            mt = str(cache[m].get("matchType") or "")
            if mt in {"NEW", "FAILED"} or mt.startswith("REJECTED_") or mt == "VERNACULAR_NOT_FOUND":
                cache[m]["matchType"] = "DEDUP_ONLY"
                cache[m]["source"] = "dedup"


# -------------------------
# Corpus string extraction
# -------------------------

def _extract_strings(x: Any, dict_keys: Optional[Sequence[str]] = None, split_comma: bool = False) -> List[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    out: List[str] = []
    if isinstance(x, str):
        if split_comma and "," in x:
            out.extend([s.strip() for s in x.split(",") if s.strip()])
        else:
            s = x.strip()
            if s:
                out.append(s)
    elif isinstance(x, dict):
        if dict_keys:
            for k in dict_keys:
                if k in x:
                    out.extend(_extract_strings(x[k], dict_keys, split_comma))
                    break
        else:
            for v in x.values():
                out.extend(_extract_strings(v, dict_keys, split_comma))
    elif isinstance(x, (list, tuple, set)):
        for it in x:
            out.extend(_extract_strings(it, dict_keys, split_comma))
    return out


# -------------------------
# Main species cache build (NO LLM)
# -------------------------

def build_species_cache_from_df(df: pd.DataFrame, raw_yaml: Dict[str, Any]) -> Dict[str, Any]:
    cfg = cfg_species_from_yaml(raw_yaml)
    cache_path = cfg["cache_file"]
    cache = load_species_cache(cache_path)

    sp_series = df["species"] if "species" in df.columns else pd.Series([None] * len(df))
    raw_fields: List[str] = []
    for val in sp_series:
        for s in _extract_strings(val, dict_keys=["name", "species"], split_comma=True):
            if s:
                raw_fields.append(s)

    cleaned: List[str] = []
    for r in tqdm(raw_fields, desc="Species clean"):
        cm = smart_clean_mention(r)
        if cm:
            cleaned.append(cm)

    seen: Set[str] = set()
    uniq: List[str] = []
    for s in cleaned:
        if s not in seen:
            seen.add(s)
            uniq.append(s)

    print(f"[INFO] species raw fields: {len(raw_fields):,} | cleaned unique mentions: {len(uniq):,}")

    for m in uniq:
        if m not in cache:
            cache[m] = {
                "canonical": m,
                "usageKey": None,
                "rank": None,
                "confidence": None,
                "matchType": "NEW",
                "source": "raw",
                "gn_canonical": None,
            }

    # Split candidates out based on strict scientific structure
    candidates = [m for m in uniq if not looks_strainish(m)]
    sci_like = [m for m in candidates if is_strict_scientific(m)]
    vern_like = [m for m in candidates if m not in sci_like]

    print(f"[INFO] strict scientific-like: {len(sci_like):,} | vernacular/other: {len(vern_like):,}")

    if sci_like and gnparser_health(cfg["gn_port"]):
        sci_inputs = [to_scientific_case(m) for m in sci_like]
        gn_map = gnparser_canonicalize(
            sci_inputs,
            gn_port=cfg["gn_port"],
            timeout=cfg["gn_timeout"],
            batch=cfg["gn_batch"],
        )
        for orig, key in zip(sci_like, sci_inputs):
            cache[orig]["gn_canonical"] = gn_map.get(key)
            if cache[orig]["gn_canonical"]:
                cache[orig]["source"] = "gnparser"
    else:
        if sci_like:
            print("[WARN] GNparser not healthy; skipping GNparser canonicalization")

    if sci_like and gbif_health(cfg["gbif_port"]):
        gbif_inputs: List[str] = []
        input_to_orig: Dict[str, List[str]] = {}
        for orig in sci_like:
            gn = cache[orig].get("gn_canonical")
            inp = to_scientific_case(gn) if isinstance(gn, str) and gn.strip() else to_scientific_case(orig)
            gbif_inputs.append(inp)
            input_to_orig.setdefault(inp, []).append(orig)

        gbif_self_test(cfg["gbif_port"])

        gbif_res = gbif_parallel_scientific(
            gbif_inputs,
            gbif_port=cfg["gbif_port"],
            timeout=cfg["gbif_timeout"],
            concurrency=cfg["gbif_conc"],
            accept_matchtypes=cfg["gbif_accept_matchtypes"],
            accept_ranks=cfg["gbif_accept_ranks"],
            min_conf=cfg["gbif_min_conf"],
        )

        for inp, entry in gbif_res.items():
            for orig in input_to_orig.get(inp, []):
                if entry.get("usageKey") is not None:
                    cache[orig].update(entry)
                else:
                    cache[orig]["matchType"] = entry.get("matchType")
                    if str(entry.get("matchType", "")).startswith("REJECTED_"):
                        cache[orig]["rank"] = None
                    else:
                        cache[orig]["rank"] = entry.get("rank")

                    cache[orig]["confidence"] = entry.get("confidence")
                    cache[orig]["source"] = cache[orig].get("source") or "gbif-local"
    else:
        if sci_like:
            print("[WARN] GBIF not healthy; skipping GBIF match")

    # Important Fix: Give anything that failed GBIF a second chance through the vernacular resolver!
    vernacular_candidates = vern_like + [m for m in sci_like if cache.get(m, {}).get("usageKey") is None]

    _assert(os.path.exists(cfg["vernacular_sqlite"]), f"Missing vernacular sqlite: {cfg['vernacular_sqlite']}")
    resolver = VernacularResolver(
        sqlite_path=cfg["vernacular_sqlite"],
        drop_geo=bool(_get(raw_yaml, "species_pipeline.vernacular_drop_geo", False)),
        two_pass=bool(_get(raw_yaml, "species_pipeline.vernacular_two_pass", True)),
        k=int(_get(raw_yaml, "species_pipeline.vernacular_k", 5)),
    )

    vern_res = resolver.resolve_many(vernacular_candidates)
    for orig, entry in vern_res.items():
        if entry.get("usageKey") is not None:
            cache[orig].update(entry)

    lexical_dedupe_unresolved(uniq, cache, dedup_threshold=cfg["dedup_threshold"])
    save_species_cache(cache_path, cache)

    total = len(cache)
    with_key = sum(1 for _, v in cache.items() if v.get("usageKey") is not None)
    vern_count = sum(1 for _, v in cache.items() if v.get("source") == "vernacular-sqlite" and v.get("usageKey") is not None)

    print(f"[INFO] Species cache updated and saved.")
    print(f"[INFO] cache_entries={total} with_usageKey={with_key} ({(with_key/max(1,total)):.1%})")
    print(f"[INFO] vernacular-sqlite with usageKey={vern_count}")

    return cache


# -------------------------
# Build per-doc species_raw + species (coarse)
# -------------------------

def build_species_fields_per_doc(
    df: pd.DataFrame,
    species_cache: Dict[str, Any],
    scientific_to_genus: bool,
    common_collapse: bool,
) -> pd.DataFrame:
    sp_series = df["species"] if "species" in df.columns else pd.Series([None] * len(df))

    species_raw_col: List[List[str]] = []
    species_coarse_col: List[List[str]] = []

    print("[INFO] Building species_raw (fine) and species (coarse) per document")

    for val in tqdm(sp_series, total=len(df), desc="Species fields"):
        fine_seen: Set[str] = set()
        fine_list: List[str] = []
        coarse_seen: Set[str] = set()
        coarse_list: List[str] = []

        raws = _extract_strings(val, dict_keys=["name", "species"], split_comma=True)
        for raw in raws:
            raw = (raw or "").strip()
            if not raw:
                continue
            cm = smart_clean_mention(raw)
            if not cm:
                continue

            entry = species_cache.get(cm)
            fine = (entry.get("canonical") if isinstance(entry, dict) else None) or cm

            if fine not in fine_seen:
                fine_seen.add(fine)
                fine_list.append(fine)

            # Determine whether it's safe to use scientific rules based strictly on resolution history
            is_sci = False
            if isinstance(entry, dict) and entry.get("usageKey") is not None:
                is_sci = entry.get("source") in ("gbif-local", "gnparser")

            coarse = to_coarse_label(
                fine,
                scientific_to_genus=scientific_to_genus,
                common_collapse=common_collapse,
                is_scientific=is_sci
            )

            if coarse and coarse not in coarse_seen:
                coarse_seen.add(coarse)
                coarse_list.append(coarse)

        if not fine_list:
            fine_list.append("Others")
        if not coarse_list:
            coarse_list.append("Others")

        species_raw_col.append(fine_list)
        species_coarse_col.append(coarse_list)

    out = df.copy()
    out["species_raw"] = species_raw_col
    out["species"] = species_coarse_col
    return out


# -------------------------
# Main indexing pipeline
# -------------------------

def main() -> None:
    from anispeakarchive.core.config import load_settings
    settings = load_settings("configs/default.yaml")
    raw_yaml = load_raw_yaml("configs/default.yaml")
    cfg = cfg_species_from_yaml(raw_yaml)

    print("[DEBUG] Using config:", os.path.abspath("configs/default.yaml"))

    ensure_services_running()

    n = build_web_corpus(
        final_relevant_jsonl=settings.paths.final_relevant_jsonl,
        web_corpus_jsonl=settings.paths.web_corpus_jsonl,
        include_abstract=settings.index.include_abstract,
        include_affiliations=settings.index.include_affiliations,
        max_docs=settings.index.max_docs,
    )
    _assert(n >= 0, "build_web_corpus returned < 0")
    print(f"[OK] web_corpus built: {n} docs -> {settings.paths.web_corpus_jsonl}")

    rows = list(iter_jsonl(settings.paths.web_corpus_jsonl))
    df = pd.DataFrame(rows)
    _assert(len(df) > 0, "No docs loaded")

    _assert(os.path.exists(cfg["vernacular_sqlite"]), f"Missing vernacular sqlite: {cfg['vernacular_sqlite']}")
    species_cache = build_species_cache_from_df(df, raw_yaml)

    df = build_species_fields_per_doc(
        df,
        species_cache=species_cache,
        scientific_to_genus=cfg["coarse_scientific_to_genus"],
        common_collapse=cfg["coarse_common_collapse"],
    )

    print("[INFO] Normalizing non-species list columns")
    list_cols = [
        "authors", "institutions",
        "species_group", "stages", "features", "modalities", "domains", "contexts",
    ]
    dict_keys_by_col = {
        "species_group": (["category", "group"], True),
        "stages":        (["stage"], True),
        "features":      (["feature"], True),
        "modalities":    (["modality"], True),
        "domains":       (["domain"], True),
        "contexts":      (["context"], True),
        "institutions":  (["display_name", "name"], False),
        "authors":       (["display_name", "name"], False),
    }

    for c in tqdm(list_cols, desc="Processing Columns"):
        if c not in df.columns:
            df[c] = [[] for _ in range(len(df))]

        col_keys, col_split = dict_keys_by_col.get(c, (None, False))

        def process_cell(val, _keys=col_keys, _split=col_split):
            seen: Set[str] = set()
            uniq: List[str] = []
            for v in _extract_strings(val, dict_keys=_keys, split_comma=_split):
                final_val = v.strip() if isinstance(v, str) else str(v)
                if final_val and final_val not in seen:
                    seen.add(final_val)
                    uniq.append(final_val)
            return uniq

        df[c] = df[c].map(process_cell)

    if "year" in df.columns:
        def _norm_year(y):
            if y is None or (isinstance(y, float) and pd.isna(y)):
                return None
            try:
                return int(y)
            except Exception:
                return None
        df["year"] = df["year"].map(_norm_year)

    print("[INFO] Initializing Dense Embedder")
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

    print("[INFO] Preparing retrieval texts for LanceDB Search")
    texts = df["text"].astype(str).tolist()

    print("[INFO] Generating Document Vectors for LanceDB Search")
    vecs = embedder.encode_texts(
        texts,
        show_progress=True,
        progress_desc="Embedding documents",
    )
    assert vecs.shape[0] == len(texts), (
        f"Embedding row count mismatch during indexing: expected {len(texts)}, got {vecs.shape[0]}"
    )
    assert embedder.embedding_dim is not None and int(embedder.embedding_dim) > 0, (
        "Embedding backend did not expose a valid embedding dimension during indexing"
    )
    df["vector"] = list(vecs)

    ldb_dir = Path(settings.paths.lancedb_dir)
    ldb_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(ldb_dir))

    table_name = settings.index.table_name
    try:
        existing = set(db.list_tables())
    except Exception:
        existing = set(db.table_names())

    if table_name in existing and settings.index.fts_replace:
        db.drop_table(table_name)
        print(f"[INFO] Dropped existing table: {table_name}")

    index_meta = build_index_metadata(
        table_name=table_name,
        model_name=embedder.model_name,
        provider=embedder.provider,
        backend_name=embedder.backend_name,
        embedding_dim=int(embedder.embedding_dim),
        embedding_fingerprint=embedder.fingerprint(),
    )
    arrow_table = pa.Table.from_pandas(df, preserve_index=False)
    arrow_table = attach_metadata_to_arrow_table(arrow_table, index_meta)

    print("[INFO] Writing to LanceDB")
    tbl = db.create_table(table_name, data=arrow_table, mode="overwrite")
    print(f"[OK] Lance table created: {table_name} rows={tbl.count_rows()}")
    write_metadata_sidecar(str(ldb_dir), table_name, index_meta)

    fts_cols = _get(settings, "index.fts_columns", ["text"]) or ["text"]
    for col in fts_cols:
        print(f"[INFO] Creating Full Text Search index on '{col}'")
        tbl.create_fts_index(col, replace=True)

    print("[DONE] Indexing Complete")


if __name__ == "__main__":
    main()
