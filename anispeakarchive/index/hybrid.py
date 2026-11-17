from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import math
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import lancedb

from anispeakarchive.index.dense import Embedder
from anispeakarchive.index.metadata import read_metadata_sidecar, read_table_metadata
from anispeakarchive.index.rerank import Reranker
from anispeakarchive.core.text_clean import strip_html

import datetime as _dt


LOGGER = logging.getLogger(__name__)
# --------------------------
# Helpers
# --------------------------

def _sql_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _where_and(clauses: List[str]) -> Optional[str]:
    clauses = [c.strip() for c in clauses if c and c.strip()]
    return " AND ".join(clauses) if clauses else None


def _rrf_scores(ranked_lists: List[List[str]], k: int) -> Dict[str, float]:
    assert isinstance(k, int) and k > 0
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst, start=1):
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def _coalesce_rows(
    lex_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    scores: Dict[str, float],
    max_out: int,
) -> pd.DataFrame:
    assert max_out > 0

    df = pd.concat([lex_df, dense_df], ignore_index=True)

    if "doc_id" not in df.columns:
        raise RuntimeError("Index missing doc_id column; rebuild index")

    df["doc_id"] = df["doc_id"].astype(str)
    df = df.drop_duplicates(subset=["doc_id"], keep="first")

    df["__rrf_score"] = df["doc_id"].map(lambda d: float(scores.get(d, 0.0)))
    df = df.sort_values("__rrf_score", ascending=False)

    # Ensure we never carry vectors forward
    if "vector" in df.columns:
        df = df.drop(columns=["vector"])

    return df.head(max_out).reset_index(drop=True)


def _json_safe(v: Any) -> Any:
    """Convert common non-JSON types (numpy/pandas/datetime) to JSON-safe values.

    Policy:
      - np scalars -> python scalars
      - np arrays -> lists
      - NaN/Inf -> None
      - pandas Timestamp/NaT and datetime/date -> ISO strings or None
    """
    # pandas NaT check (without importing pandas types at module import time)
    try:
        import pandas as _pd  # local import to avoid hard dependency at import time
        if v is _pd.NaT:
            return None
        if isinstance(v, _pd.Timestamp):
            # Timestamp can be tz-aware; isoformat handles it
            if _pd.isna(v):
                return None
            return v.isoformat()
    except Exception:
        pass

    if isinstance(v, (_dt.datetime, _dt.date)):
        return v.isoformat()

    if isinstance(v, np.ndarray):
        return [ _json_safe(x) for x in v.tolist() ]

    if isinstance(v, (np.floating, np.integer)):
        v = v.item()

    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)

    # basic JSON-safe primitives pass through
    if v is None or isinstance(v, (bool, int, str)):
        return v

    # last resort: stringify unknown objects rather than crashing the API
    return str(v)


# --------------------------
# Filters
# --------------------------

@dataclass
class SearchFilters:
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    species: Optional[List[str]] = None
    modalities: Optional[List[str]] = None
    features: Optional[List[str]] = None
    stages: Optional[List[str]] = None
    domains: Optional[List[str]] = None
    contexts: Optional[List[str]] = None
    institutions: Optional[List[str]] = None
    authors: Optional[List[str]] = None
    venues: Optional[List[str]] = None
    paper_type: Optional[str] = None
    tradition: Optional[str] = None
    has_code: Optional[bool] = None
    has_dataset: Optional[bool] = None
    has_benchmark: Optional[bool] = None


# --------------------------
# Search service
# --------------------------

class SearchService:
    def __init__(
        self,
        lancedb_dir: str,
        table_name: str,
        embedder: Embedder,
        reranker: Optional[Reranker],
        cfg: Dict[str, Any],
    ) -> None:
        assert isinstance(lancedb_dir, str) and lancedb_dir
        assert isinstance(table_name, str) and table_name

        self._db = lancedb.connect(lancedb_dir)
        self._tbl = self._db.open_table(table_name)
        self._embedder = embedder
        self._reranker = reranker
        self._cfg = cfg
        self._table_name = table_name
        self._lancedb_dir = lancedb_dir
        self._index_metadata = self._load_index_metadata()
        self._validate_index_metadata()

        # Simple embedding micro-cache (helps UI that replays same query)
        self._last_q: Optional[str] = None
        self._last_qv: Optional[np.ndarray] = None

        ui_cfg = cfg.get("ui", {}) if isinstance(cfg, dict) else {}
        self._max_facet_values = int(ui_cfg.get("max_facet_values", 400))
        self._facet_cache: Dict[str, List[str]] = {}
        self._facet_cache_time: Dict[str, float] = {}
        self._facet_ttl_sec = float(ui_cfg.get("facet_ttl_sec", 300.0))

        self._analytics_cache: Optional[Dict[str, Any]] = None
        self._analytics_cache_time = 0.0
        self._analytics_ttl_sec = float(ui_cfg.get("analytics_ttl_sec", 300.0))

    def prime_small_facet_caches(self) -> None:
        fields = ["domains", "modalities", "features", "stages", "contexts"]
        LOGGER.info("Priming small facet caches for fields=%s", fields)
        t0 = time.time()
        for field in fields:
            ft = time.time()
            try:
                values = self.facet_values(field, limit=2000)
                LOGGER.info(
                    "Primed facet cache field=%s values=%s latency_ms=%s",
                    field,
                    len(values),
                    int((time.time() - ft) * 1000),
                )
            except Exception:
                LOGGER.exception("Facet cache prime failed for field=%s", field)
        LOGGER.info("Finished priming small facet caches total_ms=%s", int((time.time() - t0) * 1000))

    def _load_index_metadata(self) -> Dict[str, Any]:
        meta = read_table_metadata(self._tbl)
        if meta is None:
            meta = read_metadata_sidecar(self._lancedb_dir, self._table_name)
        if meta is None:
            LOGGER.warning(
                "Index metadata is missing for table %r. "
                "Skipping model/dimension validation. Rebuild the index to enable checks.",
                self._table_name,
            )
            return {}
        return meta

    def _validate_index_metadata(self) -> None:
        meta = self._index_metadata
        if not meta:
            return  # no metadata — old index, skip validation

        actual_dim = int(meta.get("embedding_dim", 0))
        actual_model = str(meta.get("model_name", "")).strip()
        actual_fingerprint = str(meta.get("embedding_fingerprint", "")).strip()

        if not actual_model or actual_dim <= 0 or not actual_fingerprint:
            LOGGER.warning(
                "Index metadata for table %r is incomplete. Rebuild the index to enable checks.",
                self._table_name,
            )
            return

        runtime_dim = self._embedder.embedding_dim
        if runtime_dim is None:
            runtime_dim = int(self._embedder.encode_texts(["dimension probe"], batch_size=1)[0].shape[0])
        if runtime_dim is not None and int(runtime_dim) != actual_dim:
            raise RuntimeError(
                "Embedding dimension mismatch between configured model and index table "
                f"{self._table_name!r}: configured={runtime_dim}, indexed={actual_dim}. "
                f"Rebuild the index or point index.table_name to a matching table."
            )

        runtime_fingerprint = self._embedder.fingerprint()
        if runtime_fingerprint != actual_fingerprint:
            raise RuntimeError(
                "Configured embedding backend/model does not match the stored index metadata for "
                f"table {self._table_name!r}. configured_model={self._embedder.model_name!r}, "
                f"indexed_model={actual_model!r}. Rebuild the index or use a matching index.table_name."
            )

    # --------------------------
    # Query building
    # --------------------------

    def _build_where(self, f: SearchFilters) -> Optional[str]:
        clauses: List[str] = []

        if f.year_min is not None:
            clauses.append(f"year >= {int(f.year_min)}")
        if f.year_max is not None:
            clauses.append(f"year <= {int(f.year_max)}")

        def add_any(list_col: str, values: Optional[List[str]]) -> None:
            if not values:
                return
            safe = [v.strip() for v in values if isinstance(v, str) and v.strip()]
            if not safe:
                return
            vals = ", ".join(_sql_quote(v) for v in safe)
            clauses.append(f"array_has_any({list_col}, [{vals}])")

        add_any("species", f.species)
        add_any("modalities", f.modalities)
        add_any("features", f.features)
        add_any("stages", f.stages)
        add_any("domains", f.domains)
        add_any("contexts", f.contexts)
        add_any("institutions", f.institutions)
        add_any("authors", f.authors)

        # Venue is stored as a scalar string (not a list) in our LanceDB table.
        # Support multi-select in the UI by translating to SQL IN (...).
        if f.venues:
            safe_v = [v.strip() for v in f.venues if isinstance(v, str) and v.strip()]
            if safe_v:
                vals = ", ".join(_sql_quote(v) for v in safe_v)
                clauses.append(f"venue IN ({vals})")

        # paper_type / tradition (UI uses 'tradition', DB uses research_tradition)
        if f.paper_type and isinstance(f.paper_type, str) and f.paper_type.strip():
            v = f.paper_type.strip()
            if v == "others":
                clauses.append(
                    "(paper_type IN ('others','not_applicable','out_of_schema') OR paper_type IS NULL OR paper_type = '')"
                )
            else:
                clauses.append(f"paper_type = {_sql_quote(v)}")

        if f.tradition and isinstance(f.tradition, str) and f.tradition.strip():
            v = f.tradition.strip()
            if v == "others":
                clauses.append(
                    "(research_tradition IN ('others','not_applicable','out_of_schema') OR research_tradition IS NULL OR research_tradition = '')"
                )
            else:
                clauses.append(f"research_tradition = {_sql_quote(v)}")

        # Boolean flags
        if f.has_code is not None:
            clauses.append(f"has_code = {str(bool(f.has_code)).lower()}")
        if f.has_dataset is not None:
            clauses.append(f"has_dataset = {str(bool(f.has_dataset)).lower()}")
        if f.has_benchmark is not None:
            clauses.append(f"has_benchmark = {str(bool(f.has_benchmark)).lower()}")

        return _where_and(clauses)

    # --------------------------
    # Retrieval
    # --------------------------

    def _lexical_search(self, q: str, where: Optional[str], k: int) -> pd.DataFrame:
        assert isinstance(q, str) and q.strip()
        query = self._tbl.search(q).limit(k)
        if where:
            query = query.where(where, prefilter=True)
        return query.to_pandas()

    def _dense_search(self, q: str, where: Optional[str], k: int) -> pd.DataFrame:
        assert isinstance(q, str) and q.strip()

        if self._last_q == q and self._last_qv is not None:
            qv = self._last_qv
        else:
            qv = self._embedder.encode_texts([q])[0]
            assert np.isfinite(qv).all(), "Query embedding contains NaN or Inf values"
            assert qv.shape[0] == int(self._index_metadata["embedding_dim"]), (
                "Query embedding dimension mismatch: "
                f"expected {self._index_metadata['embedding_dim']}, got {qv.shape[0]}"
            )
            self._last_q = q
            self._last_qv = qv

        query = self._tbl.search(qv).limit(k)
        if where:
            query = query.where(where, prefilter=True)
        return query.to_pandas()

    def warmup(self) -> None:
        """Preload tokenizers/weights to reduce first-request latency."""
        try:
            _ = self._embedder.encode(["warmup"])
        except Exception:
            pass
        try:
            if self._reranker is not None:
                _ = self._reranker.score("warmup", ["warmup doc"])
        except Exception:
            pass

    def search(self, q: str, filters: SearchFilters, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        assert isinstance(q, str) and q.strip(), "q must be non-empty"
        assert isinstance(page, int) and page >= 1
        assert isinstance(page_size, int) and 1 <= page_size <= 50

        where = self._build_where(filters)

        k_lex = int(self._cfg["search"]["k_lex"])
        k_dense = int(self._cfg["search"]["k_dense"])
        rrf_k = int(self._cfg["search"]["rrf_k"])
        max_candidates = int(self._cfg["rerank"]["max_candidates"])

        rerank_cap = int(self._cfg.get("rerank", {}).get("topn", 100))
        rerank_cap = max(5, min(rerank_cap, max_candidates))

        t0 = time.time()

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_lex   = pool.submit(self._lexical_search, q, where, k_lex)
            f_dense = pool.submit(self._dense_search,   q, where, k_dense)
            lex_df   = f_lex.result()
            dense_df = f_dense.result()

        if "doc_id" not in lex_df.columns or "doc_id" not in dense_df.columns:
            raise RuntimeError("Index missing doc_id column; rebuild index")

        lex_ids = lex_df["doc_id"].astype(str).tolist()
        dense_ids = dense_df["doc_id"].astype(str).tolist()

        scores = _rrf_scores([lex_ids, dense_ids], k=rrf_k)
        if not scores:
            return {
                "query": q,
                "page": page,
                "page_size": page_size,
                "total_candidates": 0,
                "results": [],
                "debug": {"latency_ms": int((time.time() - t0) * 1000), "where": where},
            }

        fused_df = _coalesce_rows(lex_df, dense_df, scores, max_out=max_candidates)

        # Reranking
        rerank_enabled = (
            self._reranker is not None
            and bool(self._cfg["rerank"].get("enabled", False))
            and len(fused_df) > 3
        )

        if rerank_enabled:
            rerank_df = fused_df.head(rerank_cap).copy()

            if "text" in rerank_df.columns:
                docs = rerank_df["text"].astype(str).tolist()
            else:
                title = rerank_df.get("title", pd.Series([""] * len(rerank_df))).astype(str)
                summ = rerank_df.get("summary", pd.Series([""] * len(rerank_df))).astype(str)
                docs = (title + "\n\n" + summ).tolist()

            ce_scores = self._reranker.score(
                query=q,
                documents=docs,
                batch_size=int(self._cfg["rerank"]["batch_size"]),
                max_length=int(self._cfg["rerank"].get("max_length", 512)),
            )
            assert ce_scores, "Reranker returned empty scores"
            assert len(ce_scores) == len(docs), (
                "Reranker score length mismatch: "
                f"expected {len(docs)}, got {len(ce_scores)}"
            )
            rerank_df["__rerank_score"] = ce_scores
            rerank_df = rerank_df.sort_values("__rerank_score", ascending=False).reset_index(drop=True)

            rest_df = fused_df.iloc[len(rerank_df) :].copy()
            rest_df["__rerank_score"] = np.nan
            fused_df = pd.concat([rerank_df, rest_df], ignore_index=True)
        else:
            fused_df["__rerank_score"] = np.nan

        total = int(len(fused_df))
        start = (page - 1) * page_size
        end = start + page_size
        page_df = fused_df.iloc[start:end].copy()

        if "vector" in page_df.columns:
            page_df = page_df.drop(columns=["vector"])

        cols = [
            "doc_id",
            "oa_url",
            "doi",
            "title",
            "year",
            "publication_date",
            "type",
            "paper_type",
            "research_tradition",
            "social_scale",
            "venue",
            "pdf_url",
            "landing_page_url",
            "authors",
            "institutions",
            "affiliations",
            "species",
            "features",
            "modalities",
            "stages",
            "domains",
            "contexts",
            "summary",
            "abstract_text",
            "relevance_score",
            "cited_by_count",
            "has_code",
            "has_dataset",
            "has_benchmark",
            "__rrf_score",
            "__rerank_score",
        ]
        for c in cols:
            if c not in page_df.columns:
                page_df[c] = None

        records = page_df[cols].to_dict(orient="records")
        for r in records:
            for k, v in list(r.items()):
                r[k] = _json_safe(v)

            # Defensive: some upstream metadata may contain embedded HTML.
            # Keep non-English characters; only strip tags.
            if "title" in r and isinstance(r["title"], str):
                r["title"] = strip_html(r["title"]) or ""
            if "venue" in r and isinstance(r["venue"], str):
                r["venue"] = strip_html(r["venue"]) or r["venue"]

        LOGGER.debug(
            "Search debug backend=%s dim=%s reranker=%s lex=%s dense=%s fused=%s reranked=%s top_k=%s",
            self._embedder.backend_name,
            self._index_metadata["embedding_dim"],
            (self._reranker.backend_name if self._reranker is not None else "disabled"),
            len(lex_df),
            len(dense_df),
            len(fused_df),
            (min(len(fused_df), rerank_cap) if rerank_enabled else 0),
            self._cfg["search"]["top_k"],
        )

        return {
            "query": q,
            "page": page,
            "page_size": page_size,
            "total_candidates": total,
            "results": records,
            "debug": {
                "where": where,
                "k_lex": k_lex,
                "k_dense": k_dense,
                "rrf_k": rrf_k,
                "rerank_enabled": rerank_enabled,
                "rerank_cap": rerank_cap if rerank_enabled else 0,
                "embedding_backend": self._embedder.backend_name,
                "embedding_dim": int(self._index_metadata["embedding_dim"]),
                "reranker_backend": (self._reranker.backend_name if self._reranker is not None else "disabled"),
                "lexical_candidates": int(len(lex_df)),
                "dense_candidates": int(len(dense_df)),
                "fused_candidates": int(len(fused_df)),
                "reranked_candidates": int(min(len(fused_df), rerank_cap) if rerank_enabled else 0),
                "final_top_k": int(self._cfg["search"]["top_k"]),
                "latency_ms": int((time.time() - t0) * 1000),
            },
        }

    def get_paper(self, doc_id: str) -> Dict[str, Any]:
        did = doc_id.strip()
        assert did, "doc_id must be non-empty"
        df = self._tbl.search().where(f"doc_id = {_sql_quote(did)}").limit(1).to_pandas()
        if len(df) == 0:
            raise KeyError(did)
        row = df.iloc[0].to_dict()
        row.pop("vector", None)
        out = {k: _json_safe(v) for k, v in row.items()}
        if "title" in out and isinstance(out["title"], str):
            out["title"] = strip_html(out["title"]) or ""
        if "venue" in out and isinstance(out["venue"], str):
            out["venue"] = strip_html(out["venue"]) or out["venue"]
        return out

    def similar_papers(self, doc_id: str, k: int = 8) -> List[Dict[str, Any]]:
        """Return up to k papers most similar to doc_id by vector cosine distance."""
        did = doc_id.strip()
        assert did, "doc_id must be non-empty"

        vec_df = self._tbl.search().where(f"doc_id = {_sql_quote(did)}").limit(1).to_pandas()
        if len(vec_df) == 0:
            raise KeyError(did)
        if "vector" not in vec_df.columns:
            return []

        raw_vec = vec_df.iloc[0]["vector"]
        vec = raw_vec if isinstance(raw_vec, np.ndarray) else np.array(raw_vec, dtype=np.float32)

        res_df = (
            self._tbl.search(vec)
            .where(f"doc_id != {_sql_quote(did)}", prefilter=True)
            .limit(k)
            .to_pandas()
        )

        if "vector" in res_df.columns:
            res_df = res_df.drop(columns=["vector"])

        cols = [
            "doc_id", "title", "year", "venue", "authors", "summary",
            "species", "paper_type", "pdf_url", "landing_page_url", "doi",
        ]
        for c in cols:
            if c not in res_df.columns:
                res_df[c] = None

        records = res_df[cols].to_dict(orient="records")
        for r in records:
            for key, v in list(r.items()):
                r[key] = _json_safe(v)
            if isinstance(r.get("title"), str):
                r["title"] = strip_html(r["title"]) or ""
            if isinstance(r.get("venue"), str):
                r["venue"] = strip_html(r["venue"]) or r["venue"]
        return records

    # --------------------------
    # Facets
    # --------------------------

    _RX_WEIRD = re.compile(r"[^A-Za-z0-9\s\-\.'’/(),:&]", re.UNICODE)
    _RX_ORCID = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")

    def _normalize_facet_label(self, field: str, s: str) -> str:
        if not isinstance(s, str):
            return ""
        raw = s.strip()
        if not raw:
            return ""

        low = raw.lower()

        # Merge schema sentinel categories
        if low in {"not_applicable", "out_of_schema"}:
            return "others"

        # Drop obvious junk
        if low in {"true", "false", "none", "null", "nan"}:
            return ""
        if self._RX_ORCID.match(raw):
            return ""
        if raw.startswith(("*", ")", ";", ",", "[", "]")):
            return ""
        if raw.count(";") >= 2:
            return ""
        if len(raw) > 180:
            return ""

        # Field-specific cleanup
        if field == "authors":
            # Must contain letters
            if not re.search(r"[A-Za-z]", raw):
                return ""
            # Filter out affiliation-like strings that leak into authors
            if raw[0].isdigit():
                return ""
            # Drop if digit-heavy (ORCIDs/IDs/affiliation numbering)
            digits = sum(ch.isdigit() for ch in raw)
            if digits >= 4 and digits / max(1, len(raw)) > 0.20:
                return ""
            if "/" in raw or "http" in low or "@" in raw:
                return ""
            bad_kw = (
                "university",
                "institute",
                "department",
                "laboratory",
                "centre",
                "center",
                "national",
                "reserve",
                "administration",
                "zoo",
                "bureau",
                "co.",
                "ltd",
                "inc",
            )
            lo = low
            if any(k in lo for k in bad_kw):
                return ""
            # Too many tokens is usually not a name
            if len(raw.split()) > 8:
                return ""
            # Strip weird punctuation; allow common name punctuation
            if self._RX_WEIRD.search(raw):
                return ""

        if field == "institutions":
            if self._RX_WEIRD.search(raw):
                # Institutions can include punctuation, but reject control chars / odd glyph spam
                raw2 = self._RX_WEIRD.sub(" ", raw).strip()
                raw = re.sub(r"\s+", " ", raw2)

        if field == "species":
            # Conservative plural squashing for lowercase single tokens
            if raw.islower() and len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss"):
                if " " not in raw and "-" not in raw:
                    raw = raw[:-1]

        return raw

    def facet_values(self, field: str, limit: int | None = None) -> List[str]:
        """Return distinct facet values for a field.

        Handles LanceDB list columns that surface as np.ndarray inside Pandas.
        """
        f_in = (field or "").strip()
        if not f_in:
            raise KeyError("missing field")

        alias = {
            "tradition": "research_tradition",
            "research_tradition": "research_tradition",
            "paper_type": "paper_type",
            "type": "paper_type",
        }
        f = alias.get(f_in, f_in)

        allowed = {
            "species",
            "modalities",
            "features",
            "stages",
            "domains",
            "contexts",
            "institutions",
            "authors",
            "research_tradition",
            "paper_type",
            "social_scale",
            "year",
            "has_code",
            "has_dataset",
            "has_benchmark",
            "venue",
        }
        if f not in allowed:
            raise KeyError(f"unknown facet field: {f_in}")

        now = time.time()
        cache_key = f"{f}:{limit or ''}"
        if cache_key in self._facet_cache and (now - self._facet_cache_time.get(cache_key, 0.0)) < self._facet_ttl_sec:
            LOGGER.debug("Facet cache hit field=%s limit=%s", f, limit)
            return self._facet_cache[cache_key]

        t0 = time.time()
        LOGGER.info("Facet values start field=%s limit=%s", f, limit)

        try:
            df = self._tbl.search().select([f]).to_pandas()
        except Exception:
            LOGGER.exception("Facet values failed during table scan field=%s limit=%s", f, limit)
            self._facet_cache[cache_key] = []
            self._facet_cache_time[cache_key] = now
            return []

        vals: set[str] = set()

        obj_key_by_field = {
            "species": ["name", "species"],
            "modalities": ["modality", "name"],
            "features": ["feature", "name"],
            "stages": ["stage", "name"],
            "domains": ["domain", "name"],
            "contexts": ["context", "name"],
            "institutions": ["name", "institution"],
            "authors": ["name", "author"],
            "venue": ["name", "display_name", "venue"],
        }
        keys = obj_key_by_field.get(f, ["name"])

        def add_value(x: Any) -> None:
            if x is None:
                return

            # Unwrap numpy scalar/list representations
            if isinstance(x, (np.floating, np.integer)):
                x = x.item()
            if isinstance(x, np.ndarray):
                x = x.tolist()

            if isinstance(x, bool):
                vals.add("true" if x else "false")
                return

            if isinstance(x, (int, float)) and f == "year":
                try:
                    vals.add(str(int(x)))
                except Exception:
                    pass
                return

            if isinstance(x, str):
                s = self._normalize_facet_label(f, x)
                if s:
                    vals.add(s)
                return

            if isinstance(x, dict):
                for k in keys:
                    v = x.get(k)
                    if isinstance(v, str) and v.strip():
                        vv = self._normalize_facet_label(f, v)
                        if vv:
                            vals.add(vv)
                        return
                return

            if isinstance(x, (list, tuple, set, pd.Series)):
                for it in x:
                    add_value(it)
                return

        for x in df[f].tolist():
            add_value(x)

        cap = int(limit) if (isinstance(limit, int) and limit > 0) else self._max_facet_values
        out = sorted(vals)[:cap]
        self._facet_cache[cache_key] = out
        self._facet_cache_time[cache_key] = now
        LOGGER.info(
            "Facet values done field=%s limit=%s distinct=%s latency_ms=%s",
            f,
            limit,
            len(out),
            int((time.time() - t0) * 1000),
        )
        return out

    # --------------------------
    # Exports
    # --------------------------

    def export_ris(self, doc_id: str) -> str:
        """Generate a minimal RIS record for a paper."""
        p = self.get_paper(doc_id)
        title = (p.get("title") or p.get("display_name") or "").strip()
        year = p.get("year") or ""
        doi = (p.get("doi") or "").strip()
        url = (p.get("landing_page_url") or p.get("url") or "").strip()
        pdf = (p.get("pdf_url") or "").strip()

        authors = p.get("authors")
        author_list: List[str] = []
        if isinstance(authors, list):
            for a in authors:
                if isinstance(a, str) and a.strip():
                    author_list.append(a.strip())
                elif isinstance(a, dict) and isinstance(a.get("name"), str) and a["name"].strip():
                    author_list.append(a["name"].strip())
        elif isinstance(authors, str) and authors.strip():
            author_list = [x.strip() for x in authors.split(",") if x.strip()]

        paper_type = (p.get("paper_type") or p.get("type") or "").lower()
        ty = "GEN"
        if "book" in paper_type:
            ty = "BOOK"
        elif "preprint" in paper_type or "posted" in paper_type:
            ty = "PREP"
        elif "journal" in paper_type or "article" in paper_type:
            ty = "JOUR"
        elif "conference" in paper_type or "proceedings" in paper_type:
            ty = "CPAPER"

        lines: List[str] = [f"TY  - {ty}"]
        if title:
            lines.append(f"TI  - {title}")
        for a in author_list:
            lines.append(f"AU  - {a}")
        if year:
            lines.append(f"PY  - {year}")
        if doi:
            lines.append(f"DO  - {doi}")
        if url:
            lines.append(f"UR  - {url}")
        if pdf:
            lines.append(f"L1  - {pdf}")
        lines.append("ER  - ")
        return "\n".join(lines) + "\n"

    # --------------------------
    # Analytics summary (lightweight)
    # --------------------------

    def analytics_summary(self) -> Dict[str, Any]:
        now = time.time()
        if self._analytics_cache is not None and (now - self._analytics_cache_time) < self._analytics_ttl_sec:
            return self._analytics_cache

        df = self._tbl.search().select(["year", "species", "features", "domains", "institutions", "authors"]).to_pandas()

        by_year = (
            df.dropna(subset=["year"]).groupby("year").size().sort_index().reset_index(name="count").to_dict(orient="records")
        )

        def top_list(col: str, n: int = 20) -> List[Dict[str, Any]]:
            if col not in df.columns:
                return []
            tmp = df[[col]].dropna()
            tmp = tmp[tmp[col].map(lambda x: isinstance(x, (list, np.ndarray)))]
            if len(tmp) == 0:
                return []
            ex = tmp.explode(col)
            ex = ex[ex[col].map(lambda x: isinstance(x, str) and x.strip())]
            vc = ex[col].value_counts().head(n)
            return [{"value": k, "count": int(v)} for k, v in vc.items()]

        out = {
            "papers_total": int(len(df)),
            "by_year": by_year,
            "top_species": top_list("species", 25),
            "top_features": top_list("features", 25),
            "top_domains": top_list("domains", 25),
            "top_institutions": top_list("institutions", 25),
            "top_authors": top_list("authors", 25),
        }
        self._analytics_cache = out
        self._analytics_cache_time = now
        return out
