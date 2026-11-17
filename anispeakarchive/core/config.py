from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import yaml


@dataclass
class AppCfg:
    host: str
    port: int
    debug: bool
    secret_key: str


@dataclass
class PathsCfg:
    final_relevant_jsonl: str
    web_corpus_jsonl: str
    lancedb_dir: str


@dataclass
class IndexCfg:
    table_name: str
    include_abstract: bool
    include_affiliations: bool
    max_docs: Optional[int]
    fts_columns: list[str]
    fts_replace: bool


@dataclass
class EmbeddingCfg:
    provider: Optional[str]
    model_name: str
    use_gpu: bool
    batch_size: int
    normalize: bool
    max_length: int
    trust_remote_code: bool
    device: Optional[str]
    dtype: Optional[str]
    device_map: Optional[str]


@dataclass
class RerankCfg:
    enabled: bool
    provider: Optional[str]
    model_name: str
    use_gpu: bool
    batch_size: int
    max_length: int
    max_candidates: int
    trust_remote_code: bool
    device: Optional[str]
    dtype: Optional[str]
    device_map: Optional[str]


@dataclass
class AskAICfg:
    enabled: bool
    provider: str
    model_path: str
    model_name: str
    gpu_device: Optional[str]
    host: str
    port: int
    base_url: str
    api_key: str
    dtype: str
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    temperature: float
    top_p: float
    max_new_tokens: int
    max_context_records: int
    max_question_chars: int
    conversation_history_turns: int
    manage_vllm: bool
    startup_timeout_sec: int
    structured_output: bool
    query_planning: Dict[str, Any]


@dataclass
class SearchCfg:
    k_lex: int
    k_dense: int
    rrf_k: int
    top_k: int


@dataclass
class UICfg:
    max_facet_values: int


@dataclass
class Settings:
    app: AppCfg
    paths: PathsCfg
    index: IndexCfg
    embedding: EmbeddingCfg
    rerank: RerankCfg
    ask_ai: AskAICfg
    search: SearchCfg
    ui: UICfg


def _req(d: Dict[str, Any], k: str) -> Any:
    if k not in d:
        raise KeyError(f"Missing config key: {k}")
    return d[k]


def load_settings(path: str) -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a mapping")

    app = raw.get("app", {})
    paths = raw.get("paths", {})
    index = raw.get("index", {})
    emb = raw.get("embedding", {})
    rerank = raw.get("rerank", {})
    ask_ai = raw.get("ask_ai", {})
    search = raw.get("search", {})
    ui = raw.get("ui", {})

    s = Settings(
        app=AppCfg(
            host=str(_req(app, "host")),
            port=int(_req(app, "port")),
            debug=bool(_req(app, "debug")),
            secret_key=str(_req(app, "secret_key")),
        ),
        paths=PathsCfg(
            final_relevant_jsonl=str(_req(paths, "final_relevant_jsonl")),
            web_corpus_jsonl=str(_req(paths, "web_corpus_jsonl")),
            lancedb_dir=str(_req(paths, "lancedb_dir")),
        ),
        index=IndexCfg(
            table_name=str(_req(index, "table_name")),
            include_abstract=bool(_req(index, "include_abstract")),
            include_affiliations=bool(_req(index, "include_affiliations")),
            max_docs=index.get("max_docs", None),
            fts_columns=list(_req(index, "fts_columns")),
            fts_replace=bool(_req(index, "fts_replace")),
        ),
        embedding=EmbeddingCfg(
            provider=(None if emb.get("provider") is None else str(emb.get("provider"))),
            model_name=str(_req(emb, "model_name")),
            use_gpu=bool(_req(emb, "use_gpu")),
            batch_size=int(_req(emb, "batch_size")),
            normalize=bool(emb.get("normalize_embeddings", emb.get("normalize", True))),
            max_length=int(emb.get("max_length", 512)),
            trust_remote_code=bool(emb.get("trust_remote_code", False)),
            device=(None if emb.get("device") is None else str(emb.get("device"))),
            dtype=(None if emb.get("dtype") is None else str(emb.get("dtype"))),
            device_map=(None if emb.get("device_map") is None else str(emb.get("device_map"))),
        ),
        rerank=RerankCfg(
            enabled=bool(_req(rerank, "enabled")),
            provider=(None if rerank.get("provider") is None else str(rerank.get("provider"))),
            model_name=str(_req(rerank, "model_name")),
            use_gpu=bool(_req(rerank, "use_gpu")),
            batch_size=int(_req(rerank, "batch_size")),
            max_length=int(rerank.get("max_length", 512)),
            max_candidates=int(_req(rerank, "max_candidates")),
            trust_remote_code=bool(rerank.get("trust_remote_code", False)),
            device=(None if rerank.get("device") is None else str(rerank.get("device"))),
            dtype=(None if rerank.get("dtype") is None else str(rerank.get("dtype"))),
            device_map=(None if rerank.get("device_map") is None else str(rerank.get("device_map"))),
        ),
        ask_ai=AskAICfg(
            enabled=bool(ask_ai.get("enabled", False)),
            provider=str(ask_ai.get("provider", "vllm_openai")),
            model_path=str(ask_ai.get("model_path", "")),
            model_name=str(ask_ai.get("model_name", "")),
            gpu_device=(None if ask_ai.get("gpu_device") is None else str(ask_ai.get("gpu_device"))),
            host=str(ask_ai.get("host", "127.0.0.1")),
            port=int(ask_ai.get("port", 8001)),
            base_url=str(ask_ai.get("base_url", "http://127.0.0.1:8001/v1")),
            api_key=str(ask_ai.get("api_key", "EMPTY")),
            dtype=str(ask_ai.get("dtype", "bfloat16")),
            max_model_len=int(ask_ai.get("max_model_len", 8192)),
            gpu_memory_utilization=float(ask_ai.get("gpu_memory_utilization", 0.45)),
            max_num_seqs=int(ask_ai.get("max_num_seqs", 16)),
            temperature=float(ask_ai.get("temperature", 0.0)),
            top_p=float(ask_ai.get("top_p", 1.0)),
            max_new_tokens=int(ask_ai.get("max_new_tokens", 1200)),
            max_context_records=int(ask_ai.get("max_context_records", 15)),
            max_question_chars=int(ask_ai.get("max_question_chars", 2000)),
            conversation_history_turns=int(ask_ai.get("conversation_history_turns", 4)),
            manage_vllm=bool(ask_ai.get("manage_vllm", True)),
            startup_timeout_sec=int(ask_ai.get("startup_timeout_sec", 300)),
            structured_output=bool(ask_ai.get("structured_output", True)),
            query_planning=dict(ask_ai.get("query_planning", {}) or {}),
        ),
        search=SearchCfg(
            k_lex=int(_req(search, "k_lex")),
            k_dense=int(_req(search, "k_dense")),
            rrf_k=int(_req(search, "rrf_k")),
            top_k=int(_req(search, "top_k")),
        ),
        ui=UICfg(
            max_facet_values=int(_req(ui, "max_facet_values")),
        ),
    )
    return s
