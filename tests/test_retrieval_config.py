from __future__ import annotations

import json

import pyarrow as pa

from anispeakarchive.core.config import load_settings
from anispeakarchive.index.dense import infer_embedding_provider
from anispeakarchive.index.metadata import (
    METADATA_KEY,
    attach_metadata_to_arrow_table,
    build_index_metadata,
)
from anispeakarchive.index.rerank import infer_rerank_provider


def test_provider_inference_from_model_name() -> None:
    assert infer_embedding_provider("BAAI/bge-m3", None) == "sentence_transformers"
    assert infer_embedding_provider("nvidia/llama-nv-embed-reasoning-3b", None) == "nvidia_hf"
    assert infer_rerank_provider("nvidia/llama-nemotron-rerank-1b-v2", None) == "nvidia_hf"


def test_attach_metadata_to_arrow_table_roundtrip() -> None:
    table = pa.table({"doc_id": ["a"], "vector": [[1.0, 2.0]]})
    meta = build_index_metadata(
        table_name="papers_nv_reasoning_3b",
        model_name="nvidia/llama-nv-embed-reasoning-3b",
        provider="nvidia_hf",
        backend_name="nvidia_hf",
        embedding_dim=2,
        embedding_fingerprint="abc123",
    )
    table = attach_metadata_to_arrow_table(table, meta)
    payload = json.loads(table.schema.metadata[METADATA_KEY].decode("utf-8"))
    assert payload["model_name"] == "nvidia/llama-nv-embed-reasoning-3b"
    assert payload["embedding_dim"] == 2


def test_load_settings_reads_provider_aliases() -> None:
    settings = load_settings("configs/default.yaml")
    assert settings.embedding.provider == "flag_embedding"
    assert settings.embedding.normalize is True
    assert settings.embedding.max_length == 512
    assert settings.embedding.dtype is None
    assert settings.embedding.device == "cuda:1"
    assert settings.embedding.batch_size == 16
    assert settings.rerank.provider == "flag_embedding"
    assert settings.rerank.max_length == 1024
    assert settings.rerank.dtype is None
    assert settings.rerank.device == "cuda:1"
    assert settings.rerank.batch_size == 16
    assert settings.ask_ai.enabled is True
    assert settings.ask_ai.provider == "vllm_openai"
    assert settings.ask_ai.gpu_device == "1"
    assert 0 < settings.ask_ai.gpu_memory_utilization <= 1
    assert settings.ask_ai.max_context_records == 15
    assert settings.ask_ai.query_planning["enabled"] is True
    assert settings.ask_ai.query_planning["top_k_per_query"] == 15
