from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json

import pyarrow as pa


METADATA_KEY = b"asa.index.metadata"


def build_index_metadata(
    *,
    table_name: str,
    model_name: str,
    provider: str,
    backend_name: str,
    embedding_dim: int,
    embedding_fingerprint: str,
) -> Dict[str, Any]:
    return {
        "table_name": str(table_name),
        "model_name": str(model_name),
        "provider": str(provider),
        "backend_name": str(backend_name),
        "embedding_dim": int(embedding_dim),
        "embedding_fingerprint": str(embedding_fingerprint),
    }


def attach_metadata_to_arrow_table(table: pa.Table, metadata: Dict[str, Any]) -> pa.Table:
    existing = dict(table.schema.metadata or {})
    existing[METADATA_KEY] = json.dumps(metadata, sort_keys=True).encode("utf-8")
    return table.replace_schema_metadata(existing)


def read_table_metadata(table: Any) -> Optional[Dict[str, Any]]:
    schema = getattr(table, "schema", None)
    meta = getattr(schema, "metadata", None) if schema is not None else None
    if meta and METADATA_KEY in meta:
        return json.loads(meta[METADATA_KEY].decode("utf-8"))
    return None


def metadata_sidecar_path(lancedb_dir: str, table_name: str) -> Path:
    return Path(lancedb_dir) / f"{table_name}.metadata.json"


def write_metadata_sidecar(lancedb_dir: str, table_name: str, metadata: Dict[str, Any]) -> None:
    path = metadata_sidecar_path(lancedb_dir=lancedb_dir, table_name=table_name)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def read_metadata_sidecar(lancedb_dir: str, table_name: str) -> Optional[Dict[str, Any]]:
    path = metadata_sidecar_path(lancedb_dir=lancedb_dir, table_name=table_name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
