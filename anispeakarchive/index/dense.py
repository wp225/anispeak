from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional
import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def infer_embedding_provider(model_name: str, provider: Optional[str]) -> str:
    if provider is not None and str(provider).strip():
        return str(provider).strip().lower()

    lowered = str(model_name).strip().lower()
    if "bge" in lowered:
        return "sentence_transformers"
    if "llama-nv-embed" in lowered or "nemotron" in lowered:
        return "nvidia_hf"
    raise ValueError(
        "Could not infer embedding.provider from model_name="
        f"{model_name!r}. Set embedding.provider explicitly."
    )


def canonical_model_name(model_name: str) -> str:
    value = str(model_name).strip().lower()
    if not value:
        return value
    if value.startswith("/") or value.startswith("."):
        return Path(value).name.lower()
    return value.split("/")[-1]


def _pick_device(use_gpu: bool, device: Optional[str]) -> str:
    if device and str(device).strip():
        requested = str(device).strip().lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Embedding device {device!r} requested, but CUDA is not available."
            )
        return str(device).strip()
    return "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"


def _parse_torch_dtype(dtype: Optional[str]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    value = str(dtype).strip().lower()
    if not value or value == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if value not in mapping:
        raise ValueError(
            f"Unsupported dtype={dtype!r}. Expected one of: auto, float32, float16, bfloat16."
        )
    return mapping[value]


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return (vecs / norms).astype(np.float32, copy=False)


def _validate_embedding_matrix(
    vecs: np.ndarray,
    num_texts: int,
    expected_dim: Optional[int] = None,
) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float32)
    assert arr.ndim == 2, f"Expected embeddings with 2 dims, got shape={arr.shape}"
    assert arr.shape[0] == num_texts, (
        f"Embedding row count mismatch: expected {num_texts}, got {arr.shape[0]}"
    )
    assert arr.shape[1] > 0, "Embedding output has zero dimension"
    assert arr.size > 0, "Embedding output is empty"
    assert np.isfinite(arr).all(), "Embedding output contains NaN or Inf values"
    if expected_dim is not None:
        assert arr.shape[1] == expected_dim, (
            f"Embedding dimension mismatch across batches: expected {expected_dim}, "
            f"got {arr.shape[1]}"
        )
    return arr


class EmbeddingBackend(ABC):
    backend_name: str
    embedding_dim: Optional[int]

    @abstractmethod
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int,
        normalize: bool,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> np.ndarray:
        raise NotImplementedError


class FlagEmbeddingBackend(EmbeddingBackend):
    backend_name = "flag_embedding"

    def __init__(self, model_name: str, device: str, max_length: int) -> None:
        from FlagEmbedding import BGEM3FlagModel

        use_fp16 = device.startswith("cuda")
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.embedding_dim = None
        self._max_length = int(max_length)

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int,
        normalize: bool,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> np.ndarray:
        chunks: List[np.ndarray] = []
        expected_dim: Optional[int] = self.embedding_dim
        starts = range(0, len(texts), batch_size)
        if show_progress:
            starts = tqdm(
                starts,
                total=(len(texts) + batch_size - 1) // batch_size,
                desc=(progress_desc or "Embedding documents"),
                unit="batch",
            )

        for start in starts:
            batch = texts[start : start + batch_size]
            out = self._model.encode(
                batch,
                batch_size=batch_size,
                max_length=self._max_length,
            )
            arr = _validate_embedding_matrix(
                np.asarray(out["dense_vecs"], dtype=np.float32),
                num_texts=len(batch),
                expected_dim=expected_dim,
            )
            if expected_dim is None:
                expected_dim = int(arr.shape[1])
            chunks.append(arr.astype(np.float32, copy=False))

        assert chunks, "Embedding backend returned no batches"
        full = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        full = _validate_embedding_matrix(full, num_texts=len(texts), expected_dim=expected_dim)
        if normalize:
            full = _l2_normalize(full)
        self.embedding_dim = int(full.shape[1])
        return full


class SentenceTransformersEmbeddingBackend(EmbeddingBackend):
    backend_name = "sentence_transformers"

    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = int(self._model.get_sentence_embedding_dimension())

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int,
        normalize: bool,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=bool(normalize),
            show_progress_bar=bool(show_progress),
        )
        arr = _validate_embedding_matrix(vecs, num_texts=len(texts), expected_dim=self.embedding_dim)
        if normalize:
            arr = _l2_normalize(arr)
        return arr.astype(np.float32, copy=False)


class NvidiaEmbeddingBackend(EmbeddingBackend):
    backend_name = "nvidia_hf"

    def __init__(
        self,
        model_name: str,
        device: str,
        max_length: int,
        trust_remote_code: bool,
        dtype: Optional[str],
        device_map: Optional[str],
        empty_cache_min_texts: int = 128,
        empty_cache_interval_batches: int = 8,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        torch_dtype = _parse_torch_dtype(dtype)
        resolved_device_map = None if device_map is None else str(device_map).strip()
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        if torch_dtype is not None:
            model_kwargs["dtype"] = torch_dtype
        if resolved_device_map:
            model_kwargs["device_map"] = resolved_device_map

        self._model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self._model.eval()
        if resolved_device_map:
            self._device = str(next(self._model.parameters()).device)
        else:
            self._model.to(device)
            self._device = device
        self._max_length = int(max_length)
        self._empty_cache_min_texts = int(empty_cache_min_texts)
        self._empty_cache_interval_batches = int(empty_cache_interval_batches)
        self.embedding_dim = None
        self._device_map = resolved_device_map
        self._dtype = dtype

    def _pool_last_token(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        batch_idx = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
        return last_hidden_state[batch_idx, lengths]

    def _encode_batch(self, batch_texts: List[str]) -> np.ndarray:
        with torch.inference_mode():
            maybe_encode = getattr(self._model, "encode", None)
            if callable(maybe_encode):
                raw = maybe_encode(batch_texts)
                if isinstance(raw, torch.Tensor):
                    arr = raw.detach().to("cpu", dtype=torch.float32).numpy()
                else:
                    arr = np.asarray(raw, dtype=np.float32)
                del raw
                return arr

            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model(**inputs)

            if hasattr(outputs, "sentence_embeddings"):
                pooled = outputs.sentence_embeddings
            elif hasattr(outputs, "embeddings"):
                pooled = outputs.embeddings
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                pooled = self._pool_last_token(outputs.last_hidden_state, inputs["attention_mask"])
            else:
                raise RuntimeError(
                    "NVIDIA embedding model did not expose sentence embeddings, pooler output, "
                    "or last_hidden_state for pooling."
                )

            arr = pooled.detach().to("cpu", dtype=torch.float32).numpy()
            del pooled
            del outputs
            del inputs
            return arr

    def _batch_starts(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
        progress_desc: Optional[str],
    ) -> Iterator[int]:
        starts = range(0, len(texts), batch_size)
        if not show_progress:
            return iter(starts)
        total_batches = (len(texts) + batch_size - 1) // batch_size
        return iter(
            tqdm(
                starts,
                total=total_batches,
                desc=(progress_desc or "Embedding batches"),
                unit="batch",
            )
        )

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int,
        normalize: bool,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> np.ndarray:
        chunks: List[np.ndarray] = []
        expected_dim: Optional[int] = self.embedding_dim

        for batch_idx, start in enumerate(
            self._batch_starts(
                texts=texts,
                batch_size=batch_size,
                show_progress=show_progress,
                progress_desc=progress_desc,
            ),
            start=1,
        ):
            batch = texts[start : start + batch_size]
            arr = _validate_embedding_matrix(
                self._encode_batch(batch),
                num_texts=len(batch),
                expected_dim=expected_dim,
            )
            if expected_dim is None:
                expected_dim = int(arr.shape[1])
            chunks.append(arr.astype(np.float32, copy=False))
            del arr
            del batch

            if (
                self._device.startswith("cuda")
                and torch.cuda.is_available()
                and len(texts) >= self._empty_cache_min_texts
                and self._empty_cache_interval_batches > 0
                and (batch_idx % self._empty_cache_interval_batches) == 0
            ):
                torch.cuda.empty_cache()

        assert chunks, "Embedding backend returned no batches"
        full = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        full = _validate_embedding_matrix(full, num_texts=len(texts), expected_dim=expected_dim)
        if normalize:
            full = _l2_normalize(full)

        self.embedding_dim = int(full.shape[1])

        if (
            self._device.startswith("cuda")
            and torch.cuda.is_available()
            and len(texts) >= self._empty_cache_min_texts
        ):
            torch.cuda.empty_cache()

        return full


@dataclass
class Embedder:
    model_name: str
    provider: str
    device: str
    batch_size: int
    normalize: bool
    max_length: int
    trust_remote_code: bool
    dtype: Optional[str]
    device_map: Optional[str]
    _backend: EmbeddingBackend

    @property
    def backend_name(self) -> str:
        return self._backend.backend_name

    @property
    def embedding_dim(self) -> Optional[int]:
        return self._backend.embedding_dim

    @classmethod
    def create(
        cls,
        model_name: str,
        use_gpu: bool,
        batch_size: int,
        normalize: bool,
        provider: Optional[str] = None,
        max_length: int = 512,
        trust_remote_code: bool = False,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        device_map: Optional[str] = None,
    ) -> "Embedder":
        assert isinstance(model_name, str) and model_name.strip(), "embedding.model_name must be non-empty"
        assert isinstance(batch_size, int) and batch_size > 0, "embedding.batch_size must be > 0"
        provider_name = infer_embedding_provider(model_name, provider)
        resolved_device = _pick_device(use_gpu=use_gpu, device=device)

        if provider_name == "flag_embedding":
            backend = FlagEmbeddingBackend(
                model_name=model_name,
                device=resolved_device,
                max_length=max_length,
            )
        elif provider_name == "sentence_transformers":
            backend = SentenceTransformersEmbeddingBackend(model_name=model_name, device=resolved_device)
        elif provider_name == "nvidia_hf":
            backend = NvidiaEmbeddingBackend(
                model_name=model_name,
                device=resolved_device,
                max_length=max_length,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                device_map=device_map,
            )
        else:
            raise ValueError(
                f"Unsupported embedding.provider={provider_name!r}. "
                "Expected one of: flag_embedding, sentence_transformers, nvidia_hf."
            )

        LOGGER.info(
            "Initialized embedding backend=%s model=%s device=%s",
            backend.backend_name,
            model_name,
            resolved_device,
        )

        return cls(
            model_name=model_name,
            provider=provider_name,
            device=resolved_device,
            batch_size=batch_size,
            normalize=bool(normalize),
            max_length=int(max_length),
            trust_remote_code=bool(trust_remote_code),
            dtype=(None if dtype is None else str(dtype)),
            device_map=(None if device_map is None else str(device_map)),
            _backend=backend,
        )

    def encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> np.ndarray:
        assert isinstance(texts, list) and len(texts) > 0, "encode_texts expects non-empty list[str]"
        safe_texts = [t if isinstance(t, str) else str(t) for t in texts]
        arr = self._backend.encode_texts(
            safe_texts,
            batch_size=self.batch_size if batch_size is None else int(batch_size),
            normalize=self.normalize if normalize is None else bool(normalize),
            show_progress=bool(show_progress),
            progress_desc=progress_desc,
        )
        expected_dim = self._backend.embedding_dim
        arr = _validate_embedding_matrix(arr, num_texts=len(safe_texts), expected_dim=expected_dim)
        self._backend.embedding_dim = int(arr.shape[1])
        return arr.astype(np.float32, copy=False)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.encode_texts(texts=texts, batch_size=self.batch_size, normalize=self.normalize)

    def fingerprint(self) -> str:
        payload = "|".join(
            [
                self.backend_name,
                self.provider,
                canonical_model_name(self.model_name),
                str(bool(self.normalize)),
                str(int(self.embedding_dim or 0)),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
