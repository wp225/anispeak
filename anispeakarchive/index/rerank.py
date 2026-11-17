from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import logging

import numpy as np
import torch

from anispeakarchive.index.dense import infer_embedding_provider, _parse_torch_dtype


LOGGER = logging.getLogger(__name__)


def infer_rerank_provider(model_name: str, provider: Optional[str]) -> str:
    if provider is not None and str(provider).strip():
        return str(provider).strip().lower()
    return infer_embedding_provider(model_name=model_name, provider=None)


def _pick_device(use_gpu: bool, device: Optional[str]) -> str:
    if device and str(device).strip():
        requested = str(device).strip().lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Rerank device {device!r} requested, but CUDA is not available."
            )
        return str(device).strip()
    return "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"


class RerankerBackend(ABC):
    backend_name: str

    @abstractmethod
    def score(
        self,
        query: str,
        documents: List[str],
        batch_size: int,
        max_length: int,
    ) -> List[float]:
        raise NotImplementedError


class FlagEmbeddingRerankerBackend(RerankerBackend):
    backend_name = "flag_embedding"

    def __init__(self, model_name: str, use_gpu: bool) -> None:
        from FlagEmbedding import FlagReranker

        self._model = FlagReranker(model_name, use_fp16=bool(use_gpu))

    def score(
        self,
        query: str,
        documents: List[str],
        batch_size: int,
        max_length: int,
    ) -> List[float]:
        pairs = [[query, doc] for doc in documents]
        raw_scores = self._model.compute_score(pairs, batch_size=batch_size)
        return [float(s) for s in raw_scores]


class SentenceTransformersRerankerBackend(RerankerBackend):
    backend_name = "sentence_transformers"

    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=device)

    def score(
        self,
        query: str,
        documents: List[str],
        batch_size: int,
        max_length: int,
    ) -> List[float]:
        if hasattr(self._model, "max_length"):
            self._model.max_length = int(max_length)
        pairs = [(query, doc) for doc in documents]
        raw_scores = self._model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        scores = [float(s) for s in np.asarray(raw_scores, dtype=np.float32).reshape(-1)]
        return scores


class NvidiaNemotronRerankerBackend(RerankerBackend):
    backend_name = "nvidia_hf"

    def __init__(
        self,
        model_name: str,
        device: str,
        trust_remote_code: bool,
        dtype: Optional[str],
        device_map: Optional[str],
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

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

        self._model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
        self._model.eval()
        if resolved_device_map:
            self._device = str(next(self._model.parameters()).device)
        else:
            self._model.to(device)
            self._device = device
        self._device_map = resolved_device_map
        self._dtype = dtype

    def score(
        self,
        query: str,
        documents: List[str],
        batch_size: int,
        max_length: int,
    ) -> List[float]:
        scores: List[float] = []
        for start in range(0, len(documents), batch_size):
            docs = documents[start : start + batch_size]
            with torch.inference_mode():
                inputs = self._tokenizer(
                    [query] * len(docs),
                    docs,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                outputs = self._model(**inputs)
                logits = outputs.logits
                batch_scores = logits.reshape(-1).detach().to("cpu", dtype=torch.float32).tolist()
                del outputs
                del logits
                del inputs
            scores.extend(float(s) for s in batch_scores)
        return scores


@dataclass
class Reranker:
    model_name: str
    provider: str
    device: str
    batch_size: int
    max_length: int
    trust_remote_code: bool
    dtype: Optional[str]
    device_map: Optional[str]
    _backend: RerankerBackend

    @property
    def backend_name(self) -> str:
        return self._backend.backend_name

    @classmethod
    def create(
        cls,
        model_name: str,
        use_gpu: bool,
        batch_size: int,
        provider: Optional[str] = None,
        max_length: int = 512,
        trust_remote_code: bool = False,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        device_map: Optional[str] = None,
    ) -> "Reranker":
        assert isinstance(model_name, str) and model_name.strip(), "rerank.model_name must be non-empty"
        assert isinstance(batch_size, int) and batch_size > 0, "rerank.batch_size must be > 0"
        provider_name = infer_rerank_provider(model_name, provider)
        resolved_device = _pick_device(use_gpu=use_gpu, device=device)

        if provider_name == "flag_embedding":
            backend = FlagEmbeddingRerankerBackend(
                model_name=model_name,
                use_gpu=resolved_device.startswith("cuda"),
            )
        elif provider_name == "sentence_transformers":
            backend = SentenceTransformersRerankerBackend(
                model_name=model_name,
                device=resolved_device,
            )
        elif provider_name == "nvidia_hf":
            backend = NvidiaNemotronRerankerBackend(
                model_name=model_name,
                device=resolved_device,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                device_map=device_map,
            )
        else:
            raise ValueError(
                f"Unsupported rerank.provider={provider_name!r}. "
                "Expected one of: flag_embedding, sentence_transformers, nvidia_hf."
            )

        LOGGER.info(
            "Initialized reranker backend=%s model=%s device=%s",
            backend.backend_name,
            model_name,
            resolved_device,
        )

        return cls(
            model_name=model_name,
            provider=provider_name,
            device=resolved_device,
            batch_size=batch_size,
            max_length=int(max_length),
            trust_remote_code=bool(trust_remote_code),
            dtype=(None if dtype is None else str(dtype)),
            device_map=(None if device_map is None else str(device_map)),
            _backend=backend,
        )

    def score(
        self,
        query: str,
        documents: List[str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> List[float]:
        assert isinstance(query, str) and query.strip(), "query must be non-empty"
        assert isinstance(documents, list) and len(documents) > 0, "documents must be non-empty list[str]"
        safe_docs = [d if isinstance(d, str) else str(d) for d in documents]
        scores = self._backend.score(
            query=query,
            documents=safe_docs,
            batch_size=self.batch_size if batch_size is None else int(batch_size),
            max_length=self.max_length if max_length is None else int(max_length),
        )
        assert scores, "Reranker returned empty scores"
        assert len(scores) == len(documents), (
            "Reranker score length mismatch: "
            f"expected {len(documents)}, got {len(scores)}"
        )
        assert np.isfinite(np.asarray(scores, dtype=np.float32)).all(), (
            "Reranker scores contain NaN or Inf values"
        )
        return [float(s) for s in scores]

    def score_pairs(self, query: str, docs: List[str]) -> List[float]:
        return self.score(query=query, documents=docs, batch_size=self.batch_size, max_length=self.max_length)
