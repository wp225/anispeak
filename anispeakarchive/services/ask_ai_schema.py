from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class AskAIModelOutput(BaseModel):
    answer_markdown: str = Field(
        description="Grounded markdown answer using citation IDs like [1], [2], with readable sections or bullets when helpful."
    )
    cited_ids: list[int] = Field(
        description="List of citation IDs used in answer_markdown."
    )
    confidence: Literal["high", "medium", "low"]
    limitations: str | None = Field(
        default=None,
        description="Short caveat if evidence is weak, indirect, or abstract-level only.",
    )

    @field_validator("answer_markdown")
    @classmethod
    def _validate_answer_markdown(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("answer_markdown must be non-empty")
        low = text.lower()
        if "@article{" in low or "@inproceedings{" in low or "@misc{" in low:
            raise ValueError("answer_markdown must not contain raw BibTeX")
        if "references:" in low or "bibliography:" in low:
            raise ValueError("answer_markdown must not contain a fake bibliography")
        return text

    @field_validator("cited_ids")
    @classmethod
    def _validate_cited_ids(cls, value: list[int]) -> list[int]:
        if not isinstance(value, list):
            raise ValueError("cited_ids must be a list")
        out: list[int] = []
        seen: set[int] = set()
        for item in value:
            idx = int(item)
            if idx <= 0:
                raise ValueError("cited_ids must contain only positive integers")
            if idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
        return out


class AskAIQueryPlan(BaseModel):
    search_queries: list[str] = Field(
        description="Non-overlapping scholarly search strings for ASA search."
    )
    reason: str | None = Field(
        default=None,
        description="Brief explanation of why these queries were selected."
    )

    @field_validator("search_queries")
    @classmethod
    def _validate_search_queries(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("search_queries must be a list")
        out: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            key = " ".join(text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        if not out:
            raise ValueError("search_queries must be non-empty")
        return out
