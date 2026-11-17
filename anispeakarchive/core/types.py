from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class WebDoc(TypedDict, total=False):
    doc_id: str
    oa_url: str
    doi: Optional[str]
    title: str
    year: Optional[int]
    type: Optional[str]
    authors: List[str]
    institutions: List[str]
    affiliations: List[str]
    abstract_text: Optional[str]
    summary: str
    text: str

    species: List[str]
    features: List[str]
    modalities: List[str]
    domains: List[str]
    stages: List[str]
    contexts: List[str]
