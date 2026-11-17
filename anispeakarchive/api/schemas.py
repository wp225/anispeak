from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    q: str = Field(min_length=1)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=50)
    model_config = {"protected_namespaces": ()}
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
