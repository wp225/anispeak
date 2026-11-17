from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, Any, Optional
from tqdm import tqdm

from corpus.schema import to_web_doc


def iter_jsonl(path: str, max_docs: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_web_corpus(
    final_relevant_jsonl: str,
    web_corpus_jsonl: str,
    include_abstract: bool,
    include_affiliations: bool,
    max_docs: Optional[int] = None,
) -> int:
    outp = Path(web_corpus_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with outp.open("w", encoding="utf-8") as w:
        for raw in tqdm(iter_jsonl(final_relevant_jsonl, max_docs=max_docs), desc="web_corpus"):
            if raw.get("is_relevant") is False:
                continue
            doc = to_web_doc(raw, include_abstract=include_abstract, include_affiliations=include_affiliations)
            w.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    return n
