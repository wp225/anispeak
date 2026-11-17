"""Fetch a small, topical sample of papers from the OpenAlex API for the static demo.

Writes demo/build/raw/openalex.jsonl (one OpenAlex work per line).

The full AniSpeakArchive pipeline selects papers with a fine-tuned BERT classifier
and annotates them with an LLM. The demo skips both: it uses targeted OpenAlex
searches instead, so the sample is curated by query rather than by the classifier.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "raw" / "openalex.jsonl"

# Each query pulls the most relevant works with abstracts.
QUERIES = [
    "humpback whale song",
    "cetacean vocal communication",
    "dolphin signature whistle",
    "birdsong learning",
    "songbird vocal learning tutor",
    "elephant infrasound communication",
    "elephant vocalization",
    "animal call syntax combinatorial",
    "compositional syntax bird calls",
    "primate alarm calls",
    "primate vocal communication",
    "bat social calls echolocation",
    "frog acoustic communication chorus",
    "insect acoustic communication",
    "honeybee waggle dance communication",
    "bioacoustics deep learning classification",
    "passive acoustic monitoring animals",
    "animal vocal individual recognition",
    "chemical communication pheromone animals",
    "vibrational communication insects",
]

SELECT = ",".join([
    "id", "doi", "title", "display_name", "publication_year", "publication_date", "type",
    "cited_by_count", "authorships", "primary_location", "best_oa_location",
    "abstract_inverted_index", "primary_topic", "open_access",
])


def fetch(query: str, per_query: int) -> list[dict]:
    params = {
        "search": query,
        "filter": "has_abstract:true,type:article,language:en",
        "per-page": str(per_query),
        "select": SELECT,
    }
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "anispeakarchive-demo-builder"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    return payload.get("results", [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=int, default=25)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for q in QUERIES:
            works = fetch(q, args.per_query)
            added = 0
            for w in works:
                wid = w.get("id")
                if not wid or wid in seen or not (w.get("title") or w.get("display_name")):
                    continue
                seen.add(wid)
                w["_demo_query"] = q
                f.write(json.dumps(w, ensure_ascii=False) + "\n")
                added += 1
            n += added
            print(f"{q!r}: +{added}")
            time.sleep(0.3)
    print(f"wrote {n} works -> {OUT}")


if __name__ == "__main__":
    main()
