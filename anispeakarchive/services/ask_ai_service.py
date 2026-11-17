from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any

from anispeakarchive.index.hybrid import SearchFilters
from anispeakarchive.services.ask_ai_client import AskAIClient, AskAIResponseError, AskAIUnavailableError
from anispeakarchive.services.ask_ai_refs import make_bibtex, make_bibtex_key, reference_url
from anispeakarchive.services.ask_ai_render import render_answer_markdown
from anispeakarchive.services.ask_ai_schema import AskAIModelOutput, AskAIQueryPlan


LOGGER = logging.getLogger(__name__)
_RX_CITATION = re.compile(r"\[(\d+)\]")
_RX_WS = re.compile(r"\s+")

SYSTEM_PROMPT = """You are AniSpeakArchive Ask AI.

Answer the user's question using only the retrieved AniSpeakArchive records.

Rules:
1. Use only the provided records.
2. Do not use outside knowledge.
3. Do not invent papers, DOI values, species, methods, or conclusions.
4. Every factual claim about papers must cite one or more records using [1], [2], etc.
5. If the retrieved records do not support a confident answer, say so.
6. If evidence is only from titles, abstracts, or extracted metadata, say "based on indexed metadata and abstracts."
7. Write clean markdown that is easy to scan.
8. Use short sections with bold lead-ins when helpful, such as **Answer** and **Evidence**.
9. Group multi-part questions by topic or query instead of blending everything into one block.
10. Use short bullets when listing multiple papers, datasets, or contrasts.
11. When multiple search queries were used, cover each query that has relevant evidence.
12. If a query has multiple clearly relevant papers, mention more than one when the evidence supports it.
13. Prefer synthesis over dumping records, but do not make the answer too short to be useful.
14. Target roughly 2-6 short paragraphs or bullets unless the evidence is very limited.
15. Output only valid JSON matching the required schema.
16. Do not output BibTeX.
17. Do not output a bibliography.
18. Do not cite records that were not provided.
"""

QUERY_PLANNER_SYSTEM_PROMPT = """You are an AniSpeakArchive search-query planner.

AniSpeakArchive is a scholarly search engine for animal communication, bioacoustics, computational ethology, vocalization datasets, animal signal analysis, and related computational biology papers.

Your job is to convert the user's message into clean search strings for the existing AniSpeakArchive search engine.

Rules:
1. Output only valid JSON.
2. Output search strings only.
3. Do not output filters.
4. Do not answer the question.
5. Do not cite papers.
6. Do not invent paper titles, authors, DOI values, or claims.
7. Split multi-part questions into separate search queries.
8. Keep queries non-overlapping when possible.
9. Use the fewest queries that cover the user's intent.
10. Usually return 1-4 queries for simple requests.
11. Return more than 4 when the request is broad, ambiguous, multi-part, or likely to miss papers without related terminology.
12. Include semantically adjacent scholarly search terms when they improve recall.
13. For concepts that may appear under multiple labels, include close variants and paraphrases rather than only the user's wording.
14. Never return more than 15 queries.
15. Correct obvious spelling errors only when safe.
16. Preserve domain-specific terms.
17. Prefer scholarly search terms over conversational phrasing.
18. If the user asks a simple clear question, return one query.

Example 1:
User message:
"What papers discuss canine vocalization classification datasets?"

Output:
{
  "search_queries": [
    "canine vocalization classification dataset"
  ],
  "reason": "The question is a single focused dataset-retrieval request."
}

Example 2:
User message:
"What are some papers where docgs can vocalizae and communicate with their female breed, and are tehre any breed difference in dogs"

Output:
{
  "search_queries": [
    "dog vocal communication",
    "canine vocalization sex differences female dogs",
    "dog vocalization breed differences",
    "canine acoustic communication breed variation"
  ],
  "reason": "The user asks about dog vocal communication, possible sex-related differences, and breed differences. Obvious spelling errors were corrected."
}

Example 3:
User message:
"Are there papers on whale syntax and also bird dialects with datasets?"

Output:
{
  "search_queries": [
    "whale song syntax sequential structure",
    "bird song dialect vocal learning dataset",
    "animal vocalization dataset syntax dialect"
  ],
  "reason": "The request has separate whale-syntax, bird-dialect, and dataset aspects."
}

Example 4:
User message:
"which of those have datasets?"

Recent conversation context:
The previous assistant answer discussed papers on canine vocalization classification and dog communication.

Output:
{
  "search_queries": [
    "canine vocalization dataset",
    "dog vocalization classification dataset"
  ],
  "reason": "The follow-up depends on the previous canine vocalization context and asks specifically about datasets."
}

Example 5:
User message:
"papers about emotion in cat meows"

Output:
{
  "search_queries": [
    "cat meow emotion affect vocalization"
  ],
  "reason": "The request is simple and focused, so one query is enough."
}

Example 6:
User message:
"What are some papers where dogs can vocalize and communicate with their female breed, and are there any breed differences in dogs?"

Output:
{
  "search_queries": [
    "dog vocal communication",
    "canine vocalization sex differences",
    "female dog vocalization",
    "dog vocalization breed differences",
    "canine acoustic communication breed variation"
  ],
  "reason": "The request mixes dog vocal communication, sex or female-related differences, and breed differences, so the search should be split into those aspects."
}
"""

STREAM_SYSTEM_PROMPT = """You are AniSpeakArchive Ask AI.

Answer the user's question using only the retrieved AniSpeakArchive records.

Rules:
1. Use only the provided records.
2. Do not use outside knowledge.
3. Do not invent papers, DOI values, species, methods, or conclusions.
4. Every factual claim about papers must cite one or more records using [1], [2], etc.
5. If the retrieved records do not support a confident answer, say so.
6. If evidence is only from titles, abstracts, or extracted metadata, say "based on indexed metadata and abstracts."
7. Write clean markdown that is easy to scan.
8. Use short sections with bold lead-ins when helpful, such as **Answer** and **Evidence**.
9. Group multi-part questions by topic or query instead of blending everything into one block.
10. Use short bullets when listing multiple papers, datasets, or contrasts.
11. When multiple search queries were used, cover each query that has relevant evidence.
12. If a query has multiple clearly relevant papers, mention more than one when the evidence supports it.
13. Prefer synthesis over dumping records, but do not make the answer too short to be useful.
14. Target roughly 2-6 short paragraphs or bullets unless the evidence is very limited.
15. Output markdown only, not JSON.
16. Do not output BibTeX.
17. Do not output a bibliography section.
18. Do not cite records that were not provided.
"""


class AskAIService:
    def __init__(self, ask_ai_cfg: Any, search_service: Any, client: AskAIClient) -> None:
        self._cfg = ask_ai_cfg
        self._search = search_service
        self._client = client
        qp_cfg = dict(getattr(self._cfg, "query_planning", {}) or {})
        self._planner_cfg = {
            "enabled": bool(qp_cfg.get("enabled", True)),
            "min_search_queries": int(qp_cfg.get("min_search_queries", 1)),
            "max_search_queries": int(qp_cfg.get("max_search_queries", 10)),
            "max_query_chars": int(qp_cfg.get("max_query_chars", 160)),
            "top_k_per_query": int(qp_cfg.get("top_k_per_query", 15)),
            "max_tokens": int(qp_cfg.get("max_tokens", 256)),
            "temperature": float(qp_cfg.get("temperature", 0.0)),
            "structured_output": bool(qp_cfg.get("structured_output", True)),
        }

    def normalize_message(self, message: str) -> str:
        text = unicodedata.normalize("NFKC", str(message or ""))
        text = _RX_WS.sub(" ", text).strip()
        return text

    def _is_useful_record(self, record: dict[str, Any]) -> bool:
        title = str(record.get("title") or "").strip()
        if not title:
            return False
        has_content = any(
            str(record.get(field) or "").strip()
            for field in ("abstract_text", "summary", "venue", "paper_type", "research_tradition")
        )
        has_structured = any(record.get(field) for field in ("species", "features", "modalities", "stages", "domains", "contexts"))
        return bool(has_content or has_structured)

    def _resource_summary(self, record: dict[str, Any]) -> str:
        parts: list[str] = []
        if record.get("has_dataset"):
            parts.append("has dataset")
        if record.get("has_code"):
            parts.append("has code")
        if record.get("has_benchmark"):
            parts.append("has benchmark")
        return ", ".join(parts)

    def _compact_record(self, record: dict[str, Any], citation_id: int) -> dict[str, Any]:
        out = {
            "citation_id": citation_id,
            "doc_id": record.get("doc_id"),
            "title": record.get("title"),
            "authors": record.get("authors") or [],
            "year": record.get("year"),
            "venue": record.get("venue"),
            "doi": record.get("doi"),
            "oa_url": record.get("oa_url"),
            "landing_page_url": record.get("landing_page_url"),
            "pdf_url": record.get("pdf_url"),
            "summary": record.get("summary"),
            "abstract_text": record.get("abstract_text"),
            "species": record.get("species") or [],
            "species_group": record.get("species_group") or [],
            "stages": record.get("stages") or [],
            "features": record.get("features") or [],
            "modalities": record.get("modalities") or [],
            "domains": record.get("domains") or [],
            "contexts": record.get("contexts") or [],
            "research_tradition": record.get("research_tradition"),
            "paper_type": record.get("paper_type"),
            "social_scale": record.get("social_scale"),
            "has_dataset": bool(record.get("has_dataset")),
            "has_code": bool(record.get("has_code")),
            "has_benchmark": bool(record.get("has_benchmark")),
        }
        return out

    def _format_record_for_prompt(self, record: dict[str, Any]) -> str:
        abstract = str(record.get("abstract_text") or "").strip()
        if len(abstract) > 900:
            abstract = abstract[:900].rstrip() + "..."
        lines = [
            f"[{record['citation_id']}]",
            f"Title: {record.get('title') or ''}",
            f"Authors: {', '.join(record.get('authors') or [])}",
            f"Year: {record.get('year') or ''}",
            f"Venue: {record.get('venue') or ''}",
            f"DOI: {record.get('doi') or ''}",
            f"Species: {', '.join(record.get('species') or [])}",
            f"Species group: {', '.join(record.get('species_group') or [])}",
            f"Features: {', '.join(record.get('features') or [])}",
            f"Modalities: {', '.join(record.get('modalities') or [])}",
            f"Stages: {', '.join(record.get('stages') or [])}",
            f"Domains: {', '.join(record.get('domains') or [])}",
            f"Contexts: {', '.join(record.get('contexts') or [])}",
            f"Research tradition: {record.get('research_tradition') or ''}",
            f"Paper type: {record.get('paper_type') or ''}",
            f"Resources: {self._resource_summary(record)}",
            f"Matched queries: {', '.join(record.get('matched_queries') or [])}",
            f"Summary: {record.get('summary') or ''}",
            f"Abstract: {abstract}",
        ]
        return "\n".join(lines)

    def _history_text(self, recent_messages: list[dict[str, Any]]) -> str:
        max_msgs = max(0, int(self._cfg.conversation_history_turns) * 2)
        sliced = recent_messages[-max_msgs:] if max_msgs else []
        lines: list[str] = []
        for msg in sliced:
            role = str(msg.get("role") or "")
            content = self.normalize_message(str(msg.get("content") or ""))
            if role in {"user", "assistant"} and content:
                lines.append(f"{role.title()}: {content}")
        return "\n".join(lines) if lines else "(none)"

    def _prompt(self, question: str, recent_messages: list[dict[str, Any]], records: list[dict[str, Any]]) -> str:
        record_blob = "\n\n".join(self._format_record_for_prompt(r) for r in records)
        planned_queries = sorted({query for record in records for query in (record.get("matched_queries") or [])})
        return (
            f"Recent conversation context:\n{self._history_text(recent_messages)}\n\n"
            f"Current question:\n{question}\n\n"
            f"Planned search queries:\n{chr(10).join(f'- {q}' for q in planned_queries) if planned_queries else '(none)'}\n\n"
            f"Retrieved AniSpeakArchive records:\n{record_blob}\n\n"
            "Return JSON with:\n"
            "- answer_markdown\n- cited_ids\n- confidence\n- limitations\n\n"
            "answer_markdown must use citation IDs like [1] and [2].\n"
            "cited_ids must contain exactly the citation IDs used in answer_markdown.\n"
            "Format answer_markdown as readable markdown.\n"
            "Use bold lead-ins and grouped bullets when that improves clarity.\n"
            "For multi-part questions, organize the answer by sub-question or theme.\n"
            "If planned search queries are listed above, use them to organize the answer when helpful.\n"
            "If more than one paper is relevant to a query, include more than one instead of naming only one paper.\n"
            "Do not write BibTeX."
        )

    def _planner_prompt(self, question: str, recent_messages: list[dict[str, Any]]) -> str:
        return (
            f"Recent conversation context:\n{self._history_text(recent_messages)}\n\n"
            f"Current user message:\n{question}\n\n"
            "Return JSON with:\n"
            "{\n"
            '  "search_queries": ["..."],\n'
            '  "reason": "..."\n'
            "}"
        )

    def _stream_prompt(self, question: str, recent_messages: list[dict[str, Any]], records: list[dict[str, Any]]) -> str:
        record_blob = "\n\n".join(self._format_record_for_prompt(r) for r in records)
        planned_queries = sorted({query for record in records for query in (record.get("matched_queries") or [])})
        return (
            f"Recent conversation context:\n{self._history_text(recent_messages)}\n\n"
            f"Current question:\n{question}\n\n"
            f"Planned search queries:\n{chr(10).join(f'- {q}' for q in planned_queries) if planned_queries else '(none)'}\n\n"
            f"Retrieved AniSpeakArchive records:\n{record_blob}\n\n"
            "Write a readable markdown answer grounded only in the provided records.\n"
            "Use bold lead-ins and grouped bullets when that improves clarity.\n"
            "For multi-part questions, organize the answer by sub-question or theme.\n"
            "If planned search queries are listed above, use them to organize the answer when helpful.\n"
            "If more than one paper is relevant to a query, include more than one instead of naming only one paper.\n"
            "Use inline citation IDs like [1] and [2].\n"
            "Do not output JSON.\n"
            "Do not output BibTeX.\n"
            "Do not output a bibliography."
        )

    def _retrieval_confidence(self, useful_count: int) -> str:
        if useful_count < 3:
            return "low"
        if useful_count < 8:
            return "medium"
        return "high"

    def _parse_citation_ids(self, answer_markdown: str) -> list[int]:
        seen: set[int] = set()
        out: list[int] = []
        for match in _RX_CITATION.findall(str(answer_markdown or "")):
            idx = int(match)
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
        return out

    def _validate_query_plan(self, plan: AskAIQueryPlan, fallback_query: str) -> AskAIQueryPlan:
        min_q = int(self._planner_cfg["min_search_queries"])
        max_q = int(self._planner_cfg["max_search_queries"])
        max_chars = int(self._planner_cfg["max_query_chars"])
        cleaned: list[str] = []
        seen: set[str] = set()
        generic_block = {"animal communication", "bioacoustics", "vocalization"}
        for query in plan.search_queries:
            text = self.normalize_message(query)
            if not text:
                continue
            if len(text) > max_chars:
                raise ValueError("Query plan produced an overlength query")
            key = text.lower()
            if key in seen:
                continue
            if key in generic_block and fallback_query.lower() not in generic_block:
                continue
            seen.add(key)
            cleaned.append(text)
        if len(cleaned) < min_q:
            raise ValueError("Query plan returned too few queries")
        cleaned = cleaned[:max_q]
        return AskAIQueryPlan(search_queries=cleaned, reason=plan.reason)

    def _plan_queries(self, query: str, recent_messages: list[dict[str, Any]]) -> tuple[list[str], str | None, bool]:
        if not bool(self._planner_cfg["enabled"]):
            return [query], None, False
        try:
            plan = self._client.generate_structured(
                AskAIQueryPlan,
                QUERY_PLANNER_SYSTEM_PROMPT,
                self._planner_prompt(query, recent_messages),
                max_tokens=int(self._planner_cfg["max_tokens"]),
                temperature=float(self._planner_cfg["temperature"]),
                structured_output=bool(self._planner_cfg["structured_output"]),
            )
            plan = self._validate_query_plan(plan, query)
            return list(plan.search_queries), plan.reason, False
        except Exception as exc:
            LOGGER.warning("Query planner fallback triggered error=%s", type(exc).__name__)
            return [query], None, True

    def _merged_sort_key(self, record: dict[str, Any]) -> tuple[float, int, str]:
        year = record.get("year")
        try:
            year_value = int(year)
        except Exception:
            year_value = -1
        title = str(record.get("title") or "")
        return (-float(record.get("__merged_score", 0.0)), -year_value, title.lower())

    def _rerank_merged_records(self, question: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reranker = getattr(self._search, "_reranker", None)
        search_cfg = getattr(self._search, "_cfg", {}) or {}
        rerank_cfg = dict(search_cfg.get("rerank", {}) or {})
        if reranker is None or not bool(rerank_cfg.get("enabled", False)) or len(records) <= 3:
            return records
        docs = []
        for rec in records:
            text = str(rec.get("text") or "").strip()
            if not text:
                text = f"{rec.get('title') or ''}\n\n{rec.get('summary') or ''}\n\n{rec.get('abstract_text') or ''}"
            docs.append(text)
        scores = reranker.score(
            query=question,
            documents=docs,
            batch_size=int(rerank_cfg.get("batch_size", 16)),
            max_length=int(rerank_cfg.get("max_length", 512)),
        )
        for rec, score in zip(records, scores):
            rec["__merged_rerank_score"] = float(score)
        return sorted(
            records,
            key=lambda rec: (
                -float(rec.get("__merged_rerank_score", 0.0)),
                -float(rec.get("__merged_score", 0.0)),
                -(int(rec.get("year")) if str(rec.get("year") or "").isdigit() else -1),
                str(rec.get("title") or "").lower(),
            ),
        )

    def _run_retrieval(self, message: str, recent_messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any]]:
        query = self.normalize_message(message)
        planned_queries, plan_reason, fallback_query_planning = self._plan_queries(query, recent_messages)
        top_k_per_query = int(self._planner_cfg["top_k_per_query"])
        merged: dict[str, dict[str, Any]] = {}
        per_query_num_results: dict[str, int] = {}
        for planned_query in planned_queries:
            out = self._search.search(planned_query, filters=SearchFilters(), page=1, page_size=top_k_per_query)
            results = list(out.get("results") or [])[:top_k_per_query]
            per_query_num_results[planned_query] = len(results)
            for rank, record in enumerate(results, start=1):
                doc_id = str(record.get("doc_id") or "").strip()
                if not doc_id:
                    continue
                score = 1.0 / (60.0 + rank)
                if doc_id not in merged:
                    merged[doc_id] = dict(record)
                    merged[doc_id]["matched_queries"] = [planned_query]
                    merged[doc_id]["per_query_ranks"] = {planned_query: rank}
                    merged[doc_id]["__merged_score"] = score
                else:
                    merged[doc_id]["__merged_score"] = float(merged[doc_id]["__merged_score"]) + score
                    matched_queries = list(merged[doc_id].get("matched_queries") or [])
                    if planned_query not in matched_queries:
                        matched_queries.append(planned_query)
                    merged[doc_id]["matched_queries"] = matched_queries
                    per_query_ranks = dict(merged[doc_id].get("per_query_ranks") or {})
                    per_query_ranks[planned_query] = rank
                    merged[doc_id]["per_query_ranks"] = per_query_ranks
        all_results = sorted(merged.values(), key=self._merged_sort_key)
        useful = [r for r in all_results if self._is_useful_record(r)]
        useful = self._rerank_merged_records(query, useful)
        useful = useful[: int(self._cfg.max_context_records)]
        records = [self._compact_record(r, idx) for idx, r in enumerate(useful, start=1)]
        for rec, src in zip(records, useful):
            rec["matched_queries"] = list(src.get("matched_queries") or [])
            rec["per_query_ranks"] = dict(src.get("per_query_ranks") or {})
        retrieval_conf = self._retrieval_confidence(len(records))
        debug_meta = {
            "planned_search_queries": planned_queries,
            "query_plan_reason": plan_reason,
            "fallback_query_planning": bool(fallback_query_planning),
            "per_query_num_results": per_query_num_results,
            "top_k_per_query": top_k_per_query,
            "merged_num_results": len(all_results),
        }
        return query, all_results, records, retrieval_conf, debug_meta

    def _build_references(self, records: list[dict[str, Any]], cited_ids: list[int]) -> list[dict[str, Any]]:
        by_id = {int(r["citation_id"]): r for r in records}
        used_keys: set[str] = set()
        refs: list[dict[str, Any]] = []
        for citation_id in cited_ids:
            record = by_id.get(int(citation_id))
            if record is None:
                continue
            url = reference_url(record)
            ref_record = dict(record)
            ref_record["url"] = url
            citation_key = make_bibtex_key(ref_record, used_keys)
            refs.append(
                {
                    "citation_id": citation_id,
                    "title": record.get("title"),
                    "authors": record.get("authors") or [],
                    "year": record.get("year"),
                    "venue": record.get("venue"),
                    "doi": record.get("doi"),
                    "url": url,
                    "doc_id": record.get("doc_id"),
                    "bibtex": make_bibtex(ref_record, citation_key),
                }
            )
        return refs

    def _reindex_citations(self, answer_markdown: str, references: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        ordered_ids = self._parse_citation_ids(answer_markdown)
        if not ordered_ids or not references:
            return answer_markdown, references

        ref_by_old_id = {int(ref["citation_id"]): dict(ref) for ref in references}
        display_ids = [citation_id for citation_id in ordered_ids if citation_id in ref_by_old_id]
        if not display_ids:
            return answer_markdown, references

        id_map = {old_id: new_id for new_id, old_id in enumerate(display_ids, start=1)}

        def repl(match: re.Match[str]) -> str:
            old_id = int(match.group(1))
            new_id = id_map.get(old_id)
            return f"[{new_id}]" if new_id is not None else match.group(0)

        reindexed_answer = _RX_CITATION.sub(repl, answer_markdown)
        reindexed_refs: list[dict[str, Any]] = []
        for old_id in display_ids:
            ref = dict(ref_by_old_id[old_id])
            ref["citation_id"] = id_map[old_id]
            ref["source_citation_id"] = old_id
            reindexed_refs.append(ref)
        return reindexed_answer, reindexed_refs

    def _fallback_answer(self, query: str, records: list[dict[str, Any]], num_results: int) -> dict[str, Any]:
        top_ids = [int(r["citation_id"]) for r in records[: min(3, len(records))]]
        refs = self._build_references(records, top_ids)
        retrieval = {
            "query": query,
            "num_results": int(num_results),
            "num_context_records": int(len(records)),
            "context_record_doc_ids": [str(r.get("doc_id") or "") for r in records],
            "confidence": "low",
        }
        return {
            "answer": "AniSpeakArchive did not find strong matches for this query.\n\nTry queries like:\n- canine vocalization classification dataset\n- dog bark dataset classification\n- canine acoustic communication dataset",
            "answer_html": render_answer_markdown(
                "AniSpeakArchive did not find strong matches for this query.\n\n"
                "Try queries like:\n"
                "- canine vocalization classification dataset\n"
                "- dog bark dataset classification\n"
                "- canine acoustic communication dataset"
            ),
            "model_output": {"confidence": "low", "limitations": "Too few strong archive matches were retrieved."},
            "references": refs,
            "retrieval": retrieval,
            "retrieval_json": json.dumps(retrieval, sort_keys=True),
        }

    def _finalize_streamed_answer(
        self,
        answer_text: str,
        records: list[dict[str, Any]],
        query: str,
        num_results: int,
        retrieval_conf: str,
        retrieval_debug: dict[str, Any],
    ) -> dict[str, Any]:
        text = str(answer_text or "").strip()
        if not text:
            raise AskAIResponseError("Ask AI returned empty answer.")
        parsed_ids = self._parse_citation_ids(text)
        valid_ids = {int(r["citation_id"]) for r in records}
        cited_ids = [idx for idx in parsed_ids if idx in valid_ids]
        limitations = None if retrieval_conf == "high" else "Based on indexed metadata and abstracts."
        confidence = retrieval_conf
        if not cited_ids:
            cited_ids = [int(r["citation_id"]) for r in records[: min(3, len(records))]]
            if cited_ids:
                text = text.rstrip() + "\n\nRelevant records: " + " ".join(f"[{idx}]" for idx in cited_ids)
            confidence = "low"
            limitations = "No valid citations were produced during streaming; attached top evidence records."
        refs = self._build_references(records, cited_ids)
        text, refs = self._reindex_citations(text, refs)
        return {
            "answer": text,
            "answer_html": render_answer_markdown(text),
            "model_output": {
                "confidence": confidence,
                "limitations": limitations,
            },
            "references": refs,
            "retrieval": {
                "query": query,
                "num_results": int(num_results),
                "num_context_records": int(len(records)),
                "context_record_doc_ids": [str(r.get("doc_id") or "") for r in records],
                "matched_queries": {str(r.get("doc_id") or ""): list(r.get("matched_queries") or []) for r in records},
                "per_query_ranks": {str(r.get("doc_id") or ""): dict(r.get("per_query_ranks") or {}) for r in records},
                "confidence": confidence,
                **retrieval_debug,
            },
            "retrieval_json": json.dumps(
                {
                    "query": query,
                    "num_results": int(num_results),
                    "num_context_records": int(len(records)),
                    "context_record_doc_ids": [str(r.get("doc_id") or "") for r in records],
                    "matched_queries": {str(r.get("doc_id") or ""): list(r.get("matched_queries") or []) for r in records},
                    "per_query_ranks": {str(r.get("doc_id") or ""): dict(r.get("per_query_ranks") or {}) for r in records},
                    "confidence": confidence,
                    **retrieval_debug,
                },
                sort_keys=True,
            ),
        }

    def _validate_and_repair_output(
        self,
        model_output: AskAIModelOutput,
        records: list[dict[str, Any]],
        question: str,
        recent_messages: list[dict[str, Any]],
    ) -> AskAIModelOutput:
        valid_ids = {int(r["citation_id"]) for r in records}
        parsed_ids = self._parse_citation_ids(model_output.answer_markdown)
        cited_ids = list(dict.fromkeys(model_output.cited_ids))
        mismatch = set(parsed_ids) != set(cited_ids)
        invalid = [idx for idx in set(parsed_ids or cited_ids) if idx not in valid_ids]
        if not mismatch and not invalid and parsed_ids:
            return AskAIModelOutput(
                answer_markdown=model_output.answer_markdown,
                cited_ids=parsed_ids,
                confidence=model_output.confidence,
                limitations=model_output.limitations,
            )

        repair_prompt = self._prompt(question, recent_messages, records) + (
            "\n\nYour previous answer had invalid or missing citation IDs. "
            "Use only valid citation IDs from the provided records."
        )
        repaired = self._client.generate(SYSTEM_PROMPT, repair_prompt)
        parsed_ids = self._parse_citation_ids(repaired.answer_markdown)
        final_ids = [idx for idx in parsed_ids if idx in valid_ids]
        if final_ids:
            return AskAIModelOutput(
                answer_markdown=repaired.answer_markdown,
                cited_ids=final_ids,
                confidence=repaired.confidence,
                limitations=repaired.limitations,
            )

        top_ids = [int(r["citation_id"]) for r in records[: min(3, len(records))]]
        suffix = ""
        if top_ids:
            suffix = "\n\nRelevant records: " + " ".join(f"[{idx}]" for idx in top_ids)
        return AskAIModelOutput(
            answer_markdown=repaired.answer_markdown.strip() + suffix,
            cited_ids=top_ids,
            confidence="low",
            limitations=repaired.limitations or "Citations could not be validated against retrieved evidence.",
        )

    def answer_question(self, message: str, recent_messages: list[dict[str, Any]]) -> dict[str, Any]:
        query, all_results, records, retrieval_conf, retrieval_debug = self._run_retrieval(message, recent_messages)

        if len(records) < 3:
            out = self._fallback_answer(query, records, len(all_results))
            out["retrieval"].update(retrieval_debug)
            out["retrieval_json"] = json.dumps(out["retrieval"], sort_keys=True)
            return out

        model_output = self._client.generate(SYSTEM_PROMPT, self._prompt(query, recent_messages, records))
        model_output = self._validate_and_repair_output(model_output, records, query, recent_messages)
        cited_ids = self._parse_citation_ids(model_output.answer_markdown) or list(model_output.cited_ids)
        cited_ids = [idx for idx in cited_ids if any(int(r["citation_id"]) == int(idx) for r in records)]
        if not cited_ids:
            cited_ids = [int(r["citation_id"]) for r in records[: min(3, len(records))]]
            model_output = AskAIModelOutput(
                answer_markdown=model_output.answer_markdown + "\n\nRelevant records: " + " ".join(f"[{idx}]" for idx in cited_ids),
                cited_ids=cited_ids,
                confidence="low",
                limitations=model_output.limitations or "No valid citations were produced; attached top evidence records.",
            )

        refs = self._build_references(records, cited_ids)
        answer_markdown, refs = self._reindex_citations(model_output.answer_markdown, refs)
        return {
            "answer": answer_markdown,
            "answer_html": render_answer_markdown(answer_markdown),
            "model_output": {
                "confidence": model_output.confidence,
                "limitations": model_output.limitations,
            },
            "references": refs,
            "retrieval": {
                "query": query,
                "num_results": int(len(all_results)),
                "num_context_records": int(len(records)),
                "context_record_doc_ids": [str(r.get("doc_id") or "") for r in records],
                "matched_queries": {str(r.get("doc_id") or ""): list(r.get("matched_queries") or []) for r in records},
                "per_query_ranks": {str(r.get("doc_id") or ""): dict(r.get("per_query_ranks") or {}) for r in records},
                "confidence": retrieval_conf if model_output.confidence != "low" else "low",
                **retrieval_debug,
            },
            "raw_model_output": model_output.model_dump(),
            "retrieval_json": json.dumps(
                {
                    "query": query,
                    "num_results": int(len(all_results)),
                    "num_context_records": int(len(records)),
                    "context_record_doc_ids": [str(r.get("doc_id") or "") for r in records],
                    "matched_queries": {str(r.get("doc_id") or ""): list(r.get("matched_queries") or []) for r in records},
                    "per_query_ranks": {str(r.get("doc_id") or ""): dict(r.get("per_query_ranks") or {}) for r in records},
                    "confidence": retrieval_conf if model_output.confidence != "low" else "low",
                    **retrieval_debug,
                },
                sort_keys=True,
            ),
        }

    def answer_question_stream_events(
        self,
        message: str,
        recent_messages: list[dict[str, Any]],
    ):
        query, all_results, records, retrieval_conf, retrieval_debug = self._run_retrieval(message, recent_messages)
        if len(records) < 3:
            out = self._fallback_answer(query, records, len(all_results))
            out["retrieval"].update(retrieval_debug)
            out["retrieval_json"] = json.dumps(out["retrieval"], sort_keys=True)
            yield {"type": "final", "payload": out}
            return
        chunks: list[str] = []
        for chunk in self._client.iter_stream_answer(
            STREAM_SYSTEM_PROMPT,
            self._stream_prompt(query, recent_messages, records),
        ):
            chunks.append(chunk)
            yield {"type": "token", "text": chunk}
        answer = "".join(chunks).strip()
        if not answer:
            raise AskAIResponseError("LLM returned empty streaming content")
        yield {
            "type": "final",
            "payload": self._finalize_streamed_answer(answer, records, query, len(all_results), retrieval_conf, retrieval_debug),
        }
