from __future__ import annotations

from types import SimpleNamespace

from flask import Flask
import sqlite3

from anispeakarchive.api.routes_ask import bp as ask_bp
from anispeakarchive.auth.db import exec_all, init_db
from anispeakarchive.services.ask_ai_client import AskAIClient, AskAIResponseError
from anispeakarchive.services.ask_ai_refs import doi_to_url, escape_bibtex_value, make_bibtex, make_bibtex_key
from anispeakarchive.services.ask_ai_schema import AskAIModelOutput, AskAIQueryPlan
from anispeakarchive.services.ask_ai_service import AskAIService
from anispeakarchive.services.ask_ai_render import render_answer_markdown


class FakeSearch:
    def __init__(self, results):
        self._results = results

    def search(self, q, filters, page, page_size):
        return {"results": self._results[:page_size], "total_candidates": len(self._results)}


class FakeClient:
    def __init__(self, outputs, plan=None):
        self.outputs = list(outputs)
        self.prompts = []
        self.plan = plan

    def generate(self, system_prompt, user_prompt):
        self.prompts.append(user_prompt)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    def generate_structured(self, schema_cls, system_prompt, user_prompt, **kwargs):
        self.prompts.append(user_prompt)
        if schema_cls is AskAIQueryPlan:
            if isinstance(self.plan, Exception):
                raise self.plan
            if self.plan is None:
                raise AssertionError("Planner output not configured")
            return self.plan
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _cfg():
    return SimpleNamespace(
        provider="vllm_openai",
        max_context_records=15,
        max_question_chars=2000,
        conversation_history_turns=4,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=768,
        model_name="dummy",
        structured_output=True,
        base_url="http://127.0.0.1:8001/v1",
        api_key="EMPTY",
        query_planning={"enabled": False, "top_k_per_query": 15, "min_search_queries": 1, "max_search_queries": 10, "max_query_chars": 160, "max_tokens": 256, "temperature": 0.0, "structured_output": True},
    )


def _record(i: int):
    return {
        "doc_id": f"W{i}",
        "title": f"Paper {i}",
        "authors": [f"Author {i}"],
        "year": 2025,
        "venue": "Venue",
        "doi": f"10.1234/{i}",
        "oa_url": f"https://openalex.org/W{i}",
        "landing_page_url": f"https://example.org/{i}",
        "pdf_url": f"https://example.org/{i}.pdf",
        "summary": "Summary",
        "abstract_text": "Abstract",
        "species": ["dog"],
        "species_group": ["mammal"],
        "stages": ["classification"],
        "features": ["pitch"],
        "modalities": ["audio"],
        "domains": ["bioacoustics"],
        "contexts": ["lab"],
        "research_tradition": "computational",
        "paper_type": "dataset",
        "social_scale": "individual",
        "has_dataset": True,
        "has_code": False,
        "has_benchmark": False,
    }


def test_pydantic_validates_output():
    out = AskAIModelOutput(
        answer_markdown="Answer [1]",
        cited_ids=[1, 1],
        confidence="high",
        limitations=None,
    )
    assert out.cited_ids == [1]


def test_doi_and_bibtex_helpers():
    assert doi_to_url("doi:10.1/abc") == "https://doi.org/10.1/abc"
    assert "\\&" in escape_bibtex_value("A&B")
    used = set()
    key1 = make_bibtex_key({"authors": ["Jane Doe"], "year": 2025, "title": "Dog Speak"}, used)
    key2 = make_bibtex_key({"authors": ["Jane Doe"], "year": 2025, "title": "Dog Speak"}, used)
    assert key1 != key2
    bib = make_bibtex({"authors": ["Jane Doe"], "year": 2025, "title": "Dog Speak", "venue": "Venue", "url": "https://x"}, key1)
    assert "@article" in bib
    assert "title" in bib


def test_render_answer_markdown_renders_basic_blocks():
    html = render_answer_markdown("# Title\n\n- one\n- two\n\n`code`")
    assert "<h1>" in html
    assert "<ul>" in html
    assert "<code>" in html


def test_only_top_15_records_are_passed_to_llm():
    results = [_record(i) for i in range(1, 21)]
    client = FakeClient([AskAIModelOutput(answer_markdown="Grounded [1][15]", cited_ids=[1, 15], confidence="high", limitations=None)])
    svc = AskAIService(_cfg(), FakeSearch(results), client)
    out = svc.answer_question("dog datasets", [])
    assert out["retrieval"]["num_context_records"] == 15
    prompt = client.prompts[0]
    assert "[15]" in prompt
    assert "[16]" not in prompt


def test_invalid_model_json_is_retried_once():
    results = [_record(i) for i in range(1, 5)]
    client = FakeClient([
        AskAIModelOutput(answer_markdown="No citations", cited_ids=[], confidence="medium", limitations=None),
        AskAIModelOutput(answer_markdown="Fixed [1]", cited_ids=[1], confidence="medium", limitations=None),
    ])
    svc = AskAIService(_cfg(), FakeSearch(results), client)
    out = svc.answer_question("dog datasets", [])
    assert out["answer"].endswith("[1]")
    assert len(client.prompts) == 2


def test_answer_citations_are_reindexed_to_match_filtered_reference_list():
    results = [_record(i) for i in range(1, 6)]
    client = FakeClient(
        [AskAIModelOutput(answer_markdown="**Answer** Paper links [2] and [5].", cited_ids=[2, 5], confidence="high", limitations=None)]
    )
    svc = AskAIService(_cfg(), FakeSearch(results), client)
    out = svc.answer_question("dog datasets", [])
    assert out["answer"] == "**Answer** Paper links [1] and [2]."
    assert [ref["citation_id"] for ref in out["references"]] == [1, 2]
    assert [ref["source_citation_id"] for ref in out["references"]] == [2, 5]


def test_prompt_requests_structured_markdown_for_readability():
    results = [_record(i) for i in range(1, 5)]
    client = FakeClient([AskAIModelOutput(answer_markdown="Answer [1]", cited_ids=[1], confidence="high", limitations=None)])
    svc = AskAIService(_cfg(), FakeSearch(results), client)
    svc.answer_question("compare datasets and methods", [])
    prompt = client.prompts[0]
    assert "Format answer_markdown as readable markdown." in prompt
    assert "Use bold lead-ins and grouped bullets when that improves clarity." in prompt
    assert "For multi-part questions, organize the answer by sub-question or theme." in prompt
    assert "If more than one paper is relevant to a query, include more than one instead of naming only one paper." in prompt


def test_prompt_includes_planned_queries_and_matched_query_context():
    cfg = _cfg()
    cfg.query_planning = {
        "enabled": True,
        "min_search_queries": 1,
        "max_search_queries": 10,
        "max_query_chars": 160,
        "top_k_per_query": 2,
        "max_tokens": 256,
        "temperature": 0.0,
        "structured_output": True,
    }

    class PlannedSearch:
        def search(self, q, filters, page, page_size):
            if q == "dog vocal communication":
                return {"results": [_record(1), _record(2)]}
            return {"results": [_record(2), _record(3)]}

    plan = AskAIQueryPlan(
        search_queries=["dog vocal communication", "dog vocalization breed differences"],
        reason="Split into query groups.",
    )
    client = FakeClient(
        [AskAIModelOutput(answer_markdown="Answer [1][2]", cited_ids=[1, 2], confidence="high", limitations=None)],
        plan=plan,
    )
    svc = AskAIService(cfg, PlannedSearch(), client)
    svc.answer_question("compare dog questions", [])
    prompt = client.prompts[-1]
    assert "Planned search queries:" in prompt
    assert "- dog vocal communication" in prompt
    assert "- dog vocalization breed differences" in prompt
    assert "Matched queries: dog vocal communication, dog vocalization breed differences" in prompt


def test_route_validates_and_reuses_conversation(tmp_path):
    db_path = str(tmp_path / "ask.sqlite")
    init_db(db_path)

    class RouteService:
        def normalize_message(self, value):
            return " ".join(str(value).split())

        def _payload(self, message):
            return {
                "answer": "Answer [1]",
                "model_output": {"confidence": "high", "limitations": None},
                "references": [{"citation_id": 1, "title": "Paper", "authors": ["A"], "year": 2025, "venue": "Venue", "doi": None, "url": "https://x", "doc_id": "W1", "bibtex": "@article{A,}"}],
                "retrieval": {"query": message, "num_results": 5, "num_context_records": 3, "context_record_doc_ids": ["W1"], "confidence": "high"},
                "retrieval_json": "{\"query\":\"x\"}",
            }

        def answer_question(self, message, recent):
            return self._payload(message)

        def answer_question_stream_events(self, message, recent):
            yield {"type": "token", "text": "Answer "}
            yield {"type": "token", "text": "[1]"}
            yield {"type": "final", "payload": self._payload(message)}

    app = Flask(__name__)
    app.secret_key = "test-secret-key"
    app.register_blueprint(ask_bp)
    app.config["AUTH_CFG"] = {"db_path": db_path}
    app.config["ASK_AI_CFG"] = _cfg()
    app.config["ASK_AI_SERVICE"] = RouteService()

    client = app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = 1

    bad = client.post("/api/ask", json={"message": ""})
    assert bad.status_code == 400

    over = client.post("/api/ask", json={"message": "x" * 2001})
    assert over.status_code == 400

    r1 = client.post("/api/ask", json={"message": "hello"})
    assert r1.status_code == 200
    conv_id = r1.get_json()["conversation_id"]

    r2 = client.post("/api/ask", json={"message": "follow up", "conversation_id": conv_id})
    assert r2.status_code == 200
    assert r2.get_json()["conversation_id"] == conv_id

    rows = exec_all(db_path, "SELECT role, content FROM ask_messages ORDER BY id")
    assert [str(r["role"]) for r in rows] == ["user", "assistant", "user", "assistant"]

    msgs = client.get(f"/api/ask/conversations/{conv_id}/messages")
    assert msgs.status_code == 200
    assistant_msgs = [m for m in msgs.get_json()["messages"] if m["role"] == "assistant"]
    assert assistant_msgs
    assert "response_json" in assistant_msgs[0]
    assert assistant_msgs[0]["response_json"]


def test_stream_route_returns_final_payload_and_persists_response_json(tmp_path):
    db_path = str(tmp_path / "ask.sqlite")
    init_db(db_path)

    class RouteService:
        def normalize_message(self, value):
            return " ".join(str(value).split())

        def _payload(self, message):
            return {
                "answer": "Answer [1]",
                "model_output": {"confidence": "high", "limitations": None},
                "references": [{"citation_id": 1, "title": "Paper", "authors": ["A"], "year": 2025, "venue": "Venue", "doi": None, "url": "https://x", "doc_id": "W1", "bibtex": "@article{A,}"}],
                "retrieval": {"query": message, "num_results": 5, "num_context_records": 3, "context_record_doc_ids": ["W1"], "confidence": "high"},
                "retrieval_json": "{\"query\":\"x\"}",
            }

        def answer_question(self, message, recent):
            return self._payload(message)

        def answer_question_stream_events(self, message, recent):
            yield {"type": "token", "text": "Answer "}
            yield {"type": "token", "text": "[1]"}
            yield {"type": "final", "payload": self._payload(message)}

    app = Flask(__name__)
    app.secret_key = "test-secret-key"
    app.register_blueprint(ask_bp)
    app.config["AUTH_CFG"] = {"db_path": db_path}
    app.config["ASK_AI_CFG"] = _cfg()
    app.config["ASK_AI_SERVICE"] = RouteService()

    client = app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = 1

    res = client.post("/api/ask/stream", json={"message": "hello"})
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "event: conversation" in body
    assert "event: token" in body
    assert "event: final" in body
    assert "\"references\"" in body

    conv_id = exec_all(db_path, "SELECT id FROM ask_conversations ORDER BY created_at DESC LIMIT 1")[0]["id"]
    msgs = client.get(f"/api/ask/conversations/{conv_id}/messages")
    payload = msgs.get_json()["messages"][-1]
    assert payload["role"] == "assistant"
    assert payload["response_json"]


def test_client_retries_invalid_json(monkeypatch):
    cfg = _cfg()
    client = AskAIClient(cfg)

    class FakeResp:
        def __init__(self, text):
            self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]

    calls = []

    def fake_create(messages, force_json_fallback=False, **kwargs):
        calls.append(force_json_fallback)
        return FakeResp("not json" if not force_json_fallback else '{"answer_markdown":"Ok [1]","cited_ids":[1],"confidence":"high","limitations":null}')

    monkeypatch.setattr(client, "_create_completion", fake_create)
    out = client.generate("sys", "user")
    assert out.cited_ids == [1]
    assert calls == [False, True]


def test_client_skips_guided_json_for_vllm_provider(monkeypatch):
    cfg = _cfg()
    client = AskAIClient(cfg)

    class FakeResp:
        def __init__(self, text):
            self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return FakeResp('{"answer_markdown":"Ok [1]","cited_ids":[1],"confidence":"high","limitations":null}')

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    out = client.generate("sys", "user")
    assert out.cited_ids == [1]
    assert "extra_body" not in captured


def test_client_raises_after_second_invalid_json(monkeypatch):
    cfg = _cfg()
    client = AskAIClient(cfg)

    class FakeResp:
        def __init__(self, text):
            self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]

    monkeypatch.setattr(client, "_create_completion", lambda messages, force_json_fallback=False, **kwargs: FakeResp("nope"))
    try:
        client.generate("sys", "user")
    except AskAIResponseError:
        return
    raise AssertionError("Expected AskAIResponseError")


def test_query_planner_splits_messy_question_and_merges_results():
    cfg = _cfg()
    cfg.query_planning = {
        "enabled": True,
        "min_search_queries": 1,
        "max_search_queries": 10,
        "max_query_chars": 160,
        "top_k_per_query": 2,
        "max_tokens": 256,
        "temperature": 0.0,
        "structured_output": True,
    }

    class PlannedSearch:
        def __init__(self):
            self.calls = []

        def search(self, q, filters, page, page_size):
            self.calls.append((q, page_size))
            if q == "dog vocal communication":
                return {"results": [_record(1), _record(2)]}
            if q == "dog vocalization breed differences":
                return {"results": [_record(2), _record(3)]}
            return {"results": []}

    plan = AskAIQueryPlan(
        search_queries=["dog vocal communication", "dog vocalization breed differences"],
        reason="Split into communication and breed-difference aspects.",
    )
    client = FakeClient(
        [AskAIModelOutput(answer_markdown="Answer [1][2]", cited_ids=[1, 2], confidence="high", limitations=None)],
        plan=plan,
    )
    svc = AskAIService(cfg, PlannedSearch(), client)
    out = svc.answer_question("messy dog question", [])
    assert out["retrieval"]["planned_search_queries"] == ["dog vocal communication", "dog vocalization breed differences"]
    assert out["retrieval"]["per_query_num_results"]["dog vocal communication"] == 2
    assert out["retrieval"]["merged_num_results"] == 3
    assert out["retrieval"]["num_context_records"] <= 15
    assert out["retrieval"]["context_record_doc_ids"][0] == "W2"




def test_query_planner_invalid_falls_back_to_original_query():
    cfg = _cfg()
    cfg.query_planning = {
        "enabled": True,
        "min_search_queries": 1,
        "max_search_queries": 10,
        "max_query_chars": 20,
        "top_k_per_query": 15,
        "max_tokens": 256,
        "temperature": 0.0,
        "structured_output": True,
    }

    class FallbackSearch:
        def __init__(self):
            self.calls = []

        def search(self, q, filters, page, page_size):
            self.calls.append(q)
            return {"results": [_record(1), _record(2), _record(3)]}

    bad_plan = AskAIQueryPlan(search_queries=["x" * 40], reason="too long")
    client = FakeClient(
        [AskAIModelOutput(answer_markdown="Answer [1]", cited_ids=[1], confidence="medium", limitations=None)],
        plan=bad_plan,
    )
    search = FallbackSearch()
    svc = AskAIService(cfg, search, client)
    out = svc.answer_question("original query", [])
    assert search.calls == ["original query"]
    assert out["retrieval"]["fallback_query_planning"] is True
    assert out["retrieval"]["planned_search_queries"] == ["original query"]


def test_init_db_migrates_old_ask_tables(tmp_path):
    db_path = str(tmp_path / "old.sqlite")
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("CREATE TABLE ask_conversations (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, title TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)")
        cur.execute("CREATE TABLE ask_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at INTEGER NOT NULL)")
        con.commit()
    finally:
        con.close()

    init_db(db_path)

    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("PRAGMA table_info(ask_conversations)")
        ask_conv_cols = {row[1] for row in cur.fetchall()}
        cur.execute("PRAGMA table_info(ask_messages)")
        ask_msg_cols = {row[1] for row in cur.fetchall()}
        assert "user_id" in ask_conv_cols
        assert "retrieval_json" in ask_msg_cols
        assert "response_json" in ask_msg_cols
        cur.execute("PRAGMA index_list(ask_conversations)")
        index_names = {row[1] for row in cur.fetchall()}
        assert "idx_ask_conv_user" in index_names
    finally:
        con.close()
