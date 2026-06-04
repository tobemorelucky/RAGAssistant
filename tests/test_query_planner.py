import importlib
import json
import sys
import types


def _load_query_planner():
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv

    langchain_chat_models = types.ModuleType("langchain.chat_models")
    langchain_chat_models.init_chat_model = lambda *args, **kwargs: None
    sys.modules["langchain.chat_models"] = langchain_chat_models

    if "backend.query_planner" in sys.modules:
        del sys.modules["backend.query_planner"]
    return importlib.import_module("backend.query_planner")


def test_plan_retrieval_queries_falls_back_when_model_unavailable(monkeypatch):
    module = _load_query_planner()
    monkeypatch.setattr(module, "_get_planner_model", lambda: None)

    plan = module.plan_retrieval_queries("What was Adobe operating margin in 2022?")

    assert plan["enabled"] is False
    assert plan["dense_queries"] == []
    assert plan["keyword_queries"] == []
    assert plan["table_queries"] == []
    assert plan["parse_error"] == "planner_model_unavailable"


def test_plan_retrieval_queries_sanitizes_invalid_generated_entities(monkeypatch):
    module = _load_query_planner()

    class _FakeModel:
        def invoke(self, messages):
            payload = {
                "intent": "numeric_lookup",
                "must_keep_terms": ["Adobe", "2022"],
                "dense_queries": ["Adobe operating margin 2022", "AES operating margin 2022"],
                "keyword_queries": ["Adobe margin 2022"],
                "table_queries": ["Adobe operating margin table 2022"],
                "expected_evidence_type": "table_or_text",
                "constraints": ["preserve entities"],
            }
            return types.SimpleNamespace(content=json.dumps(payload))

    monkeypatch.setattr(module, "_get_planner_model", lambda: _FakeModel())

    plan = module.plan_retrieval_queries("What was Adobe operating margin in 2022?")

    assert plan["enabled"] is True
    assert plan["dense_queries"] == ["Adobe operating margin 2022"]
    assert plan["keyword_queries"] == ["Adobe margin 2022"]
    assert plan["table_queries"] == ["Adobe operating margin table 2022"]


def test_plan_retrieval_queries_rejects_queries_missing_original_numbers(monkeypatch):
    module = _load_query_planner()

    class _FakeModel:
        def invoke(self, messages):
            payload = {
                "intent": "numeric_lookup",
                "must_keep_terms": ["AMD", "2022"],
                "dense_queries": ["AMD quick ratio"],
                "keyword_queries": ["AMD quick ratio 2022"],
                "table_queries": [],
                "expected_evidence_type": "text",
                "constraints": [],
            }
            return types.SimpleNamespace(content=json.dumps(payload))

    monkeypatch.setattr(module, "_get_planner_model", lambda: _FakeModel())

    plan = module.plan_retrieval_queries("Does AMD have a healthy quick ratio for FY2022?")

    assert plan["dense_queries"] == []
    assert plan["keyword_queries"] == ["AMD quick ratio 2022"]
