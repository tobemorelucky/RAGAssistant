import importlib.util
import sys
import types
from pathlib import Path


def _install_rag_utils_stubs():
    requests = types.ModuleType("requests")
    requests.post = lambda *args, **kwargs: None
    sys.modules["requests"] = requests

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv

    langchain_chat_models = types.ModuleType("langchain.chat_models")
    langchain_chat_models.init_chat_model = lambda *args, **kwargs: None
    sys.modules["langchain.chat_models"] = langchain_chat_models

    document_page_store = types.ModuleType("document_page_store")
    document_page_store.DocumentPageStore = type("DocumentPageStore", (), {})
    sys.modules["document_page_store"] = document_page_store

    parent_chunk_store = types.ModuleType("parent_chunk_store")
    parent_chunk_store.ParentChunkStore = type("ParentChunkStore", (), {})
    sys.modules["parent_chunk_store"] = parent_chunk_store

    embedding = types.ModuleType("embedding")

    class _FakeEmbeddingService:
        def get_all_embeddings(self, texts):
            return [[0.1, 0.2] for _ in texts], [{0: 1.0} for _ in texts]

    embedding.embedding_service = _FakeEmbeddingService()
    sys.modules["embedding"] = embedding

    finance = types.ModuleType("finance_rag_features")
    finance.COMPANY_ALIASES = {}
    finance.FINANCE_METRIC_HINTS = set()
    finance.extract_keyword_tokens = lambda text: set()
    finance.extract_metric_hints = lambda text: set()
    finance.extract_numbers = lambda text: set()
    finance.extract_years = lambda text: set()
    finance.infer_doc_type = lambda text: ""
    finance.normalize_doc_name = lambda text: text
    finance.parse_finance_query = lambda query: {"company": "", "years": [], "metrics": []}
    sys.modules["finance_rag_features"] = finance

    query_parser = types.ModuleType("query_parser")
    query_parser.company_aliases_for = lambda company: []
    query_parser.matches_company_text = lambda *args, **kwargs: False
    sys.modules["query_parser"] = query_parser

    milvus_client = types.ModuleType("milvus_client")
    milvus_client.MilvusManager = type("MilvusManager", (), {})
    sys.modules["milvus_client"] = milvus_client

    table_store = types.ModuleType("table_store")
    table_store.TableStore = type("TableStore", (), {})
    sys.modules["table_store"] = table_store

    path = Path(__file__).resolve().parents[1] / "backend" / "rag_utils.py"
    spec = importlib.util.spec_from_file_location("rag_utils_table_aware_test_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_table_aware_retrieval_off_does_not_call_evidence_search(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FailMilvus:
        def hybrid_retrieve(self, **kwargs):
            raise AssertionError("should not call table evidence retrieval when off")

    monkeypatch.setattr(module, "_milvus_manager", _FailMilvus())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "off")

    doc, meta = module._build_table_context_doc("net sales")

    assert doc is None
    assert meta["table_aware_retrieval_mode"] == "off"
    assert meta["table_evidence_hit_count"] == 0


def test_table_aware_retrieval_auto_query_metric_or_number_triggers(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            return []

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc("What was Amcor's net sales in fiscal year 2023?")

    assert doc is None
    assert meta["table_aware_retrieval_mode"] == "auto"
    assert meta["table_aware_auto_triggered"] is True
    assert "query_metric_or_number" in meta["table_aware_trigger_reason"]


def test_table_aware_retrieval_auto_query_contact_or_number_triggers(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            return []

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc("What is the US and Canada conference call number?")

    assert doc is None
    assert meta["table_aware_auto_triggered"] is True
    assert "query_contact_or_number" in meta["table_aware_trigger_reason"]


def test_table_aware_retrieval_auto_summary_query_does_not_trigger(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FailMilvus:
        def hybrid_retrieve(self, **kwargs):
            raise AssertionError("should not call table evidence retrieval when auto does not trigger")

    monkeypatch.setattr(module, "_milvus_manager", _FailMilvus())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc("Summarize this document.")

    assert doc is None
    assert meta["table_aware_auto_triggered"] is False
    assert meta["table_aware_trigger_reason"] == []


def test_table_aware_retrieval_auto_business_query_does_not_trigger(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FailMilvus:
        def hybrid_retrieve(self, **kwargs):
            raise AssertionError("should not call table evidence retrieval when auto does not trigger")

    monkeypatch.setattr(module, "_milvus_manager", _FailMilvus())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc("Explain Amcor's business.")

    assert doc is None
    assert meta["table_aware_auto_triggered"] is False
    assert meta["table_aware_trigger_reason"] == []


def test_table_aware_retrieval_auto_retrieved_table_like_chunk_triggers(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            return []

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc(
        "Explain this section.",
        retrieved_docs=[
            {
                "text": "Net sales 3,909 3,673 14,544 14,694\nCost of sales (3,115) (2,951) (11,724) (11,969)"
            }
        ],
    )

    assert doc is None
    assert meta["table_aware_auto_triggered"] is True
    assert "retrieved_table_like_chunk" in meta["table_aware_trigger_reason"]


def test_table_aware_retrieval_force_dedupes_ids_and_limits_tables(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            assert kwargs["filter_expr"] == 'evidence_type != "text_chunk"'
            return [
                {"table_id": "t1", "score": 0.9, "text": "row one", "evidence_type": "table_row", "row_id": "row_1", "page_number": 8},
                {"table_id": "t2", "score": 0.8, "text": "row two", "evidence_type": "table_row", "row_id": "row_2", "page_number": 9},
                {"table_id": "t1", "score": 0.7, "text": "row one duplicate", "evidence_type": "table_row", "row_id": "row_3", "page_number": 8},
                {"table_id": "", "score": 0.6, "text": "empty", "evidence_type": "table_row", "row_id": "", "page_number": 10},
            ]

    class _FakeTableStore:
        def __init__(self):
            self.called_with = None

        def get_tables_by_ids(self, table_ids):
            self.called_with = list(table_ids)
            return [
                {"table_id": "t1", "filename": "demo.pdf", "page_number": 8, "columns": ["Metric"], "rows": [{"Metric": "Net sales"}], "csv_text": ""},
            ]

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    fake_table_store = _FakeTableStore()
    monkeypatch.setattr(module, "_table_store", fake_table_store)
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "force")
    monkeypatch.setenv("TABLE_AWARE_EVIDENCE_TOP_K", "5")
    monkeypatch.setenv("TABLE_AWARE_MAX_TABLES", "1")
    monkeypatch.setenv("TABLE_AWARE_MAX_ROWS", "5")
    monkeypatch.setenv("TABLE_AWARE_MAX_CONTEXT_CHARS", "4000")

    doc, meta = module._build_table_context_doc("net sales")

    assert fake_table_store.called_with == ["t1"]
    assert meta["table_aware_retrieval_mode"] == "force"
    assert meta["table_evidence_hit_count"] == 4
    assert meta["table_context_table_count"] == 1
    assert meta["table_ids"] == ["t1"]
    assert meta["table_context_source"] == "document_scoped_search"
    assert "Additional structured table evidence:" in doc["text"]


def test_table_aware_retrieval_uses_candidate_filename_filter(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            assert (
                kwargs["filter_expr"]
                == 'evidence_type != "text_chunk" and filename in ["JPMORGAN_2021Q1_10Q.pdf", "JPMORGAN_2021_10K.pdf"]'
            )
            return []

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    monkeypatch.setattr(module, "_table_store", type("_FakeTableStore", (), {"get_tables_by_ids": lambda self, ids: []})())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "force")

    doc, meta = module._build_table_context_doc(
        "conference call number",
        retrieved_docs=[
            {"filename": "JPMORGAN_2021Q1_10Q.pdf"},
            {"filename": "JPMORGAN_2021Q1_10Q.pdf"},
            {"filename": "JPMORGAN_2021_10K.pdf"},
        ],
    )

    assert doc is None
    assert meta["table_candidate_filenames"] == ["JPMORGAN_2021Q1_10Q.pdf", "JPMORGAN_2021_10K.pdf"]


def test_table_aware_retrieval_candidate_filenames_are_ranked_deduped_and_limited(monkeypatch):
    module = _install_rag_utils_stubs()
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "force")

    filenames = module._extract_table_candidate_filenames(
        [
            {"filename": "AES_2022_10K.pdf"},
            {"filename": "ADOBE_2022_10K.pdf"},
            {"filename": "ADOBE_2022_10K.pdf"},
            {"filename": "ADOBE_2021_10K.pdf"},
            {"filename": "EXTRA.pdf"},
        ],
        max_files=2,
    )

    assert filenames == ["ADOBE_2022_10K.pdf", "AES_2022_10K.pdf"]


def test_table_aware_retrieval_falls_back_to_global_when_no_candidate_filenames(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            assert kwargs["filter_expr"] == 'evidence_type != "text_chunk"'
            return []

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    monkeypatch.setattr(module, "_table_store", type("_FakeTableStore", (), {"get_tables_by_ids": lambda self, ids: []})())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "force")

    doc, meta = module._build_table_context_doc("net sales", retrieved_docs=[{"filename": ""}, {"text": "chunk"}])

    assert doc is None
    assert meta["table_candidate_filenames"] == []


def test_table_aware_retrieval_prefers_retrieved_table_chunks(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FailMilvus:
        def hybrid_retrieve(self, **kwargs):
            raise AssertionError("should not search table evidence when retrieved table chunk already exists")

    class _FakeTableStore:
        def get_tables_by_ids(self, table_ids):
            return [
                {
                    "table_id": "amd::table::1",
                    "filename": "AMD_2022_10K.pdf",
                    "page_number": 42,
                    "columns": ["Metric", "2022"],
                    "rows": [{"Metric": "Quick ratio", "2022": "1.84"}],
                    "csv_text": "",
                }
            ]

    monkeypatch.setattr(module, "_milvus_manager", _FailMilvus())
    monkeypatch.setattr(module, "_table_store", _FakeTableStore())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc(
        "What was AMD's quick ratio in 2022?",
        retrieved_docs=[
            {
                "filename": "AMD_2022_10K.pdf",
                "page_number": 42,
                "table_id": "amd::table::1",
                "evidence_type": "table_row",
                "text": "Document: AMD_2022_10K.pdf\nRow Values: Metric: Quick ratio; 2022: 1.84",
            }
        ],
    )

    assert doc is not None
    assert meta["table_context_source"] == "retrieved_table_chunk"
    assert meta["table_ids"] == ["amd::table::1"]


def test_table_aware_retrieval_uses_same_page_table_before_search(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FailMilvus:
        def hybrid_retrieve(self, **kwargs):
            raise AssertionError("should not search table evidence when same-page table is available")

    class _FakeTableStore:
        def get_tables_by_ids(self, table_ids):
            return []

        def get_tables_by_filename(self, filename):
            assert filename == "ADOBE_2022_10K.pdf"
            return [
                {
                    "table_id": "adobe::table::1",
                    "filename": "ADOBE_2022_10K.pdf",
                    "page_number": 88,
                    "columns": ["Metric", "2022"],
                    "rows": [{"Metric": "Operating margin", "2022": "35%"}],
                    "csv_text": "",
                }
            ]

    monkeypatch.setattr(module, "_milvus_manager", _FailMilvus())
    monkeypatch.setattr(module, "_table_store", _FakeTableStore())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc(
        "What was Adobe operating margin in 2022?",
        retrieved_docs=[
            {
                "filename": "ADOBE_2022_10K.pdf",
                "page_number": 88,
                "text": "Operating margin 35% 34% 33%\nRevenue 100 90 80",
            }
        ],
    )

    assert doc is not None
    assert meta["table_context_source"] == "same_page_table"
    assert meta["table_ids"] == ["adobe::table::1"]
    assert meta["table_candidate_pages"] == [{"filename": "ADOBE_2022_10K.pdf", "page_number": 88}]


def test_table_aware_retrieval_auto_document_scoped_fallback_requires_query_trigger(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FailMilvus:
        def hybrid_retrieve(self, **kwargs):
            raise AssertionError("should not fall back to document-scoped table search for non-data query")

    class _FakeTableStore:
        def get_tables_by_ids(self, table_ids):
            return []

        def get_tables_by_filename(self, filename):
            return []

    monkeypatch.setattr(module, "_milvus_manager", _FailMilvus())
    monkeypatch.setattr(module, "_table_store", _FakeTableStore())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc(
        "Explain Amcor's business.",
        retrieved_docs=[{"filename": "AMCOR_2023_10K.pdf", "page_number": 12, "text": "Business overview and strategy."}],
    )

    assert doc is None
    assert meta["table_context_source"] == "none"


def test_table_context_respects_max_context_chars(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            return [{"table_id": "t1", "score": 0.9, "text": "x" * 500, "evidence_type": "table_row", "row_id": "row_1", "page_number": 8}]

    class _FakeTableStore:
        def get_tables_by_ids(self, table_ids):
            return [
                {
                    "table_id": "t1",
                    "filename": "demo.pdf",
                    "page_number": 8,
                    "columns": ["Metric", "2023"],
                    "rows": [{"Metric": "Net sales", "2023": "3,909"} for _ in range(20)],
                    "csv_text": "Net sales," + ("3,909," * 100),
                }
            ]

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    monkeypatch.setattr(module, "_table_store", _FakeTableStore())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "force")
    monkeypatch.setenv("TABLE_AWARE_MAX_CONTEXT_CHARS", "200")

    doc, meta = module._build_table_context_doc("net sales")

    assert meta["table_context_char_count"] <= 200
    assert "... table evidence truncated ..." in doc["text"]


def test_table_aware_retrieval_exception_does_not_break(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FailMilvus:
        def hybrid_retrieve(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(module, "_milvus_manager", _FailMilvus())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "force")

    doc, meta = module._build_table_context_doc("net sales")

    assert doc is None
    assert meta["table_aware_retrieval_mode"] == "force"
    assert meta["table_evidence_hit_count"] == 0


def test_debug_retrieval_pipeline_exposes_table_aware_trace_fields(monkeypatch):
    module = _install_rag_utils_stubs()

    monkeypatch.setattr(
        module,
        "retrieve_documents",
        lambda question, top_k=10: {
            "final_retrieved_docs": [],
            "context_docs": [],
            "meta": {
                "retrieval_mode": "baseline",
                "table_aware_retrieval_mode": "force",
                "table_aware_auto_triggered": True,
                "table_aware_trigger_reason": ["force"],
                "table_context_source": "document_scoped_search",
                "table_evidence_hit_count": 2,
                "table_context_table_count": 1,
                "table_context_char_count": 321,
                "table_candidate_filenames": ["demo.pdf"],
                "table_candidate_pages": [{"filename": "demo.pdf", "page_number": 8}],
                "table_ids": ["t1"],
                "latency_breakdown": {"total_retrieval_ms": 12.3},
            },
        },
    )

    result = module.debug_retrieval_pipeline("net sales", top_k=5)
    trace = result["rag_trace"]

    assert trace["table_aware_retrieval_mode"] == "force"
    assert trace["table_aware_auto_triggered"] is True
    assert trace["table_aware_trigger_reason"] == ["force"]
    assert trace["table_context_source"] == "document_scoped_search"
    assert trace["table_evidence_hit_count"] == 2
    assert trace["table_context_table_count"] == 1
    assert trace["table_context_char_count"] == 321
    assert trace["table_candidate_filenames"] == ["demo.pdf"]
    assert trace["table_candidate_pages"] == [{"filename": "demo.pdf", "page_number": 8}]
    assert trace["table_ids"] == ["t1"]
