import importlib.util
from pathlib import Path
import sys
import types


fake_embedding_root = types.ModuleType("embedding")
fake_embedding_root.EmbeddingService = object
fake_embedding_root.embedding_service = object()
sys.modules["embedding"] = fake_embedding_root

fake_milvus_client_root = types.ModuleType("milvus_client")


class _FakeMilvusManagerRoot:
    def __init__(self, *args, **kwargs):
        pass


fake_milvus_client_root.MilvusManager = _FakeMilvusManagerRoot
sys.modules["milvus_client"] = fake_milvus_client_root

fake_backend_embedding = types.ModuleType("backend.embedding")


class _FakeEmbeddingService:
    def get_all_embeddings(self, texts):
        return [[0.1, 0.2] for _ in texts], [{0: 1.0} for _ in texts]


fake_backend_embedding.embedding_service = _FakeEmbeddingService()
sys.modules["backend.embedding"] = fake_backend_embedding

fake_backend_milvus_client = types.ModuleType("backend.milvus_client")


class _FakeMilvusManager:
    def __init__(self, *args, **kwargs):
        pass

    def hybrid_retrieve(self, dense_embedding, sparse_embedding, top_k):
        return []


fake_backend_milvus_client.MilvusManager = _FakeMilvusManager
sys.modules["backend.milvus_client"] = fake_backend_milvus_client


WRITE_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "debug_write_table_evidence_to_milvus.py"
WRITE_SPEC = importlib.util.spec_from_file_location("debug_write_table_evidence_to_milvus", WRITE_MODULE_PATH)
debug_write_table_evidence_to_milvus = importlib.util.module_from_spec(WRITE_SPEC)
assert WRITE_SPEC and WRITE_SPEC.loader
WRITE_SPEC.loader.exec_module(debug_write_table_evidence_to_milvus)

SEARCH_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "debug_search_table_evidence.py"
SEARCH_SPEC = importlib.util.spec_from_file_location("debug_search_table_evidence", SEARCH_MODULE_PATH)
debug_search_table_evidence = importlib.util.module_from_spec(SEARCH_SPEC)
assert SEARCH_SPEC and SEARCH_SPEC.loader
SEARCH_SPEC.loader.exec_module(debug_search_table_evidence)


def test_collection_safety_check():
    assert debug_write_table_evidence_to_milvus.is_safe_test_collection_name("table_evidence_test")
    assert debug_write_table_evidence_to_milvus.is_safe_test_collection_name("my_debug_collection")
    assert debug_write_table_evidence_to_milvus.is_safe_test_collection_name("abc_test_xyz")
    assert not debug_write_table_evidence_to_milvus.is_safe_test_collection_name("embeddings_collection")
    assert not debug_write_table_evidence_to_milvus.is_safe_test_collection_name("prod_collection")


def test_evidence_type_distribution_summary():
    summary = debug_write_table_evidence_to_milvus.summarize_evidence_types(
        [
            {"evidence_type": "table_summary"},
            {"evidence_type": "table_row"},
            {"evidence_type": "table_row"},
            {"evidence_type": "table_raw"},
        ]
    )

    assert summary == {
        "table_summary": 1,
        "table_row": 2,
        "table_raw": 1,
    }


def test_search_report_contains_expected_fields():
    report = debug_search_table_evidence.build_report(
        "net sales",
        "table_evidence_test",
        [
            {
                "score": 0.92,
                "evidence_type": "table_row",
                "table_id": "demo.pdf::table::p3::1",
                "row_id": "row_1",
                "page_number": 3,
                "table_title": "Condensed Consolidated Statements of Income",
                "text": "Document: demo.pdf\nPage: 3\nRow Values: Metric: Net sales; 2023: 120",
            }
        ],
    )

    assert "query: net sales" in report
    assert "collection name: table_evidence_test" in report
    assert "score: 0.92" in report
    assert "evidence_type: table_row" in report
    assert "table_id: demo.pdf::table::p3::1" in report
    assert "row_id: row_1" in report
    assert "page_number: 3" in report
    assert "table_title: Condensed Consolidated Statements of Income" in report
