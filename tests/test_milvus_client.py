import importlib
import sys
import types


def _load_milvus_client_module():
    fake_pymilvus = types.ModuleType("pymilvus")

    class _FakeMilvusClient:
        def __init__(self, uri=None):
            self.uri = uri

    class _FakeDataType:
        INT64 = "INT64"
        FLOAT_VECTOR = "FLOAT_VECTOR"
        SPARSE_FLOAT_VECTOR = "SPARSE_FLOAT_VECTOR"
        VARCHAR = "VARCHAR"

    class _FakeAnnSearchRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeRRFRanker:
        def __init__(self, k):
            self.k = k

    fake_pymilvus.MilvusClient = _FakeMilvusClient
    fake_pymilvus.DataType = _FakeDataType
    fake_pymilvus.AnnSearchRequest = _FakeAnnSearchRequest
    fake_pymilvus.RRFRanker = _FakeRRFRanker
    sys.modules["pymilvus"] = fake_pymilvus

    module = importlib.import_module("backend.milvus_client")
    return importlib.reload(module)


class _FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, name, dtype, **kwargs):
        self.fields.append((name, dtype, kwargs))


class _FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class _FakeClient:
    def __init__(self):
        self.last_output_fields = None
        self.search_output_fields = None
        self.created_schema = None
        self.created_collection_kwargs = None

    def has_collection(self, _collection_name):
        return False

    def create_schema(self, **_kwargs):
        schema = _FakeSchema()
        self.created_schema = schema
        return schema

    def prepare_index_params(self):
        return _FakeIndexParams()

    def create_collection(self, **kwargs):
        self.created_collection_kwargs = kwargs

    def hybrid_search(self, **kwargs):
        self.last_output_fields = kwargs["output_fields"]
        return [[{
            "id": 1,
            "text": "chunk text",
            "filename": "demo.pdf",
            "file_type": "PDF",
            "page_number": 3,
            "evidence_type": "table_summary",
            "table_id": "table-1",
            "row_id": "",
            "table_title": "Summary Table",
            "chunk_id": "chunk-1",
            "parent_chunk_id": "parent-1",
            "root_chunk_id": "root-1",
            "chunk_level": 3,
            "chunk_idx": 0,
            "distance": 0.9,
        }]]

    def search(self, **kwargs):
        self.search_output_fields = kwargs["output_fields"]
        return [[{
            "id": 2,
            "entity": {
                "text": "chunk text",
                "filename": "demo.pdf",
                "file_type": "PDF",
                "page_number": 4,
                "evidence_type": "text_chunk",
                "table_id": "",
                "row_id": "",
                "table_title": "",
                "chunk_id": "chunk-2",
                "parent_chunk_id": "parent-2",
                "root_chunk_id": "root-2",
                "chunk_level": 3,
                "chunk_idx": 1,
            },
            "distance": 0.8,
        }]]


def test_init_collection_schema_includes_table_evidence_fields():
    module = _load_milvus_client_module()
    manager = module.MilvusManager()
    fake_client = _FakeClient()
    manager._run_with_reconnect = lambda operation: operation(fake_client)

    manager.init_collection(dense_dim=8)

    field_names = [name for name, _dtype, _kwargs in fake_client.created_schema.fields]
    field_types = {name: dtype for name, dtype, _kwargs in fake_client.created_schema.fields}
    assert "evidence_type" in field_names
    assert "table_id" in field_names
    assert "row_id" in field_names
    assert "table_title" in field_names
    assert field_types["evidence_type"] == module.DataType.VARCHAR
    assert field_types["table_title"] == module.DataType.VARCHAR


def test_hybrid_retrieve_output_fields_and_formatted_results_include_table_metadata():
    module = _load_milvus_client_module()
    manager = module.MilvusManager()
    fake_client = _FakeClient()
    manager._run_with_reconnect = lambda operation: operation(fake_client)

    results = manager.hybrid_retrieve([0.1, 0.2], {0: 1.0}, top_k=2)

    assert "evidence_type" in fake_client.last_output_fields
    assert "table_id" in fake_client.last_output_fields
    assert "row_id" in fake_client.last_output_fields
    assert "table_title" in fake_client.last_output_fields
    assert results[0]["evidence_type"] == "table_summary"
    assert results[0]["table_id"] == "table-1"
    assert results[0]["row_id"] == ""
    assert results[0]["table_title"] == "Summary Table"
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["parent_chunk_id"] == "parent-1"
    assert results[0]["root_chunk_id"] == "root-1"
    assert results[0]["chunk_level"] == 3


def test_dense_retrieve_output_fields_and_defaults_include_table_metadata():
    module = _load_milvus_client_module()
    manager = module.MilvusManager()
    fake_client = _FakeClient()
    manager._run_with_reconnect = lambda operation: operation(fake_client)

    results = manager.dense_retrieve([0.1, 0.2], top_k=2)

    assert "evidence_type" in fake_client.search_output_fields
    assert "table_id" in fake_client.search_output_fields
    assert "row_id" in fake_client.search_output_fields
    assert "table_title" in fake_client.search_output_fields
    assert results[0]["evidence_type"] == "text_chunk"
    assert results[0]["table_id"] == ""
    assert results[0]["row_id"] == ""
    assert results[0]["table_title"] == ""
    assert results[0]["chunk_id"] == "chunk-2"
    assert results[0]["parent_chunk_id"] == "parent-2"
    assert results[0]["root_chunk_id"] == "root-2"
    assert results[0]["chunk_level"] == 3
