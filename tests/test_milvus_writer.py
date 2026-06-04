import importlib
import sys
import types


class _FakeEmbeddingService:
    def __init__(self):
        self.increment_calls = []
        self.embedding_calls = []

    def increment_add_documents(self, texts):
        self.increment_calls.append(list(texts))

    def get_all_embeddings(self, texts):
        self.embedding_calls.append(list(texts))
        dense = [[0.1, 0.2] for _ in texts]
        sparse = [{0: 1.0} for _ in texts]
        return dense, sparse


class _FakeMilvusManager:
    def __init__(self):
        self.init_called = 0
        self.insert_calls = []

    def init_collection(self):
        self.init_called += 1

    def insert(self, data):
        self.insert_calls.append(data)


def _load_milvus_writer_module():
    fake_embedding_module = types.ModuleType("embedding")
    fake_embedding_module.EmbeddingService = _FakeEmbeddingService
    fake_embedding_module.embedding_service = _FakeEmbeddingService()
    sys.modules["embedding"] = fake_embedding_module

    fake_milvus_client_module = types.ModuleType("milvus_client")
    fake_milvus_client_module.MilvusManager = _FakeMilvusManager
    sys.modules["milvus_client"] = fake_milvus_client_module

    module = importlib.import_module("backend.milvus_writer")
    return importlib.reload(module)


def test_write_documents_defaults_text_chunk_metadata():
    module = _load_milvus_writer_module()
    embedding_service = _FakeEmbeddingService()
    milvus_manager = _FakeMilvusManager()
    writer = module.MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)

    writer.write_documents(
        [
            {
                "text": "Revenue increased.",
                "filename": "demo.pdf",
                "file_type": "PDF",
                "chunk_id": "chunk-1",
                "parent_chunk_id": "parent-1",
                "root_chunk_id": "root-1",
                "chunk_level": 3,
            }
        ]
    )

    inserted = milvus_manager.insert_calls[0][0]
    assert inserted["evidence_type"] == "text_chunk"
    assert inserted["table_id"] == ""
    assert inserted["row_id"] == ""
    assert inserted["table_title"] == ""
    assert inserted["chunk_id"] == "chunk-1"
    assert inserted["parent_chunk_id"] == "parent-1"
    assert inserted["root_chunk_id"] == "root-1"
    assert inserted["chunk_level"] == 3


def test_write_documents_keeps_table_evidence_metadata():
    module = _load_milvus_writer_module()
    embedding_service = _FakeEmbeddingService()
    milvus_manager = _FakeMilvusManager()
    writer = module.MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)

    writer.write_documents(
        [
            {
                "text": "Document: demo.pdf\nPage: 3\nTable ID: demo::1",
                "filename": "demo.pdf",
                "file_type": "PDF",
                "evidence_type": "table_row",
                "table_id": "demo::1",
                "row_id": "row_2",
                "table_title": "Condensed Statements",
                "chunk_id": "demo::1::row::row_2",
                "parent_chunk_id": "",
                "root_chunk_id": "",
                "chunk_level": 3,
            }
        ]
    )

    inserted = milvus_manager.insert_calls[0][0]
    assert inserted["evidence_type"] == "table_row"
    assert inserted["table_id"] == "demo::1"
    assert inserted["row_id"] == "row_2"
    assert inserted["table_title"] == "Condensed Statements"
    assert inserted["chunk_id"] == "demo::1::row::row_2"


def test_write_documents_truncates_long_text_and_uses_trimmed_text_for_embeddings(monkeypatch):
    module = _load_milvus_writer_module()
    embedding_service = _FakeEmbeddingService()
    milvus_manager = _FakeMilvusManager()
    writer = module.MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)
    long_text = "A" * 30

    monkeypatch.setenv("MILVUS_TEXT_MAX_LENGTH", "20")

    writer.write_documents(
        [
            {
                "text": long_text,
                "filename": "demo.pdf",
                "file_type": "PDF",
                "chunk_id": "chunk-1",
                "parent_chunk_id": "",
                "root_chunk_id": "",
                "chunk_level": 3,
            }
        ]
    )

    expected_text = "AAAA ... [truncated]"
    assert embedding_service.increment_calls == [[expected_text]]
    assert embedding_service.embedding_calls == [[expected_text]]
    inserted = milvus_manager.insert_calls[0][0]
    assert inserted["text"] == expected_text


def test_write_documents_keeps_short_text_unchanged(monkeypatch):
    module = _load_milvus_writer_module()
    embedding_service = _FakeEmbeddingService()
    milvus_manager = _FakeMilvusManager()
    writer = module.MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)

    monkeypatch.setenv("MILVUS_TEXT_MAX_LENGTH", "20")

    writer.write_documents(
        [
            {
                "text": "Short text",
                "filename": "demo.pdf",
                "file_type": "PDF",
                "chunk_id": "chunk-1",
                "parent_chunk_id": "",
                "root_chunk_id": "",
                "chunk_level": 3,
            }
        ]
    )

    assert embedding_service.increment_calls == [["Short text"]]
    assert embedding_service.embedding_calls == [["Short text"]]
    inserted = milvus_manager.insert_calls[0][0]
    assert inserted["text"] == "Short text"
