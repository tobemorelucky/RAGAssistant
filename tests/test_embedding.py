import importlib.util
import sys
import types
from pathlib import Path


def _load_embedding_module():
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv

    langchain_huggingface = types.ModuleType("langchain_huggingface")

    class _StubHuggingFaceEmbeddings:
        def __init__(self, *args, **kwargs):
            pass

        def embed_documents(self, texts):
            return [[float(index)] for index, _ in enumerate(texts)]

    langchain_huggingface.HuggingFaceEmbeddings = _StubHuggingFaceEmbeddings
    sys.modules["langchain_huggingface"] = langchain_huggingface

    path = Path(__file__).resolve().parents[1] / "backend" / "embedding.py"
    spec = importlib.util.spec_from_file_location("embedding_test_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class _FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[text] for text in texts]


def _build_service(module, embedder):
    service = object.__new__(module.EmbeddingService)
    service._embedder = embedder
    return service


def test_get_embeddings_empty_returns_empty(monkeypatch):
    module = _load_embedding_module()
    service = _build_service(module, _FakeEmbedder())

    monkeypatch.delenv("EMBEDDING_BATCH_SIZE", raising=False)

    assert service.get_embeddings([]) == []


def test_get_embeddings_batches_and_preserves_order(monkeypatch):
    module = _load_embedding_module()
    embedder = _FakeEmbedder()
    service = _build_service(module, embedder)
    texts = [f"text-{index}" for index in range(10)]

    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "4")

    embeddings = service.get_embeddings(texts)

    assert len(embedder.calls) == 3
    assert embedder.calls == [texts[0:4], texts[4:8], texts[8:10]]
    assert embeddings == [[text] for text in texts]


def test_invalid_embedding_batch_size_falls_back_to_default(monkeypatch):
    module = _load_embedding_module()
    embedder = _FakeEmbedder()
    service = _build_service(module, embedder)
    texts = [f"text-{index}" for index in range(10)]

    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "abc")

    embeddings = service.get_embeddings(texts)

    assert len(embedder.calls) == 2
    assert embedder.calls == [texts[0:8], texts[8:10]]
    assert embeddings == [[text] for text in texts]


def test_non_positive_embedding_batch_size_falls_back_to_default(monkeypatch):
    module = _load_embedding_module()
    embedder = _FakeEmbedder()
    service = _build_service(module, embedder)
    texts = [f"text-{index}" for index in range(10)]

    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "0")

    embeddings = service.get_embeddings(texts)

    assert len(embedder.calls) == 2
    assert embedder.calls == [texts[0:8], texts[8:10]]
    assert embeddings == [[text] for text in texts]


def test_batch_failure_includes_batch_range(monkeypatch):
    module = _load_embedding_module()

    class _FailingEmbedder:
        def __init__(self):
            self.calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("oom")
            return [[text] for text in texts]

    service = _build_service(module, _FailingEmbedder())
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "4")

    try:
        service.get_embeddings([f"text-{index}" for index in range(10)])
        assert False, "expected batch failure"
    except Exception as exc:
        message = str(exc)
        assert "batch=5-8" in message
        assert "oom" in message
