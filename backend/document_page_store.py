from datetime import datetime
from typing import List

from cache import cache
from database import SessionLocal, engine
from embedding import embedding_service
from finance_rag_features import build_embedding_cache_key, compute_page_features
from models import DocumentPage
from sqlalchemy import inspect, text
from text_sanitizer import sanitize_text


class DocumentPageStore:
    """Store per-page aggregated text for page-level FinanceBench retrieval."""

    def __init__(self):
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        try:
            inspector = inspect(engine)
            if "document_pages" not in inspector.get_table_names():
                return
            columns = {item["name"] for item in inspector.get_columns("document_pages")}
            alter_sql = {
                "embedding_cache_key": "ALTER TABLE document_pages ADD COLUMN embedding_cache_key VARCHAR(255) NOT NULL DEFAULT ''",
                "page_dense_embedding": "ALTER TABLE document_pages ADD COLUMN page_dense_embedding JSON NOT NULL DEFAULT '[]'",
                "page_tokens": "ALTER TABLE document_pages ADD COLUMN page_tokens JSON NOT NULL DEFAULT '[]'",
                "page_numbers": "ALTER TABLE document_pages ADD COLUMN page_numbers JSON NOT NULL DEFAULT '[]'",
                "page_years": "ALTER TABLE document_pages ADD COLUMN page_years JSON NOT NULL DEFAULT '[]'",
                "page_metric_tokens": "ALTER TABLE document_pages ADD COLUMN page_metric_tokens JSON NOT NULL DEFAULT '[]'",
            }
            for name, sql in alter_sql.items():
                if name in columns:
                    continue
                with engine.begin() as conn:
                    conn.execute(text(sql))
        except Exception:
            return

    @staticmethod
    def _to_dict(item: DocumentPage) -> dict:
        return {
            "doc_name": item.doc_name,
            "filename": item.filename,
            "file_type": item.file_type,
            "file_path": item.file_path,
            "page_number": item.page_number,
            "page_text": item.page_text,
            "table_text": item.table_text,
            "chunk_ids": list(item.chunk_ids or []),
            "embedding_cache_key": item.embedding_cache_key,
            "page_dense_embedding": list(item.page_dense_embedding or []),
            "page_tokens": list(item.page_tokens or []),
            "page_numbers": list(item.page_numbers or []),
            "page_years": list(item.page_years or []),
            "page_metric_tokens": list(item.page_metric_tokens or []),
        }

    @staticmethod
    def _cache_key(filename: str, page_number: int) -> str:
        return f"document_page:{filename}:{page_number}"

    def upsert_pages(self, pages: List[dict]) -> int:
        if not pages:
            return 0

        normalized_pages = []
        page_texts = []
        for page in pages:
            filename = (page.get("filename") or "").strip()
            if not filename:
                continue
            page_number = int(page.get("page_number", 0) or 0)
            page_text = sanitize_text(page.get("page_text", ""))
            table_text = sanitize_text(page.get("table_text", ""))
            features = compute_page_features(page_text, table_text)
            normalized_pages.append(
                {
                    "doc_name": page.get("doc_name", ""),
                    "filename": filename,
                    "file_type": page.get("file_type", ""),
                    "file_path": page.get("file_path", ""),
                    "page_number": page_number,
                    "page_text": page_text,
                    "table_text": table_text,
                    "chunk_ids": list(page.get("chunk_ids") or []),
                    "embedding_cache_key": build_embedding_cache_key(filename, page_number, page_text),
                    **features,
                }
            )
            page_texts.append(page_text)

        if not normalized_pages:
            return 0

        page_embeddings = embedding_service.get_embeddings(page_texts)

        db = SessionLocal()
        upserted = 0
        try:
            for page, page_embedding in zip(normalized_pages, page_embeddings):
                filename = page["filename"]
                page_number = page["page_number"]
                record = (
                    db.query(DocumentPage)
                    .filter(DocumentPage.filename == filename, DocumentPage.page_number == page_number)
                    .first()
                )
                payload = {
                    "doc_name": page["doc_name"],
                    "file_type": page["file_type"],
                    "file_path": page["file_path"],
                    "page_text": page["page_text"],
                    "table_text": page["table_text"],
                    "chunk_ids": page["chunk_ids"],
                    "embedding_cache_key": page["embedding_cache_key"],
                    "page_dense_embedding": list(page_embedding or []),
                    "page_tokens": page["page_tokens"],
                    "page_numbers": page["page_numbers"],
                    "page_years": page["page_years"],
                    "page_metric_tokens": page["page_metric_tokens"],
                    "updated_at": datetime.utcnow(),
                }
                cache_payload = {
                    "doc_name": payload["doc_name"],
                    "filename": filename,
                    "file_type": payload["file_type"],
                    "file_path": payload["file_path"],
                    "page_number": page_number,
                    "page_text": payload["page_text"],
                    "table_text": payload["table_text"],
                    "chunk_ids": payload["chunk_ids"],
                    "embedding_cache_key": payload["embedding_cache_key"],
                    "page_dense_embedding": payload["page_dense_embedding"],
                    "page_tokens": payload["page_tokens"],
                    "page_numbers": payload["page_numbers"],
                    "page_years": payload["page_years"],
                    "page_metric_tokens": payload["page_metric_tokens"],
                }
                if record:
                    for key, value in payload.items():
                        setattr(record, key, value)
                else:
                    db.add(DocumentPage(filename=filename, page_number=page_number, **payload))

                cache.set_json(self._cache_key(filename, page_number), cache_payload)
                upserted += 1

            db.commit()
        finally:
            db.close()

        return upserted

    def get_pages_by_filenames(self, filenames: List[str]) -> List[dict]:
        normalized = [item.strip() for item in filenames if item and item.strip()]
        if not normalized:
            return []

        db = SessionLocal()
        try:
            rows = db.query(DocumentPage).filter(DocumentPage.filename.in_(normalized)).all()
            results = [self._to_dict(row) for row in rows]
            for payload in results:
                cache.set_json(
                    self._cache_key(payload["filename"], int(payload.get("page_number", 0) or 0)),
                    payload,
                )
            return sorted(results, key=lambda item: ((item.get("filename") or "").lower(), int(item.get("page_number", 0) or 0)))
        finally:
            db.close()

    def delete_by_filename(self, filename: str) -> int:
        if not filename:
            return 0

        db = SessionLocal()
        try:
            rows = db.query(DocumentPage).filter(DocumentPage.filename == filename).all()
            deleted = len(rows)
            if deleted > 0:
                db.query(DocumentPage).filter(DocumentPage.filename == filename).delete(synchronize_session=False)
                db.commit()
                for row in rows:
                    cache.delete(self._cache_key(row.filename, row.page_number))
            return deleted
        finally:
            db.close()
