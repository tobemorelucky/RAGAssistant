from datetime import datetime
from typing import List

from cache import cache
from database import SessionLocal
from models import DocumentPage
from text_sanitizer import sanitize_text


class DocumentPageStore:
    """Store per-page aggregated text for page-level FinanceBench retrieval."""

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
        }

    @staticmethod
    def _cache_key(filename: str, page_number: int) -> str:
        return f"document_page:{filename}:{page_number}"

    def upsert_pages(self, pages: List[dict]) -> int:
        if not pages:
            return 0

        db = SessionLocal()
        upserted = 0
        try:
            for page in pages:
                filename = (page.get("filename") or "").strip()
                if not filename:
                    continue

                page_number = int(page.get("page_number", 0) or 0)
                record = (
                    db.query(DocumentPage)
                    .filter(DocumentPage.filename == filename, DocumentPage.page_number == page_number)
                    .first()
                )
                payload = {
                    "doc_name": page.get("doc_name", ""),
                    "file_type": page.get("file_type", ""),
                    "file_path": page.get("file_path", ""),
                    "page_text": sanitize_text(page.get("page_text", "")),
                    "table_text": sanitize_text(page.get("table_text", "")),
                    "chunk_ids": list(page.get("chunk_ids") or []),
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
