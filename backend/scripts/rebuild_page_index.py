import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from document_loader import DocumentLoader
from document_page_store import DocumentPageStore
from milvus_client import MilvusManager


def main() -> None:
    started_at = time.perf_counter()
    milvus_manager = MilvusManager()
    page_store = DocumentPageStore()
    milvus_manager.init_collection()

    rows = milvus_manager.query_all(
        filter_expr="chunk_level == 3",
        output_fields=["text", "filename", "file_type", "file_path", "page_number", "chunk_id", "chunk_idx"],
    )
    grouped = defaultdict(list)
    for row in rows:
        filename = row.get("filename") or ""
        page_number = int(row.get("page_number", 0) or 0)
        if not filename:
            continue
        grouped[(filename, page_number)].append(row)

    pages = []
    for (filename, page_number), items in grouped.items():
        ordered = sorted(items, key=lambda item: int(item.get("chunk_idx", 0) or 0))
        page_text = "\n".join((item.get("text") or "").strip() for item in ordered if (item.get("text") or "").strip())
        pages.append(
            {
                "doc_name": Path(filename).stem,
                "filename": filename,
                "file_type": ordered[0].get("file_type", "") if ordered else "",
                "file_path": ordered[0].get("file_path", "") if ordered else "",
                "page_number": page_number,
                "page_text": page_text,
                "table_text": DocumentLoader._extract_table_text(page_text),
                "chunk_ids": [item.get("chunk_id", "") for item in ordered if item.get("chunk_id")],
            }
        )

    batch_size = 64
    upserted = 0
    embedding_batches = 0
    for start in range(0, len(pages), batch_size):
        batch = pages[start : start + batch_size]
        upserted += page_store.upsert_pages(batch)
        embedding_batches += 1
        print(
            f"processed_pages={min(start + len(batch), len(pages))}/{len(pages)} "
            f"batch_size={len(batch)} embedding_batch={embedding_batches}"
        )

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    print(
        f"rebuild_page_index_done pages={len(pages)} upserted={upserted} "
        f"embedding_batches={embedding_batches} elapsed_ms={elapsed_ms}"
    )


if __name__ == "__main__":
    main()
