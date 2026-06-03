"""调试 table evidence 检索后回捞完整表格上下文。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            continue


_configure_console_encoding()


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.embedding import embedding_service
from backend.milvus_client import MilvusManager
from backend.table_context_builder import (
    build_table_context_preview,
    dedupe_table_ids,
    fetch_tables_for_results,
    format_matched_evidence_hits,
    format_table_preview,
    group_hits_by_table_id,
)
from backend.table_store import TableStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调试 table evidence 检索后的完整表格上下文")
    parser.add_argument("query", help="查询文本")
    parser.add_argument(
        "--collection",
        default=(os.getenv("MILVUS_COLLECTION", "").strip() or "table_upload_test"),
        help="Milvus collection 名称",
    )
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数，默认 5")
    parser.add_argument("--preview-rows", type=int, default=5, help="完整表格预览行数，默认 5")
    parser.add_argument("--preview-chars", type=int, default=1200, help="文本预览最大字符数，默认 1200")
    return parser.parse_args(argv)


def _normalize_limit(value: int | None, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def build_report(
    query: str,
    results: list[dict],
    tables: list[dict],
    *,
    preview_rows: int,
    preview_chars: int,
) -> str:
    grouped_hits = group_hits_by_table_id(results)
    unique_table_ids = dedupe_table_ids(results)
    lines = [
        f"query: {query}",
        f"retrieved table evidence count: {len(results)}",
        f"unique table_id count: {len(unique_table_ids)}",
        f"fetched table count: {len(tables)}",
    ]

    for table_id in unique_table_ids:
        lines.append(f"table_id: {table_id}")
        hits = grouped_hits.get(table_id, [])
        if hits:
            lines.append("Matched Evidence:")
            lines.append(format_matched_evidence_hits(hits, preview_chars=preview_chars))

    for table in tables:
        lines.append("[Full Table Preview]")
        lines.append(format_table_preview(table, preview_rows=preview_rows, preview_chars=preview_chars))

    lines.append("[Table Context Preview]")
    lines.append(
        build_table_context_preview(
            results,
            tables,
            preview_rows=preview_rows,
            preview_chars=preview_chars,
        )
        or "(no table context)"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    query = (args.query or "").strip()
    collection_name = (args.collection or "").strip() or "table_upload_test"
    top_k = _normalize_limit(args.top_k, 5)
    preview_rows = _normalize_limit(args.preview_rows, 5)
    preview_chars = _normalize_limit(args.preview_chars, 1200)

    if not query:
        print("查询不能为空", file=sys.stderr)
        return 1

    os.environ["MILVUS_COLLECTION"] = collection_name

    try:
        dense_embeddings, sparse_embeddings = embedding_service.get_all_embeddings([query])
        manager = MilvusManager()
        table_store = TableStore()
        results = manager.hybrid_retrieve(
            dense_embedding=dense_embeddings[0],
            sparse_embedding=sparse_embeddings[0],
            top_k=top_k,
            filter_expr='evidence_type != "text_chunk"',
        )
        tables = fetch_tables_for_results(results, table_store)
    except Exception as exc:
        print(f"table context retrieval 失败: {exc}", file=sys.stderr)
        return 1

    print(
        build_report(
            query,
            results,
            tables,
            preview_rows=preview_rows,
            preview_chars=preview_chars,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
