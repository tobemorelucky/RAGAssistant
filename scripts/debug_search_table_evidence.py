"""在独立测试 Milvus collection 中调试搜索表格证据。"""

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在独立测试 Milvus collection 中搜索表格证据")
    parser.add_argument("query", help="查询文本")
    parser.add_argument("--collection", default="table_evidence_test", help="Milvus collection 名称")
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数，默认 5")
    return parser.parse_args(argv)


def _normalize_limit(value: int | None, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _preview_text(value: str, limit: int = 240) -> str:
    text = (value or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def build_report(query: str, collection_name: str, results: list[dict]) -> str:
    lines = [
        f"query: {query}",
        f"collection name: {collection_name}",
        f"results count: {len(results)}",
    ]
    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                f"result #{index}",
                f"- score: {result.get('score', 0.0)}",
                f"  evidence_type: {result.get('evidence_type', '')}",
                f"  table_id: {result.get('table_id', '')}",
                f"  row_id: {result.get('row_id', '')}",
                f"  page_number: {result.get('page_number', '')}",
                f"  table_title: {result.get('table_title', '')}",
                f"  text_preview: {_preview_text(result.get('text', ''))}",
            ]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    top_k = _normalize_limit(args.top_k, 5)
    collection_name = (args.collection or "table_evidence_test").strip()
    query = (args.query or "").strip()

    if not query:
        print("查询不能为空", file=sys.stderr)
        return 1

    os.environ["MILVUS_COLLECTION"] = collection_name

    try:
        dense_embedding, sparse_embedding = embedding_service.get_all_embeddings([query])
        manager = MilvusManager()
        results = manager.hybrid_retrieve(
            dense_embedding=dense_embedding[0],
            sparse_embedding=sparse_embedding[0],
            top_k=top_k,
        )
    except Exception as exc:
        print(f"搜索测试 collection 失败: {exc}", file=sys.stderr)
        return 1

    print(build_report(query, collection_name, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
