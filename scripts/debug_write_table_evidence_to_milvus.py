"""将表格证据 dry-run 写入独立测试 Milvus collection。"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for path in (ROOT, BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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

from backend.milvus_writer import MilvusWriter
from backend.table_indexer import build_table_evidence_docs
from backend.table_parser import TableAwareParser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将表格证据写入独立测试 Milvus collection")
    parser.add_argument("pdf_path", help="PDF 文件路径")
    parser.add_argument(
        "--backend",
        choices=["auto", "pdfplumber", "pdfplumber_words", "docling"],
        default="pdfplumber_words",
        help="表格解析后端，默认 pdfplumber_words",
    )
    parser.add_argument("--max-pages", type=int, default=None, help="最多解析前多少页")
    parser.add_argument("--max-tables", type=int, default=20, help="最多保留多少张表，默认 20")
    parser.add_argument("--collection", default="table_evidence_test", help="Milvus 测试 collection 名称")
    parser.add_argument("--recreate", action="store_true", help="仅对测试 collection 允许重建")
    return parser.parse_args(argv)


def _normalize_limit(value: int | None, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _normalize_optional_limit(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return None


def is_safe_test_collection_name(collection_name: str) -> bool:
    name = (collection_name or "").strip().lower()
    if not name or name == "embeddings_collection":
        return False
    return any(token in name for token in ("test", "debug", "table_evidence_test"))


def summarize_evidence_types(evidence_docs: list[dict]) -> dict[str, int]:
    counter = Counter()
    for doc in evidence_docs:
        counter[str(doc.get("evidence_type", "") or "")] += 1
    return dict(counter)


def build_report(
    *,
    collection_name: str,
    tables_count: int,
    evidence_docs: list[dict],
) -> str:
    distribution = summarize_evidence_types(evidence_docs)
    lines = [
        f"collection name: {collection_name}",
        f"tables count: {tables_count}",
        f"evidence docs count: {len(evidence_docs)}",
        "evidence_type distribution:",
    ]
    for key, value in sorted(distribution.items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    file_path = Path(args.pdf_path).expanduser().resolve()
    max_pages = _normalize_optional_limit(args.max_pages)
    max_tables = _normalize_limit(args.max_tables, 20)
    collection_name = (args.collection or "table_evidence_test").strip()

    if not file_path.exists():
        print(f"文件不存在: {file_path}", file=sys.stderr)
        return 1
    if file_path.suffix.lower() != ".pdf":
        print(f"仅支持 PDF 文件: {file_path}", file=sys.stderr)
        return 1
    if args.recreate and not is_safe_test_collection_name(collection_name):
        print(
            f"出于安全考虑，--recreate 只允许测试 collection，当前名称不允许: {collection_name}",
            file=sys.stderr,
        )
        return 1

    os.environ["TABLE_AWARE_INGESTION"] = "true"
    os.environ["TABLE_PARSER_BACKEND"] = args.backend
    os.environ["MILVUS_COLLECTION"] = collection_name

    try:
        writer = MilvusWriter()
        if args.recreate:
            manager = writer.milvus_manager
            if manager.has_collection():
                manager.drop_collection()

        parser = TableAwareParser()
        tables = parser.extract_tables(
            str(file_path),
            file_path.name,
            parser_backend=args.backend,
            max_pages=max_pages,
        )
        tables = tables[:max_tables]
        evidence_docs = build_table_evidence_docs(tables)
        writer.write_documents(evidence_docs)
    except Exception as exc:
        print(f"写入测试 collection 失败: {exc}", file=sys.stderr)
        return 1

    print(
        build_report(
            collection_name=collection_name,
            tables_count=len(tables),
            evidence_docs=evidence_docs,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
