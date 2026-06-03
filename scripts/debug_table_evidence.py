"""调试表格文本证据 dry-run 结果。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.table_indexer import build_table_evidence_docs
from backend.table_parser import TableAwareParser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调试表格文本证据构造结果")
    parser.add_argument("pdf_path", help="PDF 文件路径")
    parser.add_argument(
        "--backend",
        choices=["auto", "pdfplumber", "pdfplumber_words", "docling"],
        default=None,
        help="临时覆盖 TABLE_PARSER_BACKEND",
    )
    parser.add_argument("--max-pages", type=int, default=None, help="最多解析前多少页")
    parser.add_argument("--max-tables", type=int, default=20, help="最多展示多少张表，默认 20")
    parser.add_argument("--max-docs", type=int, default=30, help="最多展示多少条证据，默认 30")
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


def _preview_text(value: str, limit: int = 240) -> str:
    text = (value or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def apply_runtime_overrides(args: argparse.Namespace) -> dict:
    overrides = {}
    if args.backend:
        os.environ["TABLE_PARSER_BACKEND"] = args.backend
        overrides["backend"] = args.backend
    return overrides


def resolve_runtime_config(args: argparse.Namespace) -> dict:
    backend = args.backend or os.getenv("TABLE_PARSER_BACKEND", "auto").strip().lower() or "auto"
    if backend not in {"auto", "pdfplumber", "pdfplumber_words", "docling"}:
        backend = "auto"
    return {
        "backend": backend,
        "max_pages": _normalize_optional_limit(args.max_pages),
    }


def build_report(
    file_path: Path,
    tables: list[dict],
    evidence_docs: list[dict],
    *,
    max_tables: int,
    max_docs: int,
    runtime_config: dict,
) -> str:
    lines = [
        f"filename: {file_path.name}",
        f"backend: {runtime_config.get('backend', 'auto')}",
        f"max_pages: {runtime_config.get('max_pages')}",
        f"tables count: {len(tables)}",
        f"evidence docs count: {len(evidence_docs)}",
    ]

    for index, table in enumerate(tables[:max_tables], start=1):
        lines.extend(
            [
                f"table #{index}",
                f"- table_id: {table.get('table_id', '')}",
                f"  page_number: {table.get('page_number', '')}",
                f"  parser_backend: {table.get('parser_backend', '')}",
                f"  accepted: {table.get('accepted', True)}",
                f"  title: {table.get('normalized_title') or table.get('title') or ''}",
            ]
        )
    if len(tables) > max_tables:
        lines.append(f"... {len(tables) - max_tables} more tables not shown")

    for index, doc in enumerate(evidence_docs[:max_docs], start=1):
        lines.extend(
            [
                f"evidence #{index}",
                f"- evidence_type: {doc.get('evidence_type', '')}",
                f"  table_id: {doc.get('table_id', '')}",
                f"  row_id: {doc.get('row_id', '')}",
                f"  page_number: {doc.get('page_number', '')}",
                f"  text_preview: {_preview_text(doc.get('text', ''))}",
            ]
        )
    if len(evidence_docs) > max_docs:
        lines.append(f"... {len(evidence_docs) - max_docs} more evidence docs not shown")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    file_path = Path(args.pdf_path).expanduser().resolve()
    max_tables = _normalize_limit(args.max_tables, 20)
    max_docs = _normalize_limit(args.max_docs, 30)

    if not file_path.exists():
        print(f"文件不存在: {file_path}", file=sys.stderr)
        return 1
    if file_path.suffix.lower() != ".pdf":
        print(f"仅支持 PDF 文件: {file_path}", file=sys.stderr)
        return 1

    if os.getenv("TABLE_AWARE_INGESTION", "").strip().lower() not in {"true", "1", "yes", "on"}:
        print("TABLE_AWARE_INGESTION is not enabled; set TABLE_AWARE_INGESTION=true to parse tables.", file=sys.stderr)
        os.environ["TABLE_AWARE_INGESTION"] = "true"

    apply_runtime_overrides(args)
    runtime_config = resolve_runtime_config(args)

    try:
        parser = TableAwareParser()
        tables = parser.extract_tables(
            str(file_path),
            file_path.name,
            parser_backend=runtime_config["backend"],
            max_pages=runtime_config["max_pages"],
        )
        evidence_docs = build_table_evidence_docs(tables)
    except Exception as exc:
        print(f"表格证据构造失败: {exc}", file=sys.stderr)
        return 1

    print(
        build_report(
            file_path,
            tables,
            evidence_docs,
            max_tables=max_tables,
            max_docs=max_docs,
            runtime_config=runtime_config,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
