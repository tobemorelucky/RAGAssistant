"""调试 TableAwareParser 在真实 PDF 上的表格解析效果。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.table_parser import TableAwareParser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调试 PDF 表格解析结果")
    parser.add_argument("pdf_path", help="PDF 文件路径")
    parser.add_argument("--max-tables", type=int, default=5, help="最多展示多少张表，默认 5")
    parser.add_argument("--max-rows", type=int, default=5, help="每张表最多展示多少行，默认 5")
    parser.add_argument(
        "--backend",
        choices=["auto", "pdfplumber", "pdfplumber_words", "docling"],
        default=None,
        help="临时覆盖 TABLE_PARSER_BACKEND",
    )
    parser.add_argument(
        "--docling-ocr",
        dest="docling_ocr",
        action="store_true",
        help="临时开启 Docling OCR",
    )
    parser.add_argument(
        "--no-docling-ocr",
        dest="docling_ocr",
        action="store_false",
        help="临时关闭 Docling OCR",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="临时覆盖 TABLE_DOCLING_TIMEOUT_SECONDS",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="最多解析前多少页，默认不限制",
    )
    parser.add_argument(
        "--show-rejected",
        action="store_true",
        help="鏄剧ず rejected 琛ㄦ牸鍊欓€夌殑棰勮",
    )
    parser.set_defaults(docling_ocr=None)
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
    if args.docling_ocr is not None:
        os.environ["TABLE_DOCLING_OCR"] = "true" if args.docling_ocr else "false"
        overrides["docling_ocr"] = args.docling_ocr
    if args.timeout_seconds is not None:
        os.environ["TABLE_DOCLING_TIMEOUT_SECONDS"] = str(max(1, args.timeout_seconds))
        overrides["timeout_seconds"] = max(1, args.timeout_seconds)
    return overrides


def resolve_runtime_config(args: argparse.Namespace) -> dict:
    backend = args.backend or os.getenv("TABLE_PARSER_BACKEND", "auto").strip().lower() or "auto"
    if backend not in {"auto", "pdfplumber", "pdfplumber_words", "docling"}:
        backend = "auto"

    env_ocr = os.getenv("TABLE_DOCLING_OCR", "").strip().lower()
    if args.docling_ocr is not None:
        docling_ocr = bool(args.docling_ocr)
    else:
        docling_ocr = env_ocr in {"true", "1", "yes", "on"}

    try:
        timeout_seconds = max(1, int(args.timeout_seconds if args.timeout_seconds is not None else os.getenv("TABLE_DOCLING_TIMEOUT_SECONDS", "120")))
    except (TypeError, ValueError):
        timeout_seconds = 120

    return {
        "backend": backend,
        "docling_ocr": docling_ocr,
        "timeout_seconds": timeout_seconds,
        "max_pages": _normalize_optional_limit(args.max_pages),
    }


def _format_table_block(table: dict, max_rows: int) -> str:
    preview_rows = list(table.get("rows") or [])[:max_rows]
    lines = [
        f"- parser_backend: {table.get('parser_backend', '')}",
        f"  accepted: {table.get('accepted', True)}",
        f"  quality_score: {table.get('quality_score', 0.0)}",
        f"  reject_reason: {table.get('reject_reason', '')}",
        f"  numeric_cell_ratio: {table.get('numeric_cell_ratio', 0.0)}",
        f"  non_empty_cell_ratio: {table.get('non_empty_cell_ratio', 0.0)}",
        f"  effective_col_count: {table.get('effective_col_count', 0)}",
        f"  data_row_count: {table.get('data_row_count', 0)}",
        f"  page_number: {table.get('page_number', '')}",
        f"  table_index: {table.get('table_index', '')}",
        f"  table_id: {table.get('table_id', '')}",
        f"  title: {table.get('title', '')}",
        f"  caption: {table.get('caption', '')}",
        f"  columns: {table.get('columns', [])}",
        f"  rows_preview({len(preview_rows)}): {preview_rows}",
        f"  csv_text_preview: {_preview_text(table.get('csv_text', ''))}",
    ]
    return "\n".join(lines)


def build_report(
    file_path: Path,
    tables: list[dict],
    max_tables: int,
    max_rows: int,
    runtime_config: dict,
    *,
    show_rejected: bool = False,
) -> str:
    accepted_tables = [table for table in tables if table.get("accepted", True)]
    rejected_tables = [table for table in tables if not table.get("accepted", True)]
    lines = [
        f"filename: {file_path.name}",
        f"backend: {runtime_config.get('backend', 'auto')}",
        f"docling_ocr: {runtime_config.get('docling_ocr', False)}",
        f"timeout_seconds: {runtime_config.get('timeout_seconds', 120)}",
        f"max_pages: {runtime_config.get('max_pages')}",
        f"raw candidates: {len(tables)}",
        f"accepted tables: {len(accepted_tables)}",
        f"rejected tables: {len(rejected_tables)}",
    ]
    for index, table in enumerate(accepted_tables[:max_tables], start=1):
        lines.append(f"accepted table #{index}")
        lines.append(_format_table_block(table, max_rows))
    if len(accepted_tables) > max_tables:
        lines.append(f"... {len(accepted_tables) - max_tables} more accepted tables not shown")
    if show_rejected and rejected_tables:
        for index, table in enumerate(rejected_tables[:max_tables], start=1):
            lines.append(f"rejected table #{index}")
            lines.append(_format_table_block(table, max_rows))
        if len(rejected_tables) > max_tables:
            lines.append(f"... {len(rejected_tables) - max_tables} more rejected tables not shown")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    file_path = Path(args.pdf_path).expanduser().resolve()
    max_tables = _normalize_limit(args.max_tables, 5)
    max_rows = _normalize_limit(args.max_rows, 5)

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
            docling_ocr=runtime_config["docling_ocr"],
            timeout_seconds=runtime_config["timeout_seconds"],
            max_pages=runtime_config["max_pages"],
            include_rejected=True,
        )
    except Exception as exc:
        print(f"表格解析失败: {exc}", file=sys.stderr)
        return 1

    print(
        build_report(
            file_path,
            tables,
            max_tables=max_tables,
            max_rows=max_rows,
            runtime_config=runtime_config,
            show_rejected=args.show_rejected,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
