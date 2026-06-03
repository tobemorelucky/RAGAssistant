"""将已抽取的表格转换为仅用于调试预览的文本证据。"""

from __future__ import annotations

from typing import Iterable

try:
    from text_sanitizer import sanitize_text
except ModuleNotFoundError:
    from backend.text_sanitizer import sanitize_text


def _clean_text(value) -> str:
    return sanitize_text("" if value is None else str(value)).strip()


def _coerce_list(values) -> list:
    if isinstance(values, list):
        return values
    return []


def _get_table_title(table: dict) -> str:
    return (
        _clean_text(table.get("normalized_title"))
        or _clean_text(table.get("title"))
        or _clean_text(table.get("caption"))
    )


def _get_columns(table: dict) -> list[str]:
    columns = _coerce_list(table.get("normalized_columns"))
    if columns:
        return [_clean_text(column) for column in columns]
    return [_clean_text(column) for column in _coerce_list(table.get("columns"))]


def _stringify_row(row: dict, columns: list[str]) -> str:
    parts = []
    ordered_columns = columns or list(row.keys())
    for column in ordered_columns:
        value = _clean_text(row.get(column, ""))
        if value:
            parts.append(f"{column}: {value}")
    if not parts:
        for key, value in row.items():
            cleaned_value = _clean_text(value)
            if cleaned_value:
                parts.append(f"{_clean_text(key)}: {cleaned_value}")
    return "; ".join(parts)


def _preview_items(items: Iterable[str], limit: int) -> str:
    preview = [_clean_text(item) for item in items if _clean_text(item)]
    if not preview:
        return ""
    return " | ".join(preview[:limit])


def _base_evidence_doc(table: dict, evidence_type: str, row_id: str = "") -> dict:
    title = _get_table_title(table)
    return {
        "text": "",
        "evidence_type": evidence_type,
        "table_id": _clean_text(table.get("table_id")),
        "row_id": row_id,
        "filename": _clean_text(table.get("filename")),
        "page_number": int(table.get("page_number", 0) or 0),
        "table_title": title,
        "chunk_level": 3,
        "parent_chunk_id": "",
        "root_chunk_id": "",
    }


def _build_summary_text(table: dict) -> str:
    filename = _clean_text(table.get("filename"))
    page_number = int(table.get("page_number", 0) or 0)
    table_id = _clean_text(table.get("table_id"))
    title = _get_table_title(table)
    columns = _get_columns(table)
    row_preview = _preview_items(
        (
            _stringify_row(row, columns)
            for row in _coerce_list(table.get("normalized_rows"))[:3]
        ),
        3,
    )
    if not row_preview:
        row_preview = _preview_items(
            (
                _stringify_row(row, columns)
                for row in _coerce_list(table.get("rows"))[:3]
            ),
            3,
        )
    raw_lines_preview = _preview_items(_coerce_list(table.get("raw_lines"))[:3], 3)
    parts = [
        f"Document: {filename}",
        f"Page: {page_number}",
        f"Table ID: {table_id}",
        "Evidence Type: table_summary",
    ]
    if title:
        parts.append(f"Title: {title}")
    if columns:
        parts.append(f"Columns: {' | '.join(columns)}")
    if row_preview:
        parts.append(f"Row Preview: {row_preview}")
    if raw_lines_preview:
        parts.append(f"Raw Lines: {raw_lines_preview}")
    return "\n".join(parts)


def _build_row_text(table: dict, row: dict, row_index: int, columns: list[str]) -> str:
    filename = _clean_text(table.get("filename"))
    page_number = int(table.get("page_number", 0) or 0)
    table_id = _clean_text(table.get("table_id"))
    title = _get_table_title(table)
    row_values = _stringify_row(row, columns)
    parts = [
        f"Document: {filename}",
        f"Page: {page_number}",
        f"Table ID: {table_id}",
        f"Row ID: row_{row_index}",
        "Evidence Type: table_row",
    ]
    if title:
        parts.append(f"Title: {title}")
    if columns:
        parts.append(f"Columns: {' | '.join(columns)}")
    if row_values:
        parts.append(f"Row Values: {row_values}")
    return "\n".join(parts)


def _build_raw_text(table: dict, line: str, row_index: int, columns: list[str]) -> str:
    filename = _clean_text(table.get("filename"))
    page_number = int(table.get("page_number", 0) or 0)
    table_id = _clean_text(table.get("table_id"))
    title = _get_table_title(table)
    parts = [
        f"Document: {filename}",
        f"Page: {page_number}",
        f"Table ID: {table_id}",
        f"Row ID: raw_{row_index}",
        "Evidence Type: table_raw",
    ]
    if title:
        parts.append(f"Title: {title}")
    if columns:
        parts.append(f"Columns: {' | '.join(columns)}")
    parts.append(f"Raw Line: {_clean_text(line)}")
    return "\n".join(parts)


def build_table_evidence_docs(tables: list[dict]) -> list[dict]:
    """将 accepted table 转成仅用于 dry-run 预览的文本证据。"""

    evidence_docs: list[dict] = []
    for table in tables or []:
        if table.get("accepted", True) is False:
            continue

        columns = _get_columns(table)

        summary_doc = _base_evidence_doc(table, "table_summary")
        summary_doc["text"] = _build_summary_text(table)
        evidence_docs.append(summary_doc)

        normalized_rows = _coerce_list(table.get("normalized_rows"))
        if normalized_rows:
            for row_index, row in enumerate(normalized_rows, start=1):
                if not isinstance(row, dict):
                    continue
                row_doc = _base_evidence_doc(table, "table_row", row_id=f"row_{row_index}")
                row_doc["text"] = _build_row_text(table, row, row_index, columns)
                evidence_docs.append(row_doc)
            continue

        raw_lines = [line for line in _coerce_list(table.get("raw_lines")) if _clean_text(line)]
        if not raw_lines:
            raw_rows = _coerce_list(table.get("rows"))
            raw_lines = [
                _stringify_row(row, columns)
                for row in raw_rows
                if isinstance(row, dict) and _stringify_row(row, columns)
            ]

        for row_index, line in enumerate(raw_lines, start=1):
            raw_doc = _base_evidence_doc(table, "table_raw", row_id=f"raw_{row_index}")
            raw_doc["text"] = _build_raw_text(table, line, row_index, columns)
            evidence_docs.append(raw_doc)

    return evidence_docs
