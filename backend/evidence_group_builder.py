from __future__ import annotations

from typing import Iterable


def _clean_text(value) -> str:
    return "" if value is None else str(value).strip()


def _preview_text(value: str, limit: int) -> str:
    text = _clean_text(value).replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _table_row_preview(rows: Iterable[dict], limit: int) -> list[str]:
    preview = []
    for row in list(rows or [])[:limit]:
        if not isinstance(row, dict):
            continue
        parts = []
        for key, value in row.items():
            key_text = _clean_text(key)
            value_text = _clean_text(value)
            if key_text and value_text:
                parts.append(f"{key_text}: {value_text}")
        if parts:
            preview.append("; ".join(parts))
    return preview


def format_evidence_group(
    group: dict,
    *,
    index: int,
    preview_chars: int,
) -> str:
    filename = _clean_text(group.get("filename"))
    page_number = group.get("page_number", "")
    lines = [
        f"[Evidence Group {index}]",
        f"Source: {filename}, page {page_number}",
    ]

    relevant_table_rows = list(group.get("relevant_table_rows") or [])
    if relevant_table_rows:
        lines.append("Relevant table rows:")
        for row in relevant_table_rows:
            table_id = _clean_text(row.get("table_id"))
            row_page_number = row.get("page_number", page_number)
            row_label = _clean_text(row.get("row_label"))
            values_sequence = _clean_text(row.get("values_sequence"))
            columns = row.get("columns") or []
            row_text = _clean_text(row.get("row_text"))
            lines.append(
                f"- Table ID: {table_id} | Page: {row_page_number} | Row Label: {row_label or '(unknown)'}"
            )
            if values_sequence:
                lines.append(f"  Values: {values_sequence}")
            if columns:
                lines.append(f"  Columns: {columns}")
            if row_text:
                lines.append(f"  Row: {_preview_text(row_text, preview_chars)}")

    matched_snippets = list(group.get("matched_snippets") or [])
    if matched_snippets:
        lines.append("Matched snippets:")
        for snippet in matched_snippets:
            lines.append(f"- {_preview_text(snippet, preview_chars)}")

    expanded_snippets = list(group.get("expanded_snippets") or [])
    if expanded_snippets:
        lines.append("Expanded snippets:")
        for snippet in expanded_snippets:
            lines.append(f"- {_preview_text(snippet, preview_chars)}")

    return "\n".join(lines)


def build_group_debug_payload(group: dict) -> dict:
    return {
        "filename": _clean_text(group.get("filename")),
        "page_number": group.get("page_number", ""),
        "group_score": group.get("group_score", 0.0),
        "table_attach_reason": _clean_text(group.get("table_attach_reason")),
        "attached_table_ids": list(group.get("attached_table_ids") or []),
        "matched_queries": list(group.get("matched_queries") or []),
        "planner_sources": list(group.get("planner_sources") or []),
        "matched_snippet_count": len(group.get("matched_snippets") or []),
        "expanded_snippet_count": len(group.get("expanded_snippets") or []),
        "relevant_table_row_count": len(group.get("relevant_table_rows") or []),
    }
