"""构建 table evidence 命中后的完整表格上下文预览。"""

from __future__ import annotations

from typing import Iterable


def _clean_text(value) -> str:
    return "" if value is None else str(value).strip()


def _preview_text(value: str, limit: int) -> str:
    text = _clean_text(value).replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def dedupe_table_ids(results: list[dict]) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in results or []:
        table_id = _clean_text(item.get("table_id"))
        if not table_id or table_id in seen:
            continue
        seen.add(table_id)
        out.append(table_id)
    return out


def group_hits_by_table_id(results: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for item in results or []:
        table_id = _clean_text(item.get("table_id"))
        if not table_id:
            continue
        grouped.setdefault(table_id, []).append(item)
    return grouped


def fetch_tables_for_results(results: list[dict], table_store) -> list[dict]:
    table_ids = dedupe_table_ids(results)
    if not table_ids:
        return []
    return table_store.get_tables_by_ids(table_ids)


def _row_preview(rows: Iterable[dict], max_rows: int) -> list[str]:
    preview = []
    for row in list(rows or [])[:max_rows]:
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


def format_table_preview(
    table: dict,
    *,
    preview_rows: int,
    preview_chars: int,
    include_rows: bool = True,
    include_csv: bool = True,
) -> str:
    table_id = _clean_text(table.get("table_id"))
    filename = _clean_text(table.get("filename"))
    page_number = table.get("page_number", "")
    title = _clean_text(table.get("title")) or _clean_text(table.get("caption"))
    columns = table.get("columns") or []
    rows_preview = _row_preview(table.get("rows") or [], preview_rows)
    csv_preview = _preview_text(_clean_text(table.get("csv_text")), preview_chars)

    lines = [
        f"Document: {filename}",
        f"Page: {page_number}",
        f"Table ID: {table_id}",
    ]
    if title:
        lines.append(f"Title: {title}")
    lines.append(f"columns: {columns}")
    if include_rows and rows_preview:
        lines.append("rows:")
        lines.extend(rows_preview)
    if include_csv and csv_preview:
        lines.append("csv_text:")
        lines.append(csv_preview)
    return "\n".join(lines)


def format_matched_evidence_hits(hits: list[dict], *, preview_chars: int) -> str:
    lines = []
    for hit in hits or []:
        lines.extend(
            [
                f"- score: {hit.get('score', 0.0)}",
                f"  evidence_type: {_clean_text(hit.get('evidence_type'))}",
                f"  row_id: {_clean_text(hit.get('row_id'))}",
                f"  page_number: {hit.get('page_number', '')}",
                f"  text_preview: {_preview_text(_clean_text(hit.get('text')), preview_chars)}",
            ]
        )
    return "\n".join(lines)


def build_table_context_preview(
    results: list[dict],
    tables: list[dict],
    *,
    preview_rows: int,
    preview_chars: int,
    full_table_ids: set[str] | None = None,
    skipped_table_reasons: dict[str, str] | None = None,
) -> str:
    grouped_hits = group_hits_by_table_id(results)
    lines = []
    for table in tables or []:
        table_id = _clean_text(table.get("table_id"))
        hits = grouped_hits.get(table_id, [])
        include_full = full_table_ids is None or table_id in full_table_ids
        skipped_reason = _clean_text((skipped_table_reasons or {}).get(table_id))
        lines.extend(
            [
                "[Table Evidence]",
                format_table_preview(
                    table,
                    preview_rows=preview_rows,
                    preview_chars=preview_chars,
                    include_rows=False,
                    include_csv=False,
                ),
                "Matched Evidence:",
                format_matched_evidence_hits(hits, preview_chars=preview_chars) or "(none)",
                "Full Table:",
            ]
        )
        if include_full:
            lines.append(
                format_table_preview(
                    table,
                    preview_rows=preview_rows,
                    preview_chars=preview_chars,
                    include_rows=True,
                    include_csv=True,
                )
            )
        else:
            lines.append(f"(skipped: {skipped_reason or 'table_quality_rejected'})")
    return "\n".join(lines)


def truncate_table_context(text: str, max_chars: int) -> str:
    cleaned = _clean_text(text)
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    marker = "\n... table evidence truncated ..."
    if max_chars <= len(marker):
        return marker[:max_chars]
    return cleaned[: max_chars - len(marker)] + marker
