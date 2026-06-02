"""基于 pdfplumber word 坐标的轻量表格重建。"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
from statistics import median

try:
    from text_sanitizer import sanitize_text
except ModuleNotFoundError:
    from backend.text_sanitizer import sanitize_text

logger = logging.getLogger(__name__)

_METRIC_HINTS = {
    "revenue",
    "sales",
    "margin",
    "gross",
    "operating",
    "income",
    "eps",
    "ebitda",
    "capex",
    "cash",
    "assets",
    "liabilities",
    "equity",
    "inventory",
    "tax",
    "rate",
    "ratio",
    "shares",
}

_YEAR_PATTERN = re.compile(r"\b(?:fy(?:19|20)?\d{2}|(?:19|20)\d{2})\b", re.IGNORECASE)


def _load_pdfplumber():
    try:
        import pdfplumber
    except Exception:
        return None
    return pdfplumber


def _normalize_word(word: dict) -> dict | None:
    text = sanitize_text(word.get("text", "")).strip()
    if not text:
        return None
    try:
        x0 = float(word.get("x0", 0.0) or 0.0)
        x1 = float(word.get("x1", 0.0) or 0.0)
        top = float(word.get("top", 0.0) or 0.0)
        bottom = float(word.get("bottom", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    return {
        "text": text,
        "x0": x0,
        "x1": x1,
        "top": top,
        "bottom": bottom,
        "x_center": (x0 + x1) / 2.0,
        "y_center": (top + bottom) / 2.0,
        "width": max(0.0, x1 - x0),
        "height": max(0.0, bottom - top),
    }


def _cluster_rows(words: list[dict]) -> list[list[dict]]:
    if not words:
        return []
    sorted_words = sorted(words, key=lambda item: (item["y_center"], item["x0"]))
    heights = [item["height"] for item in sorted_words if item["height"] > 0]
    row_tolerance = max(3.0, (median(heights) * 0.6) if heights else 4.0)

    rows: list[list[dict]] = []
    current_row: list[dict] = []
    current_y: float | None = None
    for word in sorted_words:
        if current_y is None or abs(word["y_center"] - current_y) <= row_tolerance:
            current_row.append(word)
            current_y = word["y_center"] if current_y is None else ((current_y * (len(current_row) - 1)) + word["y_center"]) / len(current_row)
        else:
            rows.append(sorted(current_row, key=lambda item: item["x0"]))
            current_row = [word]
            current_y = word["y_center"]
    if current_row:
        rows.append(sorted(current_row, key=lambda item: item["x0"]))
    return rows


def _row_signal_score(row: list[dict]) -> int:
    texts = [item["text"] for item in row]
    joined = " ".join(texts).lower()
    year_count = len(_YEAR_PATTERN.findall(joined))
    number_count = len(re.findall(r"(?<![A-Za-z])[-+]?[$]?\d[\d,]*(?:\.\d+)?%?", joined))
    metric_count = sum(1 for token in re.findall(r"[a-zA-Z]+", joined) if token in _METRIC_HINTS)
    return year_count * 3 + number_count * 2 + metric_count


def _row_header_score(row: list[dict]) -> int:
    joined = " ".join(item["text"] for item in row)
    year_count = len(_YEAR_PATTERN.findall(joined))
    return year_count * 3 + len(row)


def _select_candidate_blocks(rows: list[list[dict]]) -> list[list[list[dict]]]:
    if not rows:
        return []

    signals = [_row_signal_score(row) for row in rows]
    blocks: list[list[list[dict]]] = []
    current_block: list[list[dict]] = []
    gap_count = 0

    for row, score in zip(rows, signals):
        is_candidate = score >= 2 or (score >= 1 and len(row) >= 3)
        if is_candidate:
            current_block.append(row)
            gap_count = 0
            continue
        if current_block and gap_count == 0 and len(row) >= 2:
            current_block.append(row)
            gap_count += 1
            continue
        if len(current_block) >= 2:
            blocks.append(current_block)
        current_block = []
        gap_count = 0

    if len(current_block) >= 2:
        blocks.append(current_block)

    if blocks:
        return blocks

    dense_rows = [row for row in rows if len(row) >= 3]
    if len(dense_rows) >= 2:
        return [dense_rows]
    return []


def _cluster_columns(rows: list[list[dict]]) -> list[float]:
    if rows:
        header_row = max(rows, key=_row_header_score)
        if _row_header_score(header_row) > 0 and len(header_row) >= 2:
            return [word["x0"] for word in header_row]

    x_positions = sorted(word["x0"] for row in rows for word in row)
    if not x_positions:
        return []
    gaps = [x_positions[index] - x_positions[index - 1] for index in range(1, len(x_positions))]
    positive_gaps = [gap for gap in gaps if gap > 0]
    tolerance = max(12.0, (median(positive_gaps) * 0.6) if positive_gaps else 18.0)

    clusters: list[list[float]] = [[x_positions[0]]]
    for position in x_positions[1:]:
        if abs(position - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(position)
        else:
            clusters.append([position])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _row_to_cells(row: list[dict], column_positions: list[float]) -> list[str]:
    if not column_positions:
        return []
    cells = [""] * len(column_positions)
    for word in row:
        column_index = min(range(len(column_positions)), key=lambda index: abs(word["x0"] - column_positions[index]))
        cells[column_index] = f"{cells[column_index]} {word['text']}".strip() if cells[column_index] else word["text"]
    return [sanitize_text(cell).strip() for cell in cells]


def _dedupe_columns(columns: list[str]) -> list[str]:
    deduped = []
    seen = {}
    for index, column in enumerate(columns, start=1):
        name = sanitize_text(column).strip() or f"column_{index}"
        count = seen.get(name, 0)
        deduped.append(f"{name}_{count + 1}" if count else name)
        seen[name] = count + 1
    return deduped


def _matrix_to_columns_and_rows(matrix: list[list[str]]) -> tuple[list[str], list[dict]]:
    normalized = [row for row in matrix if any(cell for cell in row)]
    if not normalized:
        return [], []
    width = max(len(row) for row in normalized)
    padded = [row + [""] * (width - len(row)) for row in normalized]
    header = padded[0]
    columns = _dedupe_columns(header if any(header) else [f"column_{index}" for index in range(1, width + 1)])
    data_rows = padded[1:] if any(header) else padded
    rows = []
    for row in data_rows:
        payload = {columns[index]: row[index] for index in range(len(columns))}
        if any(value for value in payload.values()):
            rows.append(payload)
    return columns, rows


def _build_csv_text(columns: list[str], rows: list[dict]) -> str:
    if not columns:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: sanitize_text(row.get(column, "")) for column in columns})
    return sanitize_text(buffer.getvalue()).strip()


def _build_html(columns: list[str], rows: list[dict]) -> str:
    if not columns:
        return ""
    head = "".join(f"<th>{column}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{sanitize_text(row.get(column, ''))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _extract_page_tables(page, filename: str, page_number: int) -> list[dict]:
    raw_words = page.extract_words() or []
    words = [item for item in (_normalize_word(word) for word in raw_words) if item]
    row_clusters = _cluster_rows(words)
    candidate_blocks = _select_candidate_blocks(row_clusters)

    tables = []
    for table_index, block_rows in enumerate(candidate_blocks, start=1):
        column_positions = _cluster_columns(block_rows)
        matrix = [_row_to_cells(row, column_positions) for row in block_rows]
        columns, rows = _matrix_to_columns_and_rows(matrix)
        if not columns or not rows:
            continue
        tables.append(
            {
                "table_id": f"{filename}::table::p{page_number}::{table_index}",
                "filename": filename,
                "doc_name": os.path.splitext(filename)[0],
                "file_type": "PDF",
                "file_path": "",
                "page_number": page_number,
                "table_index": table_index,
                "title": "",
                "caption": "",
                "before_context": "",
                "after_context": "",
                "columns": columns,
                "rows": rows,
                "csv_text": _build_csv_text(columns, rows),
                "html": _build_html(columns, rows),
                "parser_backend": "pdfplumber_words",
            }
        )
    return tables


def reconstruct_tables_from_words(file_path: str, filename: str, max_pages: int | None = None) -> list[dict]:
    pdfplumber = _load_pdfplumber()
    if pdfplumber is None:
        return []

    normalized_max_pages = None if max_pages is None else max(1, int(max_pages))
    tables = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(getattr(pdf, "pages", []) or [], start=1):
                if normalized_max_pages is not None and page_number > normalized_max_pages:
                    break
                tables.extend(_extract_page_tables(page, filename, page_number))
    except Exception:
        logger.exception("pdfplumber words reconstruction failed filename=%s", filename)
        return []
    return tables
