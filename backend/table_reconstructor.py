"""基于 pdfplumber word 坐标的轻量表格重建与通用财务表规范化。"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
from collections import Counter
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
    "profit",
    "debt",
    "cost",
    "flow",
}

_TITLE_HINTS = (
    "statement",
    "statements",
    "reconciliation",
    "balance sheet",
    "balance sheets",
    "cash flow",
    "cash flows",
    "income",
    "sales growth",
    "net debt",
    "components",
    "ebit",
)

_UNIT_HINTS = (
    "$ million",
    "$ millions",
    "€ million",
    "except per share amounts",
    "per share amounts",
    "us cents",
    "in millions",
)

_PERIOD_HINTS = (
    "three months",
    "twelve months",
    "six months",
    "nine months",
    "ended",
    "june",
    "december",
    "march",
    "september",
    "quarter",
    "fiscal",
    "year ended",
)

_SECTION_HINTS = (
    "reconciliation",
    "components",
    "segment",
    "adjusted",
    "non-gaap",
)

_YEAR_PATTERN = re.compile(r"\b(?:fy(?:19|20)?\d{2}|(?:19|20)\d{2})\b", re.IGNORECASE)
_DATE_PATTERN = re.compile(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", re.IGNORECASE)
_NUMERIC_PATTERN = re.compile(r"^\(?-?\$?\s*\d[\d,]*(?:\.\d+)?\s*%?\)?$|^[-—–]$|^n/?m$", re.IGNORECASE)
_BULLET_PATTERN = re.compile(r"^(?:[-*•·▪◦]|\(\w\)|\d+[.)])\s*")


def _load_pdfplumber():
    try:
        import pdfplumber
    except Exception:
        return None
    return pdfplumber


def _clean_text(value) -> str:
    return sanitize_text("" if value is None else str(value)).strip()


def _normalize_word(word: dict) -> dict | None:
    text = _clean_text(word.get("text", ""))
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


def _normalize_matrix(matrix: list[list[str]]) -> list[list[str]]:
    return [[_clean_text(cell) for cell in row] for row in matrix]


def _join_cells(cells: list[str]) -> str:
    return " ".join(cell for cell in cells if _clean_text(cell)).strip()


def _is_numeric_like(text: str) -> bool:
    value = _clean_text(text)
    if not value:
        return False
    normalized = value.replace(" ", "")
    return bool(_NUMERIC_PATTERN.match(normalized))


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+", _clean_text(text)))


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
    joined = " ".join(item["text"] for item in row).lower()
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
    return [_clean_text(cell) for cell in cells]


def _dedupe_columns(columns: list[str]) -> list[str]:
    deduped = []
    seen = {}
    for index, column in enumerate(columns, start=1):
        name = _clean_text(column) or f"column_{index}"
        count = seen.get(name, 0)
        deduped.append(f"{name}_{count + 1}" if count else name)
        seen[name] = count + 1
    return deduped


def _matrix_to_columns_and_rows(matrix: list[list[str]]) -> tuple[list[str], list[dict]]:
    normalized = [row for row in _normalize_matrix(matrix) if any(row)]
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
        writer.writerow({column: _clean_text(row.get(column, "")) for column in columns})
    return _clean_text(buffer.getvalue())


def _build_html(columns: list[str], rows: list[dict]) -> str:
    if not columns:
        return ""
    head = "".join(f"<th>{column}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{_clean_text(row.get(column, ''))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _looks_title_text(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _TITLE_HINTS)


def _looks_unit_text(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _UNIT_HINTS)


def _looks_period_text(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _PERIOD_HINTS)


def _looks_section_text(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _SECTION_HINTS)


def _looks_footnote_text(text: str) -> bool:
    lowered = text.lower()
    return bool(re.match(r"^\(?\d+\)?", text)) or "note" in lowered or len(text.split()) >= 12


def parse_numeric_tail_row(cells: list[str]) -> dict:
    normalized = [_clean_text(cell) for cell in cells]
    if not normalized:
        return {
            "metric_label": "",
            "values": [],
            "value_start_index": 0,
            "numeric_tail_count": 0,
        }

    last_index = len(normalized) - 1
    while last_index >= 0 and not normalized[last_index]:
        last_index -= 1
    if last_index < 0:
        return {
            "metric_label": "",
            "values": [],
            "value_start_index": 0,
            "numeric_tail_count": 0,
        }

    index = last_index
    values: list[str] = []
    while index >= 0 and _is_numeric_like(normalized[index]):
        values.insert(0, normalized[index])
        index -= 1
    value_start_index = index + 1
    metric_label = " ".join(cell for cell in normalized[:value_start_index] if cell).strip()

    return {
        "metric_label": metric_label,
        "values": values,
        "value_start_index": value_start_index,
        "numeric_tail_count": len(values),
    }


def classify_table_rows(matrix: list[list[str]]) -> list[dict]:
    normalized = _normalize_matrix(matrix)
    results = []
    for row_index, cells in enumerate(normalized):
        non_empty_cells = [cell for cell in cells if cell]
        line_text = _join_cells(cells)
        numeric_tail = parse_numeric_tail_row(cells)
        numeric_count = sum(1 for cell in non_empty_cells if _is_numeric_like(cell))
        numeric_ratio = round(numeric_count / max(1, len(non_empty_cells)), 4)

        if not non_empty_cells:
            row_type = "blank"
        elif _looks_unit_text(line_text):
            row_type = "unit"
        elif _looks_title_text(line_text) and numeric_tail["numeric_tail_count"] < 2:
            row_type = "title"
        elif _looks_period_text(line_text) and numeric_tail["numeric_tail_count"] < 2:
            row_type = "period_header"
        elif all(_YEAR_PATTERN.fullmatch(cell) or _DATE_PATTERN.search(cell) for cell in non_empty_cells if cell):
            row_type = "year_header"
        elif (
            len(non_empty_cells) >= 2
            and _clean_text(non_empty_cells[0]).lower() in {"metric", "metrics", "description", "item"}
            and all(_YEAR_PATTERN.search(cell) or _DATE_PATTERN.search(cell) for cell in non_empty_cells[1:])
        ):
            row_type = "year_header"
        elif (
            numeric_tail["numeric_tail_count"] >= 2
            and not numeric_tail["metric_label"]
            and all(_YEAR_PATTERN.search(cell) or _DATE_PATTERN.search(cell) for cell in non_empty_cells)
        ):
            row_type = "year_header"
        elif numeric_tail["numeric_tail_count"] >= 2 and numeric_tail["metric_label"]:
            row_type = "data"
        elif _looks_section_text(line_text):
            row_type = "section"
        elif _looks_footnote_text(line_text):
            row_type = "footnote"
        else:
            row_type = "unknown"

        results.append(
            {
                "row_index": row_index,
                "row_type": row_type,
                "cells": cells,
                "line_text": line_text,
                "numeric_tail_count": numeric_tail["numeric_tail_count"],
                "numeric_ratio": numeric_ratio,
            }
        )
    return results


def _dominant_numeric_tail_count(data_rows: list[list[str]]) -> int:
    counts = [parse_numeric_tail_row(row)["numeric_tail_count"] for row in data_rows if parse_numeric_tail_row(row)["numeric_tail_count"] >= 1]
    if not counts:
        return 0
    return Counter(counts).most_common(1)[0][0]


def _value_type_from_values(values: list[str], unit: str) -> str:
    lowered_unit = unit.lower()
    if "per share" in lowered_unit:
        return "per_share"
    if any("%" in value for value in values):
        return "percentage"
    if any("$" in value or "," in value for value in values):
        return "money"
    if any(re.search(r"\d", value) for value in values):
        return "number"
    return "unknown"


def _generic_value_labels(count: int) -> list[str]:
    return [f"value_{index}" for index in range(1, count + 1)]


def _expand_header_tail(cells: list[str], target_count: int) -> list[str]:
    padded = cells[-target_count:] if len(cells) >= target_count else [""] * (target_count - len(cells)) + list(cells)
    expanded = list(padded)
    carry = ""
    for index, cell in enumerate(expanded):
        if cell:
            carry = cell
        elif carry:
            expanded[index] = carry
    carry = ""
    for index in range(len(expanded) - 1, -1, -1):
        if expanded[index]:
            carry = expanded[index]
        elif carry:
            expanded[index] = carry
    return expanded


def infer_column_schema(header_rows: list[list[str]], data_rows: list[list[str]], normalized_unit: str = "") -> list[dict]:
    numeric_count = _dominant_numeric_tail_count(data_rows)
    if numeric_count <= 0:
        return []

    normalized_header_rows = [_normalize_matrix([row])[0] for row in header_rows if any(_normalize_matrix([row])[0])]
    expanded_headers = [_expand_header_tail(row, numeric_count) for row in normalized_header_rows]
    labels = []
    for column_index in range(numeric_count):
        path = []
        for row in expanded_headers:
            cell = _clean_text(row[column_index]) if column_index < len(row) else ""
            if cell and (not path or path[-1] != cell):
                path.append(cell)
        label = ", ".join(path) if path else ""
        labels.append({"path": path, "label": label})

    if not any(item["label"] for item in labels):
        fallback_labels = _generic_value_labels(numeric_count)
    else:
        fallback_labels = []

    sample_values_by_column = [[] for _ in range(numeric_count)]
    for row in data_rows:
        parsed = parse_numeric_tail_row(row)
        values = parsed["values"]
        if not values:
            continue
        padded = [""] * max(0, numeric_count - len(values)) + values[-numeric_count:]
        for index, value in enumerate(padded[-numeric_count:]):
            if value:
                sample_values_by_column[index].append(value)

    schema = []
    for index in range(numeric_count):
        label = labels[index]["label"] or fallback_labels[index]
        path = labels[index]["path"] or [label]
        schema.append(
            {
                "key": f"value_{index + 1}",
                "label": label,
                "path": path,
                "value_type": _value_type_from_values(sample_values_by_column[index], normalized_unit),
                "unit": normalized_unit,
            }
        )
    return schema


def normalize_financial_table(table: dict) -> dict:
    raw_matrix = _normalize_matrix(table.get("raw_matrix") or [])
    if not raw_matrix:
        return {
            **table,
            "raw_matrix": [],
            "raw_lines": [],
            "normalized": False,
            "normalization_level": "raw_only",
            "normalized_title": "",
            "normalized_unit": "",
            "header_rows": [],
            "data_rows": [],
            "footnote_rows": [],
            "column_schema": [],
            "normalized_columns": [],
            "normalized_rows": [],
            "normalization_notes": ["raw matrix unavailable"],
        }

    row_info = classify_table_rows(raw_matrix)
    title_rows = [item["cells"] for item in row_info if item["row_type"] == "title"]
    unit_rows = [item["cells"] for item in row_info if item["row_type"] == "unit"]
    header_rows = [item["cells"] for item in row_info if item["row_type"] in {"period_header", "year_header"}]
    data_rows = [item["cells"] for item in row_info if item["row_type"] == "data"]
    footnote_rows = [item["line_text"] for item in row_info if item["row_type"] == "footnote"]
    section_rows = [item["line_text"] for item in row_info if item["row_type"] == "section"]

    normalized_title = " ".join(_join_cells(row) for row in title_rows).strip()
    normalized_unit = " ".join(_join_cells(row) for row in unit_rows).strip()
    normalization_notes: list[str] = []

    if section_rows:
        normalization_notes.append("table includes section-style rows")
    if len(section_rows) >= 2:
        normalization_notes.append("possible multiple table sections detected")

    dominant_count = _dominant_numeric_tail_count(data_rows)
    parsed_rows = []
    for row in data_rows:
        parsed = parse_numeric_tail_row(row)
        if parsed["numeric_tail_count"] < 2:
            continue
        if dominant_count and parsed["numeric_tail_count"] < dominant_count - 1:
            continue
        parsed_rows.append(parsed)

    column_schema = infer_column_schema(header_rows, data_rows, normalized_unit=normalized_unit)
    normalized_columns = ["Metric"]
    if column_schema:
        normalized_columns.extend(item["label"] for item in column_schema)

    normalized_rows = []
    numeric_count = len(column_schema)
    for parsed in parsed_rows:
        if not numeric_count:
            break
        padded_values = [""] * max(0, numeric_count - len(parsed["values"])) + parsed["values"][-numeric_count:]
        row_payload = {"Metric": parsed["metric_label"]}
        for schema_item, value in zip(column_schema, padded_values[-numeric_count:]):
            row_payload[schema_item["label"]] = value
        normalized_rows.append(row_payload)

    level = "raw_only"
    normalized = False
    if len(normalized_rows) >= 2 and numeric_count >= 1:
        normalized = True
        if column_schema and all(not item["label"].startswith("value_") for item in column_schema):
            level = "full"
        else:
            level = "partial"
    elif parsed_rows:
        level = "partial"
        normalized = True
        normalization_notes.append("parsed data rows but column semantics remain incomplete")
    else:
        normalization_notes.append("unable to derive stable data rows")

    if not normalized_title and _join_cells(raw_matrix[0]):
        normalization_notes.append("title not confidently detected")
    if not header_rows:
        normalization_notes.append("header rows not confidently detected")

    return {
        **table,
        "raw_matrix": raw_matrix,
        "raw_lines": [_join_cells(row) for row in raw_matrix],
        "normalized": normalized,
        "normalization_level": level,
        "normalized_title": normalized_title,
        "normalized_unit": normalized_unit,
        "header_rows": header_rows,
        "data_rows": data_rows,
        "footnote_rows": footnote_rows,
        "column_schema": column_schema,
        "normalized_columns": normalized_columns,
        "normalized_rows": normalized_rows,
        "normalization_notes": normalization_notes,
    }


def _evaluate_candidate(matrix: list[list[str]], columns: list[str], rows: list[dict]) -> dict:
    data_row_count = len(rows)
    effective_col_count = sum(1 for column in columns if any((_clean_text(row.get(column, ""))) for row in rows)) or len(columns)
    total_cells = max(1, sum(len(row) for row in matrix))
    non_empty_cells = sum(1 for row in matrix for cell in row if _clean_text(cell))
    numeric_cells = sum(1 for row in matrix for cell in row if _is_numeric_like(cell))
    non_empty_cell_ratio = round(non_empty_cells / total_cells, 4)
    numeric_cell_ratio = round(numeric_cells / max(1, non_empty_cells), 4)

    first_column_key = columns[0] if columns else ""
    first_column_values = [(_clean_text(row.get(first_column_key, ""))) for row in rows] if first_column_key else []
    bullet_ratio = (
        sum(1 for value in first_column_values if _BULLET_PATTERN.match(value)) / max(1, len(first_column_values))
        if first_column_values
        else 0.0
    )
    average_words_per_cell = sum(_word_count(cell) for row in matrix for cell in row if _clean_text(cell)) / max(1, non_empty_cells)
    header_joined = " ".join(columns).lower()
    plain_header_words = [token for token in re.findall(r"[a-zA-Z]+", header_joined) if token not in _METRIC_HINTS]
    header_year_count = len(_YEAR_PATTERN.findall(header_joined))

    reject_reason = ""
    if bullet_ratio >= 0.5 and numeric_cell_ratio < 0.15:
        reject_reason = "bullet_list_like"
    elif (average_words_per_cell >= 3.2 and numeric_cell_ratio < 0.12) or (
        numeric_cell_ratio < 0.05 and data_row_count <= 1 and effective_col_count >= 6
    ):
        reject_reason = "paragraph_like"
    elif effective_col_count >= 8 and numeric_cell_ratio < 0.15 and len(plain_header_words) >= max(4, effective_col_count - 1):
        reject_reason = "too_many_text_columns"
    elif effective_col_count < 2:
        reject_reason = "too_few_columns"
    elif data_row_count < 2:
        reject_reason = "too_few_rows"
    elif non_empty_cell_ratio < 0.45:
        reject_reason = "mostly_empty"
    elif numeric_cell_ratio < 0.2 and header_year_count == 0:
        reject_reason = "low_numeric_density"

    quality_score = round(
        min(
            1.0,
            max(
                0.0,
                0.35
                + min(0.25, numeric_cell_ratio * 0.45)
                + min(0.2, non_empty_cell_ratio * 0.2)
                + min(0.1, effective_col_count * 0.02)
                + min(0.1, data_row_count * 0.025)
                + min(0.1, header_year_count * 0.05),
            ),
        ),
        4,
    )
    if reject_reason:
        quality_score = round(max(0.0, quality_score - 0.45), 4)

    return {
        "accepted": not reject_reason,
        "quality_score": quality_score,
        "reject_reason": reject_reason,
        "numeric_cell_ratio": numeric_cell_ratio,
        "non_empty_cell_ratio": non_empty_cell_ratio,
        "effective_col_count": int(effective_col_count),
        "data_row_count": int(data_row_count),
    }


def _extract_page_tables(page, filename: str, page_number: int, *, include_rejected: bool) -> list[dict]:
    raw_words = page.extract_words() or []
    words = [item for item in (_normalize_word(word) for word in raw_words) if item]
    row_clusters = _cluster_rows(words)
    candidate_blocks = _select_candidate_blocks(row_clusters)

    tables = []
    for table_index, block_rows in enumerate(candidate_blocks, start=1):
        column_positions = _cluster_columns(block_rows)
        raw_matrix = [_row_to_cells(row, column_positions) for row in block_rows]
        columns, rows = _matrix_to_columns_and_rows(raw_matrix)
        if not columns or not rows:
            continue

        candidate = {
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
            "raw_matrix": _normalize_matrix(raw_matrix),
            "raw_lines": [_join_cells(row) for row in raw_matrix],
        }
        candidate.update(_evaluate_candidate(candidate["raw_matrix"], columns, rows))
        candidate = normalize_financial_table(candidate)
        if candidate["accepted"] or include_rejected:
            tables.append(candidate)
    return tables


def reconstruct_tables_from_words(
    file_path: str,
    filename: str,
    max_pages: int | None = None,
    *,
    include_rejected: bool = False,
) -> list[dict]:
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
                tables.extend(_extract_page_tables(page, filename, page_number, include_rejected=include_rejected))
    except Exception:
        logger.exception("pdfplumber words reconstruction failed filename=%s", filename)
        return []
    return tables
