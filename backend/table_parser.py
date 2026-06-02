"""Optional table-aware PDF parser for future structured table ingestion."""

import csv
import io
import logging
import os
from typing import List

try:
    from table_config import get_table_aware_config
    from text_sanitizer import sanitize_text
except ModuleNotFoundError:
    from backend.table_config import get_table_aware_config
    from backend.text_sanitizer import sanitize_text

logger = logging.getLogger(__name__)


class TableAwareParser:
    """Best-effort structured table extraction for PDF files."""

    @staticmethod
    def _build_table_id(filename: str, page_number: int, table_index: int) -> str:
        return f"{filename}::table::p{page_number}::{table_index}"

    @staticmethod
    def _load_docling():
        try:
            from docling.document_converter import DocumentConverter
        except Exception:
            return None
        return DocumentConverter

    @staticmethod
    def _load_pdfplumber():
        try:
            import pdfplumber
        except Exception:
            return None
        return pdfplumber

    @staticmethod
    def _normalize_cell(value) -> str:
        return sanitize_text("" if value is None else str(value)).strip()

    @classmethod
    def _normalize_matrix(cls, matrix) -> list[list[str]]:
        rows = []
        for row in matrix or []:
            normalized_row = [cls._normalize_cell(cell) for cell in (row or [])]
            if any(cell for cell in normalized_row):
                rows.append(normalized_row)
        return rows

    @staticmethod
    def _dedupe_columns(columns: list[str]) -> list[str]:
        deduped = []
        seen = {}
        for index, column in enumerate(columns, start=1):
            name = sanitize_text(column).strip() or f"column_{index}"
            count = seen.get(name, 0)
            if count:
                deduped.append(f"{name}_{count + 1}")
            else:
                deduped.append(name)
            seen[name] = count + 1
        return deduped

    @classmethod
    def _columns_and_rows_from_matrix(cls, matrix) -> tuple[list[str], list[dict]]:
        normalized = cls._normalize_matrix(matrix)
        if not normalized:
            return [], []

        width = max(len(row) for row in normalized)
        padded = [row + [""] * (width - len(row)) for row in normalized]
        if len(padded) >= 2 and any(cell for cell in padded[0]):
            columns = cls._dedupe_columns(padded[0])
            data_rows = padded[1:]
        else:
            columns = [f"column_{index}" for index in range(1, width + 1)]
            data_rows = padded

        rows = []
        for row in data_rows:
            row_payload = {columns[index]: row[index] for index in range(len(columns))}
            if any(value for value in row_payload.values()):
                rows.append(row_payload)
        return columns, rows

    @staticmethod
    def _build_csv_text(columns: list[str], rows: list[dict]) -> str:
        if not columns:
            return ""
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: sanitize_text(row.get(column, "")) for column in columns})
        return sanitize_text(buffer.getvalue()).strip()

    @staticmethod
    def _build_html(columns: list[str], rows: list[dict]) -> str:
        if not columns:
            return ""
        head = "".join(f"<th>{column}</th>" for column in columns)
        body_rows = []
        for row in rows:
            cells = "".join(f"<td>{sanitize_text(row.get(column, ''))}</td>" for column in columns)
            body_rows.append(f"<tr>{cells}</tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"

    def _extract_with_docling(self, file_path: str, filename: str) -> list[dict]:
        document_converter_cls = self._load_docling()
        if document_converter_cls is None:
            return []

        try:
            converter = document_converter_cls()
            result = converter.convert(file_path)
            document = getattr(result, "document", result)
            tables = []
            for raw_table in getattr(document, "tables", []) or []:
                page_number = int(getattr(raw_table, "page_no", 0) or getattr(raw_table, "page_number", 0) or 0)
                table_index = len(tables) + 1
                data = getattr(raw_table, "data", None)
                if hasattr(data, "to_dict"):
                    data = data.to_dict(orient="records")
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    columns = self._dedupe_columns([sanitize_text(item) for item in data[0].keys()])
                    rows = [
                        {column: self._normalize_cell(row.get(column, "")) for column in columns}
                        for row in data
                    ]
                else:
                    columns, rows = self._columns_and_rows_from_matrix(data)
                if not columns and not rows:
                    continue
                tables.append(
                    {
                        "table_id": self._build_table_id(filename, page_number, table_index),
                        "filename": filename,
                        "doc_name": os.path.splitext(filename)[0],
                        "file_type": "PDF",
                        "file_path": file_path,
                        "page_number": page_number,
                        "table_index": table_index,
                        "title": self._normalize_cell(getattr(raw_table, "title", "")),
                        "caption": self._normalize_cell(getattr(raw_table, "caption", "")),
                        "before_context": "",
                        "after_context": "",
                        "columns": columns,
                        "rows": rows,
                        "html": self._build_html(columns, rows),
                        "csv_text": self._build_csv_text(columns, rows),
                    }
                )
            return tables
        except Exception:
            logger.exception("docling table parsing failed filename=%s", filename)
            return []

    def _extract_with_pdfplumber(self, file_path: str, filename: str) -> list[dict]:
        pdfplumber = self._load_pdfplumber()
        if pdfplumber is None:
            return []

        tables = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_number, page in enumerate(getattr(pdf, "pages", []) or [], start=1):
                    page_tables = page.extract_tables() or []
                    for table_index, matrix in enumerate(page_tables, start=1):
                        columns, rows = self._columns_and_rows_from_matrix(matrix)
                        if not columns and not rows:
                            continue
                        tables.append(
                            {
                                "table_id": self._build_table_id(filename, page_number, table_index),
                                "filename": filename,
                                "doc_name": os.path.splitext(filename)[0],
                                "file_type": "PDF",
                                "file_path": file_path,
                                "page_number": page_number,
                                "table_index": table_index,
                                "title": "",
                                "caption": "",
                                "before_context": "",
                                "after_context": "",
                                "columns": columns,
                                "rows": rows,
                                "html": self._build_html(columns, rows),
                                "csv_text": self._build_csv_text(columns, rows),
                            }
                        )
            return tables
        except Exception:
            logger.exception("pdfplumber table parsing failed filename=%s", filename)
            return []

    def extract_tables(self, file_path: str, filename: str) -> List[dict]:
        config = get_table_aware_config()
        if not config.table_aware_ingestion:
            return []
        if not (filename or "").lower().endswith(".pdf"):
            return []

        tables = self._extract_with_docling(file_path, filename)
        parser_backend = "docling"
        if not tables:
            tables = self._extract_with_pdfplumber(file_path, filename)
            parser_backend = "pdfplumber" if tables else "none"

        logger.info(
            "table parsing completed filename=%s tables=%s parser=%s",
            filename,
            len(tables),
            parser_backend,
        )
        return tables
