from dataclasses import dataclass
import os


@dataclass(frozen=True)
class TableAwareConfig:
    table_aware_ingestion: bool
    table_aware_retrieval: str
    table_evidence_top_k: int
    table_evidence_final_max: int
    table_full_fetch_enabled: bool
    enable_finance_formula_expansion: bool
    table_parser_backend: str
    table_docling_ocr: bool
    table_docling_timeout_seconds: int


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    return default


def _parse_int(value: str | None, default: int, minimum: int = 1) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _parse_retrieval_mode(value: str | None) -> str:
    mode = (value or "off").strip().lower()
    if mode not in {"off", "auto", "force"}:
        return "off"
    return mode


def _parse_table_parser_backend(value: str | None) -> str:
    backend = (value or "auto").strip().lower()
    if backend not in {"auto", "pdfplumber", "docling"}:
        return "auto"
    return backend


def get_table_aware_config() -> TableAwareConfig:
    return TableAwareConfig(
        table_aware_ingestion=_parse_bool(os.getenv("TABLE_AWARE_INGESTION"), False),
        table_aware_retrieval=_parse_retrieval_mode(os.getenv("TABLE_AWARE_RETRIEVAL")),
        table_evidence_top_k=_parse_int(os.getenv("TABLE_EVIDENCE_TOP_K"), 20, minimum=1),
        table_evidence_final_max=_parse_int(os.getenv("TABLE_EVIDENCE_FINAL_MAX"), 4, minimum=1),
        table_full_fetch_enabled=_parse_bool(os.getenv("TABLE_FULL_FETCH_ENABLED"), False),
        enable_finance_formula_expansion=_parse_bool(os.getenv("ENABLE_FINANCE_FORMULA_EXPANSION"), False),
        table_parser_backend=_parse_table_parser_backend(os.getenv("TABLE_PARSER_BACKEND")),
        table_docling_ocr=_parse_bool(os.getenv("TABLE_DOCLING_OCR"), False),
        table_docling_timeout_seconds=_parse_int(os.getenv("TABLE_DOCLING_TIMEOUT_SECONDS"), 120, minimum=1),
    )
