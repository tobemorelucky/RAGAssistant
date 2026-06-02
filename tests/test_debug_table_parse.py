import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "debug_table_parse.py"
SPEC = importlib.util.spec_from_file_location("debug_table_parse", MODULE_PATH)
debug_table_parse = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(debug_table_parse)


def test_parse_args_defaults():
    args = debug_table_parse.parse_args(["demo.pdf"])

    assert args.pdf_path == "demo.pdf"
    assert args.max_tables == 5
    assert args.max_rows == 5
    assert args.backend is None
    assert args.docling_ocr is None
    assert args.timeout_seconds is None
    assert args.max_pages is None
    assert args.show_rejected is False


def test_parse_args_override_backend_and_docling_options():
    args = debug_table_parse.parse_args(
        ["demo.pdf", "--backend", "pdfplumber_words", "--docling-ocr", "--timeout-seconds", "30", "--max-pages", "2", "--show-rejected"]
    )

    assert args.backend == "pdfplumber_words"
    assert args.docling_ocr is True
    assert args.timeout_seconds == 30
    assert args.max_pages == 2
    assert args.show_rejected is True


def test_apply_runtime_overrides_sets_environment(monkeypatch):
    monkeypatch.delenv("TABLE_PARSER_BACKEND", raising=False)
    monkeypatch.delenv("TABLE_DOCLING_OCR", raising=False)
    monkeypatch.delenv("TABLE_DOCLING_TIMEOUT_SECONDS", raising=False)

    args = debug_table_parse.parse_args(["demo.pdf", "--backend", "docling", "--no-docling-ocr", "--timeout-seconds", "45"])
    overrides = debug_table_parse.apply_runtime_overrides(args)

    assert overrides == {"backend": "docling", "docling_ocr": False, "timeout_seconds": 45}
    assert debug_table_parse.os.getenv("TABLE_PARSER_BACKEND") == "docling"
    assert debug_table_parse.os.getenv("TABLE_DOCLING_OCR") == "false"
    assert debug_table_parse.os.getenv("TABLE_DOCLING_TIMEOUT_SECONDS") == "45"


def test_build_report_contains_expected_fields():
    report = debug_table_parse.build_report(
        Path("demo.pdf"),
        [
            {
                "parser_backend": "pdfplumber_words",
                "accepted": True,
                "quality_score": 0.82,
                "reject_reason": "",
                "numeric_cell_ratio": 0.5,
                "non_empty_cell_ratio": 1.0,
                "effective_col_count": 2,
                "data_row_count": 1,
                "page_number": 3,
                "table_index": 1,
                "table_id": "demo.pdf::table::p3::1",
                "title": "Summary",
                "caption": "Quarterly results",
                "columns": ["Metric", "FY2022"],
                "rows": [{"Metric": "Revenue", "FY2022": "100"}],
                "csv_text": "Metric,FY2022\nRevenue,100",
            }
        ],
        max_tables=5,
        max_rows=5,
        runtime_config={"backend": "pdfplumber_words", "docling_ocr": False, "timeout_seconds": 120, "max_pages": 3},
        show_rejected=False,
    )

    assert "filename: demo.pdf" in report
    assert "backend: pdfplumber_words" in report
    assert "docling_ocr: False" in report
    assert "timeout_seconds: 120" in report
    assert "parser_backend: pdfplumber_words" in report
    assert "raw candidates: 1" in report
    assert "accepted tables: 1" in report
    assert "rejected tables: 0" in report
    assert "page_number: 3" in report
    assert "table_id: demo.pdf::table::p3::1" in report
    assert "columns: ['Metric', 'FY2022']" in report
    assert "Revenue" in report


def test_build_report_can_show_rejected_preview():
    report = debug_table_parse.build_report(
        Path("demo.pdf"),
        [
            {"parser_backend": "pdfplumber_words", "accepted": True, "quality_score": 0.8, "reject_reason": "", "numeric_cell_ratio": 0.5, "non_empty_cell_ratio": 1.0, "effective_col_count": 2, "data_row_count": 2, "page_number": 1, "table_index": 1, "table_id": "accepted", "title": "", "caption": "", "columns": ["A", "B"], "rows": [{"A": "Revenue", "B": "100"}], "csv_text": "A,B\nRevenue,100"},
            {"parser_backend": "pdfplumber_words", "accepted": False, "quality_score": 0.1, "reject_reason": "paragraph_like", "numeric_cell_ratio": 0.0, "non_empty_cell_ratio": 1.0, "effective_col_count": 2, "data_row_count": 2, "page_number": 2, "table_index": 1, "table_id": "rejected", "title": "", "caption": "", "columns": ["A", "B"], "rows": [{"A": "Text", "B": "More text"}], "csv_text": "A,B\nText,More text"},
        ],
        max_tables=5,
        max_rows=5,
        runtime_config={"backend": "pdfplumber_words", "docling_ocr": False, "timeout_seconds": 120, "max_pages": 3},
        show_rejected=True,
    )

    assert "accepted table #1" in report
    assert "rejected table #1" in report
    assert "reject_reason: paragraph_like" in report
