import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "debug_table_evidence.py"
SPEC = importlib.util.spec_from_file_location("debug_table_evidence", MODULE_PATH)
debug_table_evidence = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(debug_table_evidence)


def test_parse_args_defaults():
    args = debug_table_evidence.parse_args(["demo.pdf"])

    assert args.pdf_path == "demo.pdf"
    assert args.backend is None
    assert args.max_pages is None
    assert args.max_tables == 20
    assert args.max_docs == 30


def test_parse_args_accepts_backend_and_limits():
    args = debug_table_evidence.parse_args(
        ["demo.pdf", "--backend", "pdfplumber_words", "--max-pages", "8", "--max-tables", "20", "--max-docs", "30"]
    )

    assert args.backend == "pdfplumber_words"
    assert args.max_pages == 8
    assert args.max_tables == 20
    assert args.max_docs == 30


def test_build_report_contains_evidence_preview():
    report = debug_table_evidence.build_report(
        Path("demo.pdf"),
        [
            {
                "table_id": "demo.pdf::table::p2::1",
                "page_number": 2,
                "parser_backend": "pdfplumber_words",
                "accepted": True,
                "normalized_title": "Summary Table",
            }
        ],
        [
            {
                "evidence_type": "table_summary",
                "table_id": "demo.pdf::table::p2::1",
                "row_id": "",
                "page_number": 2,
                "text": "Document: demo.pdf\nPage: 2\nTable ID: demo.pdf::table::p2::1",
            }
        ],
        max_tables=20,
        max_docs=30,
        runtime_config={"backend": "pdfplumber_words", "max_pages": 8},
    )

    assert "tables count: 1" in report
    assert "evidence docs count: 1" in report
    assert "evidence_type: table_summary" in report
    assert "table_id: demo.pdf::table::p2::1" in report
