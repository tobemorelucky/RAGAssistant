from backend.table_indexer import build_table_evidence_docs
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "debug_table_evidence.py"
SPEC = importlib.util.spec_from_file_location("debug_table_evidence", MODULE_PATH)
debug_table_evidence = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(debug_table_evidence)


def test_normalized_table_generates_summary_and_table_rows():
    tables = [
        {
            "accepted": True,
            "table_id": "demo.pdf::table::p3::1",
            "filename": "demo.pdf",
            "page_number": 3,
            "normalized_title": "Condensed Consolidated Statements of Income",
            "normalized_columns": ["Metric", "2023", "2022"],
            "normalized_rows": [
                {"Metric": "Revenue", "2023": "120", "2022": "100"},
                {"Metric": "Operating margin", "2023": "20%", "2022": "18%"},
            ],
            "raw_lines": ["Revenue 120 100", "Operating margin 20% 18%"],
        }
    ]

    docs = build_table_evidence_docs(tables)

    assert [doc["evidence_type"] for doc in docs] == ["table_summary", "table_row", "table_row"]
    assert docs[0]["chunk_id"] == "demo.pdf::table::p3::1::summary"
    assert docs[1]["chunk_id"] == "demo.pdf::table::p3::1::row::row_1"
    assert docs[0]["page_number"] == 3
    assert "Document: demo.pdf" in docs[0]["text"]
    assert "Columns: Metric | 2023 | 2022" in docs[0]["text"]
    assert "Revenue" in docs[1]["text"]
    assert docs[1]["row_id"] == "row_1"
    assert docs[1]["text"].index("Row Values:") < docs[1]["text"].index("Columns:")


def test_raw_only_table_generates_summary_and_table_raw():
    tables = [
        {
            "accepted": True,
            "table_id": "demo.pdf::table::p4::1",
            "filename": "demo.pdf",
            "page_number": 4,
            "title": "Conference Call Participant List",
            "columns": ["Name", "Role"],
            "rows": [{"Name": "Alice", "Role": "CEO"}],
            "raw_lines": ["Alice CEO", "Bob CFO"],
            "normalized_rows": [],
        }
    ]

    docs = build_table_evidence_docs(tables)

    assert [doc["evidence_type"] for doc in docs] == ["table_summary", "table_raw", "table_raw"]
    assert docs[1]["chunk_id"] == "demo.pdf::table::p4::1::raw::raw_1"
    assert "Conference Call Participant List" in docs[0]["text"]
    assert "Raw Line: Alice CEO" in docs[1]["text"]
    assert docs[1]["row_id"] == "raw_1"


def test_non_financial_table_still_generates_evidence_when_accepted():
    tables = [
        {
            "accepted": True,
            "table_id": "call.pdf::table::p2::1",
            "filename": "call.pdf",
            "page_number": 2,
            "title": "Conference Call Participants",
            "columns": ["Name", "Role"],
            "rows": [{"Name": "Jane Doe", "Role": "Chief Executive Officer"}],
            "normalized_rows": [],
            "raw_lines": ["Jane Doe Chief Executive Officer"],
        }
    ]

    docs = build_table_evidence_docs(tables)

    assert len(docs) >= 2
    assert any("Conference Call Participants" in doc["text"] for doc in docs)
    assert any("Chief Executive Officer" in doc["text"] for doc in docs)


def test_rejected_tables_are_not_indexed():
    docs = build_table_evidence_docs(
        [
            {
                "accepted": False,
                "table_id": "bad.pdf::table::p1::1",
                "filename": "bad.pdf",
                "page_number": 1,
                "raw_lines": ["Not a table"],
            }
        ]
    )

    assert docs == []


def test_debug_table_evidence_parse_args_support_preview_and_output_json():
    args = debug_table_evidence.parse_args(
        ["demo.pdf", "--backend", "pdfplumber_words", "--preview-chars", "80", "--output-json", "out.json"]
    )

    assert args.backend == "pdfplumber_words"
    assert args.preview_chars == 80
    assert args.output_json == "out.json"


def test_debug_table_evidence_writes_json_output(tmp_path, monkeypatch):
    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    output_path = tmp_path / "evidence.json"

    monkeypatch.setenv("TABLE_AWARE_INGESTION", "true")
    monkeypatch.setattr(
        debug_table_evidence.TableAwareParser,
        "extract_tables",
        lambda self, file_path, filename, parser_backend, max_pages: [
            {
                "accepted": True,
                "table_id": "demo.pdf::table::p2::1",
                "filename": "demo.pdf",
                "page_number": 2,
                "normalized_title": "Summary Table",
                "normalized_columns": ["Metric", "2023"],
                "normalized_rows": [{"Metric": "Revenue", "2023": "100"}],
                "raw_lines": ["Revenue 100"],
                "parser_backend": "pdfplumber_words",
            }
        ],
    )

    exit_code = debug_table_evidence.main(
        [str(pdf_path), "--backend", "pdfplumber_words", "--preview-chars", "60", "--output-json", str(output_path)]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["evidence_type"] == "table_summary"
    assert payload[0]["chunk_id"] == "demo.pdf::table::p2::1::summary"
