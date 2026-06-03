from backend.table_indexer import build_table_evidence_docs


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
    assert docs[0]["page_number"] == 3
    assert "Document: demo.pdf" in docs[0]["text"]
    assert "Columns: Metric | 2023 | 2022" in docs[0]["text"]
    assert "Revenue" in docs[1]["text"]
    assert docs[1]["row_id"] == "row_1"


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
