from backend.table_context_builder import (
    build_table_context_preview,
    build_table_context_preview as _build_preview,
    dedupe_table_ids,
    fetch_tables_for_results,
    format_evidence_unit,
    format_table_preview,
    group_hits_by_table_id,
)


class _FakeTableStore:
    def __init__(self, tables):
        self.tables = tables
        self.called_with = None

    def get_tables_by_ids(self, table_ids):
        self.called_with = list(table_ids)
        lookup = {table["table_id"]: table for table in self.tables}
        return [lookup[table_id] for table_id in table_ids if table_id in lookup]


def test_fetch_tables_calls_store_with_deduped_non_empty_ids():
    store = _FakeTableStore(
        [
            {"table_id": "t1", "page_number": 1, "columns": ["Metric"], "rows": [{"Metric": "Revenue"}], "csv_text": "", "filename": "demo.pdf"},
            {"table_id": "t2", "page_number": 2, "columns": ["Metric"], "rows": [{"Metric": "Margin"}], "csv_text": "", "filename": "demo.pdf"},
        ]
    )
    results = [
        {"table_id": "t1"},
        {"table_id": ""},
        {"table_id": "t2"},
        {"table_id": "t1"},
        {"table_id": None},
    ]

    tables = fetch_tables_for_results(results, store)

    assert store.called_with == ["t1", "t2"]
    assert [table["table_id"] for table in tables] == ["t1", "t2"]


def test_group_hits_by_table_id_skips_empty_ids():
    grouped = group_hits_by_table_id(
        [
            {"table_id": "t1", "score": 0.9},
            {"table_id": ""},
            {"table_id": "t1", "score": 0.8},
            {"table_id": "t2", "score": 0.7},
        ]
    )

    assert list(grouped.keys()) == ["t1", "t2"]
    assert len(grouped["t1"]) == 2


def test_format_table_preview_contains_table_id_page_columns_and_rows():
    preview = format_table_preview(
        {
            "table_id": "t1",
            "filename": "demo.pdf",
            "page_number": 8,
            "columns": ["Metric", "2023", "2022"],
            "rows": [{"Metric": "Net sales", "2023": "3,909", "2022": "3,673"}],
            "csv_text": "Metric,2023,2022\nNet sales,3,909,3,673",
            "title": "Condensed Consolidated Statements of Income",
        },
        preview_rows=5,
        preview_chars=1200,
    )

    assert "Table ID: t1" in preview
    assert "Page: 8" in preview
    assert "columns: ['Metric', '2023', '2022']" in preview
    assert "Net sales" in preview
    assert "csv_text:" in preview


def test_build_table_context_preview_contains_hits_and_full_table():
    results = [
        {
            "table_id": "t1",
            "score": 0.95,
            "evidence_type": "table_row",
            "row_id": "row_1",
            "page_number": 8,
            "text": "Document: demo.pdf\nRow Values: Metric: Net sales; 2023: 3,909",
        }
    ]
    tables = [
        {
            "table_id": "t1",
            "filename": "demo.pdf",
            "page_number": 8,
            "columns": ["Metric", "2023", "2022"],
            "rows": [{"Metric": "Net sales", "2023": "3,909", "2022": "3,673"}],
            "csv_text": "Metric,2023,2022\nNet sales,3,909,3,673",
            "title": "Condensed Consolidated Statements of Income",
        }
    ]

    preview = _build_preview(results, tables, preview_rows=5, preview_chars=1200)

    assert "[Table Evidence]" in preview
    assert "Matched Evidence:" in preview
    assert "Full Table:" in preview
    assert "Table ID: t1" in preview


def test_no_table_evidence_returns_empty_results_without_error():
    store = _FakeTableStore([])
    tables = fetch_tables_for_results([], store)
    preview = build_table_context_preview([], [], preview_rows=5, preview_chars=1200)

    assert tables == []
    assert preview == ""


def test_format_evidence_unit_contains_matched_text_and_attached_table_preview():
    text = format_evidence_unit(
        {
            "filename": "demo.pdf",
            "page_number": 8,
            "text": "Net sales 3,909 3,673 14,544 14,694",
            "attached_tables": [
                {
                    "table": {
                        "table_id": "t1",
                        "filename": "demo.pdf",
                        "page_number": 8,
                        "columns": ["Metric", "2023", "2022"],
                        "rows": [{"Metric": "Net sales", "2023": "3,909", "2022": "3,673"}],
                        "csv_text": "Metric,2023,2022\nNet sales,3,909,3,673",
                    },
                    "include_full": True,
                    "skipped_reason": "",
                }
            ],
        },
        index=1,
        preview_rows=5,
        preview_chars=1200,
    )

    assert "[Evidence 1]" in text
    assert "Matched text:" in text
    assert "Attached same-page table:" in text
    assert "Table ID: t1" in text
