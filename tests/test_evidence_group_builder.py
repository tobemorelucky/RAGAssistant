from backend.evidence_group_builder import build_group_debug_payload, format_evidence_group


def test_format_evidence_group_contains_snippets_and_table_rows():
    text = format_evidence_group(
        {
            "filename": "demo.pdf",
            "page_number": 8,
            "matched_snippets": ["Net sales increased to 3,909 in 2023."],
            "expanded_snippets": ["Adobe income from operations increased year over year."],
            "relevant_table_rows": [
                {
                    "table_id": "t1",
                    "page_number": 8,
                    "row_label": "Net sales",
                    "values_sequence": "3,909 | 3,673",
                    "columns": ["Metric", "2023", "2022"],
                    "row_text": "Metric: Net sales; 2023: 3,909; 2022: 3,673",
                }
            ],
        },
        index=1,
        preview_chars=1200,
    )

    assert "[Evidence Group 1]" in text
    assert "Matched snippets:" in text
    assert "Expanded snippets:" in text
    assert "Relevant table rows:" in text
    assert "Table ID: t1" in text
    assert "Page: 8" in text
    assert "Values: 3,909 | 3,673" in text
    assert text.index("Relevant table rows:") < text.index("Matched snippets:")


def test_build_group_debug_payload_counts_rows_and_snippets():
    payload = build_group_debug_payload(
        {
            "filename": "demo.pdf",
            "page_number": 8,
            "group_score": 1.23,
            "table_attach_reason": "chunk_table_like",
            "attached_table_ids": ["t1"],
            "matched_queries": ["net sales 2023"],
            "planner_sources": ["original", "evidence_field_1"],
            "matched_snippets": ["a"],
            "expanded_snippets": ["b", "c"],
            "relevant_table_rows": [{"table_id": "t1", "row_text": "row"}],
        }
    )

    assert payload["filename"] == "demo.pdf"
    assert payload["matched_snippet_count"] == 1
    assert payload["expanded_snippet_count"] == 2
    assert payload["relevant_table_row_count"] == 1
