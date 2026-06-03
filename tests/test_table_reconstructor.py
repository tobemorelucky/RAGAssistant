import backend.table_reconstructor as table_reconstructor


class _FakePage:
    def __init__(self, words):
        self._words = words

    def extract_words(self):
        return self._words


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakePdfPlumber:
    def __init__(self, pages):
        self._pages = pages

    def open(self, _file_path):
        return _FakePdf(self._pages)


def _word(text, x0, top, x1=None, bottom=None):
    return {
        "text": text,
        "x0": x0,
        "x1": x1 if x1 is not None else x0 + max(12, len(text) * 6),
        "top": top,
        "bottom": bottom if bottom is not None else top + 8,
    }


def _table(words, monkeypatch, *, include_rejected=False, max_pages=None):
    monkeypatch.setattr(table_reconstructor, "_load_pdfplumber", lambda: _FakePdfPlumber([_FakePage(words)]))
    tables = table_reconstructor.reconstruct_tables_from_words(
        "demo.pdf",
        "demo.pdf",
        max_pages=max_pages,
        include_rejected=include_rejected,
    )
    return tables


def test_parse_numeric_tail_row_merges_metric_label():
    parsed = table_reconstructor.parse_numeric_tail_row(
        ["Net", "sales", "3,909", "3,673", "14,544", "14,694"]
    )

    assert parsed["metric_label"] == "Net sales"
    assert parsed["values"] == ["3,909", "3,673", "14,544", "14,694"]
    assert parsed["numeric_tail_count"] == 4


def test_parse_numeric_tail_row_handles_long_metric_label():
    parsed = table_reconstructor.parse_numeric_tail_row(
        ["Selling,", "general,", "and", "administrative", "expenses", "(342)", "(329)"]
    )

    assert parsed["metric_label"] == "Selling, general, and administrative expenses"
    assert parsed["values"] == ["(342)", "(329)"]


def test_classify_table_rows_marks_title_unit_and_data():
    matrix = [
        ["U.S. GAAP Condensed Consolidated Statements of Income"],
        ["($ million), except per share amounts"],
        ["Metric", "2022", "2023"],
        ["Net sales", "100", "120"],
    ]

    rows = table_reconstructor.classify_table_rows(matrix)

    assert rows[0]["row_type"] == "title"
    assert rows[1]["row_type"] == "unit"
    assert rows[2]["row_type"] == "year_header"
    assert rows[3]["row_type"] == "data"


def test_normalize_financial_table_simple_two_year_table():
    table = {
        "raw_matrix": [
            ["U.S. GAAP Condensed Consolidated Balance Sheets"],
            ["($ million)"],
            ["Metric", "2022", "2023"],
            ["Cash", "848", "1,021"],
            ["Total assets", "14,100", "14,450"],
        ],
        "columns": ["Metric", "2022", "2023"],
        "rows": [
            {"Metric": "Cash", "2022": "848", "2023": "1,021"},
            {"Metric": "Total assets", "2022": "14,100", "2023": "14,450"},
        ],
        "csv_text": "",
        "html": "",
    }

    normalized = table_reconstructor.normalize_financial_table(table)

    assert normalized["normalized"] is True
    assert normalized["normalization_level"] == "full"
    assert normalized["normalized_title"] == "U.S. GAAP Condensed Consolidated Balance Sheets"
    assert normalized["normalized_unit"] == "($ million)"
    assert normalized["normalized_columns"] == ["Metric", "2022", "2023"]
    assert normalized["normalized_rows"][0]["Metric"] == "Cash"
    assert normalized["normalized_rows"][1]["2023"] == "14,450"
    assert normalized["normalized_rows"][0]["_raw_line"] == "Cash 848 1,021"
    assert normalized["normalized_rows"][0]["_raw_row_index"] == 3
    assert normalized["columns"] == ["Metric", "2022", "2023"]
    assert normalized["rows"] == table["rows"]


def test_normalize_financial_table_four_period_columns():
    table = {
        "raw_matrix": [
            ["U.S. GAAP Condensed Consolidated Statements of Income"],
            ["($ million), except per share amounts"],
            ["", "Three Months Ended June 30", "", "Twelve Months Ended June 30", ""],
            ["Metric", "2022", "2023", "2022", "2023"],
            ["Net sales", "3,909", "3,673", "14,544", "14,694"],
            ["Gross profit", "1,020", "980", "4,020", "4,110"],
        ],
        "columns": ["Metric", "2022", "2023", "2022_2", "2023_2"],
        "rows": [],
        "csv_text": "",
        "html": "",
    }

    normalized = table_reconstructor.normalize_financial_table(table)

    assert normalized["normalized"] is True
    assert normalized["normalization_level"] == "full"
    assert normalized["normalized_columns"] == [
        "Metric",
        "Three Months Ended June 30, 2022",
        "Three Months Ended June 30, 2023",
        "Twelve Months Ended June 30, 2022",
        "Twelve Months Ended June 30, 2023",
    ]
    assert normalized["normalized_rows"][0]["Metric"] == "Net sales"
    assert normalized["normalized_rows"][0]["Twelve Months Ended June 30, 2023"] == "14,694"


def test_normalize_financial_table_complex_table_falls_back_to_partial():
    table = {
        "raw_matrix": [
            ["Adjusted EBIT reconciliation"],
            ["($ million)"],
            ["Metric", "Flexibles", "Rigid Packaging", "Total"],
            ["Adjusted EBIT", "120", "85", "205"],
            ["Margin", "12.5%", "10.1%", "11.3%"],
        ],
        "columns": ["Metric", "Flexibles", "Rigid Packaging", "Total"],
        "rows": [],
        "csv_text": "",
        "html": "",
    }

    normalized = table_reconstructor.normalize_financial_table(table)

    assert normalized["normalized"] is True
    assert normalized["normalization_level"] in {"partial", "raw_only"}
    assert normalized["columns"] == ["Metric", "Flexibles", "Rigid Packaging", "Total"]
    assert normalized["html"] == ""


def test_reconstruct_tables_from_words_builds_rows_and_columns(monkeypatch):
    words = [
        _word("Metric", 10, 10),
        _word("FY2023", 120, 10),
        _word("FY2022", 220, 10),
        _word("Revenue", 10, 26),
        _word("1200", 120, 26),
        _word("1100", 220, 26),
        _word("Operating", 10, 42),
        _word("margin", 64, 42),
        _word("20%", 120, 42),
        _word("18%", 220, 42),
    ]
    tables = _table(words, monkeypatch)

    assert len(tables) == 1
    assert tables[0]["page_number"] == 1
    assert tables[0]["table_index"] == 1
    assert tables[0]["columns"] == ["Metric", "FY2023", "FY2022"]
    assert tables[0]["rows"] == [
        {"Metric": "Revenue", "FY2023": "1200", "FY2022": "1100"},
        {"Metric": "Operating margin", "FY2023": "20%", "FY2022": "18%"},
    ]
    assert tables[0]["parser_backend"] == "pdfplumber_words"
    assert tables[0]["accepted"] is True
    assert tables[0]["reject_reason"] == ""
    assert tables[0]["raw_matrix"][0] == ["Metric", "FY2023", "FY2022"]


def test_reconstruct_tables_from_words_honors_max_pages(monkeypatch):
    words = [
        _word("Metric", 10, 10),
        _word("FY2023", 120, 10),
        _word("Revenue", 10, 26),
        _word("1200", 120, 26),
    ]
    monkeypatch.setattr(
        table_reconstructor,
        "_load_pdfplumber",
        lambda: _FakePdfPlumber([_FakePage(words), _FakePage(words)]),
    )

    tables = table_reconstructor.reconstruct_tables_from_words("demo.pdf", "demo.pdf", max_pages=1, include_rejected=True)

    assert all(table["page_number"] == 1 for table in tables)


def test_bullet_list_is_rejected(monkeypatch):
    words = [
        _word("Highlights", 10, 10),
        _word("FY2023", 150, 10),
        _word("FY2022", 250, 10),
        _word("*", 10, 26),
        _word("Revenue", 24, 26),
        _word("growth", 84, 26),
        _word("strong", 150, 26),
        _word("steady", 250, 26),
        _word("*", 10, 42),
        _word("Margin", 24, 42),
        _word("expansion", 78, 42),
        _word("healthy", 150, 42),
        _word("solid", 250, 42),
    ]
    tables = _table(words, monkeypatch, include_rejected=True)

    assert len(tables) == 1
    assert tables[0]["accepted"] is False
    assert tables[0]["reject_reason"] == "bullet_list_like"


def test_paragraph_like_candidate_is_rejected(monkeypatch):
    words = [
        _word("Commentary", 10, 10),
        _word("FY2023", 220, 10),
        _word("We", 10, 26),
        _word("delivered", 34, 26),
        _word("strong", 92, 26),
        _word("performance", 138, 26),
        _word("across", 214, 26),
        _word("the", 258, 26),
        _word("segment", 284, 26),
        _word("noted", 380, 26),
        _word("by", 426, 26),
        _word("CEO", 446, 26),
        _word("Demand", 10, 42),
        _word("remained", 66, 42),
        _word("resilient", 126, 42),
        _word("despite", 192, 42),
        _word("market", 246, 42),
        _word("volatility", 296, 42),
        _word("during", 380, 42),
        _word("the", 430, 42),
        _word("quarter", 454, 42),
    ]
    tables = _table(words, monkeypatch, include_rejected=True)

    assert len(tables) == 1
    assert tables[0]["accepted"] is False
    assert tables[0]["reject_reason"] == "paragraph_like"


def test_low_numeric_density_candidate_is_rejected(monkeypatch):
    words = [
        _word("Metric", 10, 10),
        _word("Operating", 120, 10),
        _word("Trend", 250, 10),
        _word("Revenue", 10, 26),
        _word("healthy", 120, 26),
        _word("stable", 250, 26),
        _word("Margin", 10, 42),
        _word("improving", 120, 42),
        _word("resilient", 250, 42),
    ]
    tables = _table(words, monkeypatch, include_rejected=True)

    assert len(tables) == 1
    assert tables[0]["accepted"] is False
    assert tables[0]["reject_reason"] == "low_numeric_density"


def test_reconstruct_tables_returns_only_accepted_by_default(monkeypatch):
    words = [
        _word("Highlights", 10, 10),
        _word("FY2023", 150, 10),
        _word("FY2022", 250, 10),
        _word("*", 10, 26),
        _word("Revenue", 24, 26),
        _word("growth", 84, 26),
        _word("strong", 150, 26),
        _word("steady", 250, 26),
        _word("*", 10, 42),
        _word("Margin", 24, 42),
        _word("expansion", 78, 42),
        _word("healthy", 150, 42),
        _word("solid", 250, 42),
    ]
    tables = _table(words, monkeypatch)

    assert tables == []
