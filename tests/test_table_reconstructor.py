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
    monkeypatch.setattr(table_reconstructor, "_load_pdfplumber", lambda: _FakePdfPlumber([_FakePage(words)]))

    tables = table_reconstructor.reconstruct_tables_from_words("demo.pdf", "demo.pdf")

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
    assert "Revenue" in tables[0]["csv_text"]


def test_reconstruct_tables_from_words_honors_max_pages(monkeypatch):
    words = [
        _word("Metric", 10, 10),
        _word("FY2023", 120, 10),
        _word("Revenue", 10, 26),
        _word("1200", 120, 26),
    ]
    monkeypatch.setattr(table_reconstructor, "_load_pdfplumber", lambda: _FakePdfPlumber([_FakePage(words), _FakePage(words)]))

    tables = table_reconstructor.reconstruct_tables_from_words("demo.pdf", "demo.pdf", max_pages=1, include_rejected=True)

    assert all(table["page_number"] == 1 for table in tables)


def test_bullet_list_is_rejected(monkeypatch):
    words = [
        _word("Highlights", 10, 10),
        _word("FY2023", 150, 10),
        _word("FY2022", 250, 10),
        _word("•", 10, 26),
        _word("Revenue", 24, 26),
        _word("growth", 84, 26),
        _word("strong", 150, 26),
        _word("steady", 250, 26),
        _word("•", 10, 42),
        _word("Margin", 24, 42),
        _word("expansion", 78, 42),
        _word("healthy", 150, 42),
        _word("solid", 250, 42),
    ]
    monkeypatch.setattr(table_reconstructor, "_load_pdfplumber", lambda: _FakePdfPlumber([_FakePage(words)]))

    tables = table_reconstructor.reconstruct_tables_from_words("demo.pdf", "demo.pdf", include_rejected=True)

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
    monkeypatch.setattr(table_reconstructor, "_load_pdfplumber", lambda: _FakePdfPlumber([_FakePage(words)]))

    tables = table_reconstructor.reconstruct_tables_from_words("demo.pdf", "demo.pdf", include_rejected=True)

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
    monkeypatch.setattr(table_reconstructor, "_load_pdfplumber", lambda: _FakePdfPlumber([_FakePage(words)]))

    tables = table_reconstructor.reconstruct_tables_from_words("demo.pdf", "demo.pdf", include_rejected=True)

    assert len(tables) == 1
    assert tables[0]["accepted"] is False
    assert tables[0]["reject_reason"] == "low_numeric_density"


def test_reconstruct_tables_returns_only_accepted_by_default(monkeypatch):
    words = [
        _word("Highlights", 10, 10),
        _word("FY2023", 150, 10),
        _word("FY2022", 250, 10),
        _word("•", 10, 26),
        _word("Revenue", 24, 26),
        _word("growth", 84, 26),
        _word("strong", 150, 26),
        _word("steady", 250, 26),
        _word("•", 10, 42),
        _word("Margin", 24, 42),
        _word("expansion", 78, 42),
        _word("healthy", 150, 42),
        _word("solid", 250, 42),
    ]
    monkeypatch.setattr(table_reconstructor, "_load_pdfplumber", lambda: _FakePdfPlumber([_FakePage(words)]))

    tables = table_reconstructor.reconstruct_tables_from_words("demo.pdf", "demo.pdf")

    assert tables == []
