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


def test_reconstruct_tables_from_words_builds_rows_and_columns(monkeypatch):
    words = [
        {"text": "Metric", "x0": 10, "x1": 60, "top": 10, "bottom": 18},
        {"text": "FY2023", "x0": 120, "x1": 165, "top": 10, "bottom": 18},
        {"text": "FY2022", "x0": 220, "x1": 265, "top": 10, "bottom": 18},
        {"text": "Revenue", "x0": 10, "x1": 60, "top": 26, "bottom": 34},
        {"text": "1200", "x0": 120, "x1": 150, "top": 26, "bottom": 34},
        {"text": "1100", "x0": 220, "x1": 250, "top": 26, "bottom": 34},
        {"text": "Operating", "x0": 10, "x1": 58, "top": 42, "bottom": 50},
        {"text": "margin", "x0": 64, "x1": 102, "top": 42, "bottom": 50},
        {"text": "20%", "x0": 120, "x1": 150, "top": 42, "bottom": 50},
        {"text": "18%", "x0": 220, "x1": 250, "top": 42, "bottom": 50},
    ]
    monkeypatch.setattr(
        table_reconstructor,
        "_load_pdfplumber",
        lambda: _FakePdfPlumber([_FakePage(words)]),
    )

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
    assert "Revenue" in tables[0]["csv_text"]


def test_reconstruct_tables_from_words_honors_max_pages(monkeypatch):
    words = [
        {"text": "Metric", "x0": 10, "x1": 60, "top": 10, "bottom": 18},
        {"text": "FY2023", "x0": 120, "x1": 165, "top": 10, "bottom": 18},
        {"text": "Revenue", "x0": 10, "x1": 60, "top": 26, "bottom": 34},
        {"text": "1200", "x0": 120, "x1": 150, "top": 26, "bottom": 34},
    ]
    monkeypatch.setattr(
        table_reconstructor,
        "_load_pdfplumber",
        lambda: _FakePdfPlumber([_FakePage(words), _FakePage(words)]),
    )

    tables = table_reconstructor.reconstruct_tables_from_words("demo.pdf", "demo.pdf", max_pages=1)

    assert all(table["page_number"] == 1 for table in tables)
