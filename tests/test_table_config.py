from backend.table_config import get_table_aware_config


def test_default_table_config_is_disabled(monkeypatch):
    for key in (
        "TABLE_AWARE_INGESTION",
        "TABLE_AWARE_RETRIEVAL",
        "TABLE_EVIDENCE_TOP_K",
        "TABLE_EVIDENCE_FINAL_MAX",
        "TABLE_FULL_FETCH_ENABLED",
        "ENABLE_FINANCE_FORMULA_EXPANSION",
        "TABLE_PARSER_BACKEND",
        "TABLE_DOCLING_OCR",
        "TABLE_DOCLING_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)

    config = get_table_aware_config()

    assert config.table_aware_ingestion is False
    assert config.table_aware_retrieval == "off"
    assert config.table_evidence_top_k == 20
    assert config.table_evidence_final_max == 4
    assert config.table_full_fetch_enabled is False
    assert config.enable_finance_formula_expansion is False
    assert config.table_parser_backend == "auto"
    assert config.table_docling_ocr is False
    assert config.table_docling_timeout_seconds == 120


def test_table_retrieval_invalid_value_falls_back_to_off(monkeypatch):
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "abc")

    config = get_table_aware_config()

    assert config.table_aware_retrieval == "off"


def test_table_parser_backend_invalid_value_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("TABLE_PARSER_BACKEND", "abc")

    config = get_table_aware_config()

    assert config.table_parser_backend == "auto"


def test_bool_parsing(monkeypatch):
    truthy_values = ("true", "1", "yes", "on")
    falsy_values = ("false", "0", "no", "off")

    for value in truthy_values:
        monkeypatch.setenv("TABLE_AWARE_INGESTION", value)
        assert get_table_aware_config().table_aware_ingestion is True

    for value in falsy_values:
        monkeypatch.setenv("TABLE_AWARE_INGESTION", value)
        assert get_table_aware_config().table_aware_ingestion is False

    for value in truthy_values:
        monkeypatch.setenv("TABLE_DOCLING_OCR", value)
        assert get_table_aware_config().table_docling_ocr is True

    for value in falsy_values:
        monkeypatch.setenv("TABLE_DOCLING_OCR", value)
        assert get_table_aware_config().table_docling_ocr is False


def test_int_parsing_min_value(monkeypatch):
    monkeypatch.setenv("TABLE_EVIDENCE_TOP_K", "0")
    monkeypatch.setenv("TABLE_EVIDENCE_FINAL_MAX", "-2")
    monkeypatch.setenv("TABLE_DOCLING_TIMEOUT_SECONDS", "0")

    config = get_table_aware_config()

    assert config.table_evidence_top_k == 1
    assert config.table_evidence_final_max == 1
    assert config.table_docling_timeout_seconds == 1
