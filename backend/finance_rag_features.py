import hashlib
import os
import re
from typing import Any, Dict, List

from query_parser import COMPANY_SPECS, extract_metrics as parse_metric_list, parse_query


TOKEN_STOPWORDS = {
    "what",
    "which",
    "when",
    "where",
    "from",
    "with",
    "that",
    "this",
    "were",
    "was",
    "does",
    "have",
    "has",
    "into",
    "about",
    "after",
    "before",
    "during",
    "under",
    "over",
    "than",
    "page",
    "pages",
    "please",
    "show",
    "tell",
}

FINANCE_METRIC_HINTS = (
    "gross margin",
    "operating margin",
    "gross profit",
    "operating income",
    "operating profit",
    "quick ratio",
    "current ratio",
    "ebitda",
    "adjusted ebitda",
    "capex",
    "capital expenditures",
    "capital expenditure",
    "eps",
    "adjusted eps",
    "earnings per share",
    "effective tax rate",
    "inventory",
    "revenue",
    "net sales",
    "free cash flow",
    "cash flow",
    "current liabilities",
    "cash and cash equivalents",
    "store count",
    "stores",
    "net income",
    "diluted eps",
    "basic eps",
)

COMPANY_ALIASES = {
    company: [*spec.get("name_aliases", []), *spec.get("ticker_aliases", [])]
    for company, spec in COMPANY_SPECS.items()
}


def normalize_doc_name(filename: str) -> str:
    return os.path.splitext(filename or "")[0]


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d[\d,.\-]*\b", (text or "").lower()))


def extract_years(text: str) -> set[str]:
    return set(re.findall(r"\b(?:19|20)\d{2}\b", (text or "").lower()))


def extract_keyword_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_%-]{1,}", (text or "").lower()))
    return {token for token in tokens if token not in TOKEN_STOPWORDS}


def extract_metric_hints(text: str) -> set[str]:
    lowered = (text or "").lower()
    matches = {hint for hint in FINANCE_METRIC_HINTS if hint in lowered}
    matches.update(parse_metric_list(text))
    return matches


def infer_doc_type(text: str) -> str:
    lowered = (text or "").lower()
    if "10-k" in lowered or "10 k" in lowered:
        return "10-K"
    if "10-q" in lowered or "10 q" in lowered:
        return "10-Q"
    if "8-k" in lowered or "8 k" in lowered:
        return "8-K"
    if "earnings" in lowered or "earnings release" in lowered:
        return "earnings"
    return ""


def build_embedding_cache_key(filename: str, page_number: int, page_text: str) -> str:
    digest = hashlib.sha1((page_text or "").encode("utf-8", errors="ignore")).hexdigest()
    return f"{filename}:{page_number}:{digest}"


def compute_page_features(page_text: str, table_text: str = "") -> Dict[str, Any]:
    combined = "\n".join(part for part in [page_text or "", table_text or ""] if part).strip()
    tokens = sorted(extract_keyword_tokens(combined))
    numbers = sorted(extract_numbers(combined))
    years = sorted(extract_years(combined))
    metric_tokens = sorted(extract_metric_hints(combined))
    return {
        "page_tokens": tokens,
        "page_numbers": numbers,
        "page_years": years,
        "page_metric_tokens": metric_tokens,
    }


def parse_finance_query(question: str) -> Dict[str, Any]:
    parsed = parse_query(question)
    company = parsed.get("company", "") or ""
    return {
        **parsed,
        "company_aliases": COMPANY_ALIASES.get(company, []),
        "quarter": (parsed.get("quarters") or [""])[0] if parsed.get("quarters") else "",
        "doc_type": (parsed.get("doc_types") or [""])[0] if parsed.get("doc_types") else "",
        "metric_keywords": parsed.get("metrics", []),
        "numbers": sorted(extract_numbers(question)),
        "tokens": sorted(extract_keyword_tokens(question)),
    }
