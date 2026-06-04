from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

logger = logging.getLogger(__name__)

ARK_API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

_PLANNER_MODEL = None

_ANCHOR_STOPWORDS = {
    "what",
    "how",
    "when",
    "where",
    "why",
    "which",
    "who",
    "whom",
    "whose",
    "does",
    "did",
    "do",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "explain",
    "summarize",
    "describe",
    "sales",
    "revenue",
    "margin",
    "ratio",
    "growth",
    "income",
    "fiscal",
    "year",
    "company",
    "segment",
    "ebit",
    "ebitda",
    "eps",
    "cash",
    "flow",
    "gross",
    "profit",
    "operating",
    "cost",
    "assets",
    "liabilities",
    "debt",
    "tax",
    "rate",
    "quick",
    "effective",
    "stores",
    "compare",
    "change",
    "increase",
    "decrease",
    "improving",
    "table",
    "row",
    "column",
    "data",
    "document",
}

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'.-]*")
_NUMBER_PATTERN = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?")
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_PERIOD_TOKEN_PATTERN = re.compile(r"^(?:FY)?(?:19|20)?\d{2,4}$|^Q[1-4]$", re.IGNORECASE)
_GENERIC_EXPLANATION_HINTS = (
    "financial concept",
    "general explanation",
    "definition of",
    "meaning of",
    "overview of",
)

_QUERY_GROUP_LIMITS = {
    "semantic_queries": 2,
    "evidence_field_queries": 2,
    "table_heading_queries": 2,
    "keyword_queries": 2,
}


def _get_planner_model():
    global _PLANNER_MODEL
    if _PLANNER_MODEL is not None:
        return _PLANNER_MODEL
    if not (ARK_API_KEY and MODEL and BASE_URL):
        return None
    try:
        _PLANNER_MODEL = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=ARK_API_KEY,
            base_url=BASE_URL,
            temperature=0.0,
        )
    except Exception:
        logger.exception("query planner model init failed")
        _PLANNER_MODEL = None
    return _PLANNER_MODEL


def _extract_numbers(text: str) -> list[str]:
    seen = set()
    values: list[str] = []
    for token in _NUMBER_PATTERN.findall(text or ""):
        if token not in seen:
            seen.add(token)
            values.append(token)
    for token in _YEAR_PATTERN.findall(text or ""):
        if token not in seen:
            seen.add(token)
            values.append(token)
    return values


def _extract_anchors(text: str) -> list[str]:
    if not text:
        return []
    anchors: list[str] = []
    seen = set()

    def _add(token: str) -> None:
        normalized = re.sub(r"[’']s$", "", (token or "").strip(), flags=re.IGNORECASE)
        if not normalized:
            return
        if _PERIOD_TOKEN_PATTERN.match(normalized):
            return
        lowered = normalized.lower()
        if lowered in _ANCHOR_STOPWORDS or normalized.isdigit() or lowered in seen:
            return
        seen.add(lowered)
        anchors.append(normalized)

    tokens = _TOKEN_PATTERN.findall(text)
    for token in tokens:
        normalized = re.sub(r"[’']s$", "", token, flags=re.IGNORECASE)
        lowered = normalized.lower()
        if lowered in _ANCHOR_STOPWORDS:
            continue
        if normalized.isupper() and any(ch.isalpha() for ch in normalized):
            _add(normalized)
        elif any(ch.isdigit() for ch in normalized) and any(ch.isalpha() for ch in normalized):
            _add(normalized)

    title_matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text)
    for match in title_matches:
        parts = [part for part in match.split() if part]
        if parts and all(part.lower() not in _ANCHOR_STOPWORDS for part in parts):
            _add(match)

    for token in tokens:
        normalized = re.sub(r"[’']s$", "", token, flags=re.IGNORECASE)
        if normalized[:1].isupper() and normalized[1:].islower():
            _add(normalized)

    return anchors


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _extract_json_blob(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _default_plan(question: str, parse_error: str = "") -> Dict[str, Any]:
    must_keep_terms = _extract_anchors(question) + [item for item in _extract_numbers(question) if item not in _extract_anchors(question)]
    return {
        "enabled": False,
        "intent": "",
        "must_keep_terms": must_keep_terms,
        "semantic_queries": [],
        "evidence_field_queries": [],
        "table_heading_queries": [],
        "keyword_queries": [],
        "planner_validation_dropped_queries": [],
        "expected_evidence_type": "",
        "constraints": [],
        "parse_error": parse_error,
    }


def _clean_queries(values: Any, limit: int) -> list[str]:
    queries: list[str] = []
    seen = set()
    for item in values or []:
        if not item:
            continue
        text = str(item).strip()
        lowered = text.lower()
        if len(text) < 3 or lowered in seen:
            continue
        seen.add(lowered)
        queries.append(text)
        if len(queries) >= limit:
            break
    return queries


def _validate_query(
    query: str,
    *,
    original_anchors: list[str],
    original_years: list[str],
) -> bool:
    text = (query or "").strip()
    if len(text) < 3:
        return False
    lowered = text.lower()
    if any(hint in lowered for hint in _GENERIC_EXPLANATION_HINTS):
        return False
    generated_anchors = {item.lower() for item in _extract_anchors(text)}
    original_anchor_set = {item.lower() for item in original_anchors}
    new_anchors = generated_anchors - original_anchor_set
    if new_anchors:
        return False
    if original_years:
        query_years = set(_YEAR_PATTERN.findall(text))
        new_years = query_years - set(original_years)
        if new_years:
            return False
    return True


def _sanitize_planner_output(question: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    original_anchors = _extract_anchors(question)
    original_numbers = _extract_numbers(question)
    original_years = _YEAR_PATTERN.findall(question or "")
    must_keep_terms = []
    seen_keep = set()
    for item in list(raw.get("must_keep_terms") or []) + original_anchors + original_numbers:
        text = str(item).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen_keep:
            continue
        seen_keep.add(lowered)
        must_keep_terms.append(text)

    dropped_queries: list[dict] = []

    def _sanitize_group(field_name: str) -> list[str]:
        cleaned = _clean_queries(raw.get(field_name), _QUERY_GROUP_LIMITS[field_name])
        accepted: list[str] = []
        for query in cleaned:
            if _validate_query(
                query,
                original_anchors=original_anchors,
                original_years=original_years,
            ):
                accepted.append(query)
            else:
                dropped_queries.append(
                    {
                        "field": field_name,
                        "query": query,
                        "reason": "validation_failed",
                    }
                )
        return accepted

    semantic_queries = _sanitize_group("semantic_queries")
    evidence_field_queries = _sanitize_group("evidence_field_queries")
    table_heading_queries = _sanitize_group("table_heading_queries")
    keyword_queries = _sanitize_group("keyword_queries")

    coverage_text = "\n".join(
        [
            " ".join(must_keep_terms),
            " ".join(semantic_queries),
            " ".join(evidence_field_queries),
            " ".join(table_heading_queries),
            " ".join(keyword_queries),
        ]
    )
    coverage_anchors = {item.lower() for item in _extract_anchors(coverage_text)}
    coverage_numbers = set(_extract_numbers(coverage_text))
    for anchor in original_anchors:
        if anchor.lower() not in coverage_anchors and anchor.lower() not in {item.lower() for item in must_keep_terms}:
            must_keep_terms.append(anchor)
    for number in original_numbers:
        if number not in coverage_numbers and number not in must_keep_terms:
            must_keep_terms.append(number)

    return {
        "enabled": True,
        "intent": str(raw.get("intent") or "").strip(),
        "must_keep_terms": must_keep_terms,
        "semantic_queries": semantic_queries,
        "evidence_field_queries": evidence_field_queries,
        "table_heading_queries": table_heading_queries,
        "keyword_queries": keyword_queries,
        "planner_validation_dropped_queries": dropped_queries,
        "expected_evidence_type": str(raw.get("expected_evidence_type") or "").strip(),
        "constraints": [str(item).strip() for item in (raw.get("constraints") or []) if str(item).strip()],
        "parse_error": "",
    }


def plan_retrieval_queries(question: str) -> Dict[str, Any]:
    text = (question or "").strip()
    if not text:
        return _default_plan(question)

    model = _get_planner_model()
    if model is None:
        return _default_plan(question, parse_error="planner_model_unavailable")

    system_prompt = (
        "You are a retrieval query planner for RAG. "
        "Return JSON only. Do not answer the user's question. "
        "Do not answer the question. "
        "Generate retrieval queries that match how evidence may appear in documents. "
        "Preserve original companies, entities, dates, years, and numbers. "
        "Do not invent new companies, people, products, dates, or years. "
        "You may add generic evidence fields, row labels, table headers, or metric components needed to find evidence. "
        "Do not turn the question into a generic concept explanation. "
        "Generate at most 2 semantic_queries, 2 evidence_field_queries, 2 keyword_queries, and 2 table_heading_queries. "
        "Use retrieval-oriented wording that may help find matching document text, table rows, row labels, column labels, or numeric evidence. "
        "The original user question will always be searched separately and remains highest priority."
    )
    user_prompt = (
        "Plan retrieval queries for this question.\n"
        f"Question: {text}\n\n"
        "Return JSON with keys: intent, must_keep_terms, semantic_queries, evidence_field_queries, "
        "table_heading_queries, keyword_queries, expected_evidence_type, constraints."
    )

    try:
        response = model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        content = _normalize_content(getattr(response, "content", response))
        blob = _extract_json_blob(content)
        raw = json.loads(blob)
        if not isinstance(raw, dict):
            return _default_plan(question, parse_error="planner_output_not_object")
        return _sanitize_planner_output(question, raw)
    except Exception as exc:
        logger.warning("query planner failed: %s", exc)
        return _default_plan(question, parse_error=str(exc))
