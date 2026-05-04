from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import logging
import os
import re
import time

import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from document_page_store import DocumentPageStore
from embedding import embedding_service as _embedding_service
from finance_rag_features import (
    COMPANY_ALIASES,
    FINANCE_METRIC_HINTS,
    extract_keyword_tokens as _feature_extract_keyword_tokens,
    extract_metric_hints as _feature_extract_metric_hints,
    extract_numbers as _feature_extract_numbers,
    extract_years as _feature_extract_years,
    infer_doc_type,
    normalize_doc_name,
    parse_finance_query,
)
from query_parser import company_aliases_for, matches_company_text
from milvus_client import MilvusManager
from parent_chunk_store import ParentChunkStore

load_dotenv()

logger = logging.getLogger(__name__)

ARK_API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
RERANK_MODEL = os.getenv("RERANK_MODEL")
RERANK_BINDING_HOST = os.getenv("RERANK_BINDING_HOST")
RERANK_API_KEY = os.getenv("RERANK_API_KEY")

AUTO_MERGE_ENABLED = os.getenv("AUTO_MERGE_ENABLED", "true").lower() != "false"
AUTO_MERGE_THRESHOLD = int(os.getenv("AUTO_MERGE_THRESHOLD", "2"))
LEAF_RETRIEVE_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))

TRACE_OUTPUT_FIELDS = [
    "id",
    "text",
    "filename",
    "file_type",
    "page_number",
    "chunk_id",
    "parent_chunk_id",
    "root_chunk_id",
    "chunk_level",
    "chunk_idx",
]

# 全局初始化检索依赖（与 api 共用 embedding_service，保证 BM25 状态一致）
_milvus_manager = MilvusManager()
_parent_chunk_store = ParentChunkStore()
_document_page_store = DocumentPageStore()

_stepback_model = None


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _parse_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _parse_retrieval_mode(value: str | None) -> str:
    mode = (value or "baseline").strip().lower()
    if mode not in {"baseline", "finance_experimental"}:
        return "baseline"
    return mode


def get_finance_rag_config() -> Dict[str, Any]:
    retrieval_mode = _parse_retrieval_mode(os.getenv("RAG_RETRIEVAL_MODE"))
    candidate_k = max(1, _parse_int(os.getenv("FINANCE_RAG_CANDIDATE_K"), 50))
    final_top_k = max(1, _parse_int(os.getenv("FINANCE_RAG_FINAL_TOP_K"), 10))
    experimental_two_stage = _parse_bool(os.getenv("FINANCE_RAG_TWO_STAGE_RETRIEVAL"), True)
    return {
        "retrieval_mode": retrieval_mode,
        "candidate_k": max(candidate_k, final_top_k),
        "final_top_k": final_top_k,
        "enable_step_back": _parse_bool(os.getenv("FINANCE_RAG_ENABLE_STEP_BACK"), False),
        "enable_page_merge": _parse_bool(os.getenv("FINANCE_RAG_ENABLE_PAGE_MERGE"), True),
        "adjacent_page_window": max(0, _parse_int(os.getenv("FINANCE_RAG_ADJACENT_PAGE_WINDOW"), 1)),
        "adjacent_chunk_window": max(0, _parse_int(os.getenv("FINANCE_RAG_ADJACENT_CHUNK_WINDOW"), 1)),
        "two_stage_retrieval": retrieval_mode == "finance_experimental" and experimental_two_stage,
        "doc_stage_top_n": max(1, _parse_int(os.getenv("FINANCE_RAG_DOC_STAGE_TOP_N"), 5)),
        "page_stage_top_n": max(1, _parse_int(os.getenv("FINANCE_RAG_PAGE_STAGE_TOP_N"), 10)),
        "max_evidence_pack_used": max(1, _parse_int(os.getenv("FINANCE_RAG_MAX_EVIDENCE_PACK_USED"), 10)),
        "min_evidence_pack_used": max(1, _parse_int(os.getenv("FINANCE_RAG_MIN_EVIDENCE_PACK_USED"), 6)),
        "max_page_text_chars": max(200, _parse_int(os.getenv("FINANCE_RAG_MAX_PAGE_TEXT_CHARS"), 2500)),
        "max_table_text_chars": max(200, _parse_int(os.getenv("FINANCE_RAG_MAX_TABLE_TEXT_CHARS"), 2500)),
        "w_dense": float(os.getenv("FINANCE_RAG_W_DENSE", "0.35")),
        "w_keyword": float(os.getenv("FINANCE_RAG_W_KEYWORD", "0.20")),
        "w_metric": float(os.getenv("FINANCE_RAG_W_METRIC", "0.15")),
        "w_number": float(os.getenv("FINANCE_RAG_W_NUMBER", "0.10")),
        "w_company": float(os.getenv("FINANCE_RAG_W_COMPANY", "0.15")),
        "w_year": float(os.getenv("FINANCE_RAG_W_YEAR", "0.10")),
        "w_doc_type": float(os.getenv("FINANCE_RAG_W_DOC_TYPE", "0.05")),
        "cover_toc_penalty": float(os.getenv("FINANCE_RAG_COVER_TOC_PENALTY", "0.40")),
    }


def get_doc_name(filename: str) -> str:
    return normalize_doc_name(filename or "")


def _escape_milvus_string(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace('"', '\\"')


def _doc_key(doc: dict) -> tuple:
    return (
        doc.get("chunk_id") or "",
        doc.get("filename") or "",
        doc.get("page_number"),
        doc.get("chunk_idx"),
        doc.get("text") or "",
    )


def _deduplicate_docs(docs: List[dict]) -> List[dict]:
    out: List[dict] = []
    seen = set()
    for doc in docs:
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _combined_score(doc: dict) -> float:
    for field in ("rerank_score", "score"):
        try:
            value = doc.get(field)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


_COVER_TEXT_HINTS = (
    "table of contents",
    "index to consolidated financial statements",
    "united states securities and exchange commission",
    "washington, d.c. 20549",
    "commission file number",
    "form 10-k",
    "form 10-q",
    "annual report pursuant to section 13 or 15(d)",
    "quarterly report pursuant to section 13 or 15(d)",
)

_COVER_QUERY_HINTS = (
    "cover",
    "title page",
    "filing date",
    "filed",
    "commission file number",
    "cik",
    "form 10-k",
    "form 10-q",
    "exchange commission",
)

def _query_targets_cover_metadata(query: str) -> bool:
    lowered = (query or "").strip().lower()
    return any(hint in lowered for hint in _COVER_QUERY_HINTS)


def _looks_like_cover_or_toc_chunk(doc: dict, query: str) -> bool:
    if _query_targets_cover_metadata(query):
        return False

    text = (doc.get("text") or "").lower()
    if not text:
        return False

    page_number = _coerce_int(doc.get("page_number"))
    cover_hint_hits = sum(1 for hint in _COVER_TEXT_HINTS if hint in text)

    if "table of contents" in text or "index to consolidated financial statements" in text:
        return True
    if cover_hint_hits >= 2:
        return True
    if page_number is not None and page_number <= 0 and cover_hint_hits >= 1:
        return True
    return False


def _filter_cover_or_toc_docs(query: str, docs: List[dict]) -> Tuple[List[dict], int]:
    filtered: List[dict] = []
    removed = 0
    for doc in docs:
        if _looks_like_cover_or_toc_chunk(doc, query):
            removed += 1
            continue
        filtered.append(doc)
    return filtered, removed


def _page_zero_count(docs: List[dict]) -> int:
    return sum(1 for doc in docs if (_coerce_int(doc.get("page_number")) or 0) <= 0)


def _page_distribution(docs: List[dict]) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    for doc in docs:
        page = doc.get("page_number")
        key = "" if page is None else str(page)
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def _dot_product(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))


def _sparse_dot(left: dict, right: dict) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return float(sum(float(value) * float(right.get(key, 0.0)) for key, value in left.items()))


def _extract_numbers(text: str) -> set[str]:
    return _feature_extract_numbers(text)


def _extract_years(text: str) -> set[str]:
    return _feature_extract_years(text)


def _extract_keyword_tokens(text: str) -> set[str]:
    return _feature_extract_keyword_tokens(text)


def _extract_metric_hints(text: str) -> set[str]:
    return _feature_extract_metric_hints(text)


def _score_overlap(query_values: set[str], page_values: set[str]) -> float:
    if not query_values:
        return 0.0
    return len(query_values & page_values) / max(1, len(query_values))


def _safe_preview(text: str, limit: int) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def _contains_company_alias(text: str, aliases: List[str]) -> bool:
    original = text or ""
    lowered = original.lower()
    for alias in aliases:
        if not alias:
            continue
        if alias.isupper() and len(alias) <= 4:
            import re as _re

            if _re.search(rf"(?<![A-Za-z0-9]){_re.escape(alias)}(?![A-Za-z0-9])", original):
                return True
            continue
        if alias.lower() in lowered:
            return True
    return False


def _company_match_score(query_parse: Dict[str, Any], filename: str, page_text: str, table_text: str = "") -> float:
    company = query_parse.get("company") or ""
    if not company:
        return 0.0
    haystack = "\n".join([filename or "", page_text or "", table_text or ""])
    return 1.0 if matches_company_text(haystack, company) else 0.0


def _year_match_score(query_parse: Dict[str, Any], filename: str, page_years: set[str], page_text: str) -> float:
    years = {str(year) for year in (query_parse.get("years") or [])}
    if not years:
        return 0.0
    filename_years = _extract_years(filename)
    text_years = page_years or _extract_years(page_text)
    if years & (filename_years | text_years):
        return 1.0
    return 0.0


def _doc_type_match_score(query_parse: Dict[str, Any], filename: str, page_text: str) -> float:
    query_doc_type = query_parse.get("doc_type") or ""
    if not query_doc_type:
        return 0.0
    page_doc_type = infer_doc_type(f"{filename}\n{page_text}")
    return 1.0 if page_doc_type == query_doc_type else 0.0


def _normalize_component_scores(items: List[dict], field_name: str, target_name: str) -> None:
    values = [float(item.get(field_name, 0.0) or 0.0) for item in items]
    max_value = max(values) if values else 0.0
    for item in items:
        raw = float(item.get(field_name, 0.0) or 0.0)
        item[target_name] = raw / max_value if max_value > 0 else 0.0


def _estimate_prompt_chars(entries: List[dict]) -> int:
    return sum(len((entry.get("text") or "").strip()) for entry in entries)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _build_filename_filter(filenames: List[str]) -> str:
    clauses = [
        f'filename == "{_escape_milvus_string(filename)}"'
        for filename in filenames
        if filename
    ]
    if not clauses:
        return f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    return f"({' or '.join(clauses)}) and chunk_level == {LEAF_RETRIEVE_LEVEL}"


def _get_rerank_endpoint() -> str:
    if not RERANK_BINDING_HOST:
        return ""
    host = RERANK_BINDING_HOST.strip().rstrip("/")
    return host if host.endswith("/v1/rerank") else f"{host}/v1/rerank"


def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    merge_parent_ids = [parent_id for parent_id, children in groups.items() if len(children) >= threshold]
    if not merge_parent_ids:
        return docs, 0

    parent_docs = _parent_chunk_store.get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    merged_docs: List[dict] = []
    merged_count = 0
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            continue
        parent_doc = dict(parent_map[parent_id])
        score = doc.get("score")
        if score is not None:
            parent_doc["score"] = max(float(parent_doc.get("score", score)), float(score))
        rerank_score = doc.get("rerank_score")
        if rerank_score is not None:
            parent_doc["rerank_score"] = max(float(parent_doc.get("rerank_score", rerank_score)), float(rerank_score))
        parent_doc["merged_from_children"] = True
        parent_doc["merged_child_count"] = len(groups[parent_id])
        merged_docs.append(parent_doc)
        merged_count += 1

    return _deduplicate_docs(merged_docs), merged_count


def _auto_merge_documents(docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    if not AUTO_MERGE_ENABLED or not docs:
        return docs[:top_k], {
            "auto_merge_enabled": AUTO_MERGE_ENABLED,
            "auto_merge_applied": False,
            "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }

    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(merged_docs, threshold=AUTO_MERGE_THRESHOLD)

    merged_docs.sort(key=_combined_score, reverse=True)
    merged_docs = merged_docs[:top_k]

    replaced_count = merged_count_l3_l2 + merged_count_l2_l1
    return merged_docs, {
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": replaced_count > 0,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": replaced_count,
        "auto_merge_steps": int(merged_count_l3_l2 > 0) + int(merged_count_l2_l1 > 0),
    }


def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]
    meta: Dict[str, Any] = {
        "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": _get_rerank_endpoint(),
        "rerank_error": None,
        "candidate_count": len(docs_with_rank),
    }
    if not docs_with_rank or not meta["rerank_enabled"]:
        return docs_with_rank[:top_k], meta

    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [doc.get("text", "") for doc in docs_with_rank],
        "top_n": min(top_k, len(docs_with_rank)),
        "return_documents": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RERANK_API_KEY}",
    }
    try:
        meta["rerank_applied"] = True
        response = requests.post(
            meta["rerank_endpoint"],
            headers=headers,
            json=payload,
            timeout=15,
        )
        if response.status_code >= 400:
            meta["rerank_error"] = f"HTTP {response.status_code}: {response.text}"
            return docs_with_rank[:top_k], meta

        items = response.json().get("results", [])
        reranked = []
        for item in items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs_with_rank):
                doc = dict(docs_with_rank[idx])
                score = item.get("relevance_score")
                if score is not None:
                    doc["rerank_score"] = score
                reranked.append(doc)

        if reranked:
            return reranked[:top_k], meta

        meta["rerank_error"] = "empty_rerank_results"
        return docs_with_rank[:top_k], meta
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        meta["rerank_error"] = str(e)
        return docs_with_rank[:top_k], meta


def _select_doc_stage_documents(candidate_docs: List[dict], doc_stage_top_n: int, query_parse: Dict[str, Any]) -> List[dict]:
    stats: Dict[str, dict] = {}
    target_years = {str(year) for year in (query_parse.get("years") or [])}
    target_doc_type = (query_parse.get("doc_types") or [query_parse.get("doc_type") or ""])[0]
    target_company = query_parse.get("company") or ""
    for doc in candidate_docs:
        filename = doc.get("filename") or ""
        if not filename:
            continue
        score = _combined_score(doc)
        filename_text = filename
        company_match = 1.0 if target_company and matches_company_text(filename_text, target_company) else 0.0
        year_match = 1.0 if target_years and (_extract_years(filename_text) & target_years) else 0.0
        doc_type_match = 1.0 if target_doc_type and infer_doc_type(filename_text) == target_doc_type else 0.0
        item = stats.setdefault(
            filename,
            {
                "filename": filename,
                "doc_name": get_doc_name(filename),
                "best_score": score,
                "hit_count": 0,
                "company_match_score": company_match,
                "year_match_score": year_match,
                "doc_type_match_score": doc_type_match,
            },
        )
        item["hit_count"] += 1
        item["best_score"] = max(item["best_score"], score)
        item["company_match_score"] = max(item["company_match_score"], company_match)
        item["year_match_score"] = max(item["year_match_score"], year_match)
        item["doc_type_match_score"] = max(item["doc_type_match_score"], doc_type_match)
        item["doc_stage_score"] = (
            item["best_score"]
            + 0.95 * item["company_match_score"]
            + 0.45 * item["year_match_score"]
            + 0.25 * item["doc_type_match_score"]
        )

    ranked = sorted(
        stats.values(),
        key=lambda item: (
            item.get("doc_stage_score", item["best_score"]),
            item["hit_count"],
            item["best_score"],
            item["filename"],
        ),
        reverse=True,
    )
    if target_company:
        company_ranked = [item for item in ranked if item.get("company_match_score", 0.0) > 0]
        if company_ranked:
            ranked = company_ranked + [item for item in ranked if item.get("company_match_score", 0.0) <= 0]
    if target_company and target_years:
        company_year_ranked = [
            item for item in ranked
            if item.get("company_match_score", 0.0) > 0 and item.get("year_match_score", 0.0) > 0
        ]
        if company_year_ranked:
            ranked = company_year_ranked + [
                item for item in ranked
                if not (item.get("company_match_score", 0.0) > 0 and item.get("year_match_score", 0.0) > 0)
            ]
    return ranked[:doc_stage_top_n]


def _retrieve_leaf_chunks(
    query: str,
    *,
    top_k: int,
    filter_expr: str,
    retrieval_scope: str,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "retrieval_scope": retrieval_scope,
        "retrieval_mode": "failed",
        "candidate_count": 0,
        "retrieve_error": None,
    }
    try:
        dense_embeddings = _embedding_service.get_embeddings([query])
        dense_embedding = dense_embeddings[0]
        sparse_embedding = _embedding_service.get_sparse_embedding(query)
        retrieved = _milvus_manager.hybrid_retrieve(
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            top_k=top_k,
            filter_expr=filter_expr,
        )
        meta["retrieval_mode"] = "hybrid"
        meta["candidate_count"] = len(retrieved)
        return {"docs": retrieved, "meta": meta}
    except Exception as exc:
        meta["retrieve_error"] = f"hybrid:{exc}"
        try:
            dense_embeddings = _embedding_service.get_embeddings([query])
            dense_embedding = dense_embeddings[0]
            retrieved = _milvus_manager.dense_retrieve(
                dense_embedding=dense_embedding,
                top_k=top_k,
                filter_expr=filter_expr,
            )
            meta["retrieval_mode"] = "dense_fallback"
            meta["candidate_count"] = len(retrieved)
            return {"docs": retrieved, "meta": meta}
        except Exception as dense_exc:
            meta["retrieve_error"] = f"{meta['retrieve_error']}; dense:{dense_exc}"
            return {"docs": [], "meta": meta}


def _should_trigger_two_stage_fallback(page_stage_candidates: List[dict], final_top_k: int, query: str) -> bool:
    if len(page_stage_candidates) < final_top_k:
        return True
    inspection_window = page_stage_candidates[: min(len(page_stage_candidates), max(3, final_top_k // 2))]
    if not inspection_window:
        return True

    page_zero_count = _page_zero_count(inspection_window)
    cover_like_count = sum(1 for doc in inspection_window if _looks_like_cover_or_toc_chunk(doc, query))
    return page_zero_count >= max(2, (len(inspection_window) + 1) // 2) or cover_like_count >= max(
        1,
        len(inspection_window) // 2,
    )


def _fetch_document_pages(selected_docs: List[dict]) -> List[dict]:
    return _document_page_store.get_pages_by_filenames([doc.get("filename") or "" for doc in selected_docs])


def _score_candidate_pages(
    query: str,
    page_records: List[dict],
    candidate_docs: List[dict],
    query_parse: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[List[dict], Dict[str, Any]]:
    if not page_records:
        return [], {
            "page_embedding_cache_hit_count": 0,
            "page_embedding_cache_miss_count": 0,
            "query_time_page_embedding_executed": False,
        }

    initial_page_score_map: Dict[tuple, float] = {}
    for doc in candidate_docs:
        key = (doc.get("filename") or "", _coerce_int(doc.get("page_number")) or 0)
        initial_page_score_map[key] = max(initial_page_score_map.get(key, 0.0), _combined_score(doc))

    query_embedding = _embedding_service.get_embeddings([query])[0]
    query_numbers = _extract_numbers(query)
    query_years = _extract_years(query)
    query_metrics = _extract_metric_hints(query)
    query_tokens = _extract_keyword_tokens(query)
    target_company = query_parse.get("company") or ""
    target_years = set(str(year) for year in (query_parse.get("years") or []))
    target_doc_types = set(query_parse.get("doc_types") or ([] if not query_parse.get("doc_type") else [query_parse.get("doc_type")]))

    page_details: List[dict] = []
    for page in page_records:
        page_text = page.get("page_text", "") or ""
        table_text = page.get("table_text", "") or ""
        dense_embedding = page.get("page_dense_embedding") or []
        combined_text = "\n".join([page_text, table_text]).strip()
        table_tokens = _extract_keyword_tokens(table_text)
        table_numbers = _extract_numbers(table_text)
        table_years = _extract_years(table_text)
        table_metrics = _extract_metric_hints(table_text)
        page_tokens = set(page.get("page_tokens") or _extract_keyword_tokens(combined_text))
        page_numbers = set(page.get("page_numbers") or _extract_numbers(combined_text))
        page_years = set(page.get("page_years") or _extract_years(combined_text))
        page_metrics = set(page.get("page_metric_tokens") or _extract_metric_hints(combined_text))
        company_match_score = _company_match_score(query_parse, page.get("filename", ""), page_text, table_text)
        year_match_score = _year_match_score(query_parse, page.get("filename", ""), page_years, page_text)
        doc_type_match_score = _doc_type_match_score(query_parse, page.get("filename", ""), page_text)
        token_overlap = _score_overlap(query_tokens, page_tokens)
        metric_overlap = _score_overlap(query_metrics, page_metrics)
        number_overlap = max(_score_overlap(query_numbers, page_numbers), _score_overlap(query_years, page_years))
        table_metric_overlap = _score_overlap(query_metrics, table_metrics)
        table_number_overlap = max(_score_overlap(query_numbers, table_numbers), _score_overlap(query_years, table_years))
        table_keyword_overlap = _score_overlap(query_tokens, table_tokens)
        table_signal_score = 0.0
        if table_text.strip():
            table_signal_score = max(
                table_metric_overlap,
                table_number_overlap,
                table_keyword_overlap * 0.5,
            )
        initial_chunk_hit_score = initial_page_score_map.get(
            ((page.get("filename") or ""), _coerce_int(page.get("page_number")) or 0),
            0.0,
        )
        dense_raw = max(0.0, _dot_product(query_embedding, dense_embedding)) if dense_embedding else 0.0
        keyword_raw = token_overlap + metric_overlap + number_overlap
        cover_penalty = config["cover_toc_penalty"] if _looks_like_cover_or_toc_chunk(
            {"page_number": page.get("page_number"), "text": page_text},
            query,
        ) else 0.0
        penalty_reasons: List[str] = []
        filter_reason = ""
        if cover_penalty > 0:
            penalty_reasons.append("cover_or_toc")
        if target_company and company_match_score <= 0:
            penalty_reasons.append("company_mismatch")
        if target_years and year_match_score <= 0:
            penalty_reasons.append("year_mismatch")
        if target_doc_types and doc_type_match_score <= 0:
            penalty_reasons.append("doc_type_mismatch")
        if not dense_embedding:
            penalty_reasons.append("missing_page_dense_embedding")
        if len(page_text.strip()) < 40:
            penalty_reasons.append("short_text")

        page_details.append(
            {
                **page,
                "dense_raw_score": dense_raw,
                "keyword_raw_score": keyword_raw,
                "number_overlap_score": number_overlap,
                "metric_overlap_score": metric_overlap,
                "company_match_score": company_match_score,
                "year_match_score": year_match_score,
                "doc_type_match_score": doc_type_match_score,
                "table_signal_score": table_signal_score,
                "has_table_text": bool(table_text.strip()),
                "initial_chunk_hit_score": initial_chunk_hit_score,
                "cover_toc_penalty": cover_penalty,
                "cover_like": cover_penalty > 0,
                "filter_reason": filter_reason,
                "penalty_reason": ", ".join(penalty_reasons),
                "has_dense_embedding": bool(dense_embedding),
            }
        )

    _normalize_component_scores(page_details, "dense_raw_score", "dense_score")
    _normalize_component_scores(page_details, "keyword_raw_score", "keyword_score")
    _normalize_component_scores(page_details, "initial_chunk_hit_score", "initial_chunk_hit_score_norm")

    scored_pages: List[dict] = []
    for page in page_details:
        final_score = (
            config["w_dense"] * page["dense_score"]
            + config["w_keyword"] * page["keyword_score"]
            + config["w_metric"] * page["metric_overlap_score"]
            + config["w_number"] * page["number_overlap_score"]
            + config["w_company"] * page["company_match_score"]
            + config["w_year"] * page["year_match_score"]
            + config["w_doc_type"] * page["doc_type_match_score"]
            + 0.08 * page["initial_chunk_hit_score_norm"]
            + 0.12 * page["table_signal_score"]
            - page["cover_toc_penalty"]
        )
        if target_company and page["company_match_score"] <= 0:
            final_score -= 0.85
        if target_years and page["year_match_score"] <= 0:
            final_score -= 0.45
        if target_doc_types and page["doc_type_match_score"] <= 0:
            final_score -= 0.18
        if not page["has_dense_embedding"]:
            final_score -= 0.35
            page["filter_reason"] = "missing_page_dense_embedding"
        if page["cover_like"]:
            page["filter_reason"] = "cover_or_toc" if not page["filter_reason"] else page["filter_reason"]

        scored_pages.append(
            {
                **page,
                "dense_score": round(page["dense_score"], 6),
                "keyword_score": round(page["keyword_score"], 6),
                "metric_overlap_score": round(page["metric_overlap_score"], 6),
                "number_overlap_score": round(page["number_overlap_score"], 6),
                "company_match_score": round(page["company_match_score"], 6),
                "year_match_score": round(page["year_match_score"], 6),
                "doc_type_match_score": round(page["doc_type_match_score"], 6),
                "table_signal_score": round(page["table_signal_score"], 6),
                "has_table_text": page["has_table_text"],
                "initial_chunk_hit_score": round(page["initial_chunk_hit_score_norm"], 6),
                "final_score": round(final_score, 6),
                "page_score": round(final_score, 6),
                "doc_name": page.get("doc_name") or get_doc_name(page.get("filename", "")),
                "filter_reason": page.get("filter_reason", ""),
                "penalty_reason": page.get("penalty_reason", ""),
            }
        )

    scored_pages.sort(key=lambda item: item.get("page_score", 0.0), reverse=True)
    hit_count = sum(1 for page in page_records if page.get("page_dense_embedding"))
    miss_count = max(0, len(page_records) - hit_count)
    return scored_pages, {
        "page_embedding_cache_hit_count": hit_count,
        "page_embedding_cache_miss_count": miss_count,
        "query_time_page_embedding_executed": False,
    }


def _build_evidence_pack(
    selected_pages: List[dict],
    all_page_records: List[dict],
    *,
    query_parse: Dict[str, Any],
    config: Dict[str, Any],
    final_top_k: int,
    enable_page_merge: bool,
    adjacent_page_window: int,
) -> Tuple[List[dict], Dict[str, Any]]:
    if not selected_pages:
        return [], {
            "page_merge_applied": False,
            "merged_chunk_count": 0,
            "final_context_chunk_count": 0,
            "final_evidence_pack_debug": [],
            "final_evidence_pack_used": [],
            "final_evidence_pack_debug_count": 0,
            "final_evidence_pack_used_count": 0,
            "dropped_evidence_count": 0,
            "dropped_reasons": {},
            "prompt_context_char_count_estimate": 0,
            "fallback_reason": "",
        }

    pages_by_key = {
        ((page.get("filename") or ""), _coerce_int(page.get("page_number")) or 0): page
        for page in all_page_records
    }
    evidence_pack_debug: List[dict] = []
    seen = set()

    def _append_entry(entry: dict) -> None:
        key = (
            entry.get("type") or "",
            entry.get("filename") or "",
            entry.get("page_number"),
            entry.get("chunk_id") or "",
            entry.get("text") or "",
        )
        if key in seen or not (entry.get("text") or "").strip():
            return
        seen.add(key)
        evidence_pack_debug.append(entry)

    max_context_entries = max(final_top_k * 3, 24)
    for page in selected_pages:
        filename = page.get("filename") or ""
        doc_name = page.get("doc_name") or get_doc_name(filename)
        page_number = _coerce_int(page.get("page_number")) or 0
        page_score = float(page.get("page_score", 0.0) or 0.0)

        _append_entry(
            {
                "filename": filename,
                "doc_name": doc_name,
                "page_number": page_number,
                "type": "page_text",
                "text": _safe_preview(page.get("page_text", "") or "", config["max_page_text_chars"]),
                "score": page_score,
                "rerank_score": page_score,
                "page_score": page_score,
                "source": f"{filename}#page={page_number}#page_text",
            }
        )

        table_text = (page.get("table_text") or "").strip()
        if table_text:
            _append_entry(
                {
                    "filename": filename,
                    "doc_name": doc_name,
                    "page_number": page_number,
                    "type": "table_text",
                    "text": _safe_preview(table_text, config["max_table_text_chars"]),
                    "score": page_score,
                    "rerank_score": page_score,
                    "page_score": page_score,
                    "source": f"{filename}#page={page_number}#table_text",
                }
            )

        page_window = adjacent_page_window if enable_page_merge else 0
        for target_page in range(page_number - page_window, page_number + page_window + 1):
            if target_page < 0:
                continue
            page_record = pages_by_key.get((filename, target_page))
            if not page_record:
                continue
            chunk_ids = [item for item in page_record.get("chunk_ids", []) if item]
            for chunk in _milvus_manager.get_chunks_by_ids(chunk_ids):
                _append_entry(
                    {
                        **chunk,
                        "doc_name": doc_name,
                        "page_number": _coerce_int(chunk.get("page_number")) or target_page,
                        "type": "chunk",
                        "text": _safe_preview(chunk.get("text", "") or "", 1200),
                        "score": page_score,
                        "rerank_score": page_score,
                        "page_score": page_score,
                        "source": f"{filename}#page={target_page}#chunk",
                    }
                )
                if len(evidence_pack_debug) >= max_context_entries:
                    break
            if len(evidence_pack_debug) >= max_context_entries:
                break
        if len(evidence_pack_debug) >= max_context_entries:
            break

    company_filtered = [
        item for item in evidence_pack_debug
        if not query_parse.get("company") or matches_company_text(
            "\n".join([item.get("filename", ""), item.get("doc_name", ""), item.get("text", "")]),
            query_parse.get("company") or "",
        )
    ]
    fallback_reason = ""
    working = company_filtered
    if not working:
        working = evidence_pack_debug
        if query_parse.get("company"):
            fallback_reason = "no_company_matched_evidence"

    seen_page_type = set()
    used_pack: List[dict] = []
    dropped_reasons: Dict[str, int] = defaultdict(int)
    target_years = set(str(year) for year in (query_parse.get("years") or []))
    target_doc_types = set(query_parse.get("doc_types") or [])

    def _entry_priority(entry: dict) -> tuple:
        filename = entry.get("filename", "") or ""
        text = entry.get("text", "") or ""
        entry_years = _extract_years("\n".join([filename, text]))
        year_match = 1 if (not target_years or bool(target_years & entry_years)) else 0
        company_match = 1 if (not query_parse.get("company") or matches_company_text("\n".join([filename, text]), query_parse.get("company") or "")) else 0
        doc_type_match = 1 if (not target_doc_types or infer_doc_type("\n".join([filename, text])) in target_doc_types) else 0
        table_bonus = 0
        if entry.get("type") == "table_text":
            table_bonus = 1
            table_bonus += len(_extract_metric_hints(text) & set(query_parse.get("metrics") or []))
            table_bonus += len(_extract_numbers(text) & set(query_parse.get("numbers") or []))
        metric_bonus = len(_extract_metric_hints(text) & set(query_parse.get("metrics") or []))
        number_bonus = len(_extract_numbers(text) & set(query_parse.get("numbers") or []))
        return (
            company_match,
            year_match,
            doc_type_match,
            metric_bonus,
            number_bonus,
            table_bonus,
            float(entry.get("page_score", entry.get("score", 0.0)) or 0.0),
        )

    for entry in sorted(working, key=_entry_priority, reverse=True):
        filename = entry.get("filename") or ""
        page_number = _coerce_int(entry.get("page_number")) or 0
        page_type_key = (filename, page_number, entry.get("type") or "")
        if page_type_key in seen_page_type:
            dropped_reasons["duplicate_page_type"] += 1
            continue
        if _looks_like_cover_or_toc_chunk({"page_number": page_number, "text": entry.get("text", "")}, query_parse.get("raw_question", "")):
            dropped_reasons["cover_or_toc"] += 1
            continue
        if query_parse.get("years"):
            entry_years = _extract_years("\n".join([filename, entry.get("text", "")]))
            if entry_years and not (set(query_parse.get("years") or []) & entry_years):
                dropped_reasons["year_mismatch"] += 1
                continue
        if len((entry.get("text") or "").strip()) < 30:
            dropped_reasons["too_short"] += 1
            continue
        seen_page_type.add(page_type_key)
        used_pack.append(entry)
        if len(used_pack) >= config["max_evidence_pack_used"]:
            break

    if len(used_pack) < config["min_evidence_pack_used"]:
        for entry in sorted(evidence_pack_debug, key=_entry_priority, reverse=True):
            filename = entry.get("filename") or ""
            page_number = _coerce_int(entry.get("page_number")) or 0
            page_type_key = (filename, page_number, entry.get("type") or "")
            if page_type_key in seen_page_type:
                continue
            seen_page_type.add(page_type_key)
            used_pack.append(entry)
            if len(used_pack) >= config["min_evidence_pack_used"]:
                break
        if len(used_pack) < config["min_evidence_pack_used"]:
            fallback_reason = fallback_reason or "insufficient_same_company_year_metric_evidence"

    prompt_chars = _estimate_prompt_chars(used_pack)
    merged_chunk_count = max(0, len(evidence_pack_debug[:max_context_entries]) - min(len(selected_pages), final_top_k))
    return used_pack, {
        "page_merge_applied": merged_chunk_count > 0,
        "merged_chunk_count": merged_chunk_count,
        "final_context_chunk_count": len(used_pack),
        "final_evidence_pack_debug": evidence_pack_debug[:max_context_entries],
        "final_evidence_pack_used": used_pack,
        "final_evidence_pack_debug_count": len(evidence_pack_debug[:max_context_entries]),
        "final_evidence_pack_used_count": len(used_pack),
        "dropped_evidence_count": max(0, len(evidence_pack_debug[:max_context_entries]) - len(used_pack)),
        "dropped_reasons": dict(dropped_reasons),
        "prompt_context_char_count_estimate": prompt_chars,
        "fallback_reason": fallback_reason,
    }


def _as_evidence_chunk(doc: dict, default_score: float | None = None) -> dict:
    filename = doc.get("filename", "") or ""
    page_number = _coerce_int(doc.get("page_number"))
    score = _combined_score(doc) if default_score is None else default_score
    return {
        **doc,
        "filename": filename,
        "doc_name": doc.get("doc_name") or get_doc_name(filename),
        "page_number": page_number if page_number is not None else doc.get("page_number", ""),
        "type": doc.get("type") or "chunk",
        "score": score,
        "rerank_score": doc.get("rerank_score", score),
        "source": doc.get("source") or f"{filename}#page={page_number if page_number is not None else 'unknown'}#chunk",
    }


def _run_two_stage_retrieval(
    query: str,
    candidate_docs: List[dict],
    *,
    query_parse: Dict[str, Any],
    config: Dict[str, Any],
    doc_stage_top_n: int,
    page_stage_top_n: int,
    final_top_k: int,
) -> Tuple[List[dict], Dict[str, Any]]:
    selected_docs = _select_doc_stage_documents(candidate_docs, doc_stage_top_n, query_parse)
    page_records = _fetch_document_pages(selected_docs)
    scored_pages, page_perf_meta = _score_candidate_pages(query, page_records, candidate_docs, query_parse, config)
    filtered_pages = [page for page in scored_pages if not page.get("cover_like") and page.get("filter_reason") != "missing_page_dense_embedding"]
    company_confident = float(query_parse.get("company_confidence") or 0.0) >= 0.8
    company_matched_pages = [page for page in filtered_pages if page.get("company_match_score", 0.0) > 0]
    if company_confident and company_matched_pages:
        filtered_pages = company_matched_pages
    year_matched_pages = [page for page in filtered_pages if page.get("year_match_score", 0.0) > 0]
    if company_confident and year_matched_pages:
        filtered_pages = year_matched_pages
    cover_page_filtered_count = max(0, len(scored_pages) - len(filtered_pages))
    selected_pages = (filtered_pages or scored_pages)[:page_stage_top_n]

    page_chunk_candidates: List[dict] = []
    for page in selected_pages:
        chunk_ids = [item for item in page.get("chunk_ids", []) if item]
        for chunk in _milvus_manager.get_chunks_by_ids(chunk_ids):
            page_chunk_candidates.append(
                {
                    **chunk,
                    "doc_name": page.get("doc_name") or get_doc_name(page.get("filename", "")),
                    "type": "chunk",
                    "score": page.get("page_score", 0.0),
                    "rerank_score": page.get("page_score", 0.0),
                }
            )

    page_stage_candidates = _deduplicate_docs(page_chunk_candidates)
    fallback_used = _should_trigger_two_stage_fallback(page_stage_candidates, final_top_k, query)
    logger.info(
        (
            "finance_rag_two_stage selected_docs=%s page_records=%s selected_pages=%s "
            "page_stage_candidates=%s cover_page_filtered_count=%s fallback_used=%s"
        ),
        [item.get("filename") for item in selected_docs],
        len(page_records),
        [(item.get("filename"), item.get("page_number")) for item in selected_pages],
        len(page_stage_candidates),
        cover_page_filtered_count,
        fallback_used,
    )
    meta = {
        "doc_stage_selected_docs": selected_docs,
        "selected_docs": selected_docs,
        "selected_pages": [
            {
                "filename": page.get("filename"),
                "doc_name": page.get("doc_name"),
                "page_number": page.get("page_number"),
                "page_score": page.get("page_score"),
                "company_match_score": page.get("company_match_score"),
                "year_match_score": page.get("year_match_score"),
                "doc_type_match_score": page.get("doc_type_match_score"),
                "filter_reason": page.get("filter_reason"),
                "penalty_reason": page.get("penalty_reason"),
            }
            for page in selected_pages
        ],
        "selected_page_records": selected_pages,
        "page_scores": [
            {
                "filename": page.get("filename"),
                "doc_name": page.get("doc_name"),
                "page_number": page.get("page_number"),
                "page_score": page.get("page_score"),
                "dense_score": page.get("dense_score"),
                "keyword_score": page.get("keyword_score"),
                "metric_overlap_score": page.get("metric_overlap_score"),
                "number_overlap_score": page.get("number_overlap_score"),
                "company_match_score": page.get("company_match_score"),
                "year_match_score": page.get("year_match_score"),
                "doc_type_match_score": page.get("doc_type_match_score"),
                "table_signal_score": page.get("table_signal_score"),
                "has_table_text": page.get("has_table_text"),
                "initial_chunk_hit_score": page.get("initial_chunk_hit_score"),
                "cover_toc_penalty": page.get("cover_toc_penalty"),
                "final_score": page.get("final_score"),
                "filter_reason": page.get("filter_reason"),
                "penalty_reason": page.get("penalty_reason"),
            }
            for page in scored_pages[: max(page_stage_top_n * 2, final_top_k)]
        ],
        "cover_page_filtered_count": cover_page_filtered_count,
        "page_stage_candidate_count": len(page_stage_candidates),
        "page_stage_candidates": page_stage_candidates,
        "fallback_used": fallback_used,
        "all_page_records": page_records,
        **page_perf_meta,
    }
    return page_stage_candidates, meta


def _get_stepback_model():
    global _stepback_model
    if not ARK_API_KEY or not MODEL:
        return None
    if _stepback_model is None:
        _stepback_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=ARK_API_KEY,
            base_url=BASE_URL,
            temperature=0.2,
        )
    return _stepback_model


def _generate_step_back_question(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请为财务/年报问答生成一个更适合检索的补充问题，只能做轻量改写，不能泛化丢失原始约束。\n"
        "必须保留并突出：公司名、年份、季度、报表名称、指标名、金额单位、百分比、日期、问题中的专有名词。\n"
        "不要改写成通用财务原理问题，不要加入中文解释，不要删除原问题中的数字和限定词。\n"
        "只输出一句补充检索问题。\n"
        f"原始问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def _answer_step_back_question(step_back_question: str) -> str:
    model = _get_stepback_model()
    if not model or not step_back_question:
        return ""
    prompt = (
        "请仅用一句话补充该问题涉及的同义表达或财务表述，帮助检索，不要输出通用原理解释，"
        "不要引入原问题之外的新公司、新年份或新指标。\n"
        f"补充问题：{step_back_question}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def generate_hypothetical_document(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请基于用户问题生成一段‘假设性文档’，内容应像真实资料片段，"
        "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
        "只输出文档正文，不要标题或解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def step_back_expand(query: str) -> dict:
    step_back_question = _generate_step_back_question(query)
    step_back_answer = _answer_step_back_question(step_back_question)
    parts = [query]
    if step_back_question:
        parts.append(step_back_question)
    if step_back_answer:
        parts.append(step_back_answer)
    expanded_query = "\n".join(part for part in parts if part).strip() or query
    return {
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "expanded_query": expanded_query,
    }


def retrieve_candidate_documents(query: str, candidate_k: int | None = None) -> Dict[str, Any]:
    config = get_finance_rag_config()
    candidate_k = max(1, candidate_k or config["candidate_k"])
    logger.info(
        "finance_rag_candidate_retrieve candidate_k=%s final_top_k=%s query_length=%s",
        candidate_k,
        config["final_top_k"],
        len(query or ""),
    )
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    base_meta: Dict[str, Any] = {
        "retrieval_mode": "failed",
        "candidate_k": candidate_k,
        "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
        "candidate_count": 0,
        "retrieve_error": None,
        "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": _get_rerank_endpoint(),
        "rerank_error": None,
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": False,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": 0,
        "auto_merge_steps": 0,
    }
    started_at = time.perf_counter()
    retrieved = _retrieve_leaf_chunks(
        query,
        top_k=candidate_k,
        filter_expr=filter_expr,
        retrieval_scope="global",
    )
    meta = {**base_meta, **retrieved.get("meta", {})}
    meta["latency_breakdown"] = {
        "initial_retrieval_ms": round((time.perf_counter() - started_at) * 1000, 2),
    }
    return {"docs": retrieved.get("docs", []), "meta": meta}


def _fetch_neighbor_page_docs(filename: str, page_number: int, page_window: int) -> List[dict]:
    if page_window < 0:
        return []
    out: List[dict] = []
    escaped_filename = _escape_milvus_string(filename)
    for target_page in range(page_number - page_window, page_number + page_window + 1):
        if target_page < 0:
            continue
        filter_expr = (
            f'filename == "{escaped_filename}" and '
            f"page_number == {target_page} and "
            f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
        )
        out.extend(_milvus_manager.query_all(filter_expr=filter_expr, output_fields=TRACE_OUTPUT_FIELDS))
    return out


def _fetch_neighbor_chunk_docs(filename: str, chunk_idx: int, chunk_window: int) -> List[dict]:
    if chunk_window <= 0:
        return []
    out: List[dict] = []
    escaped_filename = _escape_milvus_string(filename)
    for target_idx in range(chunk_idx - chunk_window, chunk_idx + chunk_window + 1):
        if target_idx < 0:
            continue
        filter_expr = (
            f'filename == "{escaped_filename}" and '
            f"chunk_idx == {target_idx} and "
            f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
        )
        out.extend(_milvus_manager.query_all(filter_expr=filter_expr, output_fields=TRACE_OUTPUT_FIELDS))
    return out


def _merge_context_chunks(
    final_docs: List[dict],
    *,
    enable_page_merge: bool,
    adjacent_page_window: int,
    adjacent_chunk_window: int,
    final_top_k: int,
) -> Tuple[List[dict], Dict[str, Any]]:
    if not final_docs:
        return [], {
            "page_merge_applied": False,
            "merged_chunk_count": 0,
            "final_context_chunk_count": 0,
        }
    if not enable_page_merge:
        return final_docs[:final_top_k], {
            "page_merge_applied": False,
            "merged_chunk_count": 0,
            "final_context_chunk_count": len(final_docs[:final_top_k]),
        }

    extra_docs: List[dict] = []
    seen_pages = set()
    seen_chunks = set()

    for doc in final_docs:
        filename = doc.get("filename") or ""
        if not filename:
            continue

        page_number = _coerce_int(doc.get("page_number"))
        if page_number is not None:
            page_key = (filename, page_number)
            if page_key not in seen_pages:
                extra_docs.extend(_fetch_neighbor_page_docs(filename, page_number, adjacent_page_window))
                seen_pages.add(page_key)

        chunk_idx = _coerce_int(doc.get("chunk_idx"))
        if chunk_idx is not None and adjacent_chunk_window > 0:
            chunk_key = (filename, chunk_idx)
            if chunk_key not in seen_chunks:
                extra_docs.extend(_fetch_neighbor_chunk_docs(filename, chunk_idx, adjacent_chunk_window))
                seen_chunks.add(chunk_key)

    context_docs = _deduplicate_docs(final_docs + extra_docs)
    max_context_chunks = max(final_top_k, min(40, final_top_k * 3))
    context_docs = context_docs[:max_context_chunks]

    merged_chunk_count = max(0, len(context_docs) - len(final_docs))
    return context_docs, {
        "page_merge_applied": merged_chunk_count > 0,
        "merged_chunk_count": merged_chunk_count,
        "final_context_chunk_count": len(context_docs),
    }


def finalize_retrieved_documents(
    query: str,
    candidate_docs: List[dict],
    *,
    final_top_k: int | None = None,
    enable_page_merge: bool | None = None,
    adjacent_page_window: int | None = None,
    adjacent_chunk_window: int | None = None,
) -> Dict[str, Any]:
    config = get_finance_rag_config()
    retrieval_mode = config["retrieval_mode"]
    experimental_mode = retrieval_mode == "finance_experimental"
    final_top_k = max(1, final_top_k or config["final_top_k"])
    enable_page_merge = config["enable_page_merge"] if enable_page_merge is None else enable_page_merge
    adjacent_page_window = config["adjacent_page_window"] if adjacent_page_window is None else max(0, adjacent_page_window)
    adjacent_chunk_window = config["adjacent_chunk_window"] if adjacent_chunk_window is None else max(0, adjacent_chunk_window)
    doc_stage_top_n = config["doc_stage_top_n"]
    page_stage_top_n = max(config["page_stage_top_n"], final_top_k)
    logger.info(
        (
            "finance_rag_finalize retrieval_mode=%s candidate_k=%s final_top_k=%s candidate_docs=%s "
            "enable_page_merge=%s two_stage_retrieval=%s doc_stage_top_n=%s page_stage_top_n=%s"
        ),
        retrieval_mode,
        config["candidate_k"],
        final_top_k,
        len(candidate_docs),
        enable_page_merge,
        config["two_stage_retrieval"],
        doc_stage_top_n,
        page_stage_top_n,
    )

    deduped_candidates = _deduplicate_docs(candidate_docs)
    stage_two_meta: Dict[str, Any] = {
        "retrieval_mode": retrieval_mode,
        "two_stage_retrieval": config["two_stage_retrieval"],
        "doc_stage_top_n": doc_stage_top_n,
        "page_stage_top_n": page_stage_top_n,
        "query_parse": {},
        "doc_stage_selected_docs": [],
        "selected_docs": [],
        "selected_pages": [],
        "page_scores": [],
        "final_evidence_pack": [],
        "final_evidence_pack_debug": [],
        "final_evidence_pack_used": [],
        "page_stage_candidates": [],
        "page_stage_candidate_count": 0,
        "cover_page_filtered_count": 0,
        "fallback_used": False,
        "fallback_reason": "",
    }
    rerank_meta: Dict[str, Any]
    if experimental_mode:
        t_parse = time.perf_counter()
        query_parse = parse_finance_query(query)
        query_parse["raw_question"] = query
        query_parse_ms = round((time.perf_counter() - t_parse) * 1000, 2)
        stage_two_meta["query_parse"] = query_parse
        initial_rerank_top_k = max(final_top_k * 2, page_stage_top_n)
        initial_reranked_docs, initial_rerank_meta = _rerank_documents(
            query=query,
            docs=deduped_candidates,
            top_k=initial_rerank_top_k,
        )
    else:
        query_parse_ms = 0.0
        initial_reranked_docs = []
        initial_rerank_meta = {}

    if experimental_mode and config["two_stage_retrieval"]:
        t_page = time.perf_counter()
        stage_two_candidates, stage_two_rerank_meta = _run_two_stage_retrieval(
            query,
            deduped_candidates,
            query_parse=query_parse,
            config=config,
            doc_stage_top_n=doc_stage_top_n,
            page_stage_top_n=page_stage_top_n,
            final_top_k=final_top_k,
        )
        page_rerank_ms = round((time.perf_counter() - t_page) * 1000, 2)
        stage_two_meta.update(stage_two_rerank_meta)
        selected_pages = stage_two_meta.get("selected_page_records", []) or []
        all_page_records = stage_two_meta.get("all_page_records", []) or []
        evidence_pack_used, page_merge_meta = _build_evidence_pack(
            selected_pages,
            all_page_records,
            query_parse=query_parse,
            config=config,
            final_top_k=final_top_k,
            enable_page_merge=enable_page_merge,
            adjacent_page_window=adjacent_page_window,
        )
        fallback_candidates = [_as_evidence_chunk(doc) for doc in initial_reranked_docs]
        combined_pack = _deduplicate_docs(
            evidence_pack_used + stage_two_candidates + fallback_candidates
        )
        if len(evidence_pack_used) < final_top_k:
            stage_two_meta["fallback_used"] = True
            stage_two_meta["fallback_reason"] = stage_two_meta.get("fallback_reason") or "insufficient_used_evidence"
        context_docs = evidence_pack_used[: config["max_evidence_pack_used"]]
        final_docs = (context_docs + [doc for doc in combined_pack if doc not in context_docs])[:final_top_k]
        stage_two_meta["final_evidence_pack"] = page_merge_meta.get("final_evidence_pack_debug", [])
        stage_two_meta["final_evidence_pack_debug"] = page_merge_meta.get("final_evidence_pack_debug", [])
        stage_two_meta["final_evidence_pack_used"] = page_merge_meta.get("final_evidence_pack_used", [])
        page_merge_meta["final_context_chunk_count"] = len(context_docs)
        page_merge_meta["merged_chunk_count"] = max(0, len(page_merge_meta.get("final_evidence_pack_debug", [])) - len(context_docs))
        rerank_meta = {
            "rerank_enabled": True,
            "rerank_applied": True,
            "rerank_model": "page_hybrid_rerank",
            "rerank_endpoint": "local_page_hybrid_rerank",
            "rerank_error": None,
        }
        merge_meta = {
            "auto_merge_enabled": AUTO_MERGE_ENABLED,
            "auto_merge_applied": False,
            "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }
    else:
        page_rerank_ms = 0.0
        filtered_candidates, cover_page_filtered_count = _filter_cover_or_toc_docs(query, deduped_candidates)
        rerank_input = filtered_candidates or deduped_candidates
        stage_two_meta["cover_page_filtered_count"] = cover_page_filtered_count if filtered_candidates else 0
        reranked_docs, rerank_meta = _rerank_documents(
            query=query,
            docs=rerank_input,
            top_k=max(final_top_k * 2, final_top_k),
        )
        final_docs, merge_meta = _auto_merge_documents(docs=reranked_docs, top_k=final_top_k)
        if len(final_docs) < final_top_k and len(reranked_docs) >= final_top_k:
            final_docs = reranked_docs[:final_top_k]
            merge_meta = {
                **merge_meta,
                "auto_merge_applied": False,
                "auto_merge_replaced_chunks": 0,
                "auto_merge_steps": 0,
            }
        context_docs, page_merge_meta = _merge_context_chunks(
            final_docs,
            enable_page_merge=enable_page_merge,
            adjacent_page_window=adjacent_page_window,
            adjacent_chunk_window=adjacent_chunk_window,
            final_top_k=final_top_k,
        )
        stage_two_meta["final_evidence_pack_used"] = final_docs[: config["max_evidence_pack_used"]]
        stage_two_meta["final_evidence_pack_debug"] = context_docs
        stage_two_meta["final_evidence_pack"] = context_docs

    meta = {
        **stage_two_meta,
        **(initial_rerank_meta if experimental_mode and config["two_stage_retrieval"] else {}),
        **rerank_meta,
        **merge_meta,
        **page_merge_meta,
        "final_top_k": final_top_k,
        "final_candidate_count": len(deduped_candidates),
        "latency_breakdown": {
            "query_parse_ms": query_parse_ms,
            "page_rerank_ms": page_rerank_ms,
            "final_evidence_pack_used_count": len(stage_two_meta.get("final_evidence_pack_used", []) or []),
            "prompt_context_char_count_estimate": _estimate_prompt_chars(stage_two_meta.get("final_evidence_pack_used", []) or context_docs),
            "page_embedding_cache_hit_count": stage_two_meta.get("page_embedding_cache_hit_count", 0),
            "page_embedding_cache_miss_count": stage_two_meta.get("page_embedding_cache_miss_count", 0),
            "query_time_page_embedding_executed": stage_two_meta.get("query_time_page_embedding_executed", False),
        },
    }
    if meta["latency_breakdown"]["query_time_page_embedding_executed"]:
        logger.warning("WARNING: query-time page embedding executed; this should be cached/precomputed.")
    logger.info(
        (
            "finance_rag_finalize_result retrieval_mode=%s final_retrieved_count=%s final_page_distribution=%s "
            "final_page_zero_count=%s cover_page_filtered_count=%s fallback_used=%s "
            "query_parse_ms=%s page_rerank_ms=%s pack_used=%s prompt_chars=%s cache_hit=%s cache_miss=%s"
        ),
        retrieval_mode,
        len(final_docs),
        _page_distribution(final_docs),
        _page_zero_count(final_docs),
        meta.get("cover_page_filtered_count", 0),
        meta.get("fallback_used", False),
        query_parse_ms,
        page_rerank_ms,
        len(stage_two_meta.get("final_evidence_pack_used", []) or []),
        meta["latency_breakdown"]["prompt_context_char_count_estimate"],
        meta["latency_breakdown"]["page_embedding_cache_hit_count"],
        meta["latency_breakdown"]["page_embedding_cache_miss_count"],
    )
    return {
        "final_retrieved_docs": final_docs,
        "context_docs": context_docs,
        "meta": meta,
    }


def retrieve_documents(
    query: str,
    top_k: int = 5,
    *,
    candidate_k: int | None = None,
    apply_page_merge: bool | None = None,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    candidates = retrieve_candidate_documents(query, candidate_k=candidate_k)
    finalized = finalize_retrieved_documents(
        query,
        candidates.get("docs", []),
        final_top_k=top_k,
        enable_page_merge=apply_page_merge,
    )
    meta = {**candidates.get("meta", {}), **finalized.get("meta", {})}
    meta["latency_breakdown"] = {
        **(candidates.get("meta", {}).get("latency_breakdown", {}) or {}),
        **(finalized.get("meta", {}).get("latency_breakdown", {}) or {}),
        "total_retrieval_ms": round((time.perf_counter() - started_at) * 1000, 2),
    }
    return {
        "docs": finalized.get("context_docs", []),
        "candidate_docs": candidates.get("docs", []),
        "final_retrieved_docs": finalized.get("final_retrieved_docs", []),
        "context_docs": finalized.get("context_docs", []),
        "meta": meta,
    }


def debug_retrieval_pipeline(question: str, top_k: int = 10) -> Dict[str, Any]:
    result = retrieve_documents(question, top_k=top_k)
    meta = result.get("meta", {}) or {}

    def _trace_chunk(doc: dict, rank: int) -> dict:
        filename = doc.get("filename", "") or ""
        page_number = doc.get("page_number", "")
        return {
            "rank": rank,
            "filename": filename,
            "doc_name": doc.get("doc_name") or get_doc_name(filename),
            "page_number": page_number if page_number is not None else "",
            "chunk_id": doc.get("chunk_id", "") or "",
            "type": doc.get("type", "") or "",
            "text": doc.get("text", "") or "",
            "score": _safe_float(doc.get("score")),
            "rerank_score": _safe_float(doc.get("rerank_score")),
            "source": doc.get("source") or "",
        }

    return {
        "question": question,
        "query_parse": meta.get("query_parse", {}),
        "rag_trace": {
            "retrieval_mode": meta.get("retrieval_mode", "baseline"),
            "two_stage_retrieval": meta.get("two_stage_retrieval", False),
            "selected_docs": meta.get("selected_docs", []) or meta.get("doc_stage_selected_docs", []) or [],
            "selected_pages": meta.get("selected_pages", []) or [],
            "page_scores": meta.get("page_scores", []) or [],
            "final_retrieved_chunks": [
                _trace_chunk(doc, idx)
                for idx, doc in enumerate(result.get("final_retrieved_docs", []) or [], 1)
            ],
            "final_evidence_pack_debug": [
                _trace_chunk(doc, idx)
                for idx, doc in enumerate(meta.get("final_evidence_pack_debug", []) or [], 1)
            ],
            "final_evidence_pack_used": [
                _trace_chunk(doc, idx)
                for idx, doc in enumerate(meta.get("final_evidence_pack_used", []) or result.get("context_docs", []), 1)
            ],
            "latency_breakdown": meta.get("latency_breakdown", {}) or {},
        },
    }
