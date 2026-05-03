from typing import Any
import importlib
import json
import os
import re
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langsmith import Client, evaluate
from openai import OpenAI


# ============================================================
# 0. 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

DATASET_NAME = "financebench_top40_100_v2"

EXPERIMENT_PREFIX = "FinanceBench NoisyKB StepBackOff TopK10"


# ============================================================
# 1. 导入你的 RAG Agent
# ============================================================

backend_path = PROJECT_ROOT / "backend"
if str(backend_path) not in sys.path:
    sys.path.append(str(backend_path))

chat_with_agent = importlib.import_module("agent").chat_with_agent


# ============================================================
# 2. 通用工具函数
# ============================================================

def _extract_answer(outputs: Any) -> str:
    if isinstance(outputs, dict):
        answer = outputs.get("response") or outputs.get("answer") or outputs.get("output")
        return str(answer or "").strip()

    if hasattr(outputs, "outputs") and isinstance(outputs.outputs, dict):
        answer = (
            outputs.outputs.get("response")
            or outputs.outputs.get("answer")
            or outputs.outputs.get("output")
        )
        return str(answer or "").strip()

    return ""


def _safe_json_loads(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Cannot parse JSON from judge output: {text}")


def _truncate_text(text: str, max_chars: int = 6000) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


def _normalize_doc_name(name: str) -> str:
    """
    统一 doc_name / filename，用于匹配。
    例如：
    AMD_2022_10K.pdf -> amd_2022_10k
    """
    name = str(name or "").strip()
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _to_int_or_none(x):
    if x is None:
        return None
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


def _parse_json_maybe(text: str):
    if not text:
        return None
    if isinstance(text, (dict, list)):
        return text
    text = str(text).strip()
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_evidence_page_numbers(metadata: dict) -> set[int]:
    """
    尝试从 metadata["evidence"] 或其他字段里解析 evidence page number。
    FinanceBench 的 evidence 字段可能是 JSON 字符串，也可能是普通字符串。
    这里尽量兼容。
    """
    pages = set()

    # 1. 常见 metadata 字段
    for key in [
        "evidence_page_num",
        "evidence_page",
        "page_number",
        "page_num",
        "page",
    ]:
        if key in metadata:
            v = metadata.get(key)
            if isinstance(v, list):
                for item in v:
                    page = _to_int_or_none(item)
                    if page is not None:
                        pages.add(page)
            else:
                page = _to_int_or_none(v)
                if page is not None:
                    pages.add(page)

    evidence = metadata.get("evidence", "")
    parsed = _parse_json_maybe(evidence)

    # 2. JSON 格式 evidence
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                kl = str(k).lower()
                if kl in {
                    "evidence_page_num",
                    "evidence_page",
                    "page_number",
                    "page_num",
                    "page",
                }:
                    if isinstance(v, list):
                        for item in v:
                            page = _to_int_or_none(item)
                            if page is not None:
                                pages.add(page)
                    else:
                        page = _to_int_or_none(v)
                        if page is not None:
                            pages.add(page)
                else:
                    walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    if parsed is not None:
        walk(parsed)

    # 3. 正则兜底：page 12 / page_number: 12
    evidence_text = str(evidence)
    for pattern in [
        r"evidence_page_num['\"\s:=]+(\d+)",
        r"page_number['\"\s:=]+(\d+)",
        r"page_num['\"\s:=]+(\d+)",
        r"\bpage\s+(\d+)\b",
    ]:
        for m in re.finditer(pattern, evidence_text, flags=re.IGNORECASE):
            page = _to_int_or_none(m.group(1))
            if page is not None:
                pages.add(page)

    return pages


def _extract_retrieved_chunks(outputs: dict) -> list[dict]:
    """
    从 target_function 输出中提取最终 retrieved chunks。
    兼容：
    outputs["retrieved_chunks"]
    outputs["rag_trace"]["retrieved_chunks"]
    outputs["rag_trace"]["final_retrieved_chunks"]
    outputs["rag_trace"]["initial_retrieved_chunks"]
    """
    if not isinstance(outputs, dict):
        return []

    candidates = []

    # 顶层
    for key in ["retrieved_chunks", "retrieved_docs", "contexts", "documents"]:
        val = outputs.get(key)
        if isinstance(val, list):
            candidates.extend(val)

    rag_trace = outputs.get("rag_trace") or {}
    if isinstance(rag_trace, dict):
        # 优先 final / retrieved
        for key in [
            "final_retrieved_chunks",
            "retrieved_chunks",
            "reranked_chunks",
            "initial_retrieved_chunks",
            "expanded_retrieved_chunks",
            "documents",
            "contexts",
        ]:
            val = rag_trace.get(key)
            if isinstance(val, list) and val:
                candidates.extend(val)
                # 不 break，避免不同字段都能提供信息

    chunks = []
    seen = set()

    for idx, item in enumerate(candidates):
        if isinstance(item, str):
            chunk = {
                "rank": idx + 1,
                "filename": "",
                "doc_name": "",
                "page_number": None,
                "text": item,
                "score": None,
                "rerank_score": None,
            }
        elif isinstance(item, dict):
            metadata = item.get("metadata") or {}

            filename = (
                item.get("filename")
                or item.get("file_name")
                or item.get("source")
                or item.get("document_name")
                or item.get("doc_name")
                or metadata.get("filename")
                or metadata.get("file_name")
                or metadata.get("source")
                or metadata.get("document_name")
                or metadata.get("doc_name")
                or ""
            )

            doc_name = (
                item.get("doc_name")
                or metadata.get("doc_name")
                or filename
                or ""
            )

            page_number = (
                item.get("page_number")
                or item.get("page")
                or item.get("page_num")
                or metadata.get("page_number")
                or metadata.get("page")
                or metadata.get("page_num")
            )

            text = (
                item.get("text")
                or item.get("content")
                or item.get("page_content")
                or item.get("chunk")
                or item.get("document")
                or ""
            )

            chunk = {
                "rank": item.get("rank") or idx + 1,
                "filename": str(filename or ""),
                "doc_name": str(doc_name or ""),
                "page_number": _to_int_or_none(page_number),
                "text": str(text or ""),
                "score": item.get("score"),
                "rerank_score": item.get("rerank_score"),
            }
        else:
            continue

        key = (
            _normalize_doc_name(chunk.get("filename") or chunk.get("doc_name")),
            chunk.get("page_number"),
            chunk.get("text", "")[:100],
        )
        if key not in seen:
            seen.add(key)
            chunks.append(chunk)

    # 按 rank 排一下
    def rank_key(c):
        try:
            return int(c.get("rank") or 999999)
        except Exception:
            return 999999

    chunks.sort(key=rank_key)
    return chunks


def _is_financebench_like_doc(name: str) -> bool:
    """
    粗略判断 retrieved doc 是否像 FinanceBench PDF。
    用于估计干扰文档比例。
    """
    n = _normalize_doc_name(name)
    finance_keywords = [
        "10k", "10q", "8k", "earnings",
        "amd", "americanexpress", "boeing", "pepsico", "amcor",
        "ultabeauty", "bestbuy", "cvshealth", "pfizer", "verizon",
        "johnson_johnson", "jpmorgan", "amazon", "adobe", "3m",
    ]
    return any(k in n for k in finance_keywords)


# ============================================================
# 3. Target function：调用你的 RAG
# ============================================================

def target_function(inputs: dict) -> dict:
    question = inputs["question"]

    session_id = f"financebench_eval_{uuid4().hex}"

    result = chat_with_agent(
        user_text=question,
        user_id="langsmith_financebench_eval_user",
        session_id=session_id,
    )

    if isinstance(result, dict):
        response_text = str(
            result.get("response")
            or result.get("answer")
            or result.get("output")
            or ""
        )
        rag_trace = result.get("rag_trace", {}) or {}
    else:
        response_text = str(result)
        rag_trace = {}

    return {
        "response": response_text,
        "rag_trace": rag_trace,
    }


# ============================================================
# 4. 豆包 Judge
# ============================================================

judge_client = OpenAI(
    api_key=os.getenv("DOUBAO_API_KEY"),
    base_url=os.getenv("DOUBAO_BASE_URL"),
)

JUDGE_MODEL = os.getenv("DOUBAO_MODEL")


def _call_doubao_json_judge(system_prompt: str, user_prompt: str) -> dict:
    resp = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    raw = resp.choices[0].message.content
    parsed = _safe_json_loads(raw)
    parsed["_raw"] = raw
    return parsed


# ============================================================
# 5. 检索诊断 evaluator：不调用 LLM
# ============================================================

def financebench_retrieval_diagnostics_evaluator(run, example):
    outputs = run.outputs or {}
    metadata = example.metadata or {}

    expected_doc_name = metadata.get("doc_name", "")
    expected_norm = _normalize_doc_name(expected_doc_name)
    evidence_pages = _extract_evidence_page_numbers(metadata)

    retrieved_chunks = _extract_retrieved_chunks(outputs)

    top5 = retrieved_chunks[:5]
    top10 = retrieved_chunks[:10]

    def doc_hit(chunks):
        for c in chunks:
            cand = _normalize_doc_name(c.get("doc_name") or c.get("filename"))
            if expected_norm and cand == expected_norm:
                return True
            # 兼容 filename 带路径或 pdf 后缀
            if expected_norm and expected_norm in cand:
                return True
        return False

    def page_hit(chunks):
        if not evidence_pages:
            return False
        for c in chunks:
            cand = _normalize_doc_name(c.get("doc_name") or c.get("filename"))
            page = c.get("page_number")
            same_doc = expected_norm and (cand == expected_norm or expected_norm in cand)
            if same_doc and page in evidence_pages:
                return True
        return False

    doc_hit_5 = doc_hit(top5)
    doc_hit_10 = doc_hit(top10)
    page_hit_5 = page_hit(top5)
    page_hit_10 = page_hit(top10)

    retrieved_names = [
        f"rank={i+1}, doc={c.get('doc_name') or c.get('filename')}, page={c.get('page_number')}, score={c.get('score')}, rerank={c.get('rerank_score')}"
        for i, c in enumerate(retrieved_chunks[:10])
    ]

    finance_like_count = sum(
        1 for c in top10
        if _is_financebench_like_doc(c.get("doc_name") or c.get("filename"))
    )
    finance_like_ratio = finance_like_count / max(1, len(top10))

    comment = (
        f"Expected doc_name: {expected_doc_name}\n"
        f"Expected normalized doc: {expected_norm}\n"
        f"Evidence pages: {sorted(evidence_pages)}\n\n"
        f"Retrieved chunks count: {len(retrieved_chunks)}\n\n"
        f"Top retrieved chunks:\n" + "\n".join(retrieved_names)
    )

    return [
        {
            "key": "fb_doc_hit_5",
            "score": 1 if doc_hit_5 else 0,
            "comment": comment,
        },
        {
            "key": "fb_doc_hit_10",
            "score": 1 if doc_hit_10 else 0,
            "comment": comment,
        },
        {
            "key": "fb_evidence_page_hit_5",
            "score": 1 if page_hit_5 else 0,
            "comment": comment,
        },
        {
            "key": "fb_evidence_page_hit_10",
            "score": 1 if page_hit_10 else 0,
            "comment": comment,
        },
        {
            "key": "fb_retrieved_financebench_doc_ratio",
            "score": finance_like_ratio,
            "comment": comment,
        },
    ]


# ============================================================
# 6. 合并版 LLM evaluator：一次调用，多个指标
# ============================================================

def financebench_combined_llm_evaluator(run, example):
    inputs = example.inputs or {}
    reference_outputs = example.outputs or {}
    metadata = example.metadata or {}
    outputs = run.outputs or {}

    question = str(inputs.get("question", "")).strip()
    rag_answer = _extract_answer(outputs)

    reference_answer = str(reference_outputs.get("answer", "")).strip()

    evidence = str(metadata.get("evidence", "")).strip()
    justification = str(metadata.get("justification", "")).strip()
    doc_name = str(metadata.get("doc_name", "")).strip()
    company = str(metadata.get("company", "")).strip()
    question_type = str(metadata.get("question_type", "")).strip()
    question_reasoning = str(metadata.get("question_reasoning", "")).strip()

    if not rag_answer:
        comment = (
            "Judge result: incorrect\n\n"
            "Error type: missing_answer\n\n"
            "Reason: RAG 系统返回了空答案。\n\n"
            f"Question:\n{question}\n\n"
            f"Reference answer:\n{reference_answer}\n\n"
            f"Evidence:\n{_truncate_text(evidence, 2000)}"
        )
        return [
            {"key": "fb_answer_correctness", "score": 0, "comment": comment},
            {"key": "fb_evidence_groundedness", "score": 0, "comment": comment},
            {"key": "fb_judge_confidence", "score": 1, "comment": "Empty RAG answer."},
            {"key": "fb_error_type", "value": "missing_answer", "comment": comment},
        ]

    system_prompt = """
You are a strict but fair evaluator for a financial document RAG system.

You need to evaluate ONE RAG answer using the given question, reference answer, evidence, and justification.

Return four things:
1. Whether the RAG answer is correct.
2. Whether the RAG answer is grounded in the provided evidence.
3. The main error type.
4. Your confidence.

Answer correctness rules:
- Mark answer_correct = true if the RAG answer conveys the same factual meaning as the reference answer.
- For financial numbers, units, percentages, years, company names, and financial metrics, be strict.
- Minor wording differences are acceptable.
- Extra explanation is acceptable only if it does not contradict the reference answer or evidence.
- If the RAG answer gives a wrong number, wrong year, wrong company, wrong metric, or unsupported conclusion, mark answer_correct = false.
- If the RAG answer says it cannot answer even though the reference answer and evidence are available, mark answer_correct = false.
- If the RAG answer is approximate but does not change the meaning, it can be correct.
- Do not require exact string match.

Evidence groundedness rules:
- Mark grounded = true if the key claims in the RAG answer are supported by the provided evidence.
- Mark grounded = false if the RAG answer introduces important numbers, dates, entities, metrics, or conclusions not supported by the evidence.
- Mark grounded = false if the RAG answer contradicts the evidence.
- It is acceptable if the RAG answer is shorter than the evidence.
- It is acceptable if the RAG answer paraphrases the evidence.

Error types:
- correct
- wrong_number
- wrong_entity
- wrong_time_period
- wrong_metric
- unsupported_claim
- missing_answer
- irrelevant
- over_answer
- unclear

Return JSON only, without Markdown:
{
  "answer_correct": true or false,
  "grounded": true or false,
  "error_type": "correct | wrong_number | wrong_entity | wrong_time_period | wrong_metric | unsupported_claim | missing_answer | irrelevant | over_answer | unclear",
  "confidence": 0.0 to 1.0,
  "reason": "brief but specific explanation"
}
""".strip()

    evidence_short = _truncate_text(evidence, max_chars=6000)
    justification_short = _truncate_text(justification, max_chars=2000)
    rag_answer_short = _truncate_text(rag_answer, max_chars=3000)

    user_prompt = f"""
Question:
{question}

Company:
{company}

Document name:
{doc_name}

Question type:
{question_type}

Question reasoning:
{question_reasoning}

Reference answer:
{reference_answer}

Evidence:
{evidence_short}

Justification:
{justification_short}

RAG answer:
{rag_answer_short}
""".strip()

    try:
        parsed = _call_doubao_json_judge(system_prompt, user_prompt)

        answer_correct = bool(parsed.get("answer_correct"))
        grounded = bool(parsed.get("grounded"))
        error_type = str(parsed.get("error_type", "unclear")).strip()
        reason = str(parsed.get("reason", "")).strip()
        raw = str(parsed.get("_raw", "")).strip()

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        allowed_error_types = {
            "correct",
            "wrong_number",
            "wrong_entity",
            "wrong_time_period",
            "wrong_metric",
            "unsupported_claim",
            "missing_answer",
            "irrelevant",
            "over_answer",
            "unclear",
        }
        if error_type not in allowed_error_types:
            error_type = "unclear"

        # 一致性后处理
        if answer_correct and grounded:
            error_type = "correct"

        if not answer_correct and error_type == "correct":
            error_type = "unclear"

        detailed_comment = (
            f"Answer correct: {answer_correct}\n\n"
            f"Grounded: {grounded}\n\n"
            f"Error type: {error_type}\n\n"
            f"Confidence: {confidence}\n\n"
            f"Judge reason: {reason}\n\n"
            f"Question:\n{question}\n\n"
            f"Company:\n{company}\n\n"
            f"Doc name:\n{doc_name}\n\n"
            f"Reference answer:\n{reference_answer}\n\n"
            f"Evidence:\n{evidence_short}\n\n"
            f"Justification:\n{justification_short}\n\n"
            f"RAG answer:\n{rag_answer_short}\n\n"
            f"Raw judge output:\n{raw}"
        )

        return [
            {
                "key": "fb_answer_correctness",
                "score": 1 if answer_correct else 0,
                "comment": detailed_comment,
            },
            {
                "key": "fb_evidence_groundedness",
                "score": 1 if grounded else 0,
                "comment": detailed_comment,
            },
            {
                "key": "fb_judge_confidence",
                "score": confidence,
                "comment": reason,
            },
            {
                "key": "fb_error_type",
                "value": error_type,
                "comment": reason,
            },
        ]

    except Exception as e:
        comment = (
            f"Judge failed: {type(e).__name__}: {e}\n\n"
            f"Question:\n{question}\n\n"
            f"Reference answer:\n{reference_answer}\n\n"
            f"Evidence:\n{evidence_short}\n\n"
            f"RAG answer:\n{rag_answer_short}"
        )
        return [
            {"key": "fb_answer_correctness", "score": 0, "comment": comment},
            {"key": "fb_evidence_groundedness", "score": 0, "comment": comment},
            {"key": "fb_judge_confidence", "score": 0, "comment": comment},
            {"key": "fb_error_type", "value": "judge_failed", "comment": comment},
        ]


# ============================================================
# 7. 数据集结构检查
# ============================================================

def check_dataset_structure() -> None:
    client = Client()

    examples = list(client.list_examples(dataset_name=DATASET_NAME, limit=1))
    if not examples:
        raise RuntimeError(f"数据集 {DATASET_NAME} 中没有样本。")

    ex = examples[0]

    print("\n=== 数据集结构检查 ===")
    print("inputs:", ex.inputs)
    print("outputs:", ex.outputs)
    print("metadata keys:", list((ex.metadata or {}).keys()))
    print("metadata evidence preview:", str((ex.metadata or {}).get("evidence", ""))[:300])

    if "question" not in (ex.inputs or {}):
        raise RuntimeError("数据集 inputs 中没有 question 字段。")

    if "answer" not in (ex.outputs or {}):
        raise RuntimeError("数据集 reference outputs 中没有 answer 字段。")

    if "evidence" not in (ex.metadata or {}):
        print("警告：metadata 中没有 evidence 字段。")


# ============================================================
# 8. 运行评估
# ============================================================

if __name__ == "__main__":
    print("LANGSMITH_API_KEY exists:", bool(os.getenv("LANGSMITH_API_KEY")))
    print("LANGCHAIN_API_KEY exists:", bool(os.getenv("LANGCHAIN_API_KEY")))
    print("DOUBAO_API_KEY exists:", bool(os.getenv("DOUBAO_API_KEY")))
    print("DOUBAO_BASE_URL:", os.getenv("DOUBAO_BASE_URL"))
    print("DOUBAO_MODEL:", os.getenv("DOUBAO_MODEL"))
    print("DATASET_NAME:", DATASET_NAME)

    print("\n=== RAG 检索配置建议检查 ===")
    print("FINANCE_RAG_CANDIDATE_K:", os.getenv("FINANCE_RAG_CANDIDATE_K"))
    print("FINANCE_RAG_FINAL_TOP_K:", os.getenv("FINANCE_RAG_FINAL_TOP_K"))
    print("FINANCE_RAG_ENABLE_STEP_BACK:", os.getenv("FINANCE_RAG_ENABLE_STEP_BACK"))
    print("FINANCE_RAG_ENABLE_PAGE_MERGE:", os.getenv("FINANCE_RAG_ENABLE_PAGE_MERGE"))

    check_dataset_structure()

    evaluate(
        target_function,
        data=DATASET_NAME,
        evaluators=[
            financebench_retrieval_diagnostics_evaluator,
            financebench_combined_llm_evaluator,
        ],
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=1,
    )