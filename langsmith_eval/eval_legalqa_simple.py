from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langsmith import evaluate
from openai import OpenAI


# ============================================================
# LegalQA LangSmith Evaluation
# 适配数据集字段：
# question / answer / difficulty / question_type /
# gold_docs / gold_articles / gold_keywords / evidence
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)



# ============================================================
# 禁止 transformers / huggingface_hub 访问 HuggingFace
# 必须放在 import backend.agent 之前
# ============================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# 可选：如果之前设置过 HF 镜像，也可以清掉，避免误触发联网
# os.environ.pop("HF_ENDPOINT", None)


# 基础任务
# DATASET_NAME = os.getenv("LEGALQA_DATASET_NAME", "legalqa_chinese_laws_v2")
# EXPERIMENT_PREFIX = os.getenv("LEGALQA_EXPERIMENT_PREFIX", "LegalQA50_c40_top5_stepback_off")
#场景题
DATASET_NAME = os.getenv("LEGALQA_DATASET_NAME", "legalqa_complex_scenarios_10_v1")

EXPERIMENT_PREFIX = os.getenv("LEGALQA_EXPERIMENT_PREFIX", "LegalQA10_complex_scenarios")

# 让脚本可以从 langsmith_eval/ 目录导入 backend/agent.py
BACKEND_PATH = PROJECT_ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.append(str(BACKEND_PATH))

chat_with_agent = importlib.import_module("agent").chat_with_agent


def truncate_text(text: Any, max_chars: int = 6000) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


def extract_answer(outputs: Any) -> str:
    """从 target_function 的输出中提取最终回答。"""
    if isinstance(outputs, dict):
        return str(
            outputs.get("response")
            or outputs.get("answer")
            or outputs.get("output")
            or ""
        ).strip()
    return str(outputs or "").strip()


def safe_json_loads(text: str) -> dict:
    """
    尽量解析 judge 返回的 JSON。
    兼容 ```json ... ``` 包裹或前后混入少量解释文本的情况。
    """
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


def target_function(inputs: dict) -> dict:
    """
    LangSmith 对每条样本调用一次。
    inputs 预期格式：{"question": "..."}
    """
    question = str(inputs.get("question", "")).strip()
    session_id = f"legalqa_eval_{uuid4().hex}"

    result = chat_with_agent(
        user_text=question,
        user_id="langsmith_legalqa_eval_user",
        session_id=session_id,
    )

    if isinstance(result, dict):
        response = str(
            result.get("response")
            or result.get("answer")
            or result.get("output")
            or ""
        )
        rag_trace = result.get("rag_trace", {}) or {}
    else:
        response = str(result)
        rag_trace = {}

    return {
        "response": response,
        "rag_trace": rag_trace,
    }


judge_client = OpenAI(
    api_key=os.getenv("DOUBAO_API_KEY"),
    base_url=os.getenv("DOUBAO_BASE_URL"),
)
JUDGE_MODEL = os.getenv("DOUBAO_MODEL")


def call_json_judge(system_prompt: str, user_prompt: str) -> dict:
    resp = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    parsed = safe_json_loads(raw)
    parsed["_raw"] = raw
    return parsed


LEGAL_JUDGE_SYSTEM_PROMPT = """
你是一名严格但公平的中文法律 RAG 评估员。
你需要根据【问题】、【参考答案】、【依据材料】和【RAG回答】判断回答质量。

本数据集包含：
- easy：基础法条事实题；
- medium：法条条件、程序、时限、责任类型题；
- hard：多法综合与案例适用题。

评价规则：
1. 以参考答案和依据材料为主要标准，不要求逐字一致。
2. 法律名称、主体、条件、期限、程序、责任类型必须严格。
3. 如果问题没有要求法条号，RAG回答不写法条号也可以正确；但不能引用错误法条。
4. 如果 RAG 回答的核心法律结论正确，但遗漏部分条件或适用范围，应降低 score，并视情况标记 incomplete。
5. 如果 RAG 回答包含依据材料不支持的重要结论，grounded=false。
6. 如果 RAG 回答只是泛泛而谈，没有回答问题核心，answer_correct=false。
7. 对 hard 综合题，应重点判断：法律关系识别、责任衔接、结论边界是否正确。
8. 不要因为措辞更简短或顺序不同而判错。

error_type 只能从以下值中选择：
correct
wrong_law
wrong_article
wrong_condition
wrong_time_limit
wrong_procedure
wrong_responsibility
incomplete
unsupported_claim
hallucination
missing_answer
irrelevant
unclear

只返回 JSON，不要 Markdown，不要额外解释：
{
  "answer_correct": true 或 false,
  "grounded": true 或 false,
  "score": 0.0 到 1.0,
  "error_type": "correct | wrong_law | wrong_article | wrong_condition | wrong_time_limit | wrong_procedure | wrong_responsibility | incomplete | unsupported_claim | hallucination | missing_answer | irrelevant | unclear",
  "confidence": 0.0 到 1.0,
  "reason": "用中文简要说明判断理由"
}
""".strip()


def legalqa_judge_evaluator(run, example):
    """
    单次 LLM-as-judge 评估。
    输出 4 个 LangSmith feedback：
    - legal_answer_correctness: 0/1
    - legal_groundedness: 0/1
    - legal_answer_score: 0~1
    - legal_error_type: 分类标签
    """
    inputs = example.inputs or {}
    outputs = run.outputs or {}
    reference_outputs = example.outputs or {}
    metadata = example.metadata or {}

    question = str(inputs.get("question", "")).strip()
    reference_answer = str(reference_outputs.get("answer", "")).strip()
    rag_answer = extract_answer(outputs)

    difficulty = str(metadata.get("difficulty", "")).strip()
    question_type = str(metadata.get("question_type", "")).strip()
    evidence = str(metadata.get("evidence", "")).strip()
    gold_docs = metadata.get("gold_docs", []) or []
    gold_articles = metadata.get("gold_articles", []) or []
    gold_keywords = metadata.get("gold_keywords", []) or []

    if not rag_answer:
        comment = (
            "RAG 返回空答案。\n\n"
            f"Question:\n{question}\n\n"
            f"Reference answer:\n{reference_answer}\n\n"
            f"Evidence:\n{evidence}"
        )
        return [
            {"key": "legal_answer_correctness", "score": 0, "comment": comment},
            {"key": "legal_groundedness", "score": 0, "comment": comment},
            {"key": "legal_answer_score", "score": 0, "comment": comment},
            {"key": "legal_error_type", "value": "missing_answer", "comment": comment},
        ]

    user_prompt = f"""
问题：
{question}

难度：
{difficulty}

问题类型：
{question_type}

参考答案：
{reference_answer}

依据材料：
{truncate_text(evidence, 5000)}

辅助信息：
- 期望命中文档：{gold_docs}
- 期望涉及法条：{gold_articles}
- 关键词：{gold_keywords}

RAG回答：
{truncate_text(rag_answer, 3000)}
""".strip()

    try:
        parsed = call_json_judge(LEGAL_JUDGE_SYSTEM_PROMPT, user_prompt)

        answer_correct = bool(parsed.get("answer_correct", False))
        grounded = bool(parsed.get("grounded", False))

        try:
            score = float(parsed.get("score", 1.0 if answer_correct else 0.0))
        except Exception:
            score = 1.0 if answer_correct else 0.0
        score = max(0.0, min(1.0, score))

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        allowed_error_types = {
            "correct",
            "wrong_law",
            "wrong_article",
            "wrong_condition",
            "wrong_time_limit",
            "wrong_procedure",
            "wrong_responsibility",
            "incomplete",
            "unsupported_claim",
            "hallucination",
            "missing_answer",
            "irrelevant",
            "unclear",
        }
        error_type = str(parsed.get("error_type", "unclear")).strip()
        if error_type not in allowed_error_types:
            error_type = "unclear"

        # 避免 judge 输出 answer_correct=true 但 error_type 不是 correct 的轻微不一致
        if answer_correct and grounded and score >= 0.8:
            error_type = "correct"

        reason = str(parsed.get("reason", "")).strip()
        raw = str(parsed.get("_raw", "")).strip()

        comment = (
            f"answer_correct={answer_correct}\n"
            f"grounded={grounded}\n"
            f"score={score}\n"
            f"confidence={confidence}\n"
            f"error_type={error_type}\n"
            f"reason={reason}\n\n"
            f"Question:\n{question}\n\n"
            f"Difficulty: {difficulty}\n"
            f"Question type: {question_type}\n\n"
            f"Reference answer:\n{reference_answer}\n\n"
            f"Evidence:\n{truncate_text(evidence, 5000)}\n\n"
            f"Gold docs: {gold_docs}\n"
            f"Gold articles: {gold_articles}\n"
            f"Gold keywords: {gold_keywords}\n\n"
            f"RAG answer:\n{truncate_text(rag_answer, 3000)}\n\n"
            f"Raw judge output:\n{raw}"
        )

        return [
            {
                "key": "legal_answer_correctness",
                "score": 1 if answer_correct else 0,
                "comment": comment,
            },
            {
                "key": "legal_groundedness",
                "score": 1 if grounded else 0,
                "comment": comment,
            },
            {
                "key": "legal_answer_score",
                "score": score,
                "comment": comment,
            },
            {
                "key": "legal_judge_confidence",
                "score": confidence,
                "comment": reason,
            },
            {
                "key": "legal_error_type",
                "value": error_type,
                "comment": reason,
            },
        ]

    except Exception as e:
        comment = (
            f"Judge failed: {type(e).__name__}: {e}\n\n"
            f"Question:\n{question}\n\n"
            f"Reference answer:\n{reference_answer}\n\n"
            f"Evidence:\n{truncate_text(evidence, 3000)}\n\n"
            f"RAG answer:\n{truncate_text(rag_answer, 3000)}"
        )
        return [
            {"key": "legal_answer_correctness", "score": 0, "comment": comment},
            {"key": "legal_groundedness", "score": 0, "comment": comment},
            {"key": "legal_answer_score", "score": 0, "comment": comment},
            {"key": "legal_judge_confidence", "score": 0, "comment": comment},
            {"key": "legal_error_type", "value": "judge_failed", "comment": comment},
        ]


if __name__ == "__main__":
    print(f"DATASET_NAME={DATASET_NAME}")
    print(f"EXPERIMENT_PREFIX={EXPERIMENT_PREFIX}")
    print(f"JUDGE_MODEL={JUDGE_MODEL}")

    evaluate(
        target_function,
        data=DATASET_NAME,
        evaluators=[legalqa_judge_evaluator],
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=1,
    )
