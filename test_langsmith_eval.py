from typing import Any, Optional
import importlib
import json
import os
import re
import sys
from uuid import uuid4

from dotenv import load_dotenv
from langsmith import evaluate
from openai import OpenAI


# =========================
# 0. 加载环境变量
# =========================
load_dotenv(override=True)


# =========================
# 1. 导入你的 RAG Agent
# =========================
backend_path = os.path.join(os.path.dirname(__file__), "backend")
if backend_path not in sys.path:
    sys.path.append(backend_path)

chat_with_agent = importlib.import_module("agent").chat_with_agent


# =========================
# 2. 输出提取工具
# =========================
def _extract_answer(outputs: Any) -> str:
    """
    从 target_function 的输出中提取最终回答。
    兼容 response / answer / output 三种字段。
    """
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


def _extract_reference(reference_outputs: Optional[dict]) -> str:
    """
    从 LangSmith dataset 的 reference_outputs 中提取标准答案。
    你的 CSV 列是 answer，所以主要取 answer。
    """
    if not isinstance(reference_outputs, dict):
        return ""

    for key in ("answer", "response", "output", "expected_answer"):
        value = reference_outputs.get(key)
        if value:
            return str(value).strip()

    return ""


def _safe_json_loads(text: str) -> dict:
    """
    尽量从 judge 模型输出中解析 JSON。
    兼容 ```json ... ``` 代码块和额外文本。
    """
    text = (text or "").strip()

    # 去掉 Markdown 代码块
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # 尝试截取第一个 JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Cannot parse JSON from judge output: {text}")


# =========================
# 3. 你的 RAG target function
# =========================
def target_function(inputs: dict) -> dict:
    """
    LangSmith 会对 test 数据集里的每条样本调用这个函数。
    inputs 里应该有 question 字段。
    """
    question = inputs["question"]

    # 每个样本使用独立会话，避免上下文串扰
    session_id = f"langsmith_eval_{uuid4().hex}"

    result = chat_with_agent(
        user_text=question,
        user_id="langsmith_eval_user",
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


# =========================
# 4. 豆包 LLM-as-judge evaluator
# =========================
judge_client = OpenAI(
    api_key=os.getenv("DOUBAO_API_KEY"),
    base_url=os.getenv("DOUBAO_BASE_URL"),
)

JUDGE_MODEL = os.getenv("DOUBAO_MODEL")


def doubao_llm_correctness_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    """
    使用豆包 / 火山方舟 OpenAI 兼容接口做 LLM-as-judge。
    判断 RAG answer 是否与 reference answer 在事实层面一致。
    """
    question = inputs.get("question", "")
    student_answer = _extract_answer(outputs)
    reference_answer = _extract_reference(reference_outputs)

    if not student_answer:
        return {
            "key": "doubao_llm_correctness",
            "score": 0,
            "comment": "RAG 系统返回了空答案。",
        }

    if not reference_answer:
        return {
            "key": "doubao_llm_correctness",
            "score": 0,
            "comment": "样本没有 reference answer，无法做 reference-based correctness。",
        }

    system_prompt = """
You are a strict but fair evaluator for a RAG question-answering system.

Your task is to judge whether the RAG answer is factually correct with respect to the reference answer.

Evaluation rules:
1. Mark correct if the RAG answer conveys the same factual meaning as the reference answer.
2. Different wording, aliases, abbreviations, or equivalent expressions are acceptable.
3. Extra information is acceptable only if it does not contradict the reference answer.
4. Mark incorrect if the answer contradicts the reference answer.
5. Mark incorrect if the answer avoids the question or gives irrelevant information.
6. Do not require exact string match.
7. For SQuAD-style short answers, focus on whether the required entity, date, number, location, or phrase is correct.

Return JSON only, without Markdown:
{
  "correct": true or false,
  "reason": "brief explanation"
}
""".strip()

    user_prompt = f"""
Question:
{question}

Reference answer:
{reference_answer}

RAG answer:
{student_answer}
""".strip()

    try:
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

        correct = bool(parsed.get("correct"))
        reason = str(parsed.get("reason", "")).strip()

        return {
            "key": "doubao_llm_correctness",
            "score": 1 if correct else 0,
            "comment": reason,
        }

    except Exception as e:
        return {
            "key": "doubao_llm_correctness",
            "score": 0,
            "comment": f"Judge failed: {type(e).__name__}: {e}",
        }


# =========================
# 5. 运行评估
# =========================
if __name__ == "__main__":
    print("LANGSMITH_API_KEY exists:", bool(os.getenv("LANGSMITH_API_KEY")))
    print("LANGCHAIN_API_KEY exists:", bool(os.getenv("LANGCHAIN_API_KEY")))
    print("DOUBAO_API_KEY exists:", bool(os.getenv("DOUBAO_API_KEY")))
    print("DOUBAO_BASE_URL:", os.getenv("DOUBAO_BASE_URL"))
    print("DOUBAO_MODEL:", os.getenv("DOUBAO_MODEL"))

    evaluate(
        target_function,
        data="test-1000",
        evaluators=[doubao_llm_correctness_evaluator],
        experiment_prefix="RAG Pipeline Doubao LLM Judge Evaluation",
        max_concurrency=1,
    )