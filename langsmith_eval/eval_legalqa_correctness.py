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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

DATASET_NAME = "legalqa_chinese_laws_v1"
EXPERIMENT_PREFIX = "LegalQA baseline eval"


backend_path = PROJECT_ROOT / "backend"
if str(backend_path) not in sys.path:
    sys.path.append(str(backend_path))


# 如果你的 FinanceBench 脚本里不是这样导入的，就以你原脚本为准
chat_with_agent = importlib.import_module("agent").chat_with_agent


def safe_json_loads(text: str) -> dict:
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

    raise ValueError(f"无法解析 judge JSON：{text}")


def target_function(inputs: dict) -> dict:
    question = inputs["question"]
    session_id = f"legalqa_eval_{uuid4().hex}"

    result = chat_with_agent(
        user_text=question,
        user_id="langsmith_legalqa_eval_user",
        session_id=session_id,
    )

    if isinstance(result, dict):
        response = (
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
        "response": str(response),
        "rag_trace": rag_trace,
    }


judge_client = OpenAI(
    api_key=os.getenv("DOUBAO_API_KEY"),
    base_url=os.getenv("DOUBAO_BASE_URL"),
)

JUDGE_MODEL = os.getenv("DOUBAO_MODEL")


def legal_answer_evaluator(run, example):
    question = example.inputs.get("question", "")
    reference_answer = example.outputs.get("answer", "")
    metadata = example.metadata or {}
    evidence = metadata.get("evidence", "")
    gold_articles = metadata.get("gold_articles", [])
    gold_docs = metadata.get("gold_docs", [])

    outputs = run.outputs or {}
    rag_answer = outputs.get("response", "")

    if not rag_answer:
        return {
            "key": "legal_answer_correctness",
            "score": 0,
            "comment": "RAG 返回空答案。",
        }

    system_prompt = """
你是一名严格但公平的中文法律 RAG 评估员。
请根据问题、参考答案、依据材料和 RAG 回答，判断 RAG 回答是否正确。

评分要求：
1. 如果法律结论与参考答案一致，且没有明显编造，answer_correct=true。
2. 如果回答引用了错误法律、错误责任类型、错误期限、错误程序，answer_correct=false。
3. 如果回答大体正确但不完整，可以判 true，但 reason 中说明不足。
4. grounded 表示回答是否能被依据材料支持。
5. 只返回 JSON，不要输出 Markdown。

返回格式：
{
  "answer_correct": true 或 false,
  "grounded": true 或 false,
  "score": 0 到 1,
  "error_type": "correct | wrong_law | wrong_article | wrong_condition | wrong_procedure | unsupported_claim | missing_answer | irrelevant | unclear",
  "reason": "中文简要理由"
}
""".strip()

    user_prompt = f"""
问题：
{question}

参考答案：
{reference_answer}

依据材料：
{evidence}

期望命中文档：
{gold_docs}

期望法条：
{gold_articles}

RAG回答：
{rag_answer}
""".strip()

    resp = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    raw = resp.choices[0].message.content
    parsed = safe_json_loads(raw)

    answer_correct = bool(parsed.get("answer_correct", False))
    grounded = bool(parsed.get("grounded", False))
    score = float(parsed.get("score", 1 if answer_correct else 0))
    error_type = parsed.get("error_type", "unclear")
    reason = parsed.get("reason", "")

    comment = (
        f"answer_correct={answer_correct}\n"
        f"grounded={grounded}\n"
        f"score={score}\n"
        f"error_type={error_type}\n"
        f"reason={reason}\n\n"
        f"问题：{question}\n\n"
        f"参考答案：{reference_answer}\n\n"
        f"RAG回答：{rag_answer}\n\n"
        f"依据材料：{evidence}\n\n"
        f"judge_raw：{raw}"
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
            "score": max(0, min(1, score)),
            "comment": comment,
        },
        {
            "key": "legal_error_type",
            "value": error_type,
            "comment": reason,
        },
    ]


def check_dataset():
    client = Client()
    examples = list(client.list_examples(dataset_name=DATASET_NAME, limit=1))
    if not examples:
        raise RuntimeError(f"LangSmith 数据集为空或不存在：{DATASET_NAME}")

    ex = examples[0]
    print("=== 数据集检查 ===")
    print("inputs:", ex.inputs)
    print("outputs:", ex.outputs)
    print("metadata:", ex.metadata)


if __name__ == "__main__":
    print("DATASET_NAME:", DATASET_NAME)
    print("DOUBAO_API_KEY exists:", bool(os.getenv("DOUBAO_API_KEY")))
    print("DOUBAO_BASE_URL:", os.getenv("DOUBAO_BASE_URL"))
    print("DOUBAO_MODEL:", os.getenv("DOUBAO_MODEL"))

    check_dataset()

    evaluate(
        target_function,
        data=DATASET_NAME,
        evaluators=[legal_answer_evaluator],
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=1,
    )