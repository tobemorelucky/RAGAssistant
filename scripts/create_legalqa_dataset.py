import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

DATASET_NAME = "legalqa_chinese_laws_v2"
DATA_PATH = PROJECT_ROOT / "datasets" / "legal_qa" / "legal_qa_eval_50.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSONL 第 {line_no} 行格式错误: {e}") from e
    return rows


def main():
    client = Client()
    rows = load_jsonl(DATA_PATH)

    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        dataset = existing[0]
        print(f"数据集已存在：{DATASET_NAME}")
        print("为了避免重复写入样本，建议你先去 LangSmith 页面删除旧数据集，或者改一个新 DATASET_NAME。")
        return

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="中文法律 RAG 评估集：宪法、民法典、刑法、民诉、刑诉、行政诉讼法、治安管理处罚法。",
    )

    inputs = []
    outputs = []
    metadata = []

    for row in rows:
        inputs.append({"question": row["question"]})
        outputs.append({"answer": row["answer"]})
        metadata.append(
            {
                "difficulty": row.get("difficulty", ""),
                "question_type": row.get("question_type", ""),
                "gold_docs": row.get("gold_docs", []),
                "gold_articles": row.get("gold_articles", []),
                "gold_keywords": row.get("gold_keywords", []),
                "evidence": row.get("evidence", ""),
            }
        )

    client.create_examples(
        dataset_id=dataset.id,
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
    )

    print(f"完成：已创建 LangSmith 数据集 {DATASET_NAME}，共 {len(rows)} 条样本。")


if __name__ == "__main__":
    main()