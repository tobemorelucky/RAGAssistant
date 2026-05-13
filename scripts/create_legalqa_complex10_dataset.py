import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

# 新数据集：只包含 10 条复杂场景题
DATASET_NAME = "legalqa_complex_scenarios_10_v1"
DATA_PATH = PROJECT_ROOT / "datasets" / "legal_qa" / "legal_qa_eval_10_complex_scenarios.jsonl"


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
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到数据集文件：{DATA_PATH}\n"
            "请先把 legal_qa_eval_10_complex_scenarios.jsonl 放到 datasets/legal_qa/ 目录下。"
        )

    client = Client()
    rows = load_jsonl(DATA_PATH)

    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        print(f"数据集已存在：{DATASET_NAME}")
        print("如果想重新导入，请先在 LangSmith 页面删除旧数据集，或者改一个新的 DATASET_NAME。")
        return

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="LegalQA complex scenario evaluation set: 10 hard Chinese legal multi-law reasoning questions.",
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
