import os
from dotenv import load_dotenv
from langsmith import Client

# 关键：override=True，强制 .env 覆盖系统环境变量里的旧值
load_dotenv(override=True)

def mask(value: str | None) -> str:
    if not value:
        return "None"
    if len(value) <= 12:
        return "***"
    return value[:8] + "..." + value[-6:]

print("LANGCHAIN_API_KEY =", mask(os.getenv("LANGCHAIN_API_KEY")))
print("LANGSMITH_API_KEY =", mask(os.getenv("LANGSMITH_API_KEY")))
print("LANGCHAIN_ENDPOINT =", os.getenv("LANGCHAIN_ENDPOINT"))
print("LANGSMITH_ENDPOINT =", os.getenv("LANGSMITH_ENDPOINT"))
print("LANGCHAIN_PROJECT =", os.getenv("LANGCHAIN_PROJECT"))
print("LANGSMITH_PROJECT =", os.getenv("LANGSMITH_PROJECT"))

client = Client(
    api_key=os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY"),
    api_url=os.getenv("LANGCHAIN_ENDPOINT") or os.getenv("LANGSMITH_ENDPOINT"),
)

datasets = list(client.list_datasets(limit=3))

print("\n认证成功。前几个数据集：")
for ds in datasets:
    print("-", ds.name)