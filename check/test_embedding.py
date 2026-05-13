import os
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={
        "local_files_only": True
    }
)

vec = embedding.embed_query("测试医疗RAG")
print("向量维度:", len(vec))