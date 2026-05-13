from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name=r"C:\Users\he\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\9a0624b896d81da7492a910ffa53731274b6cf3d",
    model_kwargs={
        "local_files_only": True
    }
)

vec = embedding.embed_query("测试医疗RAG")
print("向量维度:", len(vec))