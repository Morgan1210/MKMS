import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ====== 配置 ======
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "mkms.index"
CHUNK_SIZE = 600  # 每块字符数

# ====== 加载模型 ======
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# ====== 文本切块 ======
def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# ====== 构建索引 ======
def build_index(text):
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    embeddings = model.encode(chunks, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    print("Index saved:", INDEX_PATH)

    return chunks, index

# ====== 查询 ======
def search(query, chunks, index, top_k=3):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for idx in indices[0]:
        results.append(chunks[idx])

    return results


if __name__ == "__main__":

    # 读取你的 MKMS 文档
    with open("mkms_doc.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks, index = build_index(text)

    # 测试查询
    print("\nTest query:")
    results = search("TSLY理论崩溃的影响是什么？", chunks, index)

    for r in results:
        print("\n---\n", r)