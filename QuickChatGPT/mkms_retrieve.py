import sys
import os

# 把当前目录从路径里临时移除
current_dir = os.path.dirname(__file__)
if current_dir in sys.path:
    sys.path.remove(current_dir)

# 然后导入
from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np
import faiss

DB_PATH = "mkms.db"
INDEX_PATH = "mkms.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_index = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def get_index():
    global _index
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
    return _index

def fetch_chunks_by_ids(ids):
    # ids: list[int]
    if not ids:
        return []
    conn = sqlite3.connect(DB_PATH)
    qmarks = ",".join(["?"] * len(ids))
    rows = conn.execute(
        f"SELECT id, text, source, created_at FROM chunks WHERE id IN ({qmarks})",
        ids
    ).fetchall()
    conn.close()

    # 按 ids 的原始顺序排序（保持相似度排序）
    rank = {cid: i for i, cid in enumerate(ids)}
    rows.sort(key=lambda r: rank.get(r[0], 10**9))
    return rows


def retrieve(query: str, top_k: int = 5, min_score: float = 0.15, include_emails: bool = True):
    """
    返回：list[dict] = {id, score, text, source, created_at}
    同时检索文档和邮件
    """
    model = get_model()
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    all_results = []

    # 1. 查文档索引
    doc_index = get_index()  # mkms.index
    D, I = doc_index.search(q, top_k * 2)
    # ... 处理文档结果 ...

    # 2. 查邮件索引
    try:
        email_index = faiss.read_index("email.index")
        D_email, I_email = email_index.search(q, top_k * 2)
        # ... 从 email.db 获取邮件内容 ...
    except:
        pass  # 邮件索引不存在时跳过

    return all_results[:top_k]

# ==================== 邮件检索支持 ====================
EMAIL_DB_PATH = "email.db"  # 和 email_indexer.py 里保持一致


def fetch_email_chunks_by_ids(ids):
    """
    从邮件数据库获取chunks
    ids: list[int]  FAISS返回的ID列表
    """
    if not ids:
        return []

    conn = sqlite3.connect(EMAIL_DB_PATH)
    qmarks = ",".join(["?"] * len(ids))
    rows = conn.execute(
        f"""SELECT id, text, email_id, subject, from_addr, received_time, source 
           FROM email_chunks WHERE id IN ({qmarks})""",
        ids
    ).fetchall()
    conn.close()

    # 按ids顺序排序
    rank = {cid: i for i, cid in enumerate(ids)}
    rows.sort(key=lambda r: rank.get(r[0], 10 ** 9))
    return rows


def fetch_email_chunks(faiss_ids, scores):
    """
    封装好的邮件chunks获取函数
    faiss_ids: FAISS返回的ID列表（一维数组）
    scores: 对应的分数列表
    """
    ids = [int(i) for i in faiss_ids if i != -1]
    if not ids:
        return []

    rows = fetch_email_chunks_by_ids(ids)

    # 构建结果
    score_map = {ids[i]: scores[i] for i in range(len(ids))}
    results = []
    for row in rows:
        cid, text, email_id, subject, from_addr, received_time, source = row
        results.append({
            "id": int(cid),
            "score": float(score_map.get(cid, 0.0)),
            "text": text,
            "email_id": email_id,
            "subject": subject,
            "from_addr": from_addr,
            "received_time": received_time,
            "source": source,
            "type": "email"  # 标记来源类型
        })

    return results


def warm_up():
    """预热模型"""
    print("🔥 预热检索模型...")
    _ = retrieve("test", top_k=3)
    print("✅ 预热完成")


def retrieve_from_emails(query: str, top_k: int = 5, min_score: float = 0.15):
    """
    专门从邮件索引中检索
    返回：list[dict] = {id, score, text, email_id, subject, from_addr, received_time, source}
    """
    EMAIL_DB_PATH = "email.db"
    EMAIL_INDEX_PATH = "email.index"

    # 1. 加载模型（复用）
    from mkms_retrieve import get_model
    model = get_model()

    # 2. 加载邮件索引
    try:
        email_index = faiss.read_index(EMAIL_INDEX_PATH)
    except Exception as e:
        print(f"❌ 无法加载邮件索引: {e}")
        return []

    # 3. 向量化查询
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    # 4. 搜索
    D, I = email_index.search(q, top_k)

    # 5. 收集结果
    ids = []
    scores = []
    for score, cid in zip(D[0].tolist(), I[0].tolist()):
        if cid == -1 or score < min_score:
            continue
        ids.append(int(cid))
        scores.append(float(score))

    if not ids:
        return []

    # 6. 从数据库获取邮件内容
    conn = sqlite3.connect(EMAIL_DB_PATH)
    qmarks = ",".join(["?"] * len(ids))
    rows = conn.execute(
        f"""SELECT id, text, email_id, subject, from_addr, received_time, source 
           FROM email_chunks WHERE id IN ({qmarks})""",
        ids
    ).fetchall()
    conn.close()

    # 7. 按原始顺序排序
    rank = {cid: i for i, cid in enumerate(ids)}
    rows.sort(key=lambda r: rank.get(r[0], 10 ** 9))

    # 8. 格式化输出
    score_map = {ids[i]: scores[i] for i in range(len(ids))}
    results = []
    for row in rows:
        cid, text, email_id, subject, from_addr, received_time, source = row
        results.append({
            "id": int(cid),
            "score": float(score_map.get(cid, 0.0)),
            "text": text,
            "email_id": email_id,
            "subject": subject,
            "from_addr": from_addr,
            "received_time": received_time,
            "source": source,
            "type": "email"
        })

    return results


# 在 main.py 里调用一下
if __name__ == "__main__":
    warm_up()