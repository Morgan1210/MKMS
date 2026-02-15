import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

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

def retrieve(query: str, top_k: int = 5, min_score: float = 0.15):
    """
    返回：list[dict] = {id, score, text, source, created_at}
    score 是 cosine 相似度（因为你用 normalize + IP）
    """
    model = get_model()
    index = get_index()

    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    D, I = index.search(q, top_k)

    ids = []
    scored = []
    for score, cid in zip(D[0].tolist(), I[0].tolist()):
        if cid == -1:
            continue
        if score < min_score:
            continue
        ids.append(int(cid))
        scored.append(float(score))

    rows = fetch_chunks_by_ids(ids)

    # rows: [(id, text, source, created_at), ...]
    out = []
    score_map = {ids[i]: scored[i] for i in range(len(ids))}
    for cid, text, source, created_at in rows:
        out.append({
            "id": int(cid),
            "score": float(score_map.get(cid, 0.0)),
            "text": text,
            "source": source,
            "created_at": created_at
        })
    return out