import os, time, re, hashlib, sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DB_PATH = "mkms.db"
INDEX_PATH = "mkms.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TARGET_CHUNK = 600  # 你先 status quo，后面再调
BATCH_SIZE = 64

def smart_chunks(text: str, target=TARGET_CHUNK):
    parts = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks, buf = [], ""
    for p in parts:
        if len(buf) + len(p) + 1 <= target:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hash TEXT UNIQUE,
        text TEXT NOT NULL,
        created_at INTEGER,
        source TEXT
    )
    """)
    conn.commit()
    return conn

def upsert_chunks(conn, chunks, source):
    now = int(time.time())
    ids, new_texts = [], []
    for t in chunks:
        h = sha1(t)
        try:
            cur = conn.execute(
                "INSERT INTO chunks(hash, text, created_at, source) VALUES(?,?,?,?)",
                (h, t, now, source)
            )
            cid = cur.lastrowid
            ids.append(cid)
            new_texts.append(t)
        except sqlite3.IntegrityError:
            # 已存在则取旧 id；（status quo：我们重建索引时也会包含它）
            cur = conn.execute("SELECT id FROM chunks WHERE hash=?", (h,))
            cid = cur.fetchone()[0]
            ids.append(cid)
            new_texts.append(t)  # status quo：全量重建索引用
    conn.commit()
    return ids, new_texts

def build_faiss_index(ids, vecs):
    dim = vecs.shape[1]
    base = faiss.IndexFlatIP(dim)         # 归一化后点积=cosine，速度快
    index = faiss.IndexIDMap2(base)       # 关键：向量 id 映射到 SQLite id
    index.add_with_ids(vecs, np.array(ids, dtype=np.int64))
    faiss.write_index(index, INDEX_PATH)

def main():
    text = open("mkms_doc.txt", "r", encoding="utf-8").read()
    chunks = smart_chunks(text)

    conn = init_db()
    ids, texts_for_index = upsert_chunks(conn, chunks, source="mkms_doc.txt")
    conn.close()

    model = SentenceTransformer(MODEL_NAME)
    vecs = model.encode(
        texts_for_index,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype(np.float32)

    build_faiss_index(ids, vecs)
    print(f"OK: chunks={len(ids)}  index={INDEX_PATH}  db={DB_PATH}")

if __name__ == "__main__":
    main()