"""
email_indexer.py
专门处理邮件的索引构建，与 mkms_index.py 平行
"""

import sys
import os

# 把当前目录从路径里临时移除
current_dir = os.path.dirname(__file__)
if current_dir in sys.path:
    sys.path.remove(current_dir)

# 然后导入
from sentence_transformers import SentenceTransformer

# 导入完再加回来（如果需要）
sys.path.insert(0, current_dir)

import time
import re
import hashlib
import sqlite3
import json
from pathlib import Path
import numpy as np
import faiss



# ==================== 配置 ====================
EMAIL_DB_PATH = "email.db"  # 邮件专用数据库
EMAIL_INDEX_PATH = "email.index"  # 邮件专用FAISS索引
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 和文档用相同模型

TARGET_CHUNK = 600  # 每块大小
BATCH_SIZE = 64

EMAIL_SOURCE_PREFIX = "email:"  # 来源标识前缀


# ==================== 邮件处理 ====================
def parse_email(raw_email: dict) -> str:
    """
    将Graph API返回的邮件对象转换成纯文本
    """
    parts = []

    # 基本信息
    subject = raw_email.get('subject', '无主题')
    from_addr = raw_email.get('from', {}).get('emailAddress', {}).get('address', '未知')
    to_addrs = []
    for recipient in raw_email.get('toRecipients', []):
        addr = recipient.get('emailAddress', {}).get('address', '')
        if addr:
            to_addrs.append(addr)

    received_time = raw_email.get('receivedDateTime', '')
    if received_time:
        received_time = received_time.replace('T', ' ').replace('Z', '')

    # 正文
    body = raw_email.get('body', {})
    content = body.get('content', '')
    content_type = body.get('contentType', 'text')

    # 如果是HTML，简单清理标签
    if content_type == 'html':
        content = re.sub(r'<[^>]+>', ' ', content)  # 去掉HTML标签
        content = re.sub(r'\s+', ' ', content)  # 合并空白

    # 组装
    parts.append(f"【邮件】")
    parts.append(f"主题: {subject}")
    parts.append(f"发件人: {from_addr}")
    if to_addrs:
        parts.append(f"收件人: {', '.join(to_addrs)}")
    parts.append(f"时间: {received_time}")
    parts.append("")
    parts.append(f"正文:")
    parts.append(content)

    return "\n".join(parts)


def load_emails_from_file(file_path: str) -> list:
    """
    从文件加载邮件，标准 JSON 格式
    """
    print(f"📂 正在加载文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        print(f"✅ 成功加载 {len(data)} 封邮件")
        return data
    else:
        print(f"⚠️ 文件不是列表，包装成列表")
        return [data]

def emails_to_chunks(emails: list) -> list:
    """
    将邮件列表转换成文本块
    每封邮件单独作为一个chunk（不跨邮件切分）
    """
    chunks = []
    for email in emails:
        text = parse_email(email)
        # 对单封邮件内部切分（如果太长）
        email_chunks = smart_chunks(text, target=TARGET_CHUNK)
        chunks.extend(email_chunks)
    return chunks


# ==================== 复用你的工具函数 ====================
def smart_chunks(text: str, target=TARGET_CHUNK):
    """复用 mkms_index 的分块逻辑"""
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
    """初始化邮件专用数据库"""
    conn = sqlite3.connect(EMAIL_DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS email_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hash TEXT UNIQUE,
        text TEXT NOT NULL,
        email_id TEXT,              -- Graph API 的邮件ID
        subject TEXT,                -- 邮件主题
        from_addr TEXT,              -- 发件人
        received_time TEXT,          -- 接收时间
        created_at INTEGER,
        source TEXT
    )
    """)
    conn.commit()
    return conn


def upsert_chunks(conn, chunks, emails_metadata):
    """
    插入或更新chunks
    emails_metadata: 与chunks对应的元数据列表
    """
    now = int(time.time())
    ids, new_texts = [], []

    for i, chunk in enumerate(chunks):
        h = sha1(chunk)
        # 获取对应的邮件元数据（取第一个邮件，因为一个chunk可能来自一封邮件）
        meta = emails_metadata[i] if i < len(emails_metadata) else {}

        try:
            cur = conn.execute(
                """INSERT INTO email_chunks 
                   (hash, text, email_id, subject, from_addr, received_time, created_at, source) 
                   VALUES(?,?,?,?,?,?,?,?)""",
                (h, chunk,
                 meta.get('id', ''),
                 meta.get('subject', ''),
                 meta.get('from_addr', ''),
                 meta.get('received_time', ''),
                 now,
                 meta.get('source', 'email'))
            )
            cid = cur.lastrowid
            ids.append(cid)
            new_texts.append(chunk)
        except sqlite3.IntegrityError:
            # 已存在则取旧 id
            cur = conn.execute("SELECT id FROM email_chunks WHERE hash=?", (h,))
            cid = cur.fetchone()[0]
            ids.append(cid)
            new_texts.append(chunk)

    conn.commit()
    return ids, new_texts


def build_faiss_index(ids, vecs):
    """建FAISS索引"""
    dim = vecs.shape[1]
    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base)
    index.add_with_ids(vecs, np.array(ids, dtype=np.int64))
    faiss.write_index(index, EMAIL_INDEX_PATH)


# ==================== 主函数 ====================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="邮件索引构建工具")
    parser.add_argument("--input", required=False, default="email_content.json", help="邮件文件路径（支持 .json 或 .txt）")
    parser.add_argument("--db", default=EMAIL_DB_PATH, help="邮件数据库路径")
    parser.add_argument("--index", default=EMAIL_INDEX_PATH, help="邮件索引输出路径")
    args = parser.parse_args()

    # 1. 加载邮件
    print(f"📧 加载邮件: {args.input}")
    emails = load_emails_from_file(args.input)
    print(f"   共 {len(emails)} 封邮件")

    # 2. 转换成chunks
    print("✂️ 正在分块...")
    chunks = []
    metadata = []
    for email in emails:
        email_chunks = emails_to_chunks([email])
        chunks.extend(email_chunks)
        # 为每个chunk记录对应的邮件元数据
        for _ in email_chunks:
            metadata.append({
                'id': email.get('id', ''),
                'subject': email.get('subject', ''),
                'from_addr': email.get('from', {}).get('emailAddress', {}).get('address', ''),
                'received_time': email.get('receivedDateTime', ''),
                'source': f"{EMAIL_SOURCE_PREFIX}{email.get('id', '')}"
            })

    print(f"   生成 {len(chunks)} 个文本块")

    # 3. 存入数据库
    conn = init_db()
    ids, new_texts = upsert_chunks(conn, chunks, metadata)
    conn.close()
    print(f"💾 数据库: {len(ids)} 总记录, {len(new_texts)} 新增")

    # 4. 向量化
    print(f"🔮 加载模型 {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"📊 生成向量（共 {len(new_texts)} 条）...")
    vectors = model.encode(
        new_texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype(np.float32)

    # 5. 建索引（全量重建）
    # 注意：这里为了简化，只索引新增的。实际应该全量或合并
    # 简单起见，我们每次全量重建
    conn = sqlite3.connect(args.db)
    all_rows = conn.execute("SELECT id, text FROM email_chunks").fetchall()
    conn.close()

    all_ids = [row[0] for row in all_rows]
    all_texts = [row[1] for row in all_rows]

    print(f"🔨 生成全量向量（共 {len(all_texts)} 条）...")
    all_vectors = model.encode(
        all_texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype(np.float32)

    build_faiss_index(all_ids, all_vectors)

    print(f"🎉 完成！索引已保存到 {args.index}")
    print(f"   email_chunks: {len(all_ids)}, 数据库: {args.db}")


if __name__ == "__main__":
    main()