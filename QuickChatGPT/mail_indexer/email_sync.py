"""
email_sync.py - 邮件同步与向量化增量更新

功能：
1. 从 Microsoft Graph API 拉取新邮件
2. 增量向量化并存入数据库
3. 重建 FAISS 索引

用法：
    python email_sync.py                    # 同步最新邮件
    python email_sync.py --full             # 全量重建
    python email_sync.py --days 7           # 同步最近7天
"""

import sys
import os
import io

# Windows UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import re
import hashlib
import sqlite3
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# ==================== 配置 ====================
EMAIL_DB_PATH = Path(__file__).parent.parent / "email.db"
EMAIL_INDEX_PATH = Path(__file__).parent.parent / "email.index"
TOKEN_FILE = Path(__file__).parent.parent / "outlook_mail" / "ms_graph_token.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TARGET_CHUNK = 600
BATCH_SIZE = 64

CLIENT_ID = "4d7ff71c-b75b-4e45-bfeb-1b536a99c11e"
TENANT = "consumers"
SCOPES = ["Mail.Read", "Mail.ReadWrite", "Mail.Send", "offline_access"]


# ==================== Token 管理 ====================
def load_token():
    """加载并刷新 token"""
    if not TOKEN_FILE.exists():
        raise FileNotFoundError(f"Token file not found: {TOKEN_FILE}")
    
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        token_data = json.load(f)
    
    expires_in = token_data.get("expires_in", 3600)
    acquired_at = token_data.get("acquired_at", 0)
    
    # Token 还有效（预留5分钟）
    if time.time() - acquired_at < expires_in - 300:
        return token_data["access_token"]
    
    # 刷新 token
    print("🔄 Token 已过期，刷新中...")
    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        raise ValueError("No refresh_token available")
    
    refresh_url = f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token"
    refresh_params = {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": refresh_token,
        "scope": " ".join(SCOPES)
    }
    
    resp = requests.post(refresh_url, data=refresh_params)
    if resp.status_code == 200:
        new_token = resp.json()
        new_token["acquired_at"] = time.time()
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            json.dump(new_token, f, ensure_ascii=False, indent=2)
        print("✅ Token 刷新成功")
        return new_token["access_token"]
    
    raise Exception(f"Token refresh failed: {resp.status_code}")


# ==================== Graph API 调用 ====================
def fetch_emails(access_token, days=None, max_emails=500):
    """
    从 Graph API 获取邮件
    
    Args:
        access_token: 访问令牌
        days: 最近多少天（None = 全部）
        max_emails: 最大数量
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # 构建过滤条件
    filter_clause = None
    if days:
        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        filter_clause = f"receivedDateTime ge {since}"
    
    all_messages = []
    url = "https://graph.microsoft.com/v1.0/me/messages"
    params = {
        "$top": 100,
        "$orderby": "receivedDateTime DESC",
        "$select": "id,subject,from,toRecipients,receivedDateTime,isRead,body",
    }
    if filter_clause:
        params["$filter"] = filter_clause
    
    page = 1
    while url and len(all_messages) < max_emails:
        print(f"   第 {page} 页...", end="")
        resp = requests.get(url, headers=headers, params=params if page == 1 else {})
        
        if resp.status_code != 200:
            print(f" ❌ 失败: {resp.status_code}")
            break
        
        data = resp.json()
        messages = data.get("value", [])
        all_messages.extend(messages)
        print(f" 本页 {len(messages)} 封，累计 {len(all_messages)} 封")
        
        url = data.get("@odata.nextLink")
        params = None
        page += 1
    
    return all_messages[:max_emails]


# ==================== 邮件解析 ====================
def parse_email(raw_email: dict) -> str:
    """将 Graph API 邮件对象转换成纯文本"""
    parts = []
    
    subject = raw_email.get("subject", "无主题")
    from_addr = raw_email.get("from", {}).get("emailAddress", {}).get("address", "未知")
    from_name = raw_email.get("from", {}).get("emailAddress", {}).get("name", "")
    
    to_addrs = []
    for recipient in raw_email.get("toRecipients", []):
        addr = recipient.get("emailAddress", {}).get("address", "")
        if addr:
            to_addrs.append(addr)
    
    received_time = raw_email.get("receivedDateTime", "")
    if received_time:
        received_time = received_time.replace("T", " ").replace("Z", "")
    
    body = raw_email.get("body", {})
    content = body.get("content", "")
    content_type = body.get("contentType", "text")
    
    # HTML 清理
    if content_type == "html":
        content = re.sub(r"<[^>]+>", " ", content)
        content = re.sub(r"\s+", " ", content)
    
    parts.append(f"【邮件】")
    parts.append(f"主题: {subject}")
    parts.append(f"发件人: {from_name} <{from_addr}>" if from_name else f"发件人: {from_addr}")
    if to_addrs:
        parts.append(f"收件人: {', '.join(to_addrs)}")
    parts.append(f"时间: {received_time}")
    parts.append("")
    parts.append(f"正文:")
    parts.append(content.strip())
    
    return "\n".join(parts)


def smart_chunks(text: str, target=TARGET_CHUNK):
    """智能分块"""
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


# ==================== 数据库操作 ====================
def init_db():
    """初始化数据库"""
    conn = sqlite3.connect(EMAIL_DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS email_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hash TEXT UNIQUE,
        text TEXT NOT NULL,
        email_id TEXT,
        subject TEXT,
        from_addr TEXT,
        received_time TEXT,
        created_at INTEGER,
        source TEXT,
        synced_at INTEGER
    )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_email_id ON email_chunks(email_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON email_chunks(hash)")
    conn.commit()
    return conn


def get_existing_email_ids(conn):
    """获取已同步的邮件 ID"""
    rows = conn.execute("SELECT DISTINCT email_id FROM email_chunks WHERE email_id != ''").fetchall()
    return set(row[0] for row in rows)


def upsert_chunks(conn, chunks, email_meta, synced_at):
    """插入新 chunks"""
    now = int(time.time())
    ids, new_texts = [], []
    
    for i, chunk in enumerate(chunks):
        h = sha1(chunk)
        meta = email_meta[i] if i < len(email_meta) else {}
        
        try:
            cur = conn.execute(
                """INSERT INTO email_chunks 
                   (hash, text, email_id, subject, from_addr, received_time, created_at, source, synced_at) 
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (h, chunk,
                 meta.get("id", ""),
                 meta.get("subject", ""),
                 meta.get("from_addr", ""),
                 meta.get("received_time", ""),
                 now,
                 meta.get("source", "email"),
                 synced_at)
            )
            ids.append(cur.lastrowid)
            new_texts.append(chunk)
        except sqlite3.IntegrityError:
            cur = conn.execute("SELECT id FROM email_chunks WHERE hash=?", (h,))
            ids.append(cur.fetchone()[0])
    
    conn.commit()
    return ids, new_texts


# ==================== 向量化 ====================
def build_faiss_index(conn, model):
    """重建 FAISS 索引"""
    rows = conn.execute("SELECT id, text FROM email_chunks").fetchall()
    if not rows:
        print("⚠️ 没有数据，跳过索引构建")
        return
    
    ids = [row[0] for row in rows]
    texts = [row[1] for row in rows]
    
    print(f"🔨 生成向量（共 {len(texts)} 条）...")
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype(np.float32)
    
    dim = vectors.shape[1]
    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base)
    index.add_with_ids(vectors, np.array(ids, dtype=np.int64))
    faiss.write_index(index, str(EMAIL_INDEX_PATH))
    
    print(f"✅ 索引已保存: {EMAIL_INDEX_PATH}")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="邮件同步与向量化")
    parser.add_argument("--full", action="store_true", help="全量重建")
    parser.add_argument("--days", type=int, default=None, help="同步最近 N 天")
    parser.add_argument("--max", type=int, default=500, help="最大邮件数")
    args = parser.parse_args()
    
    print("=" * 60)
    print("📧 邮件同步与向量化")
    print("=" * 60)
    
    # 1. 加载模型
    print(f"🔮 加载模型: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    # 2. 获取 token
    print("🔑 获取访问令牌...")
    access_token = load_token()
    
    # 3. 获取邮件
    if args.full:
        print("📦 全量同步...")
        emails = fetch_emails(access_token, days=None, max_emails=args.max)
    else:
        print(f"📦 增量同步（最近 {args.days or 30} 天）...")
        emails = fetch_emails(access_token, days=args.days or 30, max_emails=args.max)
    
    print(f"   共获取 {len(emails)} 封邮件")
    
    # 4. 初始化数据库
    conn = init_db()
    existing_ids = get_existing_email_ids(conn)
    
    # 5. 过滤新邮件
    new_emails = [e for e in emails if e.get("id") not in existing_ids]
    print(f"   新邮件: {len(new_emails)} 封")
    
    if not new_emails and not args.full:
        print("✅ 没有新邮件，跳过")
        conn.close()
        return
    
    # 6. 处理邮件
    synced_at = int(time.time())
    all_chunks = []
    all_meta = []
    
    for email in new_emails:
        text = parse_email(email)
        chunks = smart_chunks(text)
        all_chunks.extend(chunks)
        
        for _ in chunks:
            all_meta.append({
                "id": email.get("id", ""),
                "subject": email.get("subject", ""),
                "from_addr": email.get("from", {}).get("emailAddress", {}).get("address", ""),
                "received_time": email.get("receivedDateTime", ""),
                "source": f"email:{email.get('id', '')}"
            })
    
    print(f"   生成 {len(all_chunks)} 个文本块")
    
    # 7. 存入数据库
    ids, new_texts = upsert_chunks(conn, all_chunks, all_meta, synced_at)
    print(f"💾 数据库: {len(ids)} 记录")
    
    # 8. 重建索引
    build_faiss_index(conn, model)
    
    conn.close()
    print("=" * 60)
    print("🎉 同步完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
