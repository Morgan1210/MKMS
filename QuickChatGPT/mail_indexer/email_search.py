"""
email_search.py - 邮件语义检索

功能：
1. 语义搜索邮件
2. 支持时间范围过滤
3. 支持发件人过滤

用法：
    python email_search.py "税务文件"
    python email_search.py "发票" --from_addr amazon.com
    python email_search.py "合同" --days 30 --top 10
"""

import sys
import os
import io

# Windows UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMAIL_DB_PATH = Path(__file__).parent.parent / "email.db"
EMAIL_INDEX_PATH = Path(__file__).parent.parent / "email.index"
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
        if not EMAIL_INDEX_PATH.exists():
            raise FileNotFoundError(f"Index not found: {EMAIL_INDEX_PATH}")
        _index = faiss.read_index(str(EMAIL_INDEX_PATH))
    return _index


def search(
    query: str,
    top_k: int = 5,
    min_score: float = 0.15,
    days: int = None,
    from_addr: str = None,
):
    """
    语义搜索邮件
    
    Args:
        query: 搜索查询
        top_k: 返回数量
        min_score: 最小相似度
        days: 最近多少天
        from_addr: 发件人过滤（支持部分匹配）
    
    Returns:
        list[dict]: 搜索结果
    """
    model = get_model()
    index = get_index()
    
    # 向量化查询
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    
    # 搜索（多取一些用于过滤）
    D, I = index.search(q, top_k * 3)
    
    # 收集结果
    ids = []
    scores = []
    for score, cid in zip(D[0].tolist(), I[0].tolist()):
        if cid == -1 or score < min_score:
            continue
        ids.append(int(cid))
        scores.append(float(score))
    
    if not ids:
        return []
    
    # 从数据库获取详情
    conn = sqlite3.connect(EMAIL_DB_PATH)
    qmarks = ",".join(["?"] * len(ids))
    rows = conn.execute(
        f"""SELECT id, text, email_id, subject, from_addr, received_time, source 
           FROM email_chunks WHERE id IN ({qmarks})""",
        ids
    ).fetchall()
    conn.close()
    
    # 过滤
    results = []
    score_map = {ids[i]: scores[i] for i in range(len(ids))}
    
    for row in rows:
        cid, text, email_id, subject, from_addr_val, received_time, source = row
        
        # 时间过滤
        if days and received_time:
            try:
                email_date = datetime.fromisoformat(received_time.replace("Z", "+00:00"))
                if email_date < datetime.now(email_date.tzinfo) - timedelta(days=days):
                    continue
            except:
                pass
        
        # 发件人过滤
        if from_addr and from_addr.lower() not in (from_addr_val or "").lower():
            continue
        
        results.append({
            "id": int(cid),
            "score": float(score_map.get(cid, 0.0)),
            "text": text,
            "email_id": email_id,
            "subject": subject,
            "from_addr": from_addr_val,
            "received_time": received_time,
            "source": source,
            "type": "email"
        })
    
    # 按分数排序，取 top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def main():
    parser = argparse.ArgumentParser(description="邮件语义搜索")
    parser.add_argument("query", help="搜索查询")
    parser.add_argument("--top", type=int, default=5, help="返回数量")
    parser.add_argument("--min_score", type=float, default=0.15, help="最小相似度")
    parser.add_argument("--days", type=int, default=None, help="最近多少天")
    parser.add_argument("--from_addr", type=str, default=None, help="发件人过滤")
    args = parser.parse_args()
    
    print(f"🔍 搜索: {args.query}")
    print("-" * 60)
    
    results = search(
        args.query,
        top_k=args.top,
        min_score=args.min_score,
        days=args.days,
        from_addr=args.from_addr,
    )
    
    if not results:
        print("❌ 没有找到相关邮件")
        return
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. 📧 {r['subject']}")
        print(f"   发件人: {r['from_addr']}")
        print(f"   时间: {r['received_time']}")
        print(f"   相似度: {r['score']:.3f}")
        print(f"   内容预览: {r['text'][:200]}...")


if __name__ == "__main__":
    main()
