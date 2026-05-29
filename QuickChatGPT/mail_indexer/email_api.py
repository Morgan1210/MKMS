"""
email_api.py - 邮件检索 API（供 AI 调用）

功能：
1. 提供统一的检索接口
2. 支持多条件过滤
3. 返回结构化结果

用法：
    from email_tools.email_api import EmailRetriever
    
    retriever = EmailRetriever()
    results = retriever.search("发票", top_k=5)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMAIL_DB_PATH = Path(__file__).parent.parent / "email.db"
EMAIL_INDEX_PATH = Path(__file__).parent.parent / "email.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmailRetriever:
    """邮件检索器"""
    
    _model = None
    _index = None
    
    def __init__(self, db_path=None, index_path=None):
        self.db_path = Path(db_path) if db_path else EMAIL_DB_PATH
        self.index_path = Path(index_path) if index_path else EMAIL_INDEX_PATH
    
    @property
    def model(self):
        if EmailRetriever._model is None:
            EmailRetriever._model = SentenceTransformer(MODEL_NAME)
        return EmailRetriever._model
    
    @property
    def index(self):
        if EmailRetriever._index is None:
            if not self.index_path.exists():
                raise FileNotFoundError(f"Index not found: {self.index_path}")
            EmailRetriever._index = faiss.read_index(str(self.index_path))
        return EmailRetriever._index
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.15,
        days: Optional[int] = None,
        from_addr: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> List[Dict]:
        """
        语义搜索邮件
        
        Args:
            query: 搜索查询
            top_k: 返回数量
            min_score: 最小相似度
            days: 最近多少天
            from_addr: 发件人过滤（部分匹配）
            subject: 主题过滤（部分匹配）
        
        Returns:
            list[dict]: 搜索结果
        """
        # 向量化查询
        q = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        
        # 搜索
        D, I = self.index.search(q, top_k * 3)
        
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
        conn = sqlite3.connect(self.db_path)
        qmarks = ",".join(["?"] * len(ids))
        rows = conn.execute(
            f"""SELECT id, text, email_id, subject, from_addr, received_time, source 
               FROM email_chunks WHERE id IN ({qmarks})""",
            ids
        ).fetchall()
        conn.close()
        
        # 过滤和格式化
        results = []
        score_map = {ids[i]: scores[i] for i in range(len(ids))}
        
        for row in rows:
            cid, text, email_id, subject_val, from_addr_val, received_time, source = row
            
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
            
            # 主题过滤
            if subject and subject.lower() not in (subject_val or "").lower():
                continue
            
            results.append({
                "id": int(cid),
                "score": float(score_map.get(cid, 0.0)),
                "text": text,
                "email_id": email_id,
                "subject": subject_val,
                "from_addr": from_addr_val,
                "received_time": received_time,
                "source": source,
                "type": "email"
            })
        
        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def get_by_email_id(self, email_id: str) -> List[Dict]:
        """根据邮件 ID 获取所有 chunks"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """SELECT id, text, email_id, subject, from_addr, received_time, source 
               FROM email_chunks WHERE email_id = ?""",
            (email_id,)
        ).fetchall()
        conn.close()
        
        return [
            {
                "id": row[0],
                "text": row[1],
                "email_id": row[2],
                "subject": row[3],
                "from_addr": row[4],
                "received_time": row[5],
                "source": row[6],
                "type": "email"
            }
            for row in rows
        ]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        conn = sqlite3.connect(self.db_path)
        
        total_chunks = conn.execute("SELECT COUNT(*) FROM email_chunks").fetchone()[0]
        unique_emails = conn.execute(
            "SELECT COUNT(DISTINCT email_id) FROM email_chunks WHERE email_id != ''"
        ).fetchone()[0]
        
        conn.close()
        
        return {
            "total_chunks": total_chunks,
            "unique_emails": unique_emails,
            "index_exists": self.index_path.exists(),
        }


# 便捷函数
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = EmailRetriever()
    return _retriever


def search_email(query: str, **kwargs) -> List[Dict]:
    """便捷搜索函数"""
    return get_retriever().search(query, **kwargs)


if __name__ == "__main__":
    # 测试
    r = EmailRetriever()
    print("📊 统计:", r.get_stats())
    
    results = r.search("发票", top_k=3)
    print(f"\n🔍 搜索 '发票': {len(results)} 条结果")
    for res in results:
        print(f"  - {res['subject']} (score={res['score']:.3f})")
