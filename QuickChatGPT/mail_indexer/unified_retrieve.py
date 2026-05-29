"""
统一检索入口 - 整合文档和邮件

用法：
    from unified_retrieve import UnifiedRetriever
    
    retriever = UnifiedRetriever()
    results = retriever.search("税务", top_k=10)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 路径配置
MKMS_DB_PATH = Path(__file__).parent.parent / "mkms.db"
MKMS_INDEX_PATH = Path(__file__).parent.parent / "mkms.index"
EMAIL_DB_PATH = Path(__file__).parent.parent / "email.db"
EMAIL_INDEX_PATH = Path(__file__).parent.parent / "email.index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class UnifiedRetriever:
    """统一检索器 - 文档 + 邮件"""
    
    _model = None
    _doc_index = None
    _email_index = None
    
    @property
    def model(self):
        if UnifiedRetriever._model is None:
            UnifiedRetriever._model = SentenceTransformer(MODEL_NAME)
        return UnifiedRetriever._model
    
    @property
    def doc_index(self):
        if UnifiedRetriever._doc_index is None:
            if MKMS_INDEX_PATH.exists():
                UnifiedRetriever._doc_index = faiss.read_index(str(MKMS_INDEX_PATH))
        return UnifiedRetriever._doc_index
    
    @property
    def email_index(self):
        if UnifiedRetriever._email_index is None:
            if EMAIL_INDEX_PATH.exists():
                UnifiedRetriever._email_index = faiss.read_index(str(EMAIL_INDEX_PATH))
        return UnifiedRetriever._email_index
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.15,
        include_docs: bool = True,
        include_emails: bool = True,
        email_days: Optional[int] = None,
        email_from: Optional[str] = None,
    ) -> List[Dict]:
        """
        统一检索
        
        Args:
            query: 搜索查询
            top_k: 总返回数量
            min_score: 最小相似度
            include_docs: 是否包含文档
            include_emails: 是否包含邮件
            email_days: 邮件时间过滤
            email_from: 邮件发件人过滤
        
        Returns:
            list[dict]: 混合结果（按分数排序）
        """
        # 向量化查询
        q = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        
        all_results = []
        
        # 1. 文档检索
        if include_docs and self.doc_index:
            D, I = self.doc_index.search(q, top_k)
            doc_results = self._fetch_docs(I[0].tolist(), D[0].tolist(), min_score)
            all_results.extend(doc_results)
        
        # 2. 邮件检索
        if include_emails and self.email_index:
            D, I = self.email_index.search(q, top_k)
            email_results = self._fetch_emails(
                I[0].tolist(), D[0].tolist(), min_score, email_days, email_from
            )
            all_results.extend(email_results)
        
        # 3. 统一排序
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]
    
    def _fetch_docs(self, ids, scores, min_score):
        """获取文档结果"""
        results = []
        valid = [(int(cid), float(score)) for cid, score in zip(ids, scores) 
                 if cid != -1 and score >= min_score]
        
        if not valid:
            return results
        
        ids_list = [v[0] for v in valid]
        score_map = dict(valid)
        
        conn = sqlite3.connect(MKMS_DB_PATH)
        qmarks = ",".join(["?"] * len(ids_list))
        rows = conn.execute(
            f"SELECT id, text, source, created_at FROM chunks WHERE id IN ({qmarks})",
            ids_list
        ).fetchall()
        conn.close()
        
        for row in rows:
            results.append({
                "id": row[0],
                "score": score_map.get(row[0], 0.0),
                "text": row[1],
                "source": row[2],
                "created_at": row[3],
                "type": "document"
            })
        
        return results
    
    def _fetch_emails(self, ids, scores, min_score, days, from_addr):
        """获取邮件结果"""
        results = []
        valid = [(int(cid), float(score)) for cid, score in zip(ids, scores) 
                 if cid != -1 and score >= min_score]
        
        if not valid:
            return results
        
        ids_list = [v[0] for v in valid]
        score_map = dict(valid)
        
        conn = sqlite3.connect(EMAIL_DB_PATH)
        qmarks = ",".join(["?"] * len(ids_list))
        rows = conn.execute(
            f"""SELECT id, text, email_id, subject, from_addr, received_time, source 
               FROM email_chunks WHERE id IN ({qmarks})""",
            ids_list
        ).fetchall()
        conn.close()
        
        from datetime import datetime, timedelta
        
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
        
        return results
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            "document": {"chunks": 0, "index": False},
            "email": {"chunks": 0, "emails": 0, "index": False},
        }
        
        # 文档统计
        if MKMS_DB_PATH.exists():
            conn = sqlite3.connect(MKMS_DB_PATH)
            stats["document"]["chunks"] = conn.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]
            conn.close()
        stats["document"]["index"] = MKMS_INDEX_PATH.exists()
        
        # 邮件统计
        if EMAIL_DB_PATH.exists():
            conn = sqlite3.connect(EMAIL_DB_PATH)
            stats["email"]["chunks"] = conn.execute(
                "SELECT COUNT(*) FROM email_chunks"
            ).fetchone()[0]
            stats["email"]["emails"] = conn.execute(
                "SELECT COUNT(DISTINCT email_id) FROM email_chunks WHERE email_id != ''"
            ).fetchone()[0]
            conn.close()
        stats["email"]["index"] = EMAIL_INDEX_PATH.exists()
        
        return stats


# 便捷函数
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = UnifiedRetriever()
    return _retriever


def unified_search(query: str, **kwargs) -> List[Dict]:
    """便捷检索函数"""
    return get_retriever().search(query, **kwargs)


if __name__ == "__main__":
    query = f"Donald Guilmette"

    r = UnifiedRetriever()

    results = r.search(query, top_k=5)
    print(f"\n🔍 搜索 '{query}': {len(results)} 条结果")
    for res in results:
        type_icon = "📄" if res["type"] == "document" else "📧"
        print(f"  {type_icon} {res.get('subject', res.get('source', '未知'))} (score={res['score']:.3f})")
