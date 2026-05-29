#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补全邮件向量化 - 简单直接版本
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import sqlite3
import numpy as np
import faiss
import pathlib

# 配置
DB_PATH = "email.db"
INDEX_PATH = "email.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("=" * 60)
print("📧 补全邮件向量化")
print("=" * 60)

# 1. 加载现有索引
print("\n【1】加载现有 FAISS 索引...")
try:
    index = faiss.read_index(INDEX_PATH)
    print(f"  ✅ 索引类型: {type(index).__name__}")
    print(f"  ✅ 索引向量数: {index.ntotal}")
except Exception as e:
    print(f"  ❌ 加载失败: {e}")
    sys.exit(1)

# 2. 获取数据库中所有 chunk IDs
print("\n【2】获取数据库中的 chunk IDs...")
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT id FROM email_chunks ORDER BY id")
all_db_ids = [row[0] for row in c.fetchall()]
print(f"  ✅ 数据库 chunks: {len(all_db_ids)}")

# 3. 假设索引中的 IDs 是连续的 1 到 index.ntotal
# 找出缺失的 IDs（简单策略）
indexed_count = index.ntotal
missing_ids = list(range(indexed_count + 1, len(all_db_ids) + 1))

print(f"\n【3】缺失分析:")
print(f"  📊 数据库总数: {len(all_db_ids)}")
print(f"  ✅ 已索引: {indexed_count}")
print(f"  ❌ 缺失: {len(missing_ids)}")

if not missing_ids:
    print("\n✅ 所有 chunks 已向量化！")
    conn.close()
    sys.exit(0)

print(f"\n  📝 缺失的 IDs: {missing_ids[0]} 到 {missing_ids[-1]}")

# 4. 加载模型
print("\n【4】加载 Sentence Transformer 模型...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(MODEL_NAME)
print(f"  ✅ 模型加载完成: {MODEL_NAME}")

# 5. 批量向量化缺失的 chunks
print(f"\n【5】向量化 {len(missing_ids)} 个缺失 chunks...")
BATCH_SIZE = 64

try:
    for i in range(0, len(missing_ids), BATCH_SIZE):
        batch_ids = missing_ids[i:i+BATCH_SIZE]
        
        # 从数据库获取文本
        qmarks = ",".join(["?"] * len(batch_ids))
        c.execute(f"SELECT id, text FROM email_chunks WHERE id IN ({qmarks})", batch_ids)
        rows = c.fetchall()
        
        if not rows:
            print(f"  ⚠️ 批次 {i//BATCH_SIZE + 1}: 未找到数据")
            continue
        
        # 向量化
        ids = [row[0] for row in rows]
        texts = [row[1] for row in rows]
        
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        
        # 添加到索引
        ids_array = np.array(ids, dtype=np.int64)
        index.add_with_ids(embeddings, ids_array)
        
        processed = min(i + BATCH_SIZE, len(missing_ids))
        print(f"  ✅ 处理进度: {processed}/{len(missing_ids)} ({(processed/len(missing_ids)*100):.1f}%)")
    
    print(f"\n  ✅ 向量化完成！")
    print(f"  📊 索引向量数: {index.ntotal}")
    
    # 6. 保存索引
    print("\n【6】保存 FAISS 索引...")
    faiss.write_index(index, INDEX_PATH)
    
    index_size = pathlib.Path(INDEX_PATH).stat().st_size / 1024 / 1024
    print(f"  ✅ 索引已保存: {index_size:.2f} MB")
    
except Exception as e:
    print(f"\n  ❌ 向量化失败: {e}")
    import traceback
    traceback.print_exc()
    conn.close()
    sys.exit(1)

conn.close()

print("\n" + "=" * 60)
print("✅ 补全完成！")
print("=" * 60)
print(f"  📊 索引向量数: {index.ntotal}")
print(f"  📊 数据库 chunks: {len(all_db_ids)}")
if index.ntotal == len(all_db_ids):
    print("  🎉 完美匹配！所有 chunks 已向量化")
else:
    print(f"  ⚠️ 仍有 {len(all_db_ids) - index.ntotal} 个 chunks 未向量化")
