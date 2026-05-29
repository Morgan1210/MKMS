#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补全邮件向量化 - 仅处理缺失的 chunks
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

# 2. 获取索引中已有的 IDs
print("\n【2】获取已索引的 IDs...")
indexed_ids = set()

# FAISS IndexIDMap2 的 id_map 是一个 Int64Vector
# 需要转换为 Python list
if hasattr(index, 'id_map') and index.id_map.size() > 0:
    # 方法1: 通过搜索获取一些 ID（不完整）
    # 方法2: 直接遍历 id_map
    try:
        # 尝试将 id_map 转换为列表
        id_map_size = index.id_map.size()
        print(f"  📊 id_map 大小: {id_map_size}")
        
        # 使用 faiss 的内部方法获取 IDs
        # IndexIDMap2 的 IDs 存储在 id_map 中
        indexed_ids_list = []
        for i in range(id_map_size):
            idx = index.id_map.at(i)
            if isinstance(idx, (list, tuple, np.ndarray)):
                indexed_ids_list.extend(idx)
            else:
                indexed_ids_list.append(idx)
        
        indexed_ids = set(indexed_ids_list)
        print(f"  ✅ 已索引 IDs: {len(indexed_ids)}")
    except Exception as e:
        print(f"  ⚠️ 无法读取 id_map: {e}")
        print("  💡 将采用保守策略：仅添加数据库中存在但 ID < 20536 的 chunks")
        indexed_ids = set(range(index.ntotal))
else:
    print("  ⚠️ 索引中没有 id_map，假设所有都需要重新索引")
    indexed_ids = set()

# 3. 获取数据库中所有 chunk IDs
print("\n【3】获取数据库中的 chunk IDs...")
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT id FROM email_chunks ORDER BY id")
all_db_ids = [row[0] for row in c.fetchall()]
print(f"  ✅ 数据库 chunks: {len(all_db_ids)}")

# 4. 找出缺失的 chunks
all_db_ids_set = set(all_db_ids)
missing_ids = list(all_db_ids_set - indexed_ids)
missing_ids.sort()

print(f"\n【4】缺失分析:")
print(f"  📊 数据库总数: {len(all_db_ids)}")
print(f"  ✅ 已索引: {len(indexed_ids)}")
print(f"  ❌ 缺失: {len(missing_ids)}")

if not missing_ids:
    print("\n✅ 所有 chunks 已向量化！")
    conn.close()
    sys.exit(0)

print(f"\n  📝 缺失的 IDs (前10个): {missing_ids[:10]}...")

# 5. 加载模型
print("\n【5】加载 Sentence Transformer 模型...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(MODEL_NAME)
print(f"  ✅ 模型加载完成")

# 6. 批量向量化缺失的 chunks
print(f"\n【6】向量化 {len(missing_ids)} 个缺失 chunks...")
BATCH_SIZE = 64

try:
    for i in range(0, len(missing_ids), BATCH_SIZE):
        batch_ids = missing_ids[i:i+BATCH_SIZE]
        
        # 从数据库获取文本
        qmarks = ",".join(["?"] * len(batch_ids))
        c.execute(f"SELECT id, text FROM email_chunks WHERE id IN ({qmarks})", batch_ids)
        rows = c.fetchall()
        
        if not rows:
            continue
        
        # 向量化
        ids = [row[0] for row in rows]
        texts = [row[1] for row in rows]
        
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        
        # 添加到索引
        ids_array = np.array(ids, dtype=np.int64)
        index.add_with_ids(embeddings, ids_array)
        
        print(f"  ✅ 处理 {min(i+BATCH_SIZE, len(missing_ids))}/{len(missing_ids)} (新增 {len(ids)} 个向量)")
    
    print(f"\n  ✅ 向量化完成！")
    print(f"  📊 索引向量数: {index.ntotal}")
    
    # 7. 保存索引
    print("\n【7】保存 FAISS 索引...")
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
print(f"  索引向量数: {index.ntotal}")
print(f"  数据库 chunks: {len(all_db_ids)}")
if index.ntotal == len(all_db_ids):
    print("  🎉 完美匹配！所有 chunks 已向量化")
else:
    print(f"  ⚠️ 仍有 {len(all_db_ids) - index.ntotal} 个 chunks 未向量化")
