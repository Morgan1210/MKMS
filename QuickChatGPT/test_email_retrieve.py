#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试邮件检索功能
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 切换到 QuickChatGPT 目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("📧 MKMS 邮件检索功能测试")
print("=" * 60)

# 1. 检查文件
print("\n【1】检查数据文件...")
import pathlib
db_path = pathlib.Path("email.db")
index_path = pathlib.Path("email.index")

if db_path.exists():
    print(f"  ✅ email.db: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
else:
    print("  ❌ email.db 不存在")

if index_path.exists():
    print(f"  ✅ email.index: {index_path.stat().st_size / 1024 / 1024:.2f} MB")
else:
    print("  ❌ email.index 不存在")
    sys.exit(1)

# 2. 检查数据库内容
print("\n【2】检查数据库内容...")
import sqlite3
try:
    conn = sqlite3.connect("email.db")
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM email_chunks")
    chunk_count = c.fetchone()[0]
    print(f"  ✅ email_chunks 表: {chunk_count} 条记录")
    
    # 查看样例
    c.execute("SELECT id, subject, from_addr, received_time FROM email_chunks LIMIT 3")
    samples = c.fetchall()
    print(f"  📧 样例邮件:")
    for sid, subj, sender, time in samples:
        print(f"     ID {sid}: {subj[:30]} | {sender[:20]} | {time[:10]}")
    
    conn.close()
except Exception as e:
    print(f"  ❌ 数据库错误: {e}")
    sys.exit(1)

# 3. 检查 FAISS 索引
print("\n【3】检查 FAISS 索引...")
try:
    import faiss
    index = faiss.read_index("email.index")
    print(f"  ✅ 索引向量数: {index.ntotal}")
    print(f"  ✅ 向量维度: {index.d}")
except Exception as e:
    print(f"  ❌ 索引错误: {e}")
    sys.exit(1)

# 4. 测试检索
print("\n【4】测试邮件检索...")
try:
    from mkms_retrieve import retrieve_from_emails
    
    test_queries = ["hello", "test", "meeting", "发票", "项目"]
    
    for query in test_queries:
        print(f"\n  🔍 查询: '{query}'")
        results = retrieve_from_emails(query, top_k=3, min_score=0.0)  # 降低阈值到 0
        
        if results:
            print(f"     ✅ 找到 {len(results)} 条结果")
            for i, r in enumerate(results[:2], 1):
                print(f"        [{i}] 分数: {r['score']:.4f}")
                print(f"            主题: {r.get('subject', 'N/A')[:40]}")
                print(f"            内容: {r.get('text', '')[:60]}...")
        else:
            print(f"     ❌ 未找到结果")
    
except Exception as e:
    print(f"  ❌ 检索错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ 测试完成")
print("=" * 60)
