#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 MKMS 邮件检索功能
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 切换到 QuickChatGPT 目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from mkms_retrieve import retrieve

# 测试查询
test_queries = [
    "会议安排",
    "project update",
    "发票 账单",
    "python code",
]

print("=" * 60)
print("📧 MKMS 邮件检索测试")
print("=" * 60)

for query in test_queries:
    print(f"\n🔍 查询: {query}")
    print("-" * 60)
    
    try:
        results = retrieve(query, top_k=3, min_score=0.1, include_emails=True)
        
        if not results:
            print("  ❌ 没有找到相关结果")
            continue
        
        for i, r in enumerate(results, 1):
            print(f"\n  【结果 {i}】")
            print(f"    来源: {r.get('source', 'N/A')}")
            print(f"    分数: {r.get('score', 0):.4f}")
            
            if 'subject' in r:
                print(f"    主题: {r.get('subject', 'N/A')}")
                print(f"    发件人: {r.get('from_addr', 'N/A')}")
                print(f"    时间: {r.get('received_time', 'N/A')}")
            
            text = r.get('text', '')
            if len(text) > 100:
                text = text[:100] + "..."
            print(f"    内容: {text}")
    
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("✅ 测试完成")
print("=" * 60)
