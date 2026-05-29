#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索 Hyperbola 项目报错相关邮件
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 切换到 QuickChatGPT 目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("🔍 搜索 Hyperbola 报错相关邮件")
print("=" * 60)

try:
    from mkms_retrieve import retrieve_from_emails
    
    # 多个查询关键词
    queries = [
        "Hyperbola error",
        "Hyperbola 报错",
        "training error",
        "NEAT error",
        "Gen 报错",
    ]
    
    all_results = []
    seen_ids = set()
    
    for query in queries:
        print(f"\n🔍 查询: '{query}'")
        results = retrieve_from_emails(query, top_k=3, min_score=0.1)
        
        for r in results:
            rid = r.get('id')
            if rid not in seen_ids:
                seen_ids.add(rid)
                all_results.append(r)
    
    if all_results:
        # 按分数排序
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        print(f"\n✅ 共找到 {len(all_results)} 封相关邮件\n")
        print("=" * 60)
        
        for i, r in enumerate(all_results[:5], 1):
            print(f"\n【邮件 {i}】")
            print(f"  主题: {r.get('subject', 'N/A')}")
            print(f"  发件人: {r.get('from_addr', 'N/A')}")
            print(f"  时间: {r.get('received_time', 'N/A')}")
            print(f"  分数: {r.get('score', 0):.4f}")
            
            text = r.get('text', '')
            if len(text) > 300:
                text = text[:300] + '...'
            print(f"  内容预览:\n{text}")
            print("-" * 60)
    else:
        print("\n❌ 未找到相关邮件")
        print("\n💡 建议:")
        print("  1. 检查邮件是否已同步到知识库")
        print("  2. 尝试其他关键词（如 'training', 'NEAT', 'Python error'）")
        print("  3. 检查邮件时间范围（可能不在索引范围内）")

except Exception as e:
    print(f"\n❌ 搜索失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("搜索完成")
print("=" * 60)
