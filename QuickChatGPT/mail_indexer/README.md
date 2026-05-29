# Email Tools - 邮件向量化与检索工具集

## 功能

1. **email_sync.py** - 邮件同步与向量化
   - 从 Microsoft Graph API 拉取邮件
   - 增量向量化并存入数据库
   - 重建 FAISS 索引

2. **email_search.py** - 命令行邮件搜索
   - 语义搜索邮件
   - 支持时间/发件人过滤

3. **email_api.py** - Python API（供 AI 调用）
   - 统一的检索接口
   - 结构化返回结果

4. **unified_retrieve.py** - 统一检索入口
   - 整合文档和邮件
   - 混合排序返回

## 用法

### 同步邮件

```bash
# 增量同步（最近30天）
python email_sync.py

# 全量重建
python email_sync.py --full

# 同步最近7天
python email_sync.py --days 7
```

### 搜索邮件

```bash
# 语义搜索
python email_search.py "发票"

# 带过滤
python email_search.py "合同" --days 30 --from_addr amazon.com
```

### Python 调用

```python
from email_tools.email_api import search_email

results = search_email("税务文件", top_k=5, days=90)
for r in results:
    print(r["subject"], r["score"])
```

### 统一检索

```python
from email_tools.unified_retrieve import unified_search

# 同时搜索文档和邮件
results = unified_search("税务", top_k=10)
for r in results:
    if r["type"] == "email":
        print(f"📧 {r['subject']}")
    else:
        print(f"📄 {r['source']}")
```

## 数据库结构

### email.db

```sql
CREATE TABLE email_chunks (
    id INTEGER PRIMARY KEY,
    hash TEXT UNIQUE,           -- 内容哈希
    text TEXT,                  -- 文本内容
    email_id TEXT,              -- Graph API 邮件 ID
    subject TEXT,               -- 主题
    from_addr TEXT,             -- 发件人
    received_time TEXT,         -- 接收时间
    created_at INTEGER,         -- 创建时间
    source TEXT,                -- 来源标识
    synced_at INTEGER           -- 同步时间
);
```

## 架构

```
┌─────────────────┐
│  Microsoft      │
│  Graph API      │
└────────┬────────┘
         │ fetch
         ▼
┌─────────────────┐
│  email_sync.py  │──► email.db (SQLite)
│                 │──► email.index (FAISS)
└─────────────────┘

┌─────────────────┐
│  AI / main.py   │
└────────┬────────┘
         │ query
         ▼
┌─────────────────┐
│ unified_retrieve│──► mkms.db + email.db
│                 │──► mkms.index + email.index
└─────────────────┘
```

## 定时同步

可以配置 cron 或 Windows Task Scheduler 定时运行：

```bash
# 每小时同步一次
0 * * * * python /path/to/email_sync.py
```

## 注意事项

1. 首次运行需要先执行 `email_read.py` 完成授权
2. Token 会自动刷新，无需手动维护
3. 向量化使用 `all-MiniLM-L6-v2` 模型（384维）
4. 建议定期运行 `--full` 全量重建以优化索引
