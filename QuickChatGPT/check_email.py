import sqlite3
import os

db_path = "email.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # 检查表
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = c.fetchall()
    print(f"Tables: {tables}")
    
    for table in tables:
        table_name = table[0]
        c.execute(f"SELECT COUNT(*) FROM [{table_name}]")
        count = c.fetchone()[0]
        print(f"  {table_name}: {count} rows")
    
    conn.close()
else:
    print("email.db not found")
