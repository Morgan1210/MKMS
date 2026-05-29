import sqlite3
import json
from datetime import datetime


class ChatHistory:
    def __init__(self, db_path='chat_history.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 索引加速
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON chat_history(session_id)")

    def add_message(self, session_id, role, content):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content)
            )

    def get_recent_messages(self, session_id, limit=100):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT role, content FROM chat_history 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
                """,
                (session_id, limit)
            )
            # 按时间正序返回
            messages = list(cursor)[::-1]
            return [{"role": r, "content": c} for r, c in messages]

    def clear_session(self, session_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))