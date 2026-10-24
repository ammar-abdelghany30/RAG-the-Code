# src/memory.py
import re
import sqlite3
from contextlib import closing
from typing import Dict, List, Tuple

from src.config import MEMORY_DB_PATH

MAX_TURNS = 6
MAX_CHARS_PER_TURN = 1500  # large enough to keep a previous code answer for follow-ups

LANGUAGES = ["python", "javascript", "typescript", "java", "c++", "c#", "rust",
             "golang", "php", "ruby", "kotlin", "swift", "sql"]
FRAMEWORKS = ["flask", "fastapi", "django", "react", "pytorch", "tensorflow",
              "keras", "pandas", "numpy", "streamlit", "langchain"]


def _connect() -> sqlite3.Connection:
    MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS turns (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               session_id TEXT NOT NULL,
               role TEXT NOT NULL,
               content TEXT NOT NULL,
               route TEXT,
               created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS preferences (
               session_id TEXT NOT NULL,
               key TEXT NOT NULL,
               value TEXT NOT NULL,
               PRIMARY KEY (session_id, key))"""
    )
    return conn


def _last_mention(text: str, terms: List[str]):
    """The term mentioned last in the text (so 'convert it to Java' picks Java)."""
    best, best_pos = None, -1
    for term in terms:
        # the lookarounds stop 'java' matching inside 'javascript' and 'sql' inside 'mysql'
        pattern = r"(?<![\w+#])" + re.escape(term) + r"(?![\w+#])"
        for m in re.finditer(pattern, text):
            if m.start() > best_pos:
                best, best_pos = term, m.start()
    return best


def detect_preferences(text: str) -> Dict[str, str]:
    """Cheap keyword heuristic, no LLM call."""
    text = text.lower()
    found = {}
    language = _last_mention(text, LANGUAGES)
    framework = _last_mention(text, FRAMEWORKS)
    if language:
        found["language"] = language
    if framework:
        found["framework"] = framework
    return found


def add_turn(session_id: str, role: str, content: str, route: str = "") -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO turns (session_id, role, content, route) VALUES (?, ?, ?, ?)",
            (session_id, role, content, route),
        )
        if role == "user":
            for key, value in detect_preferences(content).items():
                conn.execute(
                    "INSERT OR REPLACE INTO preferences (session_id, key, value) VALUES (?, ?, ?)",
                    (session_id, key, value),
                )


def get_recent_turns(session_id: str, limit: int = MAX_TURNS) -> List[Tuple[str, str]]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT role, content FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return list(reversed(rows))  # oldest first


def get_preferences(session_id: str) -> Dict[str, str]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT key, value FROM preferences WHERE session_id = ?", (session_id,)
        ).fetchall()
    return dict(rows)


def build_memory_text(session_id: str) -> str:
    """One text block for the LLM prompt. Empty string if there is nothing yet."""
    prefs = get_preferences(session_id)
    turns = get_recent_turns(session_id)
    if not prefs and not turns:
        return ""

    lines = []
    if prefs:
        lines.append("User preferences: " + "; ".join(f"{k}={v}" for k, v in prefs.items()))
    if turns:
        lines.append("Recent conversation:")
        for role, content in turns:
            if len(content) > MAX_CHARS_PER_TURN:
                content = content[:MAX_CHARS_PER_TURN] + " ...[truncated]"
            lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


def clear_session(session_id: str) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM preferences WHERE session_id = ?", (session_id,))