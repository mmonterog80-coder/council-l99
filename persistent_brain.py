#!/usr/bin/env python
"""
PERSISTENT BRAIN & EPISODIC RAG MEMORY — COUNCIL L99
Persistent memory and continuous retrieval module for multi-agent workflows.
Uses SQLite for storage + local vector embeddings via Ollama (nomic-embed-text).
"""

import io
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import requests

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, io.UnsupportedOperation):
        pass

BASE_DIR = Path(__file__).resolve().parent
MEMORY_DB_PATH = BASE_DIR / "council_memory.sqlite"
OLLAMA_EMBED_URL = os.environ.get("OLLAMA_EMBED_URL", "http://127.0.0.1:11434/api/embed")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text:latest")

CORE_AXIOMS = [
    (
        "COLLEGIATE_DELIBERATION_RULE",
        "Multi-agent deliberation requires cross-model verification across adversarial auditing, business constraints, and technical architecture.",
        "architecture"
    ),
    (
        "ORCHESTRATOR_WORKER_HIERARCHY",
        "The primary orchestrator handles high-level task contracts, delegating heavy code generation and static analysis to specialized workers.",
        "governance"
    ),
    (
        "NON_BLOCKING_ASYNC_DAEMONS",
        "Long-running tasks and continuous services must execute asynchronously with status and error logging redirected to disk.",
        "governance"
    ),
    (
        "STRICT_CODE_VALIDATION",
        "All generated Python code must be validated with ast.parse() and linted prior to writing or execution.",
        "quality"
    )
]


def init_memory_db():
    """Initializes SQLite schema for vector memory and deliberation records."""
    conn = sqlite3.connect(MEMORY_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_axioms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            axiom_key TEXT UNIQUE,
            content TEXT,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodic_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            summary TEXT,
            participants TEXT,
            elapsed_sec REAL,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS operational_lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lesson_key TEXT UNIQUE,
            lesson_content TEXT,
            category TEXT,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    for key, content, cat in CORE_AXIOMS:
        cursor.execute("""
            INSERT OR IGNORE INTO system_axioms (axiom_key, content, category)
            VALUES (?, ?, ?)
        """, (key, content, cat))

    conn.commit()
    conn.close()


def get_embedding(text: str) -> np.ndarray | None:
    """Generates normalized vector embedding using local Ollama model."""
    try:
        resp = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "input": text},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            embeddings = data.get("embeddings") or [data.get("embedding")]
            if embeddings and embeddings[0]:
                vec = np.array(embeddings[0], dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 0 else vec
    except (requests.RequestException, KeyError, ValueError, np.linalg.LinAlgError):
        pass
    return None


def store_lesson(lesson_key: str, lesson_content: str, category: str = "general"):
    """Stores an operational lesson learned with vector embedding."""
    init_memory_db()
    vec = get_embedding(lesson_content)
    vec_blob = vec.tobytes() if vec is not None else None

    conn = sqlite3.connect(MEMORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO operational_lessons (lesson_key, lesson_content, category, embedding)
        VALUES (?, ?, ?, ?)
    """, (lesson_key, lesson_content, category, vec_blob))
    conn.commit()
    conn.close()


def record_deliberation(topic: str, summary: str, participants: list[str], elapsed_sec: float):
    """Records a deliberation outcome for persistent context tracking."""
    init_memory_db()
    vec = get_embedding(topic + " " + summary[:400])
    vec_blob = vec.tobytes() if vec is not None else None

    conn = sqlite3.connect(MEMORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO episodic_memories (topic, summary, participants, elapsed_sec, embedding)
        VALUES (?, ?, ?, ?, ?)
    """, (topic, summary, json.dumps(participants), elapsed_sec, vec_blob))
    conn.commit()
    conn.close()


def search_memories(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Retrieves top-k relevant memories based on cosine similarity."""
    init_memory_db()
    q_vec = get_embedding(query)
    conn = sqlite3.connect(MEMORY_DB_PATH)
    cursor = conn.cursor()

    results = []
    cursor.execute("SELECT topic, summary, embedding FROM episodic_memories WHERE embedding IS NOT NULL")
    for topic, summary, emb_blob in cursor.fetchall():
        if emb_blob and q_vec is not None:
            vec = np.frombuffer(emb_blob, dtype=np.float32)
            similarity = float(np.dot(q_vec, vec))
            results.append({"type": "deliberation", "title": topic, "content": summary, "score": similarity})

    cursor.execute("SELECT lesson_key, lesson_content, embedding FROM operational_lessons WHERE embedding IS NOT NULL")
    for key, content, emb_blob in cursor.fetchall():
        if emb_blob and q_vec is not None:
            vec = np.frombuffer(emb_blob, dtype=np.float32)
            similarity = float(np.dot(q_vec, vec))
            results.append({"type": "lesson", "title": key, "content": content, "score": similarity})

    conn.close()
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def format_memory_context(query: str, top_k: int = 3) -> str:
    """Formats retrieved memories into an injection block for prompt augmentation."""
    memories = search_memories(query, top_k=top_k)
    if not memories:
        return ""
    lines = ["[RECALLED MEMORY CONTEXT]:"]
    for m in memories:
        lines.append(f"- ({m['type'].upper()} score={m['score']:.2f}) {m['title']}: {m['content'][:250]}...")
    return "\n".join(lines)


if __name__ == "__main__":
    init_memory_db()
    print(f"Council memory DB initialized at {MEMORY_DB_PATH}")
