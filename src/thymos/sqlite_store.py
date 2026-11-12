"""Optional SQLite-backed memory store.

The default store is an in-memory dict serialised to JSON.  For a personal bot
that is often enough, but a single JSON file is a poor fit for incremental
writes and crash recovery.  This adapter keeps the same ``MemoryStore``
interface while persisting every item and association in SQLite.

It is deliberately *not* a vector index: retrieval still loads the working set
into memory and scans it.  What it buys is transactional writes, a durable
record per memory, and a path to FTS/vector extensions later.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterable

from .memory import MemoryStore
from .types import MemoryItem


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, path: str | Path, items: Iterable[MemoryItem] | None = None) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS edges ("
            "a TEXT NOT NULL, b TEXT NOT NULL, weight REAL NOT NULL, "
            "PRIMARY KEY (a, b))"
        )
        self._conn.commit()
        loaded = [
            MemoryItem.from_dict(json.loads(row[0]))
            for row in self._conn.execute("SELECT data FROM memories")
        ]
        super().__init__(loaded)
        for item in items or []:
            self.add(item)

    def add(self, item: MemoryItem) -> MemoryItem:
        with self._lock:
            super().add(item)
            self._conn.execute(
                "INSERT OR REPLACE INTO memories(id, data) VALUES (?, ?)",
                (item.id, json.dumps(item.to_dict(), ensure_ascii=False)),
            )
            self._conn.commit()
            return item

    def remove(self, memory_id: str) -> bool:
        with self._lock:
            removed = super().remove(memory_id)
            if removed:
                self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
                self._conn.execute("DELETE FROM edges WHERE a=? OR b=?", (memory_id, memory_id))
                self._conn.commit()
            return removed

    def link(self, a_id: str, b_id: str, weight: float) -> None:
        with self._lock:
            super().link(a_id, b_id, weight)
            a, b = self.get(a_id), self.get(b_id)
            if not a or not b:
                return
            w = max(0.0, min(1.0, weight))
            self._conn.executemany(
                "INSERT OR REPLACE INTO edges(a, b, weight) VALUES (?, ?, ?)",
                [(a_id, b_id, w), (b_id, a_id, w)],
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SQLiteMemoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
