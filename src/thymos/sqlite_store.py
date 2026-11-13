"""Optional SQLite-backed memory and state store.

The default store is an in-memory dict serialised to one JSON file.  That is
fine for a demo, but a full rewrite per save is a poor fit for a long-running
personal bot.  This adapter keeps the same ``MemoryStore`` interface while
persisting memory items, associations and the engine state in SQLite.

It is deliberately *not* a vector index: retrieval still loads the working set
into memory and scans it.  What it buys is transactional writes, a durable
record per memory, a separate engine-state row, and a versioned schema that can
be migrated instead of silently invalidating an old JSON file.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from .memory import MemoryStore
from .types import MemoryItem


class SQLiteMemoryStore(MemoryStore):
    """Memory store plus a small engine-state table in one SQLite database."""

    SCHEMA_VERSION = 1
    STATE_KEY = "engine_state"

    def __init__(self, path: str | Path, items: Iterable[MemoryItem] | None = None) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._ensure_schema()
        loaded = [
            MemoryItem.from_dict(json.loads(row[0]))
            for row in self._conn.execute("SELECT data FROM memories")
        ]
        super().__init__(loaded)
        for item in items or []:
            self.add(item)

    # ------------------------------------------------------------- schema
    @property
    def schema_version(self) -> int:
        row = self._conn.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0

    def _ensure_schema(self) -> None:
        version = self.schema_version
        if version > self.SCHEMA_VERSION:
            raise RuntimeError(
                f"SQLite schema {version} is newer than supported {self.SCHEMA_VERSION}; "
                "upgrade Thymos instead of opening it with an older build"
            )
        if version == 0:
            with self._conn:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
                )
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS edges ("
                    "a TEXT NOT NULL, b TEXT NOT NULL, weight REAL NOT NULL, "
                    "PRIMARY KEY (a, b))"
                )
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                self._conn.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
        else:
            # Tables are created by the first migration; this also repairs an
            # interrupted old database that has a version but no meta table.
            with self._conn:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
                )
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS edges ("
                    "a TEXT NOT NULL, b TEXT NOT NULL, weight REAL NOT NULL, "
                    "PRIMARY KEY (a, b))"
                )
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )

    # ------------------------------------------------------------- state
    def save_state(self, data: dict[str, Any]) -> None:
        """Persist the engine snapshot (without duplicating memory items)."""
        payload = json.dumps(data, ensure_ascii=False)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                (self.STATE_KEY, payload),
            )

    def load_state(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key=?", (self.STATE_KEY,)
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        if not isinstance(data, dict):
            raise ValueError("engine_state is not a JSON object")
        return data

    # ------------------------------------------------------------- memory
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

    def flush(self) -> None:
        """Write current in-memory mutations (review, decay, prune) to SQLite."""
        with self._lock, self._conn:
            for item in self._items.values():
                self._conn.execute(
                    "INSERT OR REPLACE INTO memories(id, data) VALUES (?, ?)",
                    (item.id, json.dumps(item.to_dict(), ensure_ascii=False)),
                )
            self._conn.execute("DELETE FROM edges")
            for item in self._items.values():
                for other_id, weight in item.associations.items():
                    self._conn.execute(
                        "INSERT OR REPLACE INTO edges(a, b, weight) VALUES (?, ?, ?)",
                        (item.id, other_id, max(0.0, min(1.0, float(weight)))),
                    )

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                # A checkpoint is an optimization for backup, not a reason to
                # fail shutdown.
                pass
            self._conn.close()

    def __enter__(self) -> "SQLiteMemoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def migrate_json(source: str | Path, target: str | Path) -> SQLiteMemoryStore:
    """Import an existing Thymos JSON state file into a new SQLite database.

    The source file is left untouched.  The target must not already contain a
    different engine state; overwriting is explicit and is done by the caller
    choosing a fresh path.
    """
    source_path = Path(source)
    with source_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("source state is not a JSON object")

    store = SQLiteMemoryStore(target)
    if store.load_state() is not None:
        store.close()
        raise FileExistsError(f"target already has engine state: {target}")

    for raw in data.get("memory") or []:
        store.add(MemoryItem.from_dict(raw))
    state = dict(data)
    # Memory lives in the memories table; keeping a second copy in state is
    # exactly the JSON bloat this backend is meant to avoid.
    state["memory"] = []
    store.save_state(state)
    store.flush()
    return store
