"""In-memory memory store with a sparse association graph."""

from __future__ import annotations

from typing import Any, Iterable

from .mathx import cosine
from .types import MemoryItem


class MemoryStore:
    def __init__(self, items: Iterable[MemoryItem] | None = None) -> None:
        self._items: dict[str, MemoryItem] = {}
        for item in items or []:
            self._items[item.id] = item

    # ------------------------------------------------------------------ CRUD
    def add(self, item: MemoryItem) -> MemoryItem:
        self._items[item.id] = item
        return item

    def get(self, memory_id: str) -> MemoryItem | None:
        return self._items.get(memory_id)

    def remove(self, memory_id: str) -> bool:
        item = self._items.pop(memory_id, None)
        if item is None:
            return False
        for other in self._items.values():
            other.associations.pop(memory_id, None)
        return True

    def all(self) -> list[MemoryItem]:
        return list(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items.values())

    # ---------------------------------------------------------- associations
    def link(self, a_id: str, b_id: str, weight: float) -> None:
        if a_id == b_id:
            return
        a, b = self._items.get(a_id), self._items.get(b_id)
        if not a or not b:
            return
        w = max(0.0, min(1.0, weight))
        a.associations[b_id] = max(a.associations.get(b_id, 0.0), w)
        b.associations[a_id] = max(b.associations.get(a_id, 0.0), w)

    def neighbors(self, memory_id: str) -> list[tuple[MemoryItem, float]]:
        item = self._items.get(memory_id)
        if not item:
            return []
        out = []
        for other_id, weight in item.associations.items():
            other = self._items.get(other_id)
            if other:
                out.append((other, weight))
        return out

    def most_similar(
        self,
        embedding: list[float],
        *,
        exclude: str | None = None,
        limit: int = 3,
        minimum: float = 0.0,
    ) -> list[tuple[MemoryItem, float]]:
        scored = []
        for item in self._items.values():
            if item.id == exclude:
                continue
            sim = cosine(embedding, item.embedding)
            if sim >= minimum:
                scored.append((item, sim))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------ maintenace
    def prune(self, max_items: int = 5000, min_retrievability: float = 0.0, now: float | None = None) -> int:
        """Archive the weakest items when capacity is exceeded."""
        if len(self._items) <= max_items:
            return 0
        from .forgetting import retrievability

        now = now or 0.0
        ranked = sorted(
            self._items.values(),
            key=lambda m: (retrievability(max(0.0, (now - m.last_access) / 86400.0), m.stability) * m.salience),
        )
        removed = 0
        for item in ranked[: max(0, len(self._items) - max_items)]:
            item.state = "archived"
            removed += 1
        return removed

    def to_list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items.values()]

    @classmethod
    def from_list(cls, data: list[dict[str, Any]] | None) -> "MemoryStore":
        return cls(MemoryItem.from_dict(d) for d in (data or []))
