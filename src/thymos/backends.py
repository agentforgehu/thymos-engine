"""Pluggable memory backend contract.

The built-in :class:`~thymos.memory.MemoryStore` and
:class:`~thymos.sqlite_store.SQLiteMemoryStore` are the reference
implementations.  External systems (Mem0, Graphiti, Letta, a vector store)
can be adapted by implementing this protocol; the control plane then treats
them as data planes instead of competitors.

The protocol is structural: no inheritance is required.  It is intentionally
small so that an adapter can be written in a few dozen lines.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

from .types import MemoryItem


@runtime_checkable
class MemoryBackend(Protocol):
    """The minimum surface the control plane needs from a memory store."""

    def add(self, item: MemoryItem) -> MemoryItem:
        ...

    def get(self, memory_id: str) -> MemoryItem | None:
        ...

    def remove(self, memory_id: str) -> bool:
        ...

    def all(self) -> list[MemoryItem]:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterable[MemoryItem]:
        ...

    def link(self, a_id: str, b_id: str, weight: float) -> None:
        ...

    def neighbors(self, memory_id: str) -> list[tuple[MemoryItem, float]]:
        ...

    def most_similar(
        self,
        embedding: list[float],
        *,
        exclude: str | None = None,
        limit: int = 3,
        minimum: float = 0.0,
    ) -> list[tuple[MemoryItem, float]]:
        ...

    def to_list(self) -> list[dict[str, Any]]:
        ...


__all__ = ["MemoryBackend"]
