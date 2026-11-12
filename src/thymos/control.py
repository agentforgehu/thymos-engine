"""Memory control plane: intent -> policy -> operators -> plan -> audit.

The engine remains the dynamical core.  This module is the outer contract:

* classify what kind of memory the query needs;
* decide whether affect may influence retrieval;
* run a small set of memory operators (facts, episodes, graph, summary);
* render one context block under a hard budget;
* return an audit trace explaining what was selected and what was not.

It is deliberately dependency-free and deterministic.  A host can replace the
policy or add operators without changing the engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .forgetting import retrievability
from .policy import AffectDecision, AffectPolicy, QueryIntent, classify_intent
from .tokens import estimate_tokens, trim_to_tokens
from .types import RetrievalResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .engine import Thymos


@dataclass
class OperatorReport:
    """One operator's contribution to the plan."""

    name: str
    status: str = "used"  # used | empty | skipped
    candidates: int = 0
    selected: int = 0
    chars: int = 0
    reason: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "candidates": self.candidates,
            "selected": self.selected,
            "chars": self.chars,
            "reason": self.reason,
            "duration_ms": round(self.duration_ms, 3),
        }


@dataclass
class AuditTrace:
    """A machine-readable explanation of one context plan."""

    query: str
    intent: QueryIntent
    affect: AffectDecision
    operators: list[OperatorReport] = field(default_factory=list)
    candidates: int = 0
    selected_ids: list[str] = field(default_factory=list)
    fact_count: int = 0
    memory_count: int = 0
    chars: int = 0
    tokens: int = 0
    budget_chars: int | None = None
    budget_tokens: int | None = None
    duration_ms: float = 0.0
    notes: list[str] = field(default_factory=list)
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent.to_dict(),
            "affect": self.affect.to_dict(),
            "operators": [op.to_dict() for op in self.operators],
            "candidates": self.candidates,
            "selected_ids": list(self.selected_ids),
            "fact_count": self.fact_count,
            "memory_count": self.memory_count,
            "chars": self.chars,
            "tokens": self.tokens,
            "budget": {"chars": self.budget_chars, "tokens": self.budget_tokens},
            "duration_ms": round(self.duration_ms, 3),
            "notes": list(self.notes),
            "created_at": self.created_at,
        }


@dataclass
class ContextPlan:
    """The result of one planning pass."""

    query: str
    intent: QueryIntent
    affect: AffectDecision
    hits: list[RetrievalResult]
    text: str
    audit: AuditTrace

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent.to_dict(),
            "affect": self.affect.to_dict(),
            "operators": [op.to_dict() for op in self.audit.operators],
            "selected_ids": list(self.audit.selected_ids),
            "budget": {"chars": self.audit.budget_chars, "tokens": self.audit.budget_tokens},
            "context": self.text,
            "audit": self.audit.to_dict(),
        }


class MemoryController:
    """Plan context from the engine without owning the engine's state."""

    def __init__(self, engine: "Thymos", policy: AffectPolicy | None = None) -> None:
        self.engine = engine
        self.policy = policy or AffectPolicy()
        self.last_plan: ContextPlan | None = None

    # ------------------------------------------------------------------ plan
    def plan(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        update: bool = True,
        max_chars: int | None = None,
        max_tokens: int | None = None,
        include_graph: bool = True,
        include_summary: bool = True,
    ) -> ContextPlan:
        started = time.perf_counter()
        q = (query or "").strip()
        intent = classify_intent(q)
        affect = self.policy.decide(intent, self.engine.affect.state)
        notes: list[str] = []
        reports: list[OperatorReport] = []

        budget_chars = max_chars
        if max_tokens is not None:
            token_cap = max(64, int(max_tokens * 3))
            budget_chars = token_cap if budget_chars is None else min(budget_chars, token_cap)

        # 1. Facts are the smallest, highest-precision representation.  They
        #    are rendered by the existing engine renderer, but measured here so
        #    the audit can say whether they were present.
        fact_started = time.perf_counter()
        fact_ratio = getattr(self.engine.config, "fact_budget_ratio", 0.35)
        fact_block = self.engine.facts.render(
            q,
            at=self.engine.now,
            max_chars=int(budget_chars * fact_ratio) if budget_chars else None,
        )
        fact_count = sum(1 for line in fact_block.splitlines() if line.startswith("- "))
        reports.append(
            OperatorReport(
                name="fact_lookup",
                status="used" if fact_block else "empty",
                candidates=len(self.engine.facts.active(self.engine.now)),
                selected=fact_count,
                chars=len(fact_block),
                reason="current facts first" if fact_block else "no active facts",
                duration_ms=(time.perf_counter() - fact_started) * 1000.0,
            )
        )

        # 2. Episode recall uses the query-aware affect weights, not the
        #    global config default.  This is the core of the C2 gate.
        episode_started = time.perf_counter()
        base_hits: list[RetrievalResult] = []
        if q:
            base_hits = self.engine.recall(
                q,
                top_k=top_k,
                update=update,
                mood_weight=affect.mood_weight,
                emotion_weight=affect.emotion_weight,
            )
        if min_score > 0:
            base_hits = [hit for hit in base_hits if hit.score >= min_score]
        reports.append(
            OperatorReport(
                name="episode_recall",
                status="used" if base_hits else "empty",
                candidates=len(self.engine.store),
                selected=len(base_hits),
                chars=sum(len(hit.memory.content) for hit in base_hits),
                reason=f"mood_weight={affect.mood_weight:.2f}",
                duration_ms=(time.perf_counter() - episode_started) * 1000.0,
            )
        )

        # 3. Graph traversal follows association edges from the best hits.
        #    It is a cheap way to recover a related event that the query did
        #    not name directly.
        graph_hits: list[RetrievalResult] = []
        graph_started = time.perf_counter()
        if include_graph and base_hits:
            graph_hits = self._expand_graph(base_hits, limit=3)
        reports.append(
            OperatorReport(
                name="graph_traversal",
                status="used" if graph_hits else "skipped" if not include_graph else "empty",
                candidates=sum(len(self.engine.store.neighbors(hit.memory.id)) for hit in base_hits[:4]),
                selected=len(graph_hits),
                chars=sum(len(hit.memory.content) for hit in graph_hits),
                reason="association expansion" if graph_hits else "no new neighbors",
                duration_ms=(time.perf_counter() - graph_started) * 1000.0,
            )
        )

        # 4. Summary/schema memories compress repeated experience.  They are
        #    most useful for history/affective questions, not for exact facts.
        summary_hits: list[RetrievalResult] = []
        summary_started = time.perf_counter()
        if include_summary and (intent.label in {"history", "affective", "mixed"} or not base_hits):
            summary_hits = self._summary_hits(limit=2)
        reports.append(
            OperatorReport(
                name="summary",
                status="used" if summary_hits else "skipped" if not include_summary else "empty",
                candidates=sum(1 for item in self.engine.store if item.kind == "semantic"),
                selected=len(summary_hits),
                chars=sum(len(hit.memory.content) for hit in summary_hits),
                reason="schema memories for history/affect" if summary_hits else "not needed",
                duration_ms=(time.perf_counter() - summary_started) * 1000.0,
            )
        )

        hits = self._merge_hits(base_hits + graph_hits + summary_hits, top_k=top_k)
        text = self.engine.render_context(
            hits,
            query=q,
            max_chars=budget_chars,
            include_state=affect.include_state,
        )
        if max_tokens is not None:
            trimmed = trim_to_tokens(text, max_tokens)
            if trimmed != text:
                notes.append("trimmed_to_tokens")
            text = trimmed
            self.engine._last_context_stats.update(
                {
                    "chars": len(text),
                    "tokens": estimate_tokens(text),
                    "budget_tokens": max_tokens,
                }
            )

        stats = dict(self.engine._last_context_stats)
        audit = AuditTrace(
            query=q,
            intent=intent,
            affect=affect,
            operators=reports,
            candidates=len(self.engine.store) + len(self.engine.facts.active(self.engine.now)),
            selected_ids=[hit.memory.id for hit in hits],
            fact_count=fact_count,
            memory_count=int(stats.get("memories", len(hits))),
            chars=len(text),
            tokens=estimate_tokens(text),
            budget_chars=budget_chars,
            budget_tokens=max_tokens,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            notes=notes,
            created_at=self.engine.now,
        )
        plan = ContextPlan(query=q, intent=intent, affect=affect, hits=hits, text=text, audit=audit)
        self.last_plan = plan
        self.engine.last_audit = audit.to_dict()
        return plan

    def context(self, query: str, **kwargs: Any) -> str:
        """Return only the rendered context block."""
        return self.plan(query, **kwargs).text

    # -------------------------------------------------------------- operators
    def _expand_graph(self, base_hits: list[RetrievalResult], *, limit: int = 3) -> list[RetrievalResult]:
        seen = {hit.memory.id for hit in base_hits}
        out: list[RetrievalResult] = []
        for hit in base_hits[:4]:
            for neighbor, weight in self.engine.store.neighbors(hit.memory.id):
                if neighbor.id in seen or neighbor.state == "archived":
                    continue
                if len(out) >= limit:
                    return out
                r = retrievability(
                    neighbor.age_days(self.engine.now),
                    neighbor.stability,
                    self.engine.config.fsrs_decay,
                )
                out.append(
                    RetrievalResult(
                        memory=neighbor,
                        score=max(0.0, hit.score * 0.50 * weight),
                        retrievability=r,
                        activation=0.0,
                        probability=0.0,
                        components={"graph": float(weight), "from_hit": 1.0},
                    )
                )
                seen.add(neighbor.id)
        return out

    def _summary_hits(self, *, limit: int = 2) -> list[RetrievalResult]:
        items = [item for item in self.engine.store if item.kind == "semantic" and item.state != "archived"]
        items.sort(
            key=lambda item: item.salience
            * retrievability(item.age_days(self.engine.now), item.stability, self.engine.config.fsrs_decay),
            reverse=True,
        )
        out: list[RetrievalResult] = []
        for item in items[:limit]:
            r = retrievability(item.age_days(self.engine.now), item.stability, self.engine.config.fsrs_decay)
            out.append(
                RetrievalResult(
                    memory=item,
                    score=0.45 + 0.25 * r,
                    retrievability=r,
                    activation=0.0,
                    probability=0.0,
                    components={"summary": 1.0},
                )
            )
        return out

    @staticmethod
    def _merge_hits(hits: list[RetrievalResult], *, top_k: int) -> list[RetrievalResult]:
        """Deduplicate by id and content, preserving the operator priority."""
        if top_k <= 0:
            return []
        by_id: dict[str, RetrievalResult] = {}
        seen_content: set[str] = set()
        for hit in hits:
            key = hit.memory.content.strip()
            if hit.memory.id in by_id or key in seen_content:
                continue
            by_id[hit.memory.id] = hit
            seen_content.add(key)
            if len(by_id) >= top_k + 4:
                break
        return list(by_id.values())


__all__ = [
    "OperatorReport",
    "AuditTrace",
    "ContextPlan",
    "MemoryController",
]
