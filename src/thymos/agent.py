"""A small, opinionated facade for chat agents and bots.

The core :class:`~thymos.engine.Thymos` API is deliberately explicit: recall,
perceive, tick and consolidate are separate operations.  A chat bot usually
wants three calls instead:

    memory = AgentMemory(".thymos/bot.json")
    context = memory.context(user_text)       # before the LLM
    reply = llm(context, user_text)           # your LLM
    memory.observe(user_text)                 # after the LLM; user facts only
    memory.end_session()                      # nightly / on shutdown

This module is a convenience layer, not a second engine.  It keeps the same
deterministic state and exposes ``memory.mind`` when you need full control.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Mapping

from .appraisal import appraise, estimate_appraisal
from .behavior import BehaviorDirective, BehaviorMapper, DefaultBehaviorMapper
from .config import Profile, ThymosConfig
from .embedding import Embedder
from .engine import Thymos
from .facts import Fact
from .mathx import clamp
from .memory import MemoryStore
from .policy import classify_intent
from .stream import Event, WindowSummary, build_window_summary, normalize_events
from .types import Appraisal, MemoryItem, RetrievalResult
from .write_path import (
    find_merge_candidate,
    merge_decision,
    merge_memory,
    normalize_key,
    skip_decision,
    store_decision,
)

# Rules are intentionally cheap and transparent.  Pass a structured
# `appraisal` (from a host or `LLMAppraiser`) when you need nuance; this
# remains the fallback.
_REMINDER = re.compile(r"提醒|答应|约定|承诺|截止|deadline")
_PROMISE = re.compile(r"生日|面试|考试")
_PREFERENCE = re.compile(r"我不吃|我喜欢|我讨厌|过敏|忌口|最爱|不喜欢|我习惯")
_EMOTION = re.compile(r"难过|开心|崩溃|委屈|压力|害怕|焦虑|生气|孤独|失望|兴奋")
_FACT = re.compile(r"我叫|我是|我在|我住|我的|我养|我有个|我在做|我今年")
_SMALLTALK = re.compile(r"^(嗯|哦|好|哈哈|在吗|早|晚安|谢谢|没事|收到)[。！？!? ]*$")
_QUESTION = re.compile(
    r"^(什么|哪|谁|怎么|为什么|为啥|多少|几|是否|是不是|能不能|可不可以|要不要|有没有|还记得)"
    r"|什么|哪[天个]|几点|什么时候|[?？]\s*$"
)


def estimate_importance(
    text: str, *, appraisal: Appraisal | None = None
) -> float:
    """A transparent 0–1 importance prior for a chat message.

    This is deliberately rule-based: it is fast, deterministic and easy to
    audit.  It is not a claim about what a person would find important.
    When ``appraisal`` is supplied, its desirability is reused instead of
    running the lexical appraiser again.
    """
    text = (text or "").strip()
    if not text:
        return 0.0
    # A short but strongly valenced message ("闭嘴", "你真好", "项目黄了")
    # must not fall through the length heuristic.  Otherwise the most
    # affectively intense turns are exactly the ones that never reach
    # `perceive`, and mood appears not to react at all.
    if _SMALLTALK.match(text):
        return 0.20
    appraisal = appraisal or estimate_appraisal(text)
    if abs(appraisal.desirability) >= 0.30:
        return 0.85
    # A reminder/commitment is worth keeping even when phrased as a question.
    if _REMINDER.search(text):
        return 0.90
    # Questions are requests for information, not new facts about the user.
    # Keep them out of long-term memory unless the host overrides importance.
    if _QUESTION.search(text):
        return 0.25
    if _PROMISE.search(text):
        return 0.90
    if _PREFERENCE.search(text):
        return 0.85
    if _EMOTION.search(text):
        return 0.80
    if _FACT.search(text):
        return 0.70
    if len(text) < 12:
        return 0.30
    return 0.45


class AgentMemory:
    """A three-call memory facade for a chat agent.

    Parameters
    ----------
    state_path:
        JSON file used by the underlying :class:`Thymos` engine.
    profile:
        Who the bot is calibrated for.  Defaults to a generic profile.
    config:
        Model parameters.  Defaults to ``ThymosConfig(offline=True)``.
    offline:
        Ignored when ``config`` is supplied.  Defaults to True so a bot does
        not call a weather API unless the developer opts in.
    now:
        Fixed timestamp for reproducible tests; defaults to wall clock.
    embedder:
        Optional ``str -> list[float]`` function.  Defaults to the deterministic
        hash n-gram embedder; pass ``OllamaEmbedder`` for Chinese semantics.
    appraiser:
        Optional ``str -> Appraisal`` callable.  When supplied (for example
        :class:`thymos.llm_appraisal.LLMAppraiser`), it replaces the lexical
        appraisal; the lexical rules remain the fallback and the default.
    """

    def __init__(
        self,
        state_path: str = ".thymos/agent.json",
        *,
        profile: Profile | None = None,
        config: ThymosConfig | None = None,
        offline: bool = True,
        now: float | None = None,
        embedder: Embedder | None = None,
        store: MemoryStore | None = None,
        appraiser: Callable[[str], Appraisal] | None = None,
        behavior_mapper: BehaviorMapper | None = None,
    ) -> None:
        cfg = config or ThymosConfig(offline=offline)
        self.mind = Thymos(
            profile or Profile(),
            cfg,
            state_path=state_path,
            now=now,
            embedder=embedder,
            store=store,
        )
        self.appraiser = appraiser
        self.behavior_mapper = behavior_mapper or DefaultBehaviorMapper()
        self.last_window: WindowSummary | None = None
        self.last_behavior: dict[str, Any] | None = None

    # ------------------------------------------------------------------ read
    def recall(
        self,
        user_text: str,
        *,
        top_k: int = 5,
        review: bool = True,
    ) -> list[RetrievalResult]:
        """Retrieve relevant memories; by default treat them as reviewed."""
        return self.mind.recall(user_text, top_k=top_k, update=review)

    def context(
        self,
        user_text: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        review: bool = True,
        max_chars: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return a compact prompt block for the current user turn.

        ``min_score`` filters weak hits.  A value around 0.35 is a reasonable
        starting point; set it to 0 to let the engine return its top-k.

        ``max_chars`` is a hard prompt budget.  Structured facts are rendered
        first; only the remaining characters are spent on episodic memories.
        This is the recommended mode for a local model with a small context.
        ``max_tokens`` is a conservative tokenizer-independent upper bound; if
        both are supplied, the rendered block must satisfy both.
        """
        plan = self.mind.plan_context(
            user_text,
            top_k=top_k,
            min_score=min_score,
            update=review,
            max_chars=max_chars,
            max_tokens=max_tokens,
        )
        return plan.text

    def plan(
        self,
        user_text: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        review: bool = True,
        max_chars: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return the context plus the control-plane audit trace.

        The returned dict is JSON-serialisable: ``context`` is the prompt
        block, ``intent`` and ``affect`` explain the routing decision, and
        ``operators`` explains which memory representations were consulted.
        """
        plan = self.mind.plan_context(
            user_text,
            top_k=top_k,
            min_score=min_score,
            update=review,
            max_chars=max_chars,
            max_tokens=max_tokens,
        ).to_dict()
        directive = self.behavior(
            user_text,
            intent=plan["intent"]["label"],
            context={"query": user_text},
        )
        plan["behavior"] = directive.to_dict()
        self.last_behavior = plan["behavior"]
        return plan

    def behavior(
        self,
        user_text: str = "",
        *,
        intent: str | None = None,
        emotion: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> BehaviorDirective:
        """Map current affect to a host-facing behavioral suggestion.

        This is the extension point for a product's persona.  Pass a custom
        ``behavior_mapper`` to :class:`AgentMemory` to replace the default
        policy without touching the dynamical core.
        """
        if intent is None:
            intent = classify_intent(user_text).label
        if emotion is None:
            emotion = self.mind.affect.state.dominant_emotion
        return self.behavior_mapper.map(
            self.mind.affect.state,
            intent=intent,
            emotion=emotion,
            context=context,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return the full inspectable state (affect, endocrine, memory stats)."""
        data = self.mind.snapshot()
        data["last_window"] = (
            self.last_window.to_dict() if self.last_window is not None else None
        )
        data["last_behavior"] = self.last_behavior
        return data

    def facts(self, *, active_only: bool = True) -> list[Fact]:
        """List facts for a host-side correction/management UI."""
        if active_only:
            return sorted(self.mind.facts.active(self.mind.now), key=lambda f: (f.predicate, f.valid_from))
        return sorted(self.mind.facts.all(), key=lambda f: (f.predicate, f.valid_from))

    def correct_fact(
        self,
        predicate: str,
        value: str,
        *,
        old_value: str | None = None,
        subject: str = "user",
        confidence: float = 0.95,
    ) -> Fact:
        """Correct a wrong/old fact and persist it."""
        fact = self.mind.facts.correct(
            subject,
            predicate,
            value,
            self.mind.now,
            old_value=old_value,
            confidence=confidence,
        )
        self.save()
        return fact

    def delete_fact(
        self,
        *,
        predicate: str | None = None,
        value: str | None = None,
        fact_id: str | None = None,
        subject: str = "user",
    ) -> int:
        """Retire a fact (keeps history) and persist it."""
        count = self.mind.facts.delete(
            self.mind.now,
            subject=subject,
            predicate=predicate,
            value=value,
            fact_id=fact_id,
        )
        if count:
            self.save()
        return count

    # ----------------------------------------------------------------- write
    def observe(
        self,
        user_text: str,
        assistant_text: str | None = None,
        *,
        importance: float | None = None,
        tags: list[str] | None = None,
        min_importance: float = 0.35,
        learn_facts: bool = True,
        appraisal: Appraisal | None = None,
    ) -> MemoryItem | None:
        """Encode one turn if it clears the importance threshold.

        Returns the stored :class:`MemoryItem`, or ``None`` when the turn was
        treated as small talk.  Pass ``min_importance=0`` to store everything.

        ``assistant_text`` is optional and off by default.  Only include it
        when the reply contains a commitment or self-disclosure worth keeping;
        storing every model reply makes hallucinations permanent.

        ``appraisal`` can be supplied by a host that already has structured
        output; otherwise ``appraiser`` (for example ``LLMAppraiser``) is
        used, and the lexical rules are the final fallback.
        """
        text = (user_text or "").strip()
        if not text:
            return None
        # Judge once, use twice: the same appraisal decides whether the turn
        # is important and how it moves affect.  The old path ran the lexical
        # rules in `estimate_importance` and again in `perceive`.
        if (
            appraisal is None
            and importance is None
            and _SMALLTALK.match(text)
            and min_importance > 0.20
        ):
            self.mind.last_write = skip_decision("smalltalk", 0.20).to_dict()
            return None
        if appraisal is None:
            appraisal = (
                self.appraiser(text)
                if self.appraiser is not None
                else estimate_appraisal(text)
            )
        score = (
            clamp(importance)
            if importance is not None
            else estimate_importance(text, appraisal=appraisal)
        )
        if score < min_importance:
            self.mind.last_write = skip_decision("below_min_importance", score).to_dict()
            return None
        # A question is a request for information, not a self-disclosure.
        # This guard matters when a host passes min_importance=0 to keep the
        # episode: without it, "我喝咖啡加糖吗？" can be parsed as "加糖".
        if _QUESTION.search(text):
            learn_facts = False

        content = text
        if assistant_text:
            reply = (assistant_text or "").strip()
            if reply:
                content = f"用户说：{text}\n助手回应：{reply[:500]}"

        cfg = self.mind.config
        if getattr(cfg, "write_admission_enabled", True):
            candidate_embedding = self.mind.embedder(text)
            candidate, similarity, reason = find_merge_candidate(
                self.mind.store,
                text,
                candidate_embedding,
                self.mind.now,
                cfg,
                tags=tags,
            )
            if candidate is not None and similarity is not None:
                # The turn still moves affect and hormones; it just does not
                # create a second trace.  Facts are attached to the surviving
                # memory id so provenance is not lost.
                self.mind.perceive(
                    content,
                    importance=score,
                    tags=tags or ["对话"],
                    source="agent",
                    learn_facts=False,
                    appraisal=appraisal,
                    store_memory=False,
                )
                if learn_facts:
                    self.mind.facts.observe(
                        text,
                        at=self.mind.now,
                        source="agent",
                        memory_id=candidate.id,
                    )
                merge_memory(
                    candidate,
                    content,
                    candidate_embedding,
                    now=self.mind.now,
                    emotion=appraise(appraisal),
                    importance=score,
                    tags=tags,
                    config=cfg,
                )
                self.mind.last_write = merge_decision(
                    score, similarity, candidate.id, reason
                ).to_dict()
                self.mind._log("write_merge", {**self.mind.last_write, "content": text[:120]})
                return candidate

        item = self.mind.perceive(
            content,
            importance=score,
            tags=tags or ["对话"],
            source="agent",
            learn_facts=learn_facts,
            appraisal=appraisal,
        )
        if item is not None:
            item.canonical_key = normalize_key(text)
        self.mind.last_write = store_decision(score, item.id if item else None).to_dict()
        return item

    def observe_llm(
        self,
        user_text: str,
        extractor,
        *,
        importance: float | None = None,
        tags: list[str] | None = None,
        min_importance: float = 0.35,
    ) -> MemoryItem | None:
        """Store a turn and use a host-supplied LLM extractor for facts.

        The episodic memory is still written by the normal path; only the
        structured fact layer is delegated.  ``extractor.extract(text, at)``
        must return objects with ``predicate``, ``value``, ``event_at``,
        ``confidence`` and ``source`` (see :class:`thymos.llm_extract.LLMFactExtractor`).
        """
        item = self.observe(
            user_text,
            importance=importance,
            tags=tags,
            min_importance=min_importance,
            learn_facts=False,
        )
        if item is None:
            return None
        for candidate in extractor.extract(user_text, self.mind.now):
            self.mind.facts.add(
                "user",
                candidate.predicate,
                candidate.value,
                self.mind.now,
                event_at=candidate.event_at,
                source=getattr(candidate, "source", "llm"),
                memory_id=item.id,
                confidence=float(getattr(candidate, "confidence", 0.8)),
            )
        return item

    def observe_stream(
        self,
        events: Iterable[Event | Mapping[str, Any] | str],
        *,
        window: str = "session",
        summary: bool = True,
        min_importance: float = 0.35,
        tags: list[str] | None = None,
        step_seconds: float = 60.0,
    ) -> WindowSummary:
        """Ingest a continuous event stream and build one window summary.

        Each ``user`` event goes through the normal :meth:`observe` write path,
        so facts, deduplication and affect still work exactly as they do for a
        single turn.  ``assistant`` and ``system`` events are recorded in the
        window but are not stored as user memories.

        Timestamps are absolute Unix seconds.  Missing timestamps are assigned
        monotonically with ``step_seconds``.  A timestamp earlier than the
        current engine clock is clamped forward rather than rewinding affect.
        The summary is stored as a semantic memory tagged ``window`` so a later
        "今晚怎么样" query can retrieve it without replaying every turn.
        """
        normalized = normalize_events(
            events, now=self.mind.now, step_seconds=step_seconds
        )
        if not normalized:
            raise ValueError("events must not be empty")
        base_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        records: list[dict[str, Any]] = []
        event_memory_ids: list[str] = []

        for event in normalized:
            if event.at is not None and event.at > self.mind.now:
                self.mind.tick(event.at - self.mind.now)
            event_tags = list(
                dict.fromkeys(base_tags + list(event.tags) + [window])
            )
            item: MemoryItem | None = None
            if event.role == "user":
                item = self.observe(
                    event.text,
                    importance=event.importance,
                    tags=event_tags,
                    min_importance=min_importance,
                    appraisal=event.appraisal,
                )
                if item is not None:
                    event_memory_ids.append(item.id)
            state = self.mind.affect.state
            records.append(
                {
                    "at": float(event.at or self.mind.now),
                    "role": event.role,
                    "text": event.text,
                    "label": state.dominant_emotion,
                    "valence": state.valence,
                    "mood": state.mood_valence,
                    "stress": state.stress,
                    "tags": event_tags,
                    "memory_id": item.id if item is not None else None,
                }
            )

        start_at = float(normalized[0].at or self.mind.now)
        end_at = float(normalized[-1].at or self.mind.now)
        window_summary = build_window_summary(
            window,
            records,
            start_at=start_at,
            end_at=end_at,
            timezone_name=self.mind.profile.timezone,
        )

        if summary and window_summary.user_turns >= 2:
            content = window_summary.summary
            embedding = self.mind.embedder(content)
            self.mind._ensure_embedding_dim(len(embedding))
            summary_item = MemoryItem(
                content=content,
                kind="semantic",
                tags=list(
                    dict.fromkeys(
                        ["window", window] + list(window_summary.topics)
                    )
                ),
                embedding=embedding,
                context={
                    "window": window,
                    "start_at": start_at,
                    "end_at": end_at,
                    "event_count": window_summary.event_count,
                    "memory_ids": list(event_memory_ids),
                },
                created_at=end_at,
                last_access=end_at,
                access_times=[end_at],
                stability=1.2,
                difficulty=0.4,
                valence=window_summary.valence_mean,
                arousal=0.25,
                salience=0.75,
                source="stream",
            )
            self.mind.store.add(summary_item)
            for memory_id in event_memory_ids:
                self.mind.store.link(summary_item.id, memory_id, 0.80)
            window_summary.memory_id = summary_item.id
            self.mind._log(
                "window_summary",
                {
                    "window": window,
                    "memory_id": summary_item.id,
                    "event_count": window_summary.event_count,
                    "user_turns": window_summary.user_turns,
                },
            )

        self.last_window = window_summary
        return window_summary

    def end_session(self, hours: float = 7.5) -> dict[str, Any]:
        """Simulate sleep consolidation and persist the state."""
        result = self.mind.consolidate(hours)
        self.mind.save()
        return result

    def save(self) -> str:
        """Persist without consolidation."""
        return self.mind.save()

    def __repr__(self) -> str:
        return f"AgentMemory(state_path={self.mind.state_path!r}, memories={len(self.mind.store)})"
