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
from typing import Any

from .config import Profile, ThymosConfig
from .engine import Thymos
from .mathx import clamp
from .types import MemoryItem, RetrievalResult

# Rules are intentionally cheap and transparent.  Replace `importance` with a
# structured LLM appraisal when you need nuance, but keep this as the fallback.
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


def estimate_importance(text: str) -> float:
    """A transparent 0–1 importance prior for a chat message.

    This is deliberately rule-based: it is fast, deterministic and easy to
    audit.  It is not a claim about what a person would find important.
    """
    text = (text or "").strip()
    if not text:
        return 0.0
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
    if _SMALLTALK.match(text):
        return 0.20
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
    """

    def __init__(
        self,
        state_path: str = ".thymos/agent.json",
        *,
        profile: Profile | None = None,
        config: ThymosConfig | None = None,
        offline: bool = True,
        now: float | None = None,
    ) -> None:
        cfg = config or ThymosConfig(offline=offline)
        self.mind = Thymos(
            profile or Profile(),
            cfg,
            state_path=state_path,
            now=now,
        )

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
    ) -> str:
        """Return a compact prompt block for the current user turn.

        ``min_score`` filters weak hits.  A value around 0.35 is a reasonable
        starting point; set it to 0 to let the engine return its top-k.
        """
        hits = self.recall(user_text, top_k=top_k, review=review)
        if min_score > 0:
            hits = [hit for hit in hits if hit.score >= min_score]
        return self.mind.render_context(hits)

    def snapshot(self) -> dict[str, Any]:
        """Return the full inspectable state (affect, endocrine, memory stats)."""
        return self.mind.snapshot()

    # ----------------------------------------------------------------- write
    def observe(
        self,
        user_text: str,
        assistant_text: str | None = None,
        *,
        importance: float | None = None,
        tags: list[str] | None = None,
        min_importance: float = 0.35,
    ) -> MemoryItem | None:
        """Encode one turn if it clears the importance threshold.

        Returns the stored :class:`MemoryItem`, or ``None`` when the turn was
        treated as small talk.  Pass ``min_importance=0`` to store everything.

        ``assistant_text`` is optional and off by default.  Only include it
        when the reply contains a commitment or self-disclosure worth keeping;
        storing every model reply makes hallucinations permanent.
        """
        text = (user_text or "").strip()
        if not text:
            return None
        score = estimate_importance(text) if importance is None else clamp(importance)
        if score < min_importance:
            return None

        content = text
        if assistant_text:
            reply = (assistant_text or "").strip()
            if reply:
                content = f"用户说：{text}\n助手回应：{reply[:500]}"

        return self.mind.perceive(
            content,
            importance=score,
            tags=tags or ["对话"],
            source="agent",
        )

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
