"""Query intent classification and affect gating.

This module is the first piece of the control plane: it decides *whether*
affective state should influence retrieval for a given query.  The dynamical
model still lives in :mod:`thymos.affect`, :mod:`thymos.endocrine` and
:mod:`thymos.retrieval`; this layer only chooses the policy.

The rules are deliberately transparent.  They are a product-safe default, not
a claim about human intent classification.  A host can replace the policy with
a model-based classifier by implementing the same small interface.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .types import AffectState

# Order matters: history questions often contain ordinary fact words
# ("我以前住在哪里"), so they must be recognised before the factual bucket.
_HISTORY = re.compile(r"之前|以前|过去|原来|曾经|上次|上一次|搬家前|历史|旧地址|原来住|以前住")
_TECHNICAL = re.compile(
    r"代码|报错|异常|bug|Bug|BUG|API|接口|部署|配置|函数|类|SQL|数据库|"
    r"Python|JavaScript|TypeScript|模型|参数|prompt|token|日志|堆栈|依赖|"
    r"安装|编译|测试|重构|性能|延迟"
)
_FACTUAL = re.compile(
    r"我叫什么|我叫|叫什么|名字|姓名|是谁|住在|住哪|住哪儿|哪座城市|"
    r"地址|电话|生日|年龄|多大|过敏|忌口|不吃|爱吃|"
    r"喜欢什么|不喜欢什么|养了|宠物|工作|做什么工作|干什么|公司|职位|"
    r"职业|项目|项目名|计划|几点|哪天|什么时候|哪里|哪个|多少|"
    r"是不是|有没有|还记得|老家|家乡|来自|妹妹|姐姐|哥哥|弟弟|结婚"
)
_AFFECTIVE = re.compile(
    r"难过|开心|高兴|焦虑|压力|孤独|害怕|生气|委屈|崩溃|失望|"
    r"兴奋|心情|感受|情绪|陪我|安慰|抱抱|累|烦|痛苦|担心|紧张"
)
_SOCIAL = re.compile(
    r"^(在吗|早安|早上好|晚安|你好|嗨|hello|hi|谢谢|多谢|哈哈|嗯|哦|好的|收到)[。！？!? ]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryIntent:
    """A small, auditable label for the query's memory needs."""

    label: str  # factual | technical | history | affective | social | mixed
    confidence: float
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class AffectDecision:
    """How much affect may influence retrieval for this query."""

    mood_weight: float
    emotion_weight: float
    include_state: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mood_weight"] = round(float(data["mood_weight"]), 3)
        data["emotion_weight"] = round(float(data["emotion_weight"]), 3)
        return data


def classify_intent(query: str) -> QueryIntent:
    """Classify a query with cheap, deterministic rules.

    The labels are intentionally coarse.  The point is to decide whether
    affect is allowed to participate, not to build a general NLU system.
    """
    q = (query or "").strip()
    if not q:
        return QueryIntent("social", 0.20, ("empty query",))

    if _HISTORY.search(q):
        return QueryIntent("history", 0.85, ("historical cue",))
    if _TECHNICAL.search(q):
        return QueryIntent("technical", 0.85, ("technical cue",))

    affective_hits = len(_AFFECTIVE.findall(q))
    factual_hits = len(_FACTUAL.findall(q))
    if affective_hits and factual_hits:
        return QueryIntent(
            "mixed",
            min(0.9, 0.55 + 0.05 * (affective_hits + factual_hits)),
            (f"affective x{affective_hits}", f"factual x{factual_hits}"),
        )
    if affective_hits:
        return QueryIntent("affective", min(0.9, 0.65 + 0.05 * affective_hits), (f"affective x{affective_hits}",))
    if factual_hits:
        return QueryIntent("factual", min(0.9, 0.65 + 0.05 * factual_hits), (f"factual x{factual_hits}",))
    if _SOCIAL.match(q):
        return QueryIntent("social", 0.75, ("social cue",))
    return QueryIntent("mixed", 0.30, ("no explicit cue",))


class AffectPolicy:
    """Map intent to retrieval weights and prompt-state visibility.

    Factual and technical queries are deliberately *affect-neutral*: a wrong
    fact is worse than a flat tone.  Affective and social queries keep the
    mood channel, because congruence is part of the product value there.
    """

    _TABLE: dict[str, tuple[float, float, bool, str]] = {
        "factual": (0.0, 0.05, False, "facts first; mood congruence disabled"),
        "technical": (0.0, 0.0, False, "technical query; affect disabled"),
        "history": (0.0, 0.05, False, "historical facts; affect disabled"),
        "affective": (0.35, 0.22, True, "supportive query; mood congruence allowed"),
        "mixed": (0.15, 0.12, True, "mixed query; affect partially allowed"),
        "social": (0.20, 0.08, True, "social turn; light affect allowed"),
    }

    def decide(self, intent: QueryIntent, affect: AffectState | None = None) -> AffectDecision:
        mood, emotion, include_state, reason = self._TABLE.get(
            intent.label, self._TABLE["mixed"]
        )
        # A very high stress state should not be hidden from a supportive
        # query; the state block helps the model choose a gentler tone.
        if include_state and affect is not None and affect.stress >= 0.75:
            reason += "; high stress state exposed"
        return AffectDecision(
            mood_weight=mood,
            emotion_weight=emotion,
            include_state=include_state,
            reason=reason,
        )


__all__ = [
    "QueryIntent",
    "AffectDecision",
    "AffectPolicy",
    "classify_intent",
]
