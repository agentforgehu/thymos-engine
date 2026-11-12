"""Optional LLM-backed fact extraction with schema validation.

The rule extractor remains the default so the package stays dependency-free.
Hosts that already have a chat model can pass a ``chat(messages) -> str``
callable and use this adapter for paraphrases, negation and event time.  The
model output is treated as untrusted: it is parsed, allow-listed and validated
before it reaches the fact store.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .facts import ExtractedFact, extract_candidates

ALLOWED_PREDICATES = {
    "name",
    "city",
    "address",
    "pet_name",
    "pet_breed",
    "allergy",
    "project",
    "deadline",
    "job",
    "friend",
    "sleep",
    "wake",
    "coffee_sugar",
    "coffee_type",
}

SYSTEM_PROMPT = """你是一个保守的事实抽取器。只抽取用户明确陈述、未来仍然有用的个人信息。
输出 JSON 数组，不要输出解释。每项字段：
- predicate: 必须是这些之一：{predicates}
- value: 简短字符串，不要包含整句
- event_at: Unix 时间戳；如果事实从说话时立即生效，用 null；如果是“下周搬”这类未来事件，填生效时间
- confidence: 0 到 1
规则：问句、假设、否定、第三人称、引语不要抽取；不确定就不要抽。
""".format(predicates=", ".join(sorted(ALLOWED_PREDICATES)))

USER_PROMPT = """当前时间：{at}
用户文本：
{text}
"""


def _parse_json_array(raw: str) -> list[dict[str, Any]]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("no JSON array in model output")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("model output is not a JSON array")
    return [item for item in data if isinstance(item, dict)]


class LLMFactExtractor:
    def __init__(self, chat: Callable[[list[dict[str, str]]], str], *, fallback: bool = True) -> None:
        self.chat = chat
        self.fallback = fallback

    def extract(self, text: str, at: float) -> list[ExtractedFact]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(at=at, text=text)},
        ]
        try:
            raw = self.chat(messages)
            parsed = _parse_json_array(raw)
        except Exception:
            return extract_candidates(text, at) if self.fallback else []

        out: list[ExtractedFact] = []
        seen: set[tuple[str, str]] = set()
        for item in parsed:
            predicate = str(item.get("predicate", "")).strip()
            value = str(item.get("value", "")).strip()
            if predicate not in ALLOWED_PREDICATES or not value or len(value) > 80:
                continue
            if any(word in value for word in ("什么", "哪里", "吗", "？", "?")):
                continue
            event_at = item.get("event_at")
            if event_at is not None:
                try:
                    event_at = float(event_at)
                except (TypeError, ValueError):
                    event_at = None
                if event_at is not None and event_at <= 0:
                    event_at = None
            try:
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.8))))
            except (TypeError, ValueError):
                confidence = 0.8
            key = (predicate, value)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                ExtractedFact(
                    predicate=predicate,
                    value=value,
                    event_at=event_at,
                    confidence=confidence,
                    source="llm",
                )
            )
        if not out and self.fallback:
            return extract_candidates(text, at)
        return out
