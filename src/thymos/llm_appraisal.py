"""Optional LLM-backed appraisal with schema validation.

The rule appraiser remains the default so Thymos stays deterministic and
dependency-free.  Hosts that already have a chat model can pass a
``chat(messages) -> str`` callable and let the model fill the same
:class:`~thymos.types.Appraisal` fields that the rules fill.  The model is
asked for *appraisal dimensions*, not just "positive/negative": anger,
sadness and anxiety share a negative valence but differ in agency, control and
certainty.

Model output is untrusted.  It is parsed as JSON, clamped to the documented
ranges, and falls back to the lexical appraiser on any parse/runtime failure.
This is an adapter, not a claim that an LLM has a stable theory of mind.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .appraisal import estimate_appraisal
from .mathx import clamp
from .types import Appraisal

SYSTEM_PROMPT = """你是一个保守的事件评价器，只评价“用户这句话描述的事件”，不要评价用户本人。
输出一个 JSON 对象，不要输出 Markdown、解释或思考过程。字段：
- goal_relevance: 0 到 1，这件事对用户当前目标有多相关
- desirability: -1 到 1，事件对用户是好是坏；中性为 0
- novelty: 0 到 1，意外/新鲜程度
- certainty: 0 到 1，事件是否已经确定发生
- control: 0 到 1，用户对事件有多少控制力
- agency: "self" | "other" | "circumstance"，谁导致/主导了事件
- norm_compliance: -1 到 1，行为是否符合社会规范；普通事件为 0
- stakes: 0 到 1，后果大小
- note: 不超过 80 字的证据短语，只写“为什么”，不要写建议
规则：
1. 只根据这句话的显式内容判断；不要因为用户可能心情不好就推断事件很糟。
2. 讽刺、引用、转述、玩笑要降低 certainty，并在 note 中标明“可能讽刺/引用”。
3. 不要把脏话本身当成强负面；要看它是否指向用户或用户在意的人/事。
4. 不确定时把 desirability 靠近 0、certainty 降低；不要编造事实。
5. 不诊断心理疾病，不做危机干预判断。
"""

USER_PROMPT = """用户文本：
{text}
"""

_ALLOWED_AGENCY = {"self", "other", "circumstance"}


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    return data


def _number(data: dict[str, Any], key: str, default: float, low: float, high: float) -> float:
    try:
        return clamp(float(data.get(key, default)), low, high)
    except (TypeError, ValueError):
        return default


class LLMAppraiser:
    """Turn a chat model into an :class:`Appraisal` callable.

    Parameters
    ----------
    chat:
        A callable accepting OpenAI-style ``messages`` and returning text.
    fallback:
        On any parse or chat error, use :func:`estimate_appraisal` instead of
        returning a neutral appraisal.  Defaults to ``True`` so a broken model
        cannot silently zero out the affect system.
    max_note_chars:
        Hard cap for the model-provided evidence string.
    """

    def __init__(
        self,
        chat: Callable[[list[dict[str, str]]], str],
        *,
        fallback: bool = True,
        max_note_chars: int = 120,
    ) -> None:
        self.chat = chat
        self.fallback = fallback
        self.max_note_chars = max(1, int(max_note_chars))

    def __call__(self, text: str) -> Appraisal:
        return self.appraise(text)

    def appraise(self, text: str) -> Appraisal:
        text = (text or "").strip()
        if not text:
            return Appraisal(note="empty")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(text=text)},
        ]
        try:
            raw = self.chat(messages)
            data = _parse_json_object(raw)
        except Exception as exc:  # noqa: BLE001 - fallback is the contract
            if self.fallback:
                return estimate_appraisal(text)
            return Appraisal(note=f"llm-error:{type(exc).__name__}")

        agency = str(data.get("agency", "circumstance")).strip().lower()
        if agency not in _ALLOWED_AGENCY:
            agency = "circumstance"
        note = str(data.get("note", "")).strip().replace("\n", " ")
        note = note[: self.max_note_chars]
        return Appraisal(
            goal_relevance=_number(data, "goal_relevance", 0.5, 0.0, 1.0),
            desirability=_number(data, "desirability", 0.0, -1.0, 1.0),
            novelty=_number(data, "novelty", 0.2, 0.0, 1.0),
            certainty=_number(data, "certainty", 0.6, 0.0, 1.0),
            control=_number(data, "control", 0.5, 0.0, 1.0),
            agency=agency,
            norm_compliance=_number(data, "norm_compliance", 0.0, -1.0, 1.0),
            stakes=_number(data, "stakes", 0.5, 0.0, 1.0),
            note=("llm:" + note) if note else "llm",
        )
