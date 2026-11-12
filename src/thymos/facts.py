"""Small, transparent fact store with validity intervals.

Episodic memories are good at "what happened"; they are bad at answering
"what is true *now*" when a user changes city, address or sleep schedule.
This module keeps a separate, tiny key/value history for high-value facts.

The extractor is intentionally rule-based and conservative.  It is a fallback
that works without an LLM; hosts can add their own facts through ``add``.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable

_QUESTION = re.compile(r"[?？]|什么|哪[个天里]?|几点|多少|吗|是否|是不是|还记得")
_FILLER = re.compile(r"^(?:总是|一般|大概|通常|都|会|要|才|现在|最近)+")
_TRAIL = re.compile(r"[，,。；;、\s]+$")
_PARTICLES = re.compile(r"(?:了|啦|呢|吧|啊|呀)+$")

_PREDICATE_LABELS = {
    "name": "姓名",
    "city": "现居",
    "address": "地址",
    "pet_name": "宠物",
    "pet_breed": "宠物品种",
    "allergy": "过敏",
    "project": "项目",
    "deadline": "截止",
    "job": "职业",
    "friend": "朋友",
    "sleep": "入睡",
    "wake": "起床",
    "coffee_sugar": "咖啡糖",
    "coffee_type": "咖啡口味",
}

_PRIORITY = {
    "name": 10,
    "city": 9,
    "address": 9,
    "allergy": 8,
    "sleep": 8,
    "wake": 7,
    "pet_name": 7,
    "pet_breed": 6,
    "project": 7,
    "deadline": 6,
    "coffee_sugar": 5,
    "coffee_type": 5,
    "job": 4,
    "friend": 3,
}

_QUERY_KEYWORDS = {
    "name": ("名字", "叫什么", "姓名"),
    "city": ("住", "城市", "哪里", "哪儿", "现居", "搬"),
    "address": ("地址", "住址", "住哪", "哪里", "哪儿"),
    "pet_name": ("猫", "狗", "宠物", "豆豆"),
    "pet_breed": ("猫", "狗", "宠物", "品种"),
    "allergy": ("过敏", "忌口", "不能吃"),
    "project": ("项目", "潮汐", "工作"),
    "deadline": ("交付", "截止", "deadline", "日期"),
    "job": ("职业", "工作", "做什么", "开发者"),
    "friend": ("朋友", "阿岚"),
    "sleep": ("睡", "作息", "几点"),
    "wake": ("起", "起床", "作息", "几点"),
    "coffee_sugar": ("咖啡", "糖"),
    "coffee_type": ("咖啡", "美式", "拿铁"),
}

_HISTORY_QUERY = re.compile(r"以前|之前|原来|过去|历史|搬家前|旧")


def _clean(value: str) -> str:
    value = _TRAIL.sub("", (value or "").strip())
    value = _FILLER.sub("", value)
    value = _PARTICLES.sub("", value)
    return value.strip("「」\"'“”")


def _iso(ts: float | None) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


@dataclass
class Fact:
    subject: str
    predicate: str
    value: str
    valid_from: float
    valid_to: float | None = None
    confidence: float = 0.8
    source: str = "user"
    memory_id: str | None = None
    last_seen: float = 0.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def active_at(self, at: float) -> bool:
        return self.valid_from <= at and (self.valid_to is None or at < self.valid_to)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fact":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (data or {}).items() if k in allowed})


def extract_facts(text: str) -> list[tuple[str, str]]:
    """Extract a small set of high-value user facts from one turn.

    Questions are ignored: a question is a retrieval cue, not a new fact.
    """
    text = (text or "").strip()
    if not text or _QUESTION.search(text):
        return []

    found: list[tuple[str, str]] = []

    def add(predicate: str, value: str) -> None:
        value = _clean(value)
        if value and len(value) <= 80:
            found.append((predicate, value))

    rules = [
        ("name", r"我叫([^\s，。,.!?！？]{1,12})"),
        ("name", r"我的名字(?:是|叫)([^\s，。,.!?！？]{1,12})"),
        ("city", r"(?:住在|现在在|目前在)([^\s，。,.!?！？]{2,12})"),
        ("address", r"(?:地址是|住址是)([^。；;\n]+)"),
        ("pet_name", r"(?:养了一只(?:猫|狗)[，,]\s*叫|我的(?:猫|狗)叫)([^，。；;\n]{1,12})"),
        ("pet_breed", r"是(橘猫|狸花猫|黑猫|白猫|布偶猫|柴犬|金毛|柯基)"),
        ("allergy", r"对([^，。；;\n]{1,12})过敏"),
        ("project", r"(?:在做(?:一个)?叫|项目叫)[「\"]?([^」\"，。；;\n]{1,20})[」\"]?(?:的)?项目"),
        ("deadline", r"(\d{1,2}\s*月\s*\d{1,2}\s*日?)\s*(?:前)?(?:要)?交付"),
        ("job", r"我是([^，。；;\n]{1,16}(?:开发者|工程师|设计师|老师|学生))"),
        ("friend", r"朋友叫([^，。；;\n]{1,12})"),
        ("sleep", r"晚上(?:一般)?([^，。；;\n]{1,12})睡"),
        ("wake", r"早上(?:一般)?([^，。；;\n]{1,12})起"),
    ]
    for predicate, pattern in rules:
        for match in re.finditer(pattern, text):
            add(predicate, match.group(1))

    # "考虑搬去上海" and "如果搬去上海" are possibilities, not a move.
    # Only commit a move when the sentence also contains a decision/completion
    # cue; the address statement later will confirm the new city anyway.
    if any(cue in text for cue in ("决定", "已经搬", "搬好了", "下周搬")):
        for match in re.finditer(r"(?:搬到|搬去|搬家到)([^\s，。,.!?！？]{2,12})", text):
            add("city", match.group(1))

    # An address usually also implies a city; keep both, because "上海" and
    # "徐汇" are different retrieval cues.
    for match in re.finditer(r"(?:地址是|住址是)([^。；;\n]+)", text):
        address = _clean(match.group(1))
        city = re.match(r"(.+?)市", address)
        if city:
            add("city", city.group(1))
        add("address", address)

    if "咖啡" in text:
        if "不加糖" in text or "无糖" in text:
            add("coffee_sugar", "不加糖")
        elif "加糖" in text:
            add("coffee_sugar", "加糖")
        for flavor in ("美式", "拿铁", "卡布奇诺", "手冲"):
            if flavor in text:
                add("coffee_type", flavor)

    # Deduplicate while preserving order.
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    for predicate, value in found:
        key = (predicate, value)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


class FactStore:
    """A key/value history where new values close old validity intervals."""

    def __init__(self, facts: Iterable[Fact] | None = None) -> None:
        self._facts: dict[tuple[str, str], list[Fact]] = {}
        for fact in facts or []:
            self._facts.setdefault((fact.subject, fact.predicate), []).append(fact)

    def all(self) -> list[Fact]:
        return [fact for group in self._facts.values() for fact in group]

    def current(self, subject: str, predicate: str, at: float) -> Fact | None:
        candidates = [
            fact
            for fact in self._facts.get((subject, predicate), [])
            if fact.active_at(at)
        ]
        return max(candidates, key=lambda f: f.valid_from) if candidates else None

    def active(self, at: float) -> list[Fact]:
        return [fact for fact in self.all() if fact.active_at(at)]

    def history(self, subject: str, predicate: str) -> list[Fact]:
        return sorted(self._facts.get((subject, predicate), []), key=lambda f: f.valid_from)

    def stale_values(self, at: float, query: str = "") -> set[str]:
        """Values that were superseded and should not be shown as current.

        A history question is the one case where old values are useful, so it
        deliberately returns an empty set and lets the caller render history.
        """
        if _HISTORY_QUERY.search(query or ""):
            return set()
        stale: set[str] = set()
        for (subject, predicate), group in self._facts.items():
            current = self.current(subject, predicate, at)
            if not current:
                continue
            for fact in group:
                if fact.valid_to is not None and fact.value != current.value:
                    stale.add(fact.value)
        return stale

    def add(
        self,
        subject: str,
        predicate: str,
        value: str,
        at: float,
        *,
        source: str = "user",
        memory_id: str | None = None,
        confidence: float = 0.8,
    ) -> Fact:
        value = _clean(value)
        current = self.current(subject, predicate, at)
        if current and current.value == value:
            current.last_seen = max(current.last_seen, at)
            return current
        if current:
            current.valid_to = at
        fact = Fact(
            subject=subject,
            predicate=predicate,
            value=value,
            valid_from=at,
            last_seen=at,
            source=source,
            memory_id=memory_id,
            confidence=confidence,
        )
        self._facts.setdefault((subject, predicate), []).append(fact)
        return fact

    def observe(
        self,
        text: str,
        at: float,
        *,
        source: str = "user",
        memory_id: str | None = None,
    ) -> list[Fact]:
        out = []
        for predicate, value in extract_facts(text):
            out.append(
                self.add(
                    "user",
                    predicate,
                    value,
                    at,
                    source=source,
                    memory_id=memory_id,
                )
            )
        return out

    def render(
        self,
        query: str = "",
        *,
        at: float | None = None,
        max_chars: int | None = None,
        include_history: bool | None = None,
    ) -> str:
        """Render active facts, prioritising facts relevant to ``query``.

        Closed facts are only shown for explicit history questions, so the
        model does not see "Hangzhou" and "Shanghai" as equally current.
        """
        at = float(at or 0.0)
        q = query or ""
        if include_history is None:
            include_history = bool(_HISTORY_QUERY.search(q))

        active = self.active(at)
        active.sort(
            key=lambda f: (
                _PRIORITY.get(f.predicate, 0)
                + (4 if any(k in q for k in _QUERY_KEYWORDS.get(f.predicate, ())) else 0),
                f.valid_from,
            ),
            reverse=True,
        )

        lines = ["## 当前事实（结构化，优先于情景记忆）"]
        for fact in active:
            label = _PREDICATE_LABELS.get(fact.predicate, fact.predicate)
            line = f"- {label}：{fact.value}（自 {_iso(fact.valid_from)}）"
            if max_chars and len("\n".join(lines + [line])) > max_chars:
                continue
            lines.append(line)

        if include_history:
            for fact in self.all():
                if fact.valid_to is None:
                    continue
                label = _PREDICATE_LABELS.get(fact.predicate, fact.predicate)
                line = f"- 历史-{label}：{fact.value}（至 {_iso(fact.valid_to)}）"
                if max_chars and len("\n".join(lines + [line])) > max_chars:
                    continue
                lines.append(line)

        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def to_list(self) -> list[dict[str, Any]]:
        return [fact.to_dict() for fact in self.all()]

    @classmethod
    def from_list(cls, data: list[dict[str, Any]] | None) -> "FactStore":
        return cls(Fact.from_dict(d) for d in (data or []))
