"""Small, transparent fact store with validity intervals.

Episodic memories are good at "what happened"; they are bad at answering
"what is true *now*" when a user changes city, address or sleep schedule.
This module keeps a separate, tiny key/value history for high-value facts.

The extractor is intentionally rule-based and conservative.  It is a fallback
that works without an LLM; hosts can add their own facts through ``add``.
The store now supports:

* multiple concurrent values for genuinely multi-valued predicates (allergies,
  pets, projects, friends);
* a distinction between *observed at* and *effective at* (a future move does
  not rewrite the current city before it happens);
* explicit correction and deletion, so a wrong fact is not permanent;
* source/confidence metadata on every fact.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable

_QUESTION = re.compile(r"[?？]|什么|哪[个天里]?|几点|多少|吗|是否|是不是|还记得")
_QUESTION_VALUE = re.compile(
    r"^(?:什么|哪[里儿个天]?|谁|多少|几点|哪个|哪天|是否|是不是|吗|还记得)"
)
_FILLER = re.compile(r"^(?:总是|一般|大概|通常|都|会|要|才|现在|最近)+")
_TRAIL = re.compile(r"[，,。；;、\s]+$")
_PARTICLES = re.compile(r"(?:了|啦|呢|吧|啊|呀|也|还|又|都)+$")

# A single value is the right model for identity/state-like attributes.  These
# predicates can legitimately hold several values at once.
MULTI_VALUE_PREDICATES = frozenset(
    {
        "allergy",
        "pet_name",
        "pet_breed",
        "project",
        "friend",
        "food_like",
        "food_dislike",
        "food_event",
        "health_event",
    }
)
_MULTI_SPLIT = re.compile(r"[、,，和及]")

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
    "food_like": "爱吃",
    "food_dislike": "不吃",
    "food_event": "饮食事件",
    "health_event": "健康事件",
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
    "food_like": 4,
    "food_dislike": 4,
    "health_event": 4,
    "food_event": 3,
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
    "food_like": ("吃", "爱吃", "喜欢", "口味", "火锅", "食物"),
    "food_dislike": ("吃", "不吃", "忌口", "食物"),
    "food_event": ("吃", "火锅", "饭", "饮食", "最近吃"),
    "health_event": ("牙", "看牙", "医院", "体检", "健康", "手术", "医生"),
}

_HISTORY_QUERY = re.compile(r"以前|之前|原来|过去|历史|搬家前|旧")

_MOVE_CUES = ("决定", "已经搬", "搬好了", "下周搬", "搬完")
_FUTURE_DAY = 86400.0


def _clean(value: str) -> str:
    value = _TRAIL.sub("", (value or "").strip())
    value = _FILLER.sub("", value)
    value = _PARTICLES.sub("", value)
    return value.strip("「」\"'“”")


def _iso(ts: float | None) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _future_offset(text: str, at: float) -> float | None:
    """A deliberately coarse event-time parser for relative move statements.

    This is not a calendar parser.  It only prevents the worst failure mode:
    treating "I will move next week" as "I live there now".  Hosts that need
    exact scheduling should pass ``event_at`` explicitly to ``FactStore.add``.
    """
    if "已经搬" in text or "搬好了" in text or "搬完" in text:
        return at
    if "后天" in text:
        return at + 2 * _FUTURE_DAY
    if "明天" in text:
        return at + _FUTURE_DAY
    if "下周" in text or "下个星期" in text or "下星期" in text:
        return at + 7 * _FUTURE_DAY
    if "下个月" in text or "下月" in text:
        return at + 30 * _FUTURE_DAY
    return None


def _event_offset(text: str, at: float) -> float | None:
    """Coarse event time for food/health events.

    These are not calendar-grade timestamps; they only keep "yesterday" and
    "next week" from collapsing into "now".  The fact remains active so a
    later query can still find it, while the value records the event.
    """
    if "昨天" in text:
        return at - _FUTURE_DAY
    if "前天" in text:
        return at - 2 * _FUTURE_DAY
    if "上周" in text or "上个星期" in text:
        return at - 7 * _FUTURE_DAY
    if "今天" in text or "刚刚" in text:
        return at
    if "明天" in text:
        return at + _FUTURE_DAY
    if "后天" in text:
        return at + 2 * _FUTURE_DAY
    if "下周" in text or "下个星期" in text or "下星期" in text:
        return at + 7 * _FUTURE_DAY
    return None


@dataclass
class ExtractedFact:
    predicate: str
    value: str
    event_at: float | None = None
    confidence: float = 0.75
    source: str = "rule"


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
    observed_at: float = 0.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def active_at(self, at: float) -> bool:
        return self.valid_from <= at and (self.valid_to is None or at < self.valid_to)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fact":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (data or {}).items() if k in allowed})


def _valid_value(value: str) -> bool:
    value = _clean(value)
    if not value or len(value) > 80:
        return False
    if _QUESTION_VALUE.match(value):
        return False
    if value.endswith("吗") or "？" in value or "?" in value:
        return False
    return True


def extract_candidates(text: str, at: float) -> list[ExtractedFact]:
    """Extract high-value facts with optional effective times.

    Questions are no longer rejected wholesale: a statement can end with a
    question tag ("我住在杭州，你知道吗？") and still contain a fact.  Instead,
    candidate *values* that are interrogatives are rejected.
    """
    text = (text or "").strip()
    if not text:
        return []

    found: list[ExtractedFact] = []
    is_question = bool(_QUESTION.search(text))

    def add(predicate: str, value: str, *, event_at: float | None = None) -> None:
        value = _clean(value)
        if _valid_value(value):
            found.append(ExtractedFact(predicate, value, event_at=event_at))

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
        ("food_like", r"(?<!不)(?:喜欢|爱吃|常吃)吃?([^，。；;\n]{1,12})"),
        ("food_dislike", r"(?:不喜欢|不爱吃|不吃)吃?([^，。；;\n]{1,12})"),
    ]
    for predicate, pattern in rules:
        # Preference/event extraction is only safe for assertions.  A question
        # such as "我喜欢吃火锅吗？" must not become a preference.
        if is_question and predicate in {"food_like", "food_dislike"}:
            continue
        for match in re.finditer(pattern, text):
            add(predicate, match.group(1))

    if not is_question:
        event_at = _event_offset(text, at)
        for match in re.finditer(
            r"(?:昨天|今天|前天|上周|周末)?(?:去)?吃(?:了)?(?:一顿)?"
            r"(火锅|烧烤|日料|川菜|粤菜|面条|米饭|饺子|汉堡|披萨|甜品|蛋糕)",
            text,
        ):
            prefix = text[max(0, match.start() - 4) : match.start()]
            if "喜欢" in prefix or "爱吃" in prefix:
                continue
            add("food_event", match.group(1), event_at=event_at)
        for match in re.finditer(
            r"(?:去)?(看(?:了)?牙|洗牙|补牙|拔牙|看医生|体检|做手术)", text
        ):
            add("health_event", match.group(1).replace("了", ""), event_at=event_at)

    # "考虑搬去上海" and "如果搬去上海" are possibilities, not a move.  A
    # decision can be future-dated; the old city stays current until then.
    if any(cue in text for cue in _MOVE_CUES):
        event_at = _future_offset(text, at)
        for match in re.finditer(r"(?:搬到|搬去|搬家到)([^\s，。,.!?！？]{2,12})", text):
            add("city", match.group(1), event_at=event_at)

    # An address usually also implies a city; keep both, because "上海" and
    # "徐汇" are different retrieval cues.  An address statement is treated as
    # an observed fact at the current time, not as a future move.
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

    seen: set[tuple[str, str]] = set()
    result: list[ExtractedFact] = []
    for item in found:
        key = (item.predicate, item.value)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def extract_facts(text: str) -> list[tuple[str, str]]:
    """Backward-compatible tuple view of :func:`extract_candidates`."""
    return [(item.predicate, item.value) for item in extract_candidates(text, 0.0)]


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

    def active_for(self, subject: str, predicate: str, at: float) -> list[Fact]:
        return [
            fact
            for fact in self._facts.get((subject, predicate), [])
            if fact.active_at(at)
        ]

    def active(self, at: float) -> list[Fact]:
        return [fact for fact in self.all() if fact.active_at(at)]

    def history(self, subject: str, predicate: str) -> list[Fact]:
        return sorted(self._facts.get((subject, predicate), []), key=lambda f: f.valid_from)

    def stale_values(self, at: float, query: str = "") -> set[str]:
        """Values that were superseded and should not be shown as current.

        A history question is the one case where old values are useful, so it
        deliberately returns an empty set and lets the caller render history.
        Multi-valued predicates do not have a single "current" value, so they
        are never treated as stale.
        """
        if _HISTORY_QUERY.search(query or ""):
            return set()
        stale: set[str] = set()
        for (subject, predicate), group in self._facts.items():
            if predicate in MULTI_VALUE_PREDICATES:
                continue
            current = self.current(subject, predicate, at)
            if not current:
                continue
            for fact in group:
                if fact.valid_to is not None and fact.value != current.value:
                    stale.add(fact.value)
        return stale

    def _add_one(
        self,
        subject: str,
        predicate: str,
        value: str,
        at: float,
        *,
        event_at: float | None = None,
        source: str = "user",
        memory_id: str | None = None,
        confidence: float = 0.8,
    ) -> Fact:
        value = _clean(value)
        if not value:
            raise ValueError("fact value must not be empty")
        effective = float(event_at if event_at is not None else at)
        if effective < 0:
            effective = at
        group = self._facts.setdefault((subject, predicate), [])

        if predicate in MULTI_VALUE_PREDICATES:
            for existing in group:
                if existing.active_at(at) and existing.value == value:
                    existing.last_seen = max(existing.last_seen, at)
                    return existing
            fact = Fact(
                subject=subject,
                predicate=predicate,
                value=value,
                valid_from=effective,
                last_seen=at,
                observed_at=at,
                source=source,
                memory_id=memory_id,
                confidence=confidence,
            )
            group.append(fact)
            return fact

        current = self.current(subject, predicate, at)
        if current and current.value == value:
            current.last_seen = max(current.last_seen, at)
            return current

        # Close a previously scheduled future value if this new value replaces
        # it.  Without this, two future moves could overlap.
        if effective > at:
            for existing in group:
                if existing.valid_from > at and existing.valid_to is None:
                    existing.valid_to = effective
        if current:
            current.valid_to = effective if effective > at else at
        fact = Fact(
            subject=subject,
            predicate=predicate,
            value=value,
            valid_from=effective,
            last_seen=at,
            observed_at=at,
            source=source,
            memory_id=memory_id,
            confidence=confidence,
        )
        group.append(fact)
        return fact

    def add(
        self,
        subject: str,
        predicate: str,
        value: str,
        at: float,
        *,
        event_at: float | None = None,
        source: str = "user",
        memory_id: str | None = None,
        confidence: float = 0.8,
    ) -> Fact:
        """Add a fact; split genuinely multi-valued predicates.

        ``event_at`` is when the fact becomes true.  It can be in the future:
        the old value remains current until that time.  ``at`` is when the
        statement was observed.
        """
        values = [value]
        if predicate == "allergy":
            values = [part for part in _MULTI_SPLIT.split(value or "") if part.strip()]
        facts = [
            self._add_one(
                subject,
                predicate,
                part,
                at,
                event_at=event_at,
                source=source,
                memory_id=memory_id,
                confidence=confidence,
            )
            for part in values
        ]
        return facts[-1]

    def observe(
        self,
        text: str,
        at: float,
        *,
        subject: str = "user",
        source: str = "user",
        memory_id: str | None = None,
    ) -> list[Fact]:
        out = []
        for item in extract_candidates(text, at):
            out.append(
                self.add(
                    subject,
                    item.predicate,
                    item.value,
                    at,
                    event_at=item.event_at,
                    source=source,
                    memory_id=memory_id,
                    confidence=item.confidence,
                )
            )
        return out

    def correct(
        self,
        subject: str,
        predicate: str,
        value: str,
        at: float,
        *,
        old_value: str | None = None,
        source: str = "user",
        confidence: float = 0.95,
    ) -> Fact:
        """Close a wrong/old value and insert the corrected one.

        ``old_value`` is optional; when supplied only that value is closed,
        which is important for multi-valued predicates.
        """
        for fact in self.active_for(subject, predicate, at):
            if old_value is None or fact.value == old_value:
                fact.valid_to = at
        return self.add(
            subject,
            predicate,
            value,
            at,
            source=source,
            confidence=confidence,
        )

    def delete(
        self,
        at: float,
        *,
        subject: str = "user",
        predicate: str | None = None,
        value: str | None = None,
        fact_id: str | None = None,
    ) -> int:
        """Retire facts without erasing history.  Returns the number closed."""
        closed = 0
        for fact in self.all():
            if fact_id and fact.id != fact_id:
                continue
            if subject and fact.subject != subject:
                continue
            if predicate and fact.predicate != predicate:
                continue
            if value and fact.value != value:
                continue
            if fact.valid_to is None or fact.valid_to > at:
                fact.valid_to = at
                closed += 1
        return closed

    def render(
        self,
        query: str = "",
        *,
        at: float | None = None,
        max_chars: int | None = None,
        include_history: bool | None = None,
    ) -> str:
        """Render only facts relevant to ``query``, under a hard budget.

        The previous version sorted all active facts but still printed every
        one.  At a 900-char context that meant a factual query could receive a
        full profile dump while the actual answer had little room.  Selection
        is now explicit:

        * a query about a predicate gets that predicate (plus one anchor);
        * a general "what do you remember" query gets a compact core profile;
        * an unrelated query gets at most a name/city anchor, never the store.
        """
        at = float(at or 0.0)
        q = (query or "").strip()
        if include_history is None:
            include_history = bool(_HISTORY_QUERY.search(q))

        active = self.active(at)
        if not active:
            return ""

        grouped: dict[str, list[Fact]] = {}
        for fact in active:
            grouped.setdefault(fact.predicate, []).append(fact)

        summary_query = any(
            word in q for word in ("总结", "记得", "关于我", "我是谁", "介绍", "全部", "所有")
        )
        matched = {
            predicate
            for predicate in grouped
            if any(keyword in q for keyword in _QUERY_KEYWORDS.get(predicate, ()))
        }

        if summary_query or not q:
            selected = sorted(
                grouped,
                key=lambda p: (_PRIORITY.get(p, 0), max(f.valid_from for f in grouped[p])),
                reverse=True,
            )[:6]
        elif matched:
            selected = sorted(
                matched,
                key=lambda p: (_PRIORITY.get(p, 0), max(f.valid_from for f in grouped[p])),
                reverse=True,
            )[:5]
            # A name is a useful anchor, but only if it does not displace the
            # predicate the user actually asked about.
            if "name" in grouped and "name" not in selected and len(selected) < 3:
                selected.append("name")
        else:
            # No lexical match: show a minimal identity anchor, not a dump.
            selected = [p for p in ("name", "city") if p in grouped][:1]

        lines = ["## 当前事实（结构化，优先于情景记忆）"]
        for predicate in selected:
            facts = grouped[predicate]
            label = _PREDICATE_LABELS.get(predicate, predicate)
            values: list[str] = []
            for fact in sorted(facts, key=lambda f: f.valid_from):
                if fact.value not in values:
                    values.append(fact.value)
            line = f"- {label}：{'、'.join(values)}（自 {_iso(min(f.valid_from for f in facts))}）"
            if max_chars and len("\n".join(lines + [line])) > max_chars:
                break
            lines.append(line)

        if include_history:
            history_predicates = set(selected) if matched or summary_query else set(grouped)
            old = [
                fact
                for fact in self.all()
                if fact.valid_to is not None and fact.predicate in history_predicates
            ]
            old.sort(key=lambda f: (f.predicate, f.valid_from))
            for fact in old:
                label = _PREDICATE_LABELS.get(fact.predicate, fact.predicate)
                line = f"- 历史-{label}：{fact.value}（至 {_iso(fact.valid_to)}）"
                if max_chars and len("\n".join(lines + [line])) > max_chars:
                    break
                lines.append(line)

        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def to_list(self) -> list[dict[str, Any]]:
        return [fact.to_dict() for fact in self.all()]

    @classmethod
    def from_list(cls, data: list[dict[str, Any]] | None) -> "FactStore":
        return cls(Fact.from_dict(d) for d in (data or []))
