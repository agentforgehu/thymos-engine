"""Appraisal rules and the appraisal -> emotion mapping.

This module is intentionally rule-based and auditable.  A later version can
replace `estimate_appraisal` with an LLM or a trained classifier; the engine
only depends on the `Appraisal` dataclass.
"""

from __future__ import annotations

import re

from .mathx import clamp
from .types import Appraisal, Emotion

EMOTION_HALF_LIFE = {
    "surprise": 5.0,
    "fear": 12.0,
    "anger": 25.0,
    "excitement": 28.0,
    "disgust": 40.0,
    "joy": 45.0,
    "anticipation": 90.0,
    "anxiety": 110.0,
    "contentment": 180.0,
    "hope": 240.0,
    "sadness": 300.0,
    "trust": 480.0,
    "neutral": 60.0,
}

POSITIVE_WORDS = (
    "喜欢", "开心", "高兴", "成功", "完成", "顺利", "通过", "惊喜", "期待",
    "温暖", "安全", "奖励", "夸", "拥抱", "陪", "爱", "感谢", "赞", "进步",
    "很好", "好消息", "赢", "达成", "放松", "安心", "满意", "骄傲", "治愈",
    "有趣", "跑完", "搞定", "支持", "理解", "原谅",
    "good", "great", "love", "success", "win", "happy", "thanks", "proud",
    "nice", "better", "done", "pass", "safe",
)

NEGATIVE_WORDS = (
    "失败", "难过", "生气", "焦虑", "害怕", "痛", "讨厌", "糟糕", "问题",
    "错误", "崩溃", "压力", "孤独", "失去", "担心", "批评", "冲突", "吵",
    "拒绝", "委屈", "失望", "累", "烦", "危险", "攻击", "病", "伤害",
    "投诉", "崩了", "迟到", "来不及", "搞砸",
    "bad", "fail", "angry", "sad", "fear", "pain", "hate", "stress",
    "lonely", "wrong", "crash", "bug", "afraid", "lost", "sorry",
)

# A few words carry more affective weight than their neighbours.
WEIGHTS = {
    "骄傲": 1.6, "惊喜": 1.5, "成功": 1.3, "开心": 1.25, "爱": 1.4,
    "崩溃": 1.6, "孤独": 1.3, "压力": 1.25, "失败": 1.3, "投诉": 1.3,
    "伤害": 1.3, "讨厌": 1.25, "失望": 1.2,
}

HIGH_STAKES = (
    "重要", "必须", "截止", "面试", "考试", "老板", "家人", "健康", "钱",
    "合同", "发布", "上线", "手术", "搬家", "分手", "结婚", "deadline",
    "interview", "exam", "family", "health", "release", "launch", "客户",
)

NOVELTY_WORDS = (
    "第一次", "新", "突然", "意外", "从未", "竟然", "没想到", "第一次",
    "first", "new", "sudden", "unexpected",
)

CONTROL_HIGH = ("我能", "可以", "决定", "选择", "掌控", "搞定", "我来", "i can", "i will", "decide")
CONTROL_LOW = ("无法", "只能", "被迫", "没办法", "不得不", "做不到", "cannot", "forced", "no choice")

OTHER_AGENTS = ("他", "她", "他们", "她们", "老板", "同事", "客户", "老师", "朋友", "he ", "she ", "they ", "boss", "client")

UNCERTAIN = ("可能", "也许", "大概", "不确定", "或许", "maybe", "perhaps", "might", "not sure")

NEGATORS = ("不", "没", "无", "别", "未")


def appraise(appraisal: Appraisal) -> Emotion:
    """Map an appraisal to a discrete label plus a PAD impulse."""
    a = appraisal.clamped()
    d, c, ctrl, novelty = a.desirability, a.certainty, a.control, a.novelty
    label = "neutral"

    if novelty >= 0.72 and abs(d) < 0.35:
        label = "surprise"
    elif d >= 0.22:
        if c < 0.45:
            label = "hope"
        elif a.agency == "other" and a.norm_compliance > 0.25:
            label = "trust"
        elif a.stakes >= 0.62:
            label = "excitement"
        elif a.stakes <= 0.32:
            label = "contentment"
        else:
            label = "joy"
    elif d <= -0.22:
        if a.norm_compliance <= -0.45:
            label = "disgust"
        elif a.agency == "other" and ctrl >= 0.55 and a.stakes >= 0.40:
            label = "anger"
        elif c < 0.42 and ctrl < 0.55:
            label = "anxiety" if d > -0.70 else "fear"
        else:
            label = "sadness"
    elif a.goal_relevance >= 0.65:
        label = "anticipation"

    valence = clamp(d * (0.55 + 0.45 * a.goal_relevance), -1.0, 1.0)
    arousal = clamp(0.16 + 0.52 * a.stakes * abs(d) + 0.34 * novelty, 0.0, 1.0)
    dominance = clamp((ctrl - 0.5) * 1.6 * (0.5 + 0.5 * a.stakes), -1.0, 1.0)
    intensity = clamp(
        abs(d) * (0.50 + 0.50 * a.goal_relevance) * (0.55 + 0.45 * a.stakes)
        + 0.25 * novelty * (1.0 - abs(d)),
        0.02,
        1.0,
    )

    return Emotion(
        label=label,
        intensity=intensity,
        valence=valence,
        arousal=arousal,
        dominance=dominance,
        half_life_min=EMOTION_HALF_LIFE.get(label, 60.0),
        appraisal=a,
    )


def estimate_appraisal(text: str) -> Appraisal:
    """Cheap, deterministic appraisal from text.

    Negation is handled per occurrence ("不开心" flips, "没想到" does not
    negate a sentiment word), which avoids the classic bag-of-words failure.
    """
    raw = (text or "").strip()
    low = raw.lower()
    if not low:
        return Appraisal(note="empty")

    pos, neg, matched = _sentiment(low)
    total = pos + neg
    desirability = 0.0 if total == 0 else clamp((pos - neg) / (total + 0.8), -1.0, 1.0)

    stakes = 0.78 if any(w in low for w in HIGH_STAKES) else 0.45
    if "!" in raw or "！" in raw:
        stakes = min(1.0, stakes + 0.12)

    novelty = 0.30
    if any(w in low for w in NOVELTY_WORDS):
        novelty = 0.78
    elif "?" in low or "？" in low:
        novelty = 0.48

    certainty = 0.66
    if "?" in low or "？" in low or any(w in low for w in UNCERTAIN):
        certainty = 0.42

    control = 0.50
    if any(w in low for w in CONTROL_HIGH):
        control = 0.82
    if any(w in low for w in CONTROL_LOW):
        control = 0.22

    agency = "other" if any(w in low for w in OTHER_AGENTS) else "self" if any(w in low for w in CONTROL_HIGH) else "circumstance"

    # Explicit emotion words are strong evidence about the appraisal pattern.
    if any(w in low for w in ("生气", "愤怒", "火大", "气死")):
        agency, control, stakes = "other", max(control, 0.68), max(stakes, 0.75)
    if any(w in low for w in ("害怕", "恐惧", "惊恐", "吓死")):
        certainty, control, stakes = min(certainty, 0.35), min(control, 0.25), max(stakes, 0.80)
    if any(w in low for w in ("难过", "伤心", "失落", "委屈")):
        control = min(control, 0.30)

    norm = -0.5 if any(w in low for w in ("道德", "欺骗", "背叛", "恶心", "违规")) else 0.1
    relevance = clamp(0.35 + 0.03 * min(len(raw), 18) + 0.15 * stakes, 0.0, 1.0)
    note = "lexical:" + ",".join(matched[:6]) if matched else "lexical:no-cue"

    return Appraisal(
        goal_relevance=relevance,
        desirability=desirability,
        novelty=novelty,
        certainty=certainty,
        control=control,
        agency=agency,
        norm_compliance=norm,
        stakes=stakes,
        note=note,
    )


def _sentiment(text: str) -> tuple[float, float, list[str]]:
    """Weighted positive/negative scores with per-occurrence negation."""
    pos = neg = 0.0
    matched: list[str] = []
    for word in POSITIVE_WORDS:
        for idx in _occurrences(text, word):
            weight = WEIGHTS.get(word, 1.0)
            if _negated(text, idx):
                neg += weight
            else:
                pos += weight
            matched.append(("!" if _negated(text, idx) else "") + word)
    for word in NEGATIVE_WORDS:
        for idx in _occurrences(text, word):
            weight = WEIGHTS.get(word, 1.0)
            if _negated(text, idx):
                pos += weight
            else:
                neg += weight
            matched.append(("!" if _negated(text, idx) else "") + word)
    return pos, neg, matched


def _occurrences(text: str, word: str) -> list[int]:
    out, start = [], 0
    while True:
        idx = text.find(word, start)
        if idx < 0:
            return out
        out.append(idx)
        start = idx + max(1, len(word))


def _negated(text: str, index: int) -> bool:
    window = text[max(0, index - 2):index]
    return any(n in window for n in NEGATORS)
