"""Appraisal rules and the appraisal -> emotion mapping.

This module is intentionally rule-based and auditable.  A later version can
replace `estimate_appraisal` with an LLM or a trained classifier; the engine
only depends on the `Appraisal` dataclass.

The lexical layer is a *baseline*, not a claim of general language
understanding.  In particular, Chinese negation and social language are
context-sensitive: "特别" contains the character "别", "差不多" contains
"不", and "别烦我" is not the positive negation of "烦".  The rules below
therefore use clause-local negation windows with explicit exceptions, plus
phrase-level social cues for direct insults, criticism, loss events and
colloquial praise.  Ambiguous cues (sarcasm, generic profanity) lower
certainty and are recorded in `Appraisal.note`; they are not silently turned
into a strong polarity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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
    "喜欢", "开心", "高兴", "成功", "完成", "顺利", "惊喜", "期待",
    "温暖", "安全", "奖励", "夸", "拥抱", "陪", "爱", "感谢", "赞", "进步",
    "很好", "好消息", "赢", "达成", "放松", "安心", "满意", "骄傲", "治愈",
    "有趣", "跑完", "搞定", "支持", "理解", "原谅", "表扬", "赞美", "认可",
    "关心", "照顾", "安慰", "陪伴", "夸奖", "鼓励", "体贴", "可靠",
    "谢谢", "多谢", "辛苦", "辛苦啦", "真好", "真棒", "太棒", "靠谱",
    "给力", "优秀", "能干", "贴心", "不错", "做得好", "干得漂亮",
    "不辛苦", "不麻烦",
    "通过了", "审核通过", "面试通过", "评审通过", "考试通过",
    "帮大忙", "帮了大忙", "帮我大忙",
    "good", "great", "love", "success", "win", "happy", "thanks", "proud",
    "nice", "better", "done", "pass", "safe",
)

NEGATIVE_WORDS = (
    "失败", "难过", "生气", "焦虑", "害怕", "痛", "讨厌", "糟糕", "问题",
    "错误", "崩溃", "压力", "孤独", "失去", "担心", "批评", "冲突", "吵",
    "拒绝", "委屈", "失望", "累", "烦", "危险", "攻击", "病", "伤害",
    "投诉", "崩了", "迟到", "来不及", "搞砸", "骂", "指责", "嘲讽", "侮辱",
    "冷落", "忽视", "烂", "差劲", "黄了", "裁员", "被裁", "失业", "下岗",
    "分手", "离婚", "去世", "病重", "住院", "手术", "被骗", "背叛", "出轨",
    "亏了", "亏钱", "亏了钱", "赔钱", "挂科", "延毕", "被拒", "拉黑", "冷战", "恶心",
    "bad", "fail", "angry", "sad", "fear", "pain", "hate", "stress",
    "lonely", "wrong", "crash", "bug", "afraid", "lost", "sorry",
)

# A few words carry more affective weight than their neighbours.
WEIGHTS = {
    "骄傲": 1.6, "惊喜": 1.5, "成功": 1.3, "开心": 1.25, "爱": 1.4,
    "崩溃": 1.6, "孤独": 1.3, "压力": 1.25, "失败": 1.3, "投诉": 1.3,
    "伤害": 1.3, "讨厌": 1.25, "失望": 1.2, "夸": 1.3, "表扬": 1.3,
    "赞美": 1.3, "认可": 1.2, "关心": 1.2, "照顾": 1.2, "安慰": 1.15,
    "骂": 1.5, "指责": 1.4, "嘲讽": 1.4, "侮辱": 1.5, "冷落": 1.3,
    "忽视": 1.2, "烂": 1.1, "差劲": 1.2, "黄了": 1.4, "裁员": 1.5,
    "被裁": 1.5, "失业": 1.4, "分手": 1.4, "离婚": 1.4, "去世": 1.6,
    "病重": 1.5, "住院": 1.2, "手术": 1.2, "被骗": 1.4, "背叛": 1.5,
    "出轨": 1.5, "亏了": 1.2, "亏钱": 1.2, "亏了钱": 1.2, "赔钱": 1.2, "挂科": 1.2, "延毕": 1.2,
    "被拒": 1.2, "拉黑": 1.2, "冷战": 1.2, "恶心": 1.2,
    "谢谢": 1.1, "辛苦": 1.15, "真好": 1.2, "真棒": 1.3, "太棒": 1.3,
    "靠谱": 1.2, "给力": 1.2, "优秀": 1.2, "能干": 1.2, "贴心": 1.2,
    "不错": 1.1,
}

HIGH_STAKES = (
    "重要", "必须", "截止", "面试", "考试", "老板", "家人", "健康", "钱",
    "合同", "发布", "上线", "手术", "搬家", "搬去", "搬到", "分手", "结婚",
    "离婚", "裁员", "失业", "去世", "住院", "被骗", "deadline",
    "interview", "exam", "family", "health", "release", "launch", "客户",
)

NOVELTY_WORDS = (
    "第一次", "头一次", "新的", "新来的", "新项目", "新工作", "新环境",
    "新消息", "新情况", "新任务", "新公司", "新家", "突然", "意外", "竟然",
    "没想到", "从未", "刚刚",
    "first", "new", "sudden", "unexpected",
)

CONTROL_HIGH = ("我能", "可以", "决定", "选择", "掌控", "搞定", "我来", "i can", "i will", "decide")
CONTROL_LOW = ("无法", "只能", "被迫", "没办法", "不得不", "做不到", "cannot", "forced", "no choice")

OTHER_AGENTS = (
    "他", "她", "他们", "她们", "老板", "同事", "客户", "老师", "朋友",
    "有人", "别人", "大家", "被", "he ", "she ", "they ", "boss", "client",
)

UNCERTAIN = ("可能", "也许", "大概", "不确定", "或许", "maybe", "perhaps", "might", "not sure")

NEGATORS = ("不", "没", "无", "别", "未")

# These strings contain a negator character but are not negations of the
# sentiment word that follows them.  Without this list, "特别开心" becomes
# negative because "别" is found in "特别", and "差不多完成了" becomes negative
# because "不" is found in "差不多".
NEGATION_EXCEPTIONS = (
    "特别", "别人", "区别", "个别", "告别", "分别", "级别", "类别",
    "性别", "识别", "差不多", "不少", "不久", "不仅", "不但", "不断",
    "不同", "不得不", "无论", "未来", "未免", "不免", "不妨", "不愧",
    "不料", "没想到", "没料到", "没成想", "不禁", "不由得", "不管", "不论", "不过", "了不得", "了不起",
    "忍不住",
)

DOUBLE_NEGATIONS = ("不是不", "并非不", "不能不", "不得不", "无不", "没有不")

# Substring cues that need a surrounding-word exception.  Chinese has no
# spaces, so "累" inside "积累" and "病" inside "病人" otherwise create false
# sentiment.
WORD_EXCEPTIONS: dict[str, tuple[str, ...]] = {
    "累": ("积累", "累计", "累积", "牵累"),
    "病": ("病人", "病历", "病毒", "病菌", "病床", "病区"),
    "夸": ("夸克",),
    "烦": ("麻烦你", "麻烦您", "麻烦帮我", "麻烦帮"),
}

FUTURE_CUES = (
    "期待", "盼", "等不及", "下周", "明天", "后天", "即将", "马上",
    "计划", "准备", "决定", "考虑", "犹豫", "将来", "以后", "下次", "周末想", "想约",
)

UNCERTAIN_FUTURE = ("考虑", "犹豫", "也许", "可能", "不确定", "还没决定")


@dataclass(frozen=True)
class LexicalCue:
    """A phrase-level appraisal cue with an explicit social signature.

    `kind` is deliberately readable so the audit trail can say *why* a turn
    was treated as an attack, a loss event or praise.  The cue is not a
    sentiment score; it is a small, replaceable rule in the baseline.
    """

    kind: str
    pattern: re.Pattern[str]
    desirability: float
    norm_compliance: float
    agency: str
    control: float | None = None
    stakes: float | None = None
    confidence: float = 0.8


ATTACK_CUES = (
    LexicalCue(
        "profane_insult",
        re.compile(r"傻逼|傻B|煞笔|脑残|智障|去死|死开|操你|草你", re.IGNORECASE),
        -0.75, -0.55, "other", 0.65, 0.70, 0.9,
    ),
    LexicalCue(
        "direct_insult",
        re.compile(
            r"你(?:怎么|真|是不是|也太|可真|到底)?[^，。！？\s]{0,4}?"
            r"(?:(?<!不)笨(?!鸟)|(?<!不)傻(?!白)|(?<!不)蠢|废物|"
            r"垃圾(?!分类|桶|站|袋|场|回收|食品|处理|焚烧|填埋)|"
            r"菜(?!市场|谱|单|园|地|刀)|弱(?!化|势)|差劲|"
            r"没用(?!这|那|它|该|过|到|上|下|来|去|多|几|什么|任何)|"
            r"(?<!行)不行|脑子进水|脑子有坑|脑子坏掉)"
        ),
        -0.65, -0.35, "other", 0.70, 0.65, 0.85,
    ),
    LexicalCue(
        "mockery",
        re.compile(r"你就是个笑话|你是个笑话|你简直是个笑话"),
        -0.55, -0.30, "other", 0.60, 0.55, 0.8,
    ),
    LexicalCue(
        "reported_insult",
        re.compile(
            r"(?:说|骂|嫌|觉得|认为)我[^，。！？\s]{0,4}?"
            r"(?:(?<!不)笨|(?<!不)傻|(?<!不)蠢|废物|"
            r"没用(?!这|那|它|该|过|到|上|下|来|去|多|几|什么|任何)|(?<!行)不行)"
        ),
        -0.55, -0.30, "other", 0.60, 0.55, 0.75,
    ),
    LexicalCue(
        "ability_insult",
        re.compile(r"菜得抠脚|菜鸡|弱爆了|拉胯|蠢死了|笨死了|傻透了|废物一个"),
        -0.60, -0.30, "other", 0.60, 0.55, 0.8,
    ),
    LexicalCue(
        "medical_insult",
        re.compile(
            r"脑子有病|神经病(?!学|科|院|医生|患者)|有病吧|"
            r"你是不是有病|你有病(?:吧|啊|吗|？|\?|$)"
        ),
        -0.60, -0.30, "other", 0.65, 0.60, 0.8,
    ),
    LexicalCue(
        "imperative_attack",
        re.compile(
            r"闭嘴|少废话|别烦我|要你何用|没用的东西|你算什么东西|谁给你的勇气|"
            r"滚(?:蛋|开|出去|吧)|(?:^|[，。！？\s])滚(?:$|[，。！？\s])"
        ),
        -0.60, -0.30, "other", 0.65, 0.60, 0.85,
    ),
    LexicalCue(
        "dismissal",
        re.compile(r"就这(?:[？?]|$)|不过如此|一般般"),
        -0.35, -0.20, "other", 0.55, 0.50, 0.65,
    ),
)

CRITICISM_CUES = (
    LexicalCue(
        "criticism",
        re.compile(
            r"(?:这个|这|方案|计划|做法|你|他|她)[^，。！？\s]{0,8}?"
            r"(?:(?<!行)不行|不对|有问题|不靠谱|行不通|做得不好|做得差)"
        ),
        -0.40, -0.20, "other", 0.60, 0.50, 0.7,
    ),
)

LOSS_CUES = (
    LexicalCue(
        "job_loss",
        re.compile(r"被裁|裁员|失业|下岗|丢工作"),
        -0.70, 0.0, "circumstance", 0.25, 0.85, 0.9,
    ),
    LexicalCue(
        "relationship_loss",
        re.compile(r"分手|离婚|被甩|拉黑|冷战"),
        -0.65, 0.0, "circumstance", 0.35, 0.80, 0.85,
    ),
    LexicalCue(
        "health_crisis",
        re.compile(r"去世|病重|住院|手术|重病|癌症|车祸"),
        -0.75, 0.0, "circumstance", 0.25, 0.90, 0.9,
    ),
    LexicalCue(
        "betrayal",
        re.compile(r"被骗|骗了|背叛|出轨|坑了"),
        -0.65, -0.40, "other", 0.35, 0.75, 0.85,
    ),
    LexicalCue(
        "project_loss",
        re.compile(r"(?:项目|计划|方案|合作).{0,4}黄了|黄了"),
        -0.55, 0.0, "circumstance", 0.30, 0.75, 0.8,
    ),
    LexicalCue(
        "setback",
        re.compile(r"挂科|延毕|被拒|亏了|亏钱|亏了钱|赔钱|搞砸|崩了"),
        -0.50, 0.0, "circumstance", 0.40, 0.65, 0.75,
    ),
)

POSITIVE_SOCIAL_CUES = (
    LexicalCue(
        "gratitude",
        re.compile(r"谢谢|多谢|感谢|辛苦了|辛苦啦|辛苦你了|麻烦你了|麻烦您了"),
        0.45, 0.20, "other", None, 0.50, 0.85,
    ),
    LexicalCue(
        "praise",
        re.compile(
            r"你真好|真好|太厉害了|好厉害|(?<!可)真厉害|"
            r"(?<!可真)厉害啊|(?<!可真)厉害呀|牛啊|牛哇|太牛了|"
            r"靠谱|给力|真棒|太棒了|棒极了|优秀|能干|贴心|"
            r"做得好|干得漂亮|不错|点赞|赞一个|太好了"
        ),
        0.45, 0.20, "other", None, 0.50, 0.8,
    ),
    LexicalCue(
        "care",
        re.compile(r"抱抱|陪你|我在呢|有我在|我在你身边|别难过|加油|关心|照顾|安慰|陪伴|体贴|支持|理解"),
        0.35, 0.30, "other", None, 0.50, 0.75,
    ),
)

SELF_NEGATIVE_CUES = (
    LexicalCue(
        "self_deprecation",
        re.compile(
            r"我(?:真|太|好|很|怎么这么|真是|就是个|是个)[^，。！？\s]{0,3}?"
            r"(?:(?<!不)笨|(?<!不)傻|(?<!不)蠢|废物|"
            r"没用(?!这|那|它|该|过|到|上|下|来|去)|垃圾(?!分类|桶|站|袋|场))"
        ),
        -0.45, -0.20, "self", 0.25, 0.50, 0.75,
    ),
)

AMBIGUOUS_CUES = (
    ("sarcasm", re.compile(r"呵呵|你可真(?:厉害|行|棒|能干|优秀)(?:啊|呢|哦)|好棒棒")),
    ("profanity", re.compile(r"卧槽|我靠|他妈的|妈的|操(?!作|场|练|控|心)|靠(?!谱|近|山|边)")),
)


def appraise(appraisal: Appraisal) -> Emotion:
    """Map an appraisal to a discrete label plus a PAD impulse."""
    a = appraisal.clamped()
    d, c, ctrl, novelty = a.desirability, a.certainty, a.control, a.novelty
    label = "neutral"

    future = "future" in a.note
    uncertain_future = "uncertain-future" in a.note

    # Surprise should not swallow a clear negative/positive event.  A strong
    # d plus high novelty is still a loss/gain first; novelty only changes the
    # label when the valence is weak.
    if novelty >= 0.72 and abs(d) < 0.22:
        label = "surprise"
    elif future and abs(d) < 0.22:
        # "Anticipation" is a future-oriented appraisal, not the default for
        # every neutral statement.  A move being considered is uncertainty;
        # a move already decided is anticipation.
        if uncertain_future:
            label = "anxiety" if a.stakes >= 0.50 else "neutral"
        else:
            label = "anticipation"
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
    elif abs(d) < 0.12:
        label = "neutral"
    elif a.goal_relevance >= 0.65:
        # Relevance alone is not anticipation.  A neutral but relevant fact
        # ("I live in Hangzhou") stays neutral; it should not paint the whole
        # conversation with one label.
        label = "neutral"

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


def explain_appraisal(appraisal: Appraisal) -> dict[str, object]:
    """Return the dimension-to-label decision as JSON-friendly data.

    This is a debugging contract, not a second appraisal.  It is deliberately
    explicit about the thresholds so a host can see why an LLM supplied
    ``desirability=-0.3`` but the label still looked neutral.
    """
    a = appraisal.clamped()
    emotion = appraise(a)
    d = a.desirability
    if abs(d) < 0.12:
        neutral_reason = "desirability_inside_deadband"
    elif abs(d) < 0.22:
        neutral_reason = "below_effect_threshold"
    elif emotion.label == "neutral":
        neutral_reason = "unexpected_neutral_branch"
    else:
        neutral_reason = ""
    return {
        "appraisal": a.to_dict(),
        "emotion": emotion.to_dict(),
        "decision": {
            "label": emotion.label,
            "strong_positive": d >= 0.22,
            "strong_negative": d <= -0.22,
            "novelty_override": a.novelty >= 0.72 and abs(d) < 0.22,
            "future_override": "future" in a.note and abs(d) < 0.22,
            "norm_violation": a.norm_compliance <= -0.45,
            "other_agency_high_control": (
                a.agency == "other" and a.control >= 0.55 and a.stakes >= 0.40
            ),
            "low_certainty_low_control": a.certainty < 0.42 and a.control < 0.55,
            "neutral_reason": neutral_reason,
        },
    }


def estimate_appraisal(text: str) -> Appraisal:
    """Cheap, deterministic appraisal from text.

    Negation is handled per occurrence inside a clause, with explicit
    exceptions for words such as "特别", "差不多" and "别人".  Phrase-level
    social cues cover direct insults, criticism, loss events and colloquial
    praise that a bag-of-words lexicon misses.  Ambiguous cues lower certainty
    and are recorded in `note` instead of being forced into a polarity.
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

    agency = (
        "other"
        if _has_other_agent(low)
        else "self"
        if any(w in low for w in CONTROL_HIGH)
        else "circumstance"
    )

    # Explicit emotion words are strong evidence about the appraisal pattern.
    if any(w in low for w in ("生气", "愤怒", "火大", "气死")):
        agency, control, stakes = "other", max(control, 0.68), max(stakes, 0.75)
    if any(w in low for w in ("害怕", "恐惧", "惊恐", "吓死")):
        certainty, control, stakes = min(certainty, 0.35), min(control, 0.25), max(stakes, 0.80)
    if any(w in low for w in ("难过", "伤心", "失落", "委屈")):
        control = min(control, 0.30)

    norm = -0.5 if any(
        w in low for w in ("道德", "欺骗", "背叛", "恶心", "违规", "出轨")
    ) else 0.1
    # Social appraisal is not reducible to a bag of adjectives: being praised,
    # insulted and cared for have different agency/norm signatures even when
    # their valence magnitude is similar.  Keep these cues explicit so a
    # praise and an insult do not both collapse into "neutral".
    praise_cue = _has_unnegated(low, ("夸", "表扬", "赞美", "认可", "夸奖", "鼓励"))
    care_cue = _has_unnegated(low, ("关心", "照顾", "安慰", "陪伴", "体贴", "支持", "理解"))
    insult_cue = _has_unnegated(low, ("骂", "指责", "嘲讽", "侮辱", "冷落", "忽视"))
    if praise_cue:
        agency = "other"
        norm = max(norm, 0.10)
        stakes = max(stakes, 0.55)
    if care_cue:
        agency = "other"
        norm = max(norm, 0.35)
        stakes = max(stakes, 0.55)
        # Care can coexist with illness or a bad day.  Without a small
        # positivity floor, "我生病了，朋友关心我" cancels out to neutral.
        if not insult_cue and desirability > -0.20:
            desirability = max(desirability, 0.28)
    if insult_cue:
        agency = "other"
        control = max(control, 0.70)
        norm = min(norm, -0.30)
        stakes = max(stakes, 0.60)

    # Phrase-level social cues.  Negative cues win over positive ones: a
    # project failure plus "thanks" is still a loss event, not a praise turn.
    attack_hits = [
        cue for cue in ATTACK_CUES
        if (match := cue.pattern.search(low))
        and not _negated(low, match.start())
    ]
    criticism_hits = [
        cue for cue in CRITICISM_CUES
        if (match := cue.pattern.search(low))
        and not _cue_match_negated(low, match)
    ]
    loss_hits = [
        cue for cue in LOSS_CUES
        if (match := cue.pattern.search(low))
        and not _cue_match_negated(low, match)
    ]
    self_hits = [
        cue for cue in SELF_NEGATIVE_CUES
        if (match := cue.pattern.search(low))
        and not _cue_match_negated(low, match)
    ]
    positive_hits = [
        cue for cue in POSITIVE_SOCIAL_CUES
        if (match := cue.pattern.search(low))
        and not _cue_match_negated(low, match)
    ]
    ambiguous_hits = [
        (kind, cue) for kind, cue in AMBIGUOUS_CUES if cue.search(low)
    ]

    cue_kinds: list[str] = []
    negative_hits = attack_hits + criticism_hits + loss_hits + self_hits
    if negative_hits:
        cue = min(negative_hits, key=lambda item: item.desirability)
        desirability = min(desirability, cue.desirability)
        norm = min(norm, cue.norm_compliance)
        if cue.stakes is not None:
            stakes = max(stakes, cue.stakes)
        if cue.control is not None:
            if cue.kind in {
                "job_loss", "relationship_loss", "health_crisis", "project_loss",
                "setback", "self_deprecation",
            }:
                control = min(control, cue.control)
            else:
                control = max(control, cue.control)
        agency = cue.agency
        certainty = max(certainty, cue.confidence)
        cue_kinds.append(cue.kind)
    elif positive_hits and desirability > -0.35:
        cue = max(positive_hits, key=lambda item: item.desirability)
        desirability = max(desirability, cue.desirability)
        norm = max(norm, cue.norm_compliance)
        if cue.stakes is not None:
            stakes = max(stakes, cue.stakes)
        agency = cue.agency
        cue_kinds.append(cue.kind)

    if ambiguous_hits:
        certainty = min(certainty, 0.35)
        # Generic profanity and standalone sarcasm are not strong enough to
        # justify a negative label without context.  Record them, but only
        # nudge valence when no explicit cue already decided the turn.
        if not negative_hits and not positive_hits:
            if any(kind == "profanity" for kind, _ in ambiguous_hits):
                desirability = min(desirability, -0.18)
                stakes = max(stakes, 0.55)
            elif any(kind == "sarcasm" for kind, _ in ambiguous_hits):
                desirability = min(desirability, -0.12)

    # Relevance is not a proxy for text length.  The old formula saturated for
    # almost every Chinese sentence, which made "relevance" look high even for
    # "I live in Hangzhou".  Keep a modest length term and explicitly discount
    # questions, which are often about a fact rather than a goal.
    question_factor = 0.72 if ("?" in raw or "？" in raw) else 1.0
    relevance = clamp(
        (0.22 + 0.08 * stakes + 0.012 * min(len(raw), 32)) * question_factor,
        0.0,
        1.0,
    )
    if negative_hits:
        relevance = max(relevance, 0.45)
    elif positive_hits:
        relevance = max(relevance, 0.40)

    note_parts = ["lexical:" + (",".join(matched[:6]) if matched else "no-cue")]
    if any(word in low for word in FUTURE_CUES):
        note_parts.append("future")
    if any(word in low for word in UNCERTAIN_FUTURE):
        note_parts.append("uncertain-future")
    note_parts.extend("cue:" + kind for kind in cue_kinds)
    note_parts.extend("ambiguous:" + kind for kind, _ in ambiguous_hits)
    note = ";".join(note_parts)

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
    """Weighted positive/negative scores with per-occurrence negation.

    Longest cues are matched first and their character spans are consumed, so
    "陪伴" does not also count "陪", and "病重" does not also count "病".
    """
    entries: list[tuple[str, int]] = []
    entries.extend((word, 1) for word in set(POSITIVE_WORDS))
    entries.extend((word, -1) for word in set(NEGATIVE_WORDS))
    entries.sort(key=lambda item: (-len(item[0]), item[0]))

    occupied: list[tuple[int, int]] = []
    hits: list[tuple[int, str, float]] = []
    for word, sign in entries:
        for idx in _occurrences(text, word):
            if _in_word_exception(text, idx, word):
                continue
            span = (idx, idx + len(word))
            if any(not (span[1] <= start or span[0] >= end) for start, end in occupied):
                continue
            occupied.append(span)
            negated = _negated(text, idx)
            polarity = sign * (-1 if negated else 1)
            weight = WEIGHTS.get(word, 1.0)
            hits.append((idx, ("!" if negated else "") + word, polarity * weight))

    hits.sort(key=lambda item: item[0])
    pos = sum(value for _, _, value in hits if value > 0)
    neg = sum(-value for _, _, value in hits if value < 0)
    return pos, neg, [label for _, label, _ in hits]


def _occurrences(text: str, word: str) -> list[int]:
    out, start = [], 0
    while True:
        idx = text.find(word, start)
        if idx < 0:
            return out
        out.append(idx)
        start = idx + max(1, len(word))


def _in_word_exception(text: str, index: int, word: str) -> bool:
    for exception in WORD_EXCEPTIONS.get(word, ()):
        start = text.find(exception, max(0, index - len(exception) + 1))
        while start != -1 and start <= index:
            if index + len(word) <= start + len(exception):
                return True
            start = text.find(exception, start + 1)
    return False


def _negated(text: str, index: int) -> bool:
    """Whether the cue at `index` is negated inside its current clause.

    The window is deliberately small, but it is not a blind two-character
    substring search.  Negators inside "特别", "差不多", "别人" and similar
    words are ignored, and double negation is left non-negated.
    """
    context = text[max(0, index - 6):index]
    context = re.split(r"[，。！？；,.!?;]", context)[-1]
    if not context:
        return False
    if any(context.endswith(pattern) for pattern in DOUBLE_NEGATIONS):
        return False
    # The exception may contain the negator at any position: "别人" starts
    # with 别, "差不多" contains 不 in the middle.  Check the whole clause
    # ending before looking at individual negator positions.
    if any(context.endswith(exception) for exception in NEGATION_EXCEPTIONS):
        return False
    for negator in NEGATORS:
        pos = context.rfind(negator)
        if pos < 0:
            continue
        prefix = context[:pos + 1]
        if any(prefix.endswith(exception) for exception in NEGATION_EXCEPTIONS):
            continue
        if any(context.startswith(exception, pos) for exception in NEGATION_EXCEPTIONS):
            continue
        if any(
            context[max(0, pos - len(exception) + 1):pos + 1] == exception
            for exception in NEGATION_EXCEPTIONS
        ):
            continue
        return True
    return False


def _cue_match_negated(
    text: str, match: re.Match[str], *, allow_leading_imperative: bool = False
) -> bool:
    """Whether a phrase-level cue is inside a negated statement.

    Prefix negation covers "我们没有分手" and "我不觉得你笨".  The span check
    covers "项目没有黄了" and "这个方案不是不行".  "别烦我" is an imperative,
    not a negated statement, so its leading 别 is allowed.
    """
    if _negated(text, match.start()):
        return True
    span = match.group(0)
    if allow_leading_imperative and span.startswith("别"):
        return False
    return any(negator in span[:-2] for negator in NEGATORS)


def _has_other_agent(text: str) -> bool:
    """True when an explicit other agent is present.

    "其他" and "吉他" contain the character "他" but are not other agents.
    """
    for word in OTHER_AGENTS:
        for index in _occurrences(text, word):
            if word == "他" and index > 0 and text[index - 1] in {"其", "吉"}:
                continue
            return True
    return False


def _has_unnegated(text: str, words: tuple[str, ...]) -> bool:
    """True when at least one cue occurs outside a negation window."""
    return any(
        not _negated(text, index)
        for word in words
        for index in _occurrences(text, word)
    )
