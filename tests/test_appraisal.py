import pytest

from thymos.appraisal import appraise, estimate_appraisal
from thymos.types import Appraisal


def test_positive_event_maps_to_joy_or_contentment():
    e = appraise(Appraisal(desirability=0.7, certainty=0.8, stakes=0.4, goal_relevance=0.8))
    assert e.label in {"joy", "contentment", "excitement"}
    assert e.valence > 0.4


def test_low_certainty_positive_is_hope():
    e = appraise(Appraisal(desirability=0.6, certainty=0.3, stakes=0.6))
    assert e.label == "hope"


def test_other_agency_high_control_negative_is_anger():
    e = appraise(
        Appraisal(desirability=-0.7, certainty=0.8, control=0.7, agency="other", stakes=0.7)
    )
    assert e.label == "anger"


def test_uncertain_uncontrollable_negative_is_anxiety_or_fear():
    e = appraise(
        Appraisal(desirability=-0.6, certainty=0.3, control=0.2, agency="circumstance", stakes=0.8)
    )
    assert e.label in {"anxiety", "fear"}


def test_novel_neutral_event_is_surprise():
    e = appraise(Appraisal(desirability=0.1, novelty=0.95, stakes=0.3))
    assert e.label == "surprise"


def test_neutral_facts_do_not_become_anticipation():
    e = appraise(estimate_appraisal("我叫林墨，住在杭州。"))
    assert e.label == "neutral"


def test_future_decision_is_anticipation_but_uncertain_move_is_anxiety():
    decided = appraise(estimate_appraisal("我决定搬去上海了，下周搬。"))
    uncertain = appraise(estimate_appraisal("我最近在考虑搬去上海，还在犹豫。"))
    assert decided.label == "anticipation"
    assert uncertain.label == "anxiety"


def test_text_negation_flips_valence():
    positive = estimate_appraisal("今天很开心")
    negative = estimate_appraisal("今天不开心")
    assert positive.desirability > 0
    assert negative.desirability < 0


def test_unexpected_good_news_is_positive_surprise():
    a = estimate_appraisal("完全没想到会有礼物，很惊喜")
    assert a.desirability > 0.3
    assert a.novelty >= 0.7


def test_anger_words_produce_other_agency():
    a = estimate_appraisal("老板又改需求，我很生气")
    assert a.desirability < 0
    assert a.agency == "other"
    assert a.control >= 0.6


def test_fear_words_reduce_control_and_certainty():
    a = estimate_appraisal("深夜一个人走回家，有点害怕")
    assert a.control <= 0.3
    assert a.certainty <= 0.4


def test_appraisal_clamps_out_of_range_values():
    a = Appraisal(desirability=9, certainty=-3, control=5, stakes=99).clamped()
    assert a.desirability == 1.0
    assert a.certainty == 0.0
    assert 0.0 <= a.control <= 1.0
    assert a.stakes == 1.0


def test_praise_insult_and_care_have_distinct_labels():
    praise = appraise(estimate_appraisal("同事夸我今天报告讲得好"))
    insult = appraise(estimate_appraisal("客户当众骂我，说方案很烂"))
    care = appraise(estimate_appraisal("我生病了，朋友一直关心我"))

    assert praise.label in {"joy", "trust", "contentment"}
    assert insult.label in {"anger", "disgust", "sadness"}
    assert care.label in {"trust", "joy", "contentment"}
    assert praise.valence > 0 and care.valence > 0
    assert insult.valence < 0
    assert len({praise.label, insult.label, care.label}) >= 2


def test_relevance_is_not_saturated_by_sentence_length():
    fact = estimate_appraisal("我叫林墨，住在杭州。")
    event = estimate_appraisal("同事夸我今天报告讲得好")
    assert fact.goal_relevance < 0.55
    assert event.goal_relevance > fact.goal_relevance


@pytest.mark.parametrize(
    "text",
    [
        "你怎么这么笨，闭嘴",
        "你是不是傻",
        "你真是个废物",
        "你就是个垃圾",
        "蠢死了",
        "脑子有病吧",
        "神经病",
        "滚蛋",
        "滚",
        "少废话",
        "别烦我",
        "要你何用",
        "没用的东西",
        "你算什么东西",
        "谁给你的勇气",
        "菜得抠脚",
        "傻逼",
        "你是不是有病",
        "你就是个笑话",
        "你脑子进水了",
        "别人说我笨",
    ],
)
def test_colloquial_attacks_are_negative(text):
    a = estimate_appraisal(text)
    e = appraise(a)
    assert a.desirability <= -0.30, text
    assert e.valence < -0.20, text
    assert e.label in {"anger", "disgust", "sadness"}, text


@pytest.mark.parametrize(
    "text",
    [
        "今天特别开心",
        "我特别喜欢这个方案",
        "别人夸我今天的报告",
        "差不多完成了",
        "不断进步",
        "我不是不喜欢这个方案",
    ],
)
def test_negator_like_substrings_do_not_flip_sentiment(text):
    a = estimate_appraisal(text)
    assert a.desirability > 0.20, text


def test_plain_negation_still_flips_sentiment():
    assert estimate_appraisal("今天不开心").desirability < 0
    assert estimate_appraisal("我不喜欢这个方案").desirability < 0
    assert estimate_appraisal("别烦我").desirability < 0


@pytest.mark.parametrize(
    "text",
    [
        "我在看垃圾分类", "我在滚面团", "蔬菜很新鲜", "我在积累经验",
        "我在夸克上查资料", "我通过 API 调用", "你讲笑话", "你行不行？",
    ],
)
def test_attack_cues_do_not_fire_on_literal_objects(text):
    a = estimate_appraisal(text)
    assert abs(a.desirability) < 0.12, text
    assert appraise(a).label == "neutral", text


@pytest.mark.parametrize(
    "text",
    [
        "谢谢你，辛苦了", "你真好", "太厉害了", "牛啊", "靠谱", "给力",
        "麻烦你了", "不辛苦", "不麻烦你了", "你帮我大忙了",
    ],
)
def test_colloquial_praise_is_positive(text):
    a = estimate_appraisal(text)
    e = appraise(a)
    assert a.desirability > 0.20, text
    assert e.valence > 0.15, text
    assert e.label in {"joy", "trust", "contentment", "excitement"}, text


@pytest.mark.parametrize(
    "text",
    ["项目黄了", "我被裁了", "我们分手了", "家人生病了", "我亏钱了"],
)
def test_high_impact_losses_are_negative(text):
    a = estimate_appraisal(text)
    e = appraise(a)
    assert a.desirability <= -0.30, text
    assert e.valence < -0.20, text
    assert e.label in {"sadness", "anxiety", "fear"}, text


@pytest.mark.parametrize("text", ["这个方案不行", "你做得不对", "这不对"])
def test_criticism_is_negative(text):
    a = estimate_appraisal(text)
    e = appraise(a)
    assert a.desirability <= -0.20, text
    assert e.valence < -0.10, text


@pytest.mark.parametrize("text", ["呵呵", "卧槽", "他妈的", "你可真厉害啊"])
def test_ambiguous_phrases_are_not_forced_into_polarity(text):
    a = estimate_appraisal(text)
    e = appraise(a)
    assert abs(a.desirability) <= 0.20, text
    assert e.label == "neutral", text
    assert "ambiguous:" in a.note, text


@pytest.mark.parametrize("text", ["我真笨", "我太蠢了", "我是个废物"])
def test_self_deprecation_is_negative_but_self_attributed(text):
    a = estimate_appraisal(text)
    e = appraise(a)
    assert a.desirability < -0.20, text
    assert a.agency == "self", text
    assert e.label in {"sadness", "anxiety"}, text


def test_other_agent_detection_ignores_qita():
    assert estimate_appraisal("其他事情都还好").agency != "other"


@pytest.mark.parametrize(
    "text",
    ["我不觉得你笨", "我们没有分手", "我没有被裁", "项目没有黄了", "这个方案不是不行"],
)
def test_negated_cues_do_not_create_a_strong_negative(text):
    a = estimate_appraisal(text)
    assert a.desirability > -0.20, text


@pytest.mark.parametrize(
    "text", ["没想到项目黄了", "没料到他会被裁", "没成想项目黄了"]
)
def test_unexpected_loss_is_still_a_loss(text):
    a = estimate_appraisal(text)
    e = appraise(a)
    assert a.desirability <= -0.30, text
    assert e.valence < -0.20, text
