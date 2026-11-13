from thymos.policy import classify_intent


def test_third_person_fact_questions_are_factual():
    for query in ("他住哪？", "他的项目叫什么？", "他叫什么？", "他做什么工作？"):
        assert classify_intent(query).label == "factual", query


def test_fact_and_emotion_in_one_turn_is_mixed():
    assert classify_intent("我今天有点累，他住哪？").label == "mixed"


def test_colloquial_insult_is_affective_intent():
    assert classify_intent("你怎么这么笨，闭嘴").label == "affective"


def test_colloquial_praise_is_affective_intent():
    assert classify_intent("谢谢你，辛苦了").label == "affective"
    assert classify_intent("你真好").label == "affective"


def test_loss_event_is_not_factual_only():
    assert classify_intent("项目黄了").label == "mixed"
    assert classify_intent("我们分手了").label == "affective"


def test_technical_query_stays_technical():
    assert classify_intent("这个 API 报错怎么办").label == "technical"
