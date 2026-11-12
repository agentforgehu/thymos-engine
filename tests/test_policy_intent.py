from thymos.policy import classify_intent


def test_third_person_fact_questions_are_factual():
    for query in ("他住哪？", "他的项目叫什么？", "他叫什么？", "他做什么工作？"):
        assert classify_intent(query).label == "factual", query


def test_fact_and_emotion_in_one_turn_is_mixed():
    assert classify_intent("我今天有点累，他住哪？").label == "mixed"
