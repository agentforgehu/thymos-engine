from thymos.facts import FactStore, extract_facts


def _store() -> FactStore:
    store = FactStore()
    at = 1000.0
    for text in (
        "我叫林墨，住在杭州。",
        "我养了一只猫，叫豆豆，是橘猫。",
        "我喝咖啡不加糖，早上一般喝美式。",
        "我在做一个叫「潮汐」的项目，4 月 15 日前要交付。",
        "我对花生和芒果过敏。",
        "我一般晚上十一点前睡，早上七点起。",
        "我喜欢吃火锅，昨天去看了牙。",
    ):
        store.observe(text, at)
        at += 60
    return store


def test_render_only_shows_predicates_relevant_to_query():
    store = _store()
    text = store.render("我对什么过敏？", at=2000, max_chars=300)
    assert "花生" in text and "芒果" in text
    assert "潮汐" not in text
    assert "咖啡" not in text
    assert len(text) <= 300


def test_general_summary_is_compact_not_a_full_dump():
    store = _store()
    text = store.render("总结一下你记得的关于我的事。", at=2000, max_chars=300)
    assert "林墨" in text
    assert "杭州" in text
    assert len(text) <= 300
    # It may include a core profile, but not every predicate in the store.
    assert sum(1 for line in text.splitlines() if line.startswith("- ")) <= 6


def test_fact_budget_is_hard_even_for_many_facts():
    store = _store()
    text = store.render("", at=2000, max_chars=120)
    assert len(text) <= 120


def test_food_and_health_events_are_extracted():
    facts = dict(extract_facts("我喜欢吃火锅，昨天去看了牙。"))
    assert facts["food_like"] == "火锅"
    assert facts["health_event"] == "看牙"
    assert "food_event" not in facts
    event = dict(extract_facts("今天吃火锅，晚上去看牙。"))
    assert event["food_event"] == "火锅"
    assert event["health_event"] == "看牙"


def test_food_question_does_not_become_a_preference():
    assert extract_facts("我喜欢吃火锅吗？") == []


def test_small_budget_fact_block_is_query_specific():
    store = FactStore()
    at = 1000.0
    for offset, text in enumerate(
        ("我叫林墨。", "我住在杭州。", "我在做一个叫「潮汐」的项目。", "我是软件工程师。")
    ):
        store.observe(text, at + offset * 10)

    name = store.render("他叫什么？", at=1100, max_chars=300)
    project = store.render("他的项目叫什么？", at=1100, max_chars=300)
    job = store.render("他做什么工作？", at=1100, max_chars=300)

    assert "林墨" in name
    assert "潮汐" not in name and "软件工程师" not in name
    assert "潮汐" in project
    assert "软件工程师" not in project and "杭州" not in project
    assert "软件工程师" in job
    assert "潮汐" not in job


def test_unmatched_query_gets_one_line_digest_not_full_dump():
    store = _store()
    text = store.render("今天天气怎么样？", at=2000, max_chars=300)
    assert "其他已知" in text
    assert "潮汐" not in text
    assert "花生" not in text


def test_chinese_third_person_and_event_patterns():
    assert dict(extract_facts("我从恩施来上海，做软件。"))["city"] == "上海"
    assert dict(extract_facts("我从恩施来上海，做软件。"))["hometown"] == "恩施"
    assert dict(extract_facts("我从恩施来上海，做软件。"))["job"] == "软件"
    assert dict(extract_facts("他的项目叫 piko。"))["project"] == "piko"
    family = dict(extract_facts("我妹下个月结婚。"))
    assert family["family"] == "妹"
    assert "结婚" in family["family_event"]
    assert dict(extract_facts("香菜不要。"))["food_dislike"] == "香菜"


def test_family_query_does_not_fall_back_to_city():
    store = FactStore()
    store.observe("我住在杭州。", 1000)
    store.observe("我妹下个月结婚。", 1010)
    text = store.render("我妹有什么事？", at=1020, max_chars=300)
    assert "妹" in text and "结婚" in text
    assert "杭州" not in text
