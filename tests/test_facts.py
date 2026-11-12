from thymos.facts import FactStore, extract_facts


def test_extracts_high_value_user_facts():
    text = (
        "我叫林墨，住在杭州。我养了一只猫，叫豆豆，是橘猫。"
        "我喝咖啡不加糖，早上一般喝美式。"
        "我在做一个叫「潮汐」的项目，4 月 15 日前要交付。"
        "我对花生过敏。"
    )
    facts = dict(extract_facts(text))
    assert facts["name"] == "林墨"
    assert facts["city"] == "杭州"
    assert facts["pet_name"] == "豆豆"
    assert facts["pet_breed"] == "橘猫"
    assert facts["coffee_sugar"] == "不加糖"
    assert facts["coffee_type"] == "美式"
    assert facts["project"] == "潮汐"
    assert facts["deadline"] == "4 月 15 日"
    assert facts["allergy"] == "花生"


def test_questions_are_not_new_facts():
    assert extract_facts("你还记得我叫什么、住在哪吗？") == []
    assert extract_facts("我的猫叫什么？是什么品种？") == []


def test_new_value_closes_old_interval_and_history_is_explicit():
    store = FactStore()
    store.observe("我住在杭州。", 1000)
    store.observe("搬家后地址是上海市徐汇区。", 2000)
    current = store.render("我现在住在哪里？", at=2000, max_chars=500)
    assert "上海" in current
    assert "徐汇" in current
    assert "杭州" not in current

    history = store.render("你还记得我搬家前的城市吗？", at=2000, max_chars=500)
    assert "杭州" in history
    assert "历史" in history


def test_move_intention_does_not_change_current_city():
    store = FactStore()
    store.observe("我住在杭州。", 1000)
    store.observe("我最近在考虑搬去上海，还在犹豫。", 2000)
    store.observe("如果搬去上海，离阿岚就远了。", 2100)
    assert store.current("user", "city", 2200).value == "杭州"

    store.observe("我决定搬去上海了，下周搬。", 2300)
    assert store.current("user", "city", 2300).value == "上海"


def test_sleep_update_does_not_show_old_value_as_current():
    store = FactStore()
    store.observe("我一般晚上十一点前睡。", 1000)
    store.observe("我最近睡眠不太好，晚上总是十二点后才睡。", 2000)
    current = store.render("我现在一般几点睡？", at=2000, max_chars=500)
    assert "十二点" in current
    assert "十一点" not in current
    assert "十一点" in store.render("以前几点睡？", at=2000, max_chars=500)


def test_fact_store_round_trip():
    store = FactStore()
    store.observe("我叫林墨，住在杭州。", 1000)
    restored = FactStore.from_list(store.to_list())
    assert restored.current("user", "name", 1000).value == "林墨"
    assert restored.current("user", "city", 1000).value == "杭州"
