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
    # A future move does not rewrite the current city before it happens.
    assert store.current("user", "city", 2300).value == "杭州"
    assert store.current("user", "city", 2300 + 7 * 86400).value == "上海"


def test_multi_value_allergy_and_correction():
    store = FactStore()
    store.observe("我对花生和芒果过敏。", 1000)
    values = [f.value for f in store.active_for("user", "allergy", 1000)]
    assert values == ["花生", "芒果"]

    store.correct("user", "allergy", "腰果", 2000, old_value="花生")
    values = [f.value for f in store.active_for("user", "allergy", 2000)]
    assert "腰果" in values
    assert "花生" not in values
    assert "芒果" in values


def test_statement_with_question_tag_still_extracts_fact():
    facts = dict(extract_facts("我住在杭州，你知道吗？"))
    assert facts["city"] == "杭州"


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


def test_negated_location_and_job_are_not_extracted():
    assert extract_facts("我不住在上海") == []
    assert extract_facts("我不在上海工作") == []
    assert extract_facts("我不做软件") == []


def test_negated_allergy_is_not_extracted_but_dislike_is():
    assert extract_facts("我对花生不过敏") == []
    assert dict(extract_facts("我不吃香菜"))["food_dislike"] == "香菜"


def test_project_rename_and_deadline_update():
    store = FactStore()
    store.observe("我的项目叫「潮汐」，4 月 15 日前要交付。", 1000)
    store.observe("我的项目改名「潮汐 2」，截止 5 月 1 日。", 2000)

    assert store.current("user", "deadline", 2000).value == "5 月 1 日"
    assert "潮汐 2" in store.render("我的项目叫什么？", at=2000, max_chars=500)
    assert "5 月 1 日" in store.render("我的项目什么时候交付？", at=2000, max_chars=500)


def test_old_deadline_restatement_does_not_override_new_value():
    store = FactStore()
    store.observe("我的项目叫「潮汐」，4 月 15 日前要交付。", 1000)
    store.observe("我的项目改名「潮汐 2」，截止 5 月 1 日。", 2000)
    store.observe("项目 4 月 15 日交付。", 3000)

    assert store.current("user", "deadline", 3000).value == "5 月 1 日"


def test_pet_relocation_and_sleep_phrase():
    facts = dict(extract_facts("我把猫豆豆送回恩施给爸妈养了。最近睡眠好多了，十二点睡。"))
    assert facts["pet_location"] == "恩施"
    assert facts["sleep"] == "十二点"


def test_pet_name_from_name_before_noun_and_relocation():
    assert dict(extract_facts("我养了一只叫豆豆的猫。"))["pet_name"] == "豆豆"
    assert dict(extract_facts("我的猫豆豆最近不爱吃饭。"))["pet_name"] == "豆豆"
    assert dict(extract_facts("我把猫豆豆送回恩施给爸妈养了。"))["pet_name"] == "豆豆"
    assert extract_facts("我的狗叫旺财，很可爱。") == [("pet_name", "旺财")]


def test_district_is_not_stored_as_city():
    facts = dict(extract_facts("我搬到上海了，现在住在徐汇区。"))
    assert facts["city"] == "上海"


def test_specific_remember_query_does_not_dump_profile():
    store = FactStore()
    store.observe("我叫林墨，住在上海，做 AI 产品。", 1000)
    store.observe("我对花生过敏，晚上十一点睡。", 1000)
    rendered = store.render("你还记得我住在哪、做什么工作吗？", at=1000, max_chars=500)
    assert "上海" in rendered
    assert "AI 产品" in rendered
    assert "花生" not in rendered
