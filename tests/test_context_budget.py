from thymos import AgentMemory, Profile, ThymosConfig

START = 1770000000.0
PROFILE = Profile(name="林墨", location_name="杭州", timezone="Asia/Shanghai")


def _memory(tmp_path):
    return AgentMemory(
        str(tmp_path / "budget.json"),
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=7),
        now=START,
    )


def test_context_respects_hard_character_budget(tmp_path):
    memory = _memory(tmp_path)
    for text in [
        "我叫林墨，住在杭州。",
        "我养了一只猫，叫豆豆，是橘猫。",
        "我喝咖啡不加糖，早上一般喝美式。",
        "我在做一个叫「潮汐」的项目，4 月 15 日前要交付。",
        "我对花生过敏。",
        "我一般晚上十一点前睡，早上七点起。",
    ]:
        memory.observe(text, min_importance=0)
        memory.mind.tick(3600)

    context = memory.context("我叫什么？", top_k=5, max_chars=600)
    assert len(context) <= 600
    assert "林墨" in context
    assert "当前事实" in context


def test_budget_keeps_current_fact_and_drops_stale_episode(tmp_path):
    memory = _memory(tmp_path)
    for text in [
        "我住在杭州。",
        "我决定搬去上海了。",
        "搬家后地址是上海市徐汇区。",
        "我一般晚上十一点前睡。",
        "我最近睡眠不太好，晚上总是十二点后才睡。",
    ]:
        memory.observe(text, min_importance=0)
        memory.mind.tick(3600)

    current = memory.context("我现在住在哪里？", max_chars=600)
    assert "上海" in current
    assert "徐汇" in current
    assert "杭州" not in current
    assert "十一点" not in current

    history = memory.context("你还记得我搬家前的城市吗？", max_chars=600)
    assert "杭州" in history
    assert "历史" in history


def test_facts_survive_state_round_trip(tmp_path):
    path = str(tmp_path / "persist.json")
    memory = AgentMemory(
        path,
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=7),
        now=START,
    )
    memory.observe("我叫林墨，住在杭州。", min_importance=0)
    memory.save()

    reloaded = AgentMemory(
        path,
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=7),
        now=START + 60,
    )
    context = reloaded.context("我叫什么？", max_chars=600)
    assert "林墨" in context
    assert reloaded.mind.facts.current("user", "name", START + 60).value == "林墨"
