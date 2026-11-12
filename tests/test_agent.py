import pytest

from thymos import AgentMemory, Profile, ThymosConfig
from thymos.agent import estimate_importance

START = 1770000000.0
PROFILE = Profile(name="测试者", location_name="上海", timezone="Asia/Shanghai")


@pytest.fixture()
def memory(tmp_path):
    return AgentMemory(
        str(tmp_path / "agent.json"),
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=3),
        now=START,
    )


def test_agent_memory_round_trip(memory, tmp_path):
    item = memory.observe("我不吃虾，过敏", importance=0.9)
    assert item is not None

    context = memory.context("我不吃什么", top_k=3)
    assert "不吃虾" in context
    assert "相关记忆" in context

    memory.end_session(7.5)
    assert (tmp_path / "agent.json").exists()

    reloaded = AgentMemory(
        str(tmp_path / "agent.json"),
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=3),
        now=START + 8 * 3600,
    )
    assert "不吃虾" in reloaded.context("虾", top_k=3)


def test_small_talk_is_skipped_by_default(memory):
    assert memory.observe("嗯") is None
    assert len(memory.mind.store) == 0


def test_observe_can_force_small_talk(memory):
    item = memory.observe("嗯", min_importance=0)
    assert item is not None
    assert len(memory.mind.store) == 1


def test_importance_rules_are_transparent():
    assert estimate_importance("答应我周五提醒交材料") >= 0.9
    assert estimate_importance("我喜欢冰美式，不加糖") >= 0.8
    assert estimate_importance("明天面试，有点害怕") >= 0.8
    assert estimate_importance("嗯") <= 0.2


def test_questions_are_not_stored_as_preferences(memory):
    assert estimate_importance("我不吃什么？") <= 0.3
    assert estimate_importance("我面试是哪天？") <= 0.3
    assert memory.observe("我不吃什么？") is None
    assert len(memory.mind.store) == 0


def test_forced_question_storage_does_not_poison_facts(memory):
    # A host may keep every episode with min_importance=0, but a question must
    # still never become a structured fact such as "咖啡糖=加糖".
    item = memory.observe("我喝咖啡加糖吗？", min_importance=0)
    assert item is not None
    assert not any(
        fact.predicate == "coffee_sugar" and "加糖" in fact.value
        for fact in memory.facts()
    )


def test_reminder_questions_are_still_important(memory):
    assert estimate_importance("你能提醒我周五交材料吗？") >= 0.9
    assert memory.observe("你能提醒我周五交材料吗？") is not None


def test_agent_memory_accepts_custom_embedder(tmp_path):
    memory = AgentMemory(
        str(tmp_path / "custom.json"),
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=3),
        now=START,
        embedder=lambda text: [float(len(text)), 1.0],
    )
    item = memory.observe("我喜欢在周末爬山", importance=0.9)
    assert item is not None
    assert len(item.embedding) == 2
    assert "爬山" in memory.context("爬山", top_k=1)
