import json

import pytest

from thymos import AgentMemory, MemoryBackend, Profile, ThymosConfig, classify_intent
from thymos.control import MemoryController
from thymos.mcp_server import MCPServer
from thymos.memory import MemoryStore
from thymos.policy import AffectPolicy
from thymos.sqlite_store import SQLiteMemoryStore
from thymos.tokens import estimate_tokens
from thymos.types import MemoryItem

START = 1770000000.0
PROFILE = Profile(name="林墨", location_name="杭州", timezone="Asia/Shanghai")


@pytest.fixture()
def memory(tmp_path):
    return AgentMemory(
        str(tmp_path / "control.json"),
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=7),
        now=START,
    )


def test_intent_classification_covers_product_paths():
    assert classify_intent("我叫什么？").label == "factual"
    assert classify_intent("这段 Python 代码报错怎么修").label == "technical"
    assert classify_intent("今天有点难过，陪我聊聊").label == "affective"
    assert classify_intent("你还记得我搬家前的城市吗？").label == "history"
    assert classify_intent("在吗").label == "social"
    assert classify_intent("最近压力很大，工作上的事该怎么处理").label == "mixed"


def test_affect_policy_suppresses_mood_for_facts_and_keeps_it_for_support():
    policy = AffectPolicy()
    factual = policy.decide(classify_intent("我叫什么？"))
    affective = policy.decide(classify_intent("今天很难过，陪我聊聊"))
    technical = policy.decide(classify_intent("API 报错怎么修"))

    assert factual.mood_weight == 0.0
    assert factual.include_state is False
    assert technical.emotion_weight == 0.0
    assert affective.mood_weight > 0.0
    assert affective.include_state is True


def test_plan_contains_facts_audit_and_no_state_for_factual_query(memory):
    memory.observe("我叫林墨，住在杭州。", min_importance=0)
    plan = memory.plan("我叫什么？", max_chars=500)

    assert plan["intent"]["label"] == "factual"
    assert "当前事实" in plan["context"]
    assert "我叫什么" not in plan["context"]  # query itself is not a memory
    assert "当前模拟状态" not in plan["context"]
    assert plan["affect"]["mood_weight"] == 0.0
    assert len(plan["context"]) <= 500
    assert plan["audit"]["budget"]["chars"] == 500
    names = [op["name"] for op in plan["operators"]]
    assert names == ["fact_lookup", "episode_recall", "graph_traversal", "summary"]


def test_affective_query_exposes_state_and_uses_mood_gate(memory):
    memory.observe("最近工作压力很大，有点焦虑。", min_importance=0)
    plan = memory.plan("最近压力很大，陪我聊聊", max_chars=600)

    assert plan["intent"]["label"] == "affective"
    assert plan["affect"]["mood_weight"] > 0.0
    assert "当前模拟状态" in plan["context"]
    assert "压力" in plan["context"]


def test_plan_respects_character_and_token_budgets(memory):
    for text in [
        "我叫林墨，住在杭州。",
        "我对花生过敏。",
        "我养了一只猫，叫豆豆。",
        "我在做一个叫潮汐的项目。",
    ]:
        memory.observe(text, min_importance=0)

    plan = memory.plan("我叫什么，对什么过敏？", max_chars=420, max_tokens=110)
    assert len(plan["context"]) <= 420
    assert plan["audit"]["tokens"] <= 110
    assert estimate_tokens(plan["context"]) <= 110


def test_graph_operator_recovers_associated_memory(memory):
    a = memory.mind.perceive("项目评审通过了，团队很高兴", importance=0.8, tags=["工作"])
    b = memory.mind.perceive("评审后大家一起吃火锅庆祝", importance=0.7, tags=["工作"])
    memory.mind.store.link(a.id, b.id, 0.95)

    plan = memory.mind.plan_context("评审", top_k=1, max_chars=800)
    graph = next(op for op in plan.audit.operators if op.name == "graph_traversal")
    assert graph.selected >= 1
    assert b.id in plan.audit.selected_ids


def test_summary_operator_uses_semantic_schema(memory):
    schema = MemoryItem(
        content="关于「工作」的反复经验：多次项目交付，整体感受偏正面。",
        kind="semantic",
        tags=["工作", "schema"],
        created_at=START,
        last_access=START,
        access_times=[START],
        stability=8.0,
        salience=0.8,
        source="consolidation",
    )
    memory.mind.store.add(schema)

    plan = memory.mind.plan_context("以前工作上的经历", top_k=1, max_chars=800)
    summary = next(op for op in plan.audit.operators if op.name == "summary")
    assert summary.selected >= 1
    assert schema.id in plan.audit.selected_ids


def test_plan_review_false_does_not_change_reps(memory):
    item = memory.observe("我喜欢在周末爬山", min_importance=0)
    assert item is not None
    before = item.reps
    memory.mind.plan_context("爬山", top_k=1, update=False)
    assert item.reps == before


def test_audit_trace_survives_state_round_trip(memory, tmp_path):
    memory.observe("我叫林墨。", min_importance=0)
    memory.plan("我叫什么？", max_chars=400)
    memory.save()

    reloaded = AgentMemory(
        str(tmp_path / "control.json"),
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=7),
        now=START + 30,
    )
    assert reloaded.mind.last_audit["query"] == "我叫什么？"
    assert reloaded.mind.last_audit["intent"]["label"] == "factual"


def test_memory_backend_protocol_accepts_builtin_stores(tmp_path):
    assert isinstance(MemoryStore(), MemoryBackend)
    with SQLiteMemoryStore(tmp_path / "backend.sqlite") as store:
        assert isinstance(store, MemoryBackend)


def test_mcp_memory_plan_tool_returns_audit(memory):
    memory.observe("我叫林墨，对花生过敏。", min_importance=0)
    server = MCPServer(memory)
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "memory_plan",
                "arguments": {"user_text": "我叫什么？", "max_chars": 400},
            },
        }
    )
    assert response is not None
    payload = response["result"]
    assert payload["isError"] is False
    data = json.loads(payload["content"][0]["text"])
    assert data["intent"]["label"] == "factual"
    assert data["affect"]["mood_weight"] == 0.0
    assert data["context"]
    assert data["audit"]["selected_ids"]


def test_controller_can_be_constructed_directly(memory):
    controller = MemoryController(memory.mind)
    plan = controller.context("在吗", max_chars=300)
    assert isinstance(plan, str)
    assert "相关记忆" in plan
