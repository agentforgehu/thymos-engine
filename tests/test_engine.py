import json

import pytest

from thymos import Profile, Thymos, ThymosConfig
from thymos.forgetting import retrievability
from thymos.types import Appraisal

START = 1770000000.0
PROFILE = Profile(name="测试者", location_name="上海", latitude=31.23, longitude=121.47, timezone="Asia/Shanghai")


@pytest.fixture()
def engine():
    return Thymos(PROFILE, ThymosConfig(offline=True, seed=11), now=START)


def test_perceive_encodes_memory_and_changes_affect(engine):
    item = engine.perceive("项目评审通过了，我很开心", importance=0.9, tags=["工作"])
    assert len(engine.store) == 1
    assert item.valence > 0
    assert item.stability > 0
    assert engine.affect.state.valence > 0
    assert engine.events[-1]["kind"] == "perceive"


def test_recall_returns_relevant_memory(engine):
    engine.perceive("线上服务崩了，客户投诉，压力很大", importance=0.9, tags=["工作"])
    engine.perceive("周末去公园散步，阳光很好", importance=0.5, tags=["生活"])
    hits = engine.recall("客户投诉", top_k=2)
    assert hits
    assert "客户" in hits[0].memory.content


def test_recall_updates_access_statistics(engine):
    item = engine.perceive("可被回忆的事件", tags=["测试"])
    hits = engine.recall("可被回忆", top_k=1)
    assert hits
    assert item.reps == 1
    assert len(item.access_times) == 2


def test_render_context_is_plain_text_and_disclaimed(engine):
    engine.perceive("用户喜欢在周末爬山", importance=0.7, tags=["偏好"])
    hits = engine.recall("爬山", top_k=1)
    text = engine.render_context(hits)
    assert "相关记忆" in text
    assert "模拟状态" in text
    assert "不是真实生理感受" in text
    assert "用户喜欢在周末爬山" in text


def test_switching_embedder_reembeds_existing_memories(tmp_path):
    state = tmp_path / "state.json"
    first = Thymos(
        PROFILE,
        ThymosConfig(offline=True, seed=11),
        state_path=str(state),
        now=START,
        embedder=lambda text: [1.0, 0.0],
    )
    first.perceive("用户喜欢在周末爬山", importance=0.8, tags=["偏好"])
    first.save()

    second = Thymos(
        PROFILE,
        ThymosConfig(offline=True, seed=11),
        state_path=str(state),
        now=START + 3600,
        embedder=lambda text: [1.0, 0.0, 0.0],
    )
    hits = second.recall("爬山", top_k=1)
    assert hits
    assert all(len(item.embedding) == 3 for item in second.store)
    assert any(event["kind"] == "reembed" for event in second.events)


def test_forgetting_reduces_retrievability(engine):
    item = engine.perceive("会被遗忘的事件", importance=0.3)
    fresh = retrievability(item.age_days(engine.now), item.stability)
    later = retrievability(item.age_days(engine.now + 120 * 86400), item.stability)
    assert fresh > later


def test_tick_advances_time_and_state(engine):
    engine.perceive("被批评了，有点难过", importance=0.8)
    before = engine.now
    snap = engine.tick(24 * 3600)
    assert engine.now == before + 24 * 3600
    assert snap["affect"]["stress"] >= 0
    assert snap["environment"]["source"] == "synthetic"


def test_consolidation_strengthens_and_abstracts(engine):
    for i in range(3):
        engine.perceive(f"第{i + 1}次项目复盘，学到很多", importance=0.8, tags=["工作"])
    # Retrieval is competitive and would keep reinforcing the strongest trace;
    # set the minimum rehearsal count directly to test the abstraction step.
    for item in engine.store.all():
        item.reps = 2
    before = [m.stability for m in engine.store.all()]
    result = engine.consolidate(7.5)
    after = [m.stability for m in engine.store.all() if m.kind == "episodic"]
    assert result["replayed"] >= 1
    assert max(after) > min(before)
    assert result["schemas_created"] >= 1
    assert any(m.kind == "semantic" for m in engine.store.all())


def test_reappraisal_is_stronger_inside_labile_window(engine):
    item = engine.perceive("被批评了", importance=0.7)
    engine.recall("被批评", top_k=1)
    engine.reappraise(item.id, new_valence=0.5)
    strong = item.valence
    assert strong > 0.0

    fresh = engine.perceive("另一件被批评的事", importance=0.7)
    engine.reappraise(fresh.id, new_valence=0.5)
    assert fresh.valence < strong


def test_persistence_roundtrip(tmp_path, engine):
    engine.perceive("会被保存的记忆", importance=0.7, tags=["测试"])
    engine.tick(3600)
    path = tmp_path / "state.json"
    engine.save(str(path))
    assert path.exists()

    loaded = Thymos(state_path=str(path), offline=True)
    assert len(loaded.store) == len(engine.store)
    assert loaded.affect.state.mood_valence == pytest.approx(engine.affect.state.mood_valence, abs=1e-6)
    assert loaded.profile.location_name == "上海"


def test_snapshot_is_json_serializable(engine):
    engine.perceive("快照测试", importance=0.4)
    payload = engine.snapshot()
    text = json.dumps(payload, ensure_ascii=False)
    assert "memory" in payload
    assert "endocrine" in payload
    assert len(text) > 100


def test_runs_are_deterministic_with_same_seed():
    a = Thymos(PROFILE, ThymosConfig(offline=True, seed=3), now=START)
    b = Thymos(PROFILE, ThymosConfig(offline=True, seed=3), now=START)
    for e in (a, b):
        e.perceive("同样的输入", importance=0.6)
        e.tick(7200)
    assert a.affect.state.mood_valence == pytest.approx(b.affect.state.mood_valence)
    assert a.snapshot()["environment"]["temp_c"] == b.snapshot()["environment"]["temp_c"]


def test_explicit_appraisal_is_respected(engine):
    item = engine.perceive(
        "中性描述",
        appraisal=Appraisal(desirability=-0.9, stakes=0.9, control=0.1, certainty=0.3),
        importance=0.9,
    )
    assert item.valence < -0.5
    assert item.arousal > 0.4
