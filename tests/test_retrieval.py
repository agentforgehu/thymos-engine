import pytest

from thymos.config import Profile, ThymosConfig
from thymos.embedding import embed
from thymos.forgetting import retrievability
from thymos.memory import MemoryStore
from thymos.retrieval import actr_base_activation, context_similarity, retrieve, review
from thymos.types import AffectState, MemoryItem


def make_memory(content, *, created, tags=None, valence=0.0, arousal=0.2, stability=5.0, reps=0):
    return MemoryItem(
        content=content,
        embedding=embed(content),
        created_at=created,
        last_access=created,
        access_times=[created] * max(1, reps),
        tags=tags or [],
        valence=valence,
        arousal=arousal,
        stability=stability,
        reps=reps,
    )


def test_actr_prefers_recent_traces():
    now = 1_000_000.0
    recent = actr_base_activation([now - 60], now)
    old = actr_base_activation([now - 86400], now)
    assert recent > old


def test_actr_prefers_frequency():
    now = 1_000_000.0
    once = actr_base_activation([now - 3600], now)
    thrice = actr_base_activation([now - 3600, now - 7200, now - 10800], now)
    assert thrice > once


def test_context_similarity_rewards_matching_tags_and_embedding():
    now = 1_000_000.0
    mem = make_memory("项目评审通过了", created=now - 3600, tags=["工作"])
    env = None
    high, parts = context_similarity(mem, embed("项目评审通过了"), ["工作"], env, 0.0)
    low, _ = context_similarity(mem, embed("今天下雨"), ["天气"], env, 0.0)
    assert high > low
    assert parts["semantic"] > 0.5


def test_retrieval_ranks_semantic_match_first():
    now = 1_000_000.0
    store = MemoryStore(
        [
            make_memory("项目评审通过了，大家很开心", created=now - 3600, tags=["工作"]),
            make_memory("晚上下雨，取消了散步", created=now - 3600, tags=["天气"]),
            make_memory("朋友寄来礼物", created=now - 3600, tags=["朋友"]),
        ]
    )
    hits = retrieve(store, now=now, config=ThymosConfig(), query_embedding=embed("项目评审"), query_tags=["工作"])
    assert hits[0].memory.tags == ["工作"]


def test_retrieval_excludes_archived():
    now = 1_000_000.0
    mem = make_memory("归档的记忆", created=now - 100)
    mem.state = "archived"
    store = MemoryStore([mem])
    assert retrieve(store, now=now, config=ThymosConfig(), query_embedding=embed("归档")) == []


def test_review_increases_reps_and_stability():
    now = 1_000_000.0
    mem = make_memory("可复习的记忆", created=now - 86400 * 3, stability=1.0)
    store = MemoryStore([mem])
    hits = retrieve(store, now=now, config=ThymosConfig(), query_embedding=embed("记忆"))
    assert hits
    before = mem.stability
    review(hits[0], now=now, quality=0.9, config=ThymosConfig())
    assert mem.reps == 1
    assert mem.stability > before
    assert mem.labile_until == now + ThymosConfig().reconsolidation_window_min * 60


def test_retrievability_drops_with_age():
    now = 1_000_000.0
    young = make_memory("新记忆", created=now - 3600, stability=1.0)
    old = make_memory("旧记忆", created=now - 86400 * 30, stability=1.0)
    assert retrievability(young.age_days(now), young.stability) > retrievability(old.age_days(now), old.stability)


def test_mood_congruence_can_bias_selection():
    now = 1_000_000.0
    negative = make_memory("很糟糕的一天", created=now - 7200, valence=-0.8, arousal=0.7)
    positive = make_memory("很开心的一天", created=now - 7200, valence=0.8, arousal=0.7)
    store = MemoryStore([negative, positive])
    sad = retrieve(
        store,
        now=now,
        config=ThymosConfig(),
        query_embedding=embed("一天"),
        affect=AffectState(mood_valence=-0.6),
    )
    assert sad[0].memory.valence < 0
