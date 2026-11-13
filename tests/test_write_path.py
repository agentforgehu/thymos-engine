from pathlib import Path

from thymos import AgentMemory, ThymosConfig
from thymos.appraisal import estimate_appraisal
from thymos.write_path import normalize_key


def make_memory(tmp_path: Path, *, embedder=None, config=None) -> AgentMemory:
    cfg = config or ThymosConfig(offline=True, seed=7)
    return AgentMemory(
        str(tmp_path / "state.json"),
        config=cfg,
        embedder=embedder,
    )


def test_exact_duplicate_reinforces_instead_of_storing_twice(tmp_path):
    memory = make_memory(tmp_path)
    first = memory.observe("我叫林墨，住在杭州。", min_importance=0)
    second = memory.observe("我叫林墨，住在杭州。", min_importance=0)

    assert first is not None and second is not None
    assert first.id == second.id
    assert len(memory.mind.store) == 1
    assert first.merged_count == 1
    assert memory.mind.last_write["action"] == "reinforce"
    assert memory.mind.last_write["reason"] == "exact_duplicate"


def test_exact_duplicate_is_deduplicated_across_reload(tmp_path):
    path = tmp_path / "state.json"
    memory = AgentMemory(str(path), config=ThymosConfig(offline=True, seed=7))
    first = memory.observe("我对花生过敏。", min_importance=0)
    memory.save()

    reopened = AgentMemory(str(path), config=ThymosConfig(offline=True, seed=7))
    second = reopened.observe("我对花生过敏。", min_importance=0)

    assert first is not None and second is not None
    assert first.id == second.id
    assert len(reopened.mind.store) == 1
    assert second.merged_count == 1


def test_semantic_paraphrase_merges_when_embedding_is_close(tmp_path):
    # A deterministic stand-in for a real Chinese embedder: the two identity
    # paraphrases share a vector, while the unrelated sentence does not.
    def embedder(text: str) -> list[float]:
        if "林墨" in text or "杭州" in text:
            return [1.0, 0.0, 0.0]
        if "项目" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    memory = make_memory(tmp_path, embedder=embedder)
    first = memory.observe("我叫林墨，住在杭州。", min_importance=0)
    second = memory.observe("我住在杭州，名字是林墨。", min_importance=0)

    assert first is not None and second is not None
    assert first.id == second.id
    assert len(memory.mind.store) == 1
    assert memory.mind.last_write["action"] == "merge"
    assert memory.mind.last_write["similarity"] >= 0.90


def test_conflicting_numbers_are_not_merged(tmp_path):
    def embedder(text: str) -> list[float]:
        return [1.0, 0.0]

    memory = make_memory(tmp_path, embedder=embedder)
    first = memory.observe("项目 4 月 15 日交付。", min_importance=0)
    second = memory.observe("项目 5 月 1 日交付。", min_importance=0)

    assert first is not None and second is not None
    assert first.id != second.id
    assert len(memory.mind.store) == 2
    assert memory.mind.last_write["action"] == "store"


def test_low_importance_is_skipped_and_audited(tmp_path):
    memory = make_memory(tmp_path)
    item = memory.observe("今天星期三，外面在下雨。", min_importance=0.8)

    assert item is None
    assert len(memory.mind.store) == 0
    assert memory.mind.last_write["action"] == "skip"
    assert memory.mind.last_write["reason"] == "below_min_importance"


def test_merge_keeps_fact_provenance_on_surviving_memory(tmp_path):
    memory = make_memory(tmp_path)
    first = memory.observe("我对花生过敏。", min_importance=0)
    second = memory.observe("我对花生过敏。", min_importance=0)

    assert first is not None and second is not None
    facts = memory.facts()
    assert facts
    assert all(fact.memory_id == first.id for fact in facts if fact.predicate == "allergy")


def test_perceive_can_feel_without_storing(tmp_path):
    memory = make_memory(tmp_path)
    before = memory.snapshot()["affect"]["mood"]["valence"]
    result = memory.mind.perceive(
        "你怎么这么笨，闭嘴。",
        appraisal=estimate_appraisal("你怎么这么笨，闭嘴。"),
        store_memory=False,
    )
    after = memory.snapshot()["affect"]["mood"]["valence"]

    assert result is None
    assert len(memory.mind.store) == 0
    assert after != before


def test_write_admission_can_be_disabled(tmp_path):
    cfg = ThymosConfig(offline=True, seed=7, write_admission_enabled=False)
    memory = make_memory(tmp_path, config=cfg)
    memory.observe("重复的一句话。", min_importance=0)
    memory.observe("重复的一句话。", min_importance=0)

    assert len(memory.mind.store) == 2


def test_normalize_key_is_surface_only():
    assert normalize_key(" 项目黄了。 ") == normalize_key("项目黄了")
    assert normalize_key("A 项目！") == "a项目"
