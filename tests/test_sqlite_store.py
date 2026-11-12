from thymos import AgentMemory, Profile, ThymosConfig
from thymos.sqlite_store import SQLiteMemoryStore

START = 1770000000.0


def test_sqlite_store_persists_items_and_edges(tmp_path):
    db = tmp_path / "memory.sqlite"
    store = SQLiteMemoryStore(db)
    memory = AgentMemory(
        str(tmp_path / "state.json"),
        profile=Profile(name="林墨", location_name="杭州"),
        config=ThymosConfig(offline=True, seed=7),
        now=START,
        store=store,
    )
    memory.observe("我叫林墨，住在杭州。", min_importance=0)
    memory.observe("我在做一个叫潮汐的项目。", min_importance=0)
    memory.save()
    assert len(memory.mind.store) == 2
    store.close()

    reopened = SQLiteMemoryStore(db)
    assert len(reopened) == 2
    reloaded = AgentMemory(
        str(tmp_path / "state.json"),
        profile=Profile(name="林墨", location_name="杭州"),
        config=ThymosConfig(offline=True, seed=7),
        now=START + 60,
        store=reopened,
    )
    assert len(reloaded.mind.store) == 2
    assert "林墨" in reloaded.context("我叫什么？", max_chars=600)
    reopened.close()
