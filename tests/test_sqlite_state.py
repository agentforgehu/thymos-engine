import json
import sqlite3
import threading

import pytest

from thymos import AgentMemory, Profile, Thymos, ThymosConfig
from thymos.sqlite_store import SQLiteMemoryStore, migrate_json
from thymos.types import MemoryItem

START = 1770000000.0
PROFILE = Profile(name="林墨", location_name="杭州", timezone="Asia/Shanghai")


def _agent(path):
    return AgentMemory(
        str(path),
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=7),
        now=START,
    )


def test_sqlite_state_roundtrip_keeps_memories_and_mood(tmp_path):
    db = tmp_path / "agent.sqlite"
    memory = _agent(db)
    memory.observe("我叫林墨，住在杭州。", min_importance=0)
    memory.observe("项目黄了，我很难过。", importance=0.8)
    before = memory.snapshot()["affect"]["mood"]["valence"]
    memory.save()
    assert db.exists()
    assert not (tmp_path / "agent.json").exists()
    assert memory.mind.store.schema_version == 1

    state = memory.mind.store.load_state()
    assert state is not None
    assert state["memory"] == []
    memory.mind.store.close()

    reopened = _agent(db)
    assert len(reopened.mind.store) == 2
    assert "林墨" in reopened.context("我叫什么？", max_chars=500)
    assert reopened.snapshot()["affect"]["mood"]["valence"] == pytest.approx(before)
    reopened.mind.store.close()


def test_sqlite_state_survives_process_style_reopen(tmp_path):
    db = tmp_path / "agent.db"
    memory = _agent(db)
    item = memory.observe("我对花生过敏。", min_importance=0)
    assert item is not None
    memory.save()
    memory_id = item.id
    memory.mind.store.close()

    reopened = _agent(db)
    assert reopened.mind.store.get(memory_id) is not None
    assert "花生" in reopened.context("我过敏什么？", max_chars=500)
    reopened.mind.store.close()


def test_migrate_json_to_sqlite_and_keep_facts(tmp_path):
    source = tmp_path / "old.json"
    old = Thymos(PROFILE, ThymosConfig(offline=True, seed=7), state_path=str(source), now=START)
    old.perceive("我叫林墨，住在杭州。", importance=0.8)
    old.perceive("我对花生过敏。", importance=0.8)
    old.save()
    assert source.exists()

    target = tmp_path / "new.sqlite"
    store = migrate_json(source, target)
    assert len(store) == 2
    state = store.load_state()
    assert state is not None
    assert state["memory"] == []
    assert state["facts"]
    store.close()

    reopened = _agent(target)
    assert len(reopened.mind.store) == 2
    assert "花生" in reopened.context("我过敏什么？", max_chars=500)
    reopened.mind.store.close()


def test_migrate_refuses_to_overwrite_existing_engine_state(tmp_path):
    source = tmp_path / "old.json"
    old = Thymos(PROFILE, ThymosConfig(offline=True, seed=7), state_path=str(source), now=START)
    old.perceive("测试", importance=0.5)
    old.save()
    target = tmp_path / "new.sqlite"
    store = migrate_json(source, target)
    store.close()

    with pytest.raises(FileExistsError):
        migrate_json(source, target)


def test_sqlite_flush_persists_review_mutations(tmp_path):
    db = tmp_path / "memory.sqlite"
    store = SQLiteMemoryStore(db)
    item = MemoryItem(content="测试记忆", created_at=START, last_access=START)
    store.add(item)
    item.stability = 42.0
    item.salience = 0.9
    store.flush()
    store.close()

    reopened = SQLiteMemoryStore(db)
    loaded = reopened.get(item.id)
    assert loaded is not None
    assert loaded.stability == 42.0
    assert loaded.salience == 0.9
    reopened.close()


def test_sqlite_schema_version_is_explicit_and_rejects_newer(tmp_path):
    db = tmp_path / "version.sqlite"
    store = SQLiteMemoryStore(db)
    assert store.schema_version == 1
    store.close()

    conn = sqlite3.connect(db)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError):
        SQLiteMemoryStore(db)


def test_sqlite_state_version_is_checked(tmp_path):
    db = tmp_path / "state_version.sqlite"
    memory = _agent(db)
    memory.observe("测试", min_importance=0)
    memory.save()
    store = memory.mind.store
    state = store.load_state()
    assert state is not None
    state["version"] = 99
    store.save_state(state)
    store.close()

    with pytest.raises(ValueError, match="newer than supported"):
        _agent(db)


def test_sqlite_state_without_engine_state_is_a_clear_error(tmp_path):
    db = tmp_path / "empty.sqlite"
    store = SQLiteMemoryStore(db)
    store.close()
    with pytest.raises(ValueError, match="engine_state"):
        _agent(db)


def test_sqlite_store_path_must_match_state_path(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "other.sqlite")
    engine = Thymos(
        PROFILE,
        ThymosConfig(offline=True, seed=7),
        state_path=str(tmp_path / "state.sqlite"),
        store=store,
        now=START,
    )
    with pytest.raises(ValueError, match="does not match"):
        engine.save()
    store.close()


def test_sqlite_concurrent_writers_do_not_lose_rows(tmp_path):
    db = tmp_path / "concurrent.sqlite"
    stores = [SQLiteMemoryStore(db), SQLiteMemoryStore(db)]

    def writer(store, prefix):
        for i in range(20):
            store.add(
                MemoryItem(
                    content=f"{prefix}-{i}",
                    created_at=START,
                    last_access=START,
                )
            )

    threads = [
        threading.Thread(target=writer, args=(stores[0], "a")),
        threading.Thread(target=writer, args=(stores[1], "b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for store in stores:
        store.close()

    reopened = SQLiteMemoryStore(db)
    assert len(reopened) == 40
    reopened.close()


def test_sqlite_state_preserves_events_and_tick_count(tmp_path):
    db = tmp_path / "events.sqlite"
    memory = _agent(db)
    memory.observe("项目黄了。", importance=0.8)
    memory.mind.tick(3600)
    memory.save()
    ticks = memory.mind._tick_count
    memory.mind.store.close()

    reopened = _agent(db)
    assert reopened.mind._tick_count == ticks
    assert reopened.mind.events
    reopened.mind.store.close()
