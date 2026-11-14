import pytest

from thymos import AgentMemory, Event, Profile, ThymosConfig
from thymos.stream import normalize_events


START = 1770000000.0
PROFILE = Profile(name="林墨", location_name="杭州", timezone="Asia/Shanghai")


@pytest.fixture()
def memory(tmp_path):
    return AgentMemory(
        str(tmp_path / "stream.json"),
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=7),
        now=START,
    )


def _events():
    return [
        Event("今天加班到九点，累死了", at=START, tags=["工作"]),
        Event("领导还临时改需求，真的很烦", at=START + 1800, tags=["工作"]),
        Event("不过同事帮我买了饭", at=START + 3600, tags=["同事"]),
        Event("现在好多了", at=START + 5400, tags=["恢复"]),
    ]


def test_stream_builds_window_and_stores_summary_memory(memory):
    window = memory.observe_stream(_events(), window="今晚")

    assert window.event_count == 4
    assert window.user_turns == 4
    assert window.assistant_turns == 0
    assert window.start_at == START
    assert window.end_at == START + 5400
    assert window.duration_minutes == 90.0
    assert "窗口「今晚」" in window.summary
    assert window.memory_id is not None
    assert window.topics[:1] == ["工作"]

    summary = next(item for item in memory.mind.store if item.id == window.memory_id)
    assert summary.kind == "semantic"
    assert "window" in summary.tags
    assert summary.context["window"] == "今晚"
    assert summary.context["event_count"] == 4

    snapshot = memory.snapshot()
    assert snapshot["last_window"]["memory_id"] == window.memory_id
    assert snapshot["last_window"]["summary"] == window.summary


def test_stream_summary_is_retrievable_as_an_evening_context(memory):
    memory.observe_stream(_events(), window="今晚")
    context = memory.context("今晚怎么样", top_k=4, max_chars=800)
    assert "窗口「今晚」" in context
    assert "情绪标签" in context


def test_stream_uses_monotonic_timestamps_and_real_gaps(memory):
    events = normalize_events(
        [
            "第一条",
            Event("第二条"),
            {"text": "第三条", "timestamp": START + 7200},
        ],
        now=START,
        step_seconds=60,
    )
    assert [event.at for event in events] == [START, START + 60, START + 7200]

    window = memory.observe_stream(events, window="今晚")
    assert window.duration_minutes == 120.0


def test_stream_assistant_events_are_recorded_but_not_user_memories(memory):
    window = memory.observe_stream(
        [
            Event("我今天有点累", at=START),
            Event("听起来今天挺不容易的", at=START + 60, role="assistant"),
        ],
        window="今晚",
    )
    assert window.event_count == 2
    assert window.user_turns == 1
    assert window.assistant_turns == 1
    # A one-user-turn window is not worth a semantic summary.
    assert window.memory_id is None
    assert all(item.source != "stream" for item in memory.mind.store)


def test_stream_empty_is_rejected(memory):
    with pytest.raises(ValueError):
        memory.observe_stream([], window="今晚")


def test_stream_accepts_plain_strings(memory):
    window = memory.observe_stream(["第一句", "第二句"], window="短窗口")
    assert window.event_count == 2
    assert window.duration_minutes == 1.0
    assert window.user_turns == 2
