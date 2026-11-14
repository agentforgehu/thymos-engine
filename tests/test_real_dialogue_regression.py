"""Regression tests driven by real multi-turn dialogue, not generated templates.

The public default is the CPED test-split window file shipped with the
repository.  A host can point ``THYMOS_REAL_DIALOGUE_FIXTURE`` at its own
JSONL file or directory (same schema) to run the same pipeline over private,
redacted conversations.

These tests assert *pipeline invariants*: every real window is ingested, the
summary is stored and retrievable, state survives SQLite, and results are
deterministic.  They deliberately do not assert emotion accuracy on CPED; the
lexical baseline is weak and tuning against this test set would be leakage.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from thymos import AgentMemory, Profile, ThymosConfig
from thymos.appraisal import appraise, estimate_appraisal

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WINDOWS = ROOT / "evals" / "companion_zh" / "data" / "companion_windows.jsonl"
DEFAULT_UTTERANCES = ROOT / "evals" / "companion_zh" / "data" / "cped_emotion_sample.jsonl"
PROFILE = Profile(name="真实对话回归", location_name="杭州", timezone="Asia/Shanghai")
ALLOWED_LABELS = {
    "neutral",
    "joy",
    "contentment",
    "excitement",
    "trust",
    "hope",
    "anticipation",
    "surprise",
    "sadness",
    "anger",
    "anxiety",
    "fear",
    "disgust",
    "depressed",
}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _windows_path() -> Path:
    configured = os.environ.get("THYMOS_REAL_DIALOGUE_FIXTURE")
    if configured:
        path = Path(configured)
        if path.is_dir():
            path = path / "companion_windows.jsonl"
        if not path.exists():
            pytest.fail(f"THYMOS_REAL_DIALOGUE_FIXTURE does not exist: {path}")
        return path
    return DEFAULT_WINDOWS


def _windows() -> list[dict]:
    rows = _read_jsonl(_windows_path())
    assert rows, "real-dialogue fixture is empty"
    return rows


def _memory(tmp_path: Path, now: float, suffix: str = "state.json") -> AgentMemory:
    return AgentMemory(
        str(tmp_path / suffix),
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=11),
        now=now,
    )


@pytest.mark.parametrize("row", _windows(), ids=lambda row: row["id"])
def test_real_window_replays_through_the_pipeline(tmp_path, row):
    events = row["events"]
    memory = _memory(tmp_path, events[0]["at"])
    window = memory.observe_stream(events, window="今晚", step_seconds=900)

    assert window.event_count == len(events)
    assert window.user_turns == sum(1 for event in events if event.get("role", "user") == "user")
    assert window.memory_id is not None
    assert "窗口「今晚」" in window.summary
    assert window.labels
    assert math.isfinite(window.valence_mean)
    assert -1.0 <= window.valence_mean <= 1.0

    context = memory.context(row["probe"], top_k=5, max_chars=900)
    assert "窗口「今晚」" in context
    snapshot = memory.snapshot()
    assert snapshot["last_window"]["memory_id"] == window.memory_id
    assert snapshot["last_window"]["event_count"] == len(events)


def test_real_window_is_deterministic(tmp_path):
    row = _windows()[0]
    events = row["events"]
    first = _memory(tmp_path, events[0]["at"], "first.json")
    second = _memory(tmp_path, events[0]["at"], "second.json")

    first_window = first.observe_stream(events, window="今晚", step_seconds=900)
    second_window = second.observe_stream(events, window="今晚", step_seconds=900)

    assert first_window.summary == second_window.summary
    first_dict = first_window.to_dict()
    second_dict = second_window.to_dict()
    # IDs are intentionally random; compare the dynamics and the rendered
    # summary, not the storage identity.
    first_dict.pop("memory_id")
    second_dict.pop("memory_id")
    assert first_dict == second_dict
    first_last = first.snapshot()["last_window"]
    second_last = second.snapshot()["last_window"]
    first_last.pop("memory_id")
    second_last.pop("memory_id")
    assert first_last == second_last


def test_real_window_summary_survives_sqlite_reload(tmp_path):
    row = _windows()[0]
    events = row["events"]
    path = tmp_path / "real.sqlite"
    memory = AgentMemory(
        str(path),
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=11),
        now=events[0]["at"],
    )
    window = memory.observe_stream(events, window="今晚", step_seconds=900)
    memory.save()

    reloaded = AgentMemory(
        str(path),
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=11),
    )
    context = reloaded.context(row["probe"], top_k=5, max_chars=900)
    assert "窗口「今晚」" in context
    assert any(item.id == window.memory_id for item in reloaded.mind.store)


def test_all_real_utterances_are_appraisable():
    rows = _read_jsonl(DEFAULT_UTTERANCES)
    assert len(rows) == 156
    for row in rows:
        emotion = appraise(estimate_appraisal(row["text"]))
        assert emotion.label in ALLOWED_LABELS
        assert math.isfinite(emotion.valence)
        assert math.isfinite(emotion.arousal)
