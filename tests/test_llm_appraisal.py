import json

import pytest

import thymos.agent as agent_module
from thymos import AgentMemory, Appraisal, LLMAppraiser, Profile, ThymosConfig
from thymos.appraisal import appraise

START = 1770000000.0
PROFILE = Profile(name="测试者", location_name="上海", timezone="Asia/Shanghai")


def _chat_returning(payload):
    def chat(messages):
        assert messages[0]["role"] == "system"
        assert "JSON" in messages[0]["content"]
        return json.dumps(payload, ensure_ascii=False)

    return chat


def test_llm_appraiser_parses_dimensions_and_keeps_audit_note():
    payload = {
        "goal_relevance": 0.8,
        "desirability": 0.72,
        "novelty": 0.3,
        "certainty": 0.9,
        "control": 0.7,
        "agency": "other",
        "norm_compliance": 0.3,
        "stakes": 0.6,
        "note": "用户被明确表扬",
    }
    appraiser = LLMAppraiser(_chat_returning(payload))
    a = appraiser("你这次报告做得真好")
    assert a.desirability == pytest.approx(0.72)
    assert a.agency == "other"
    assert a.stakes == pytest.approx(0.6)
    assert a.note.startswith("llm:")
    assert appraise(a).label in {"joy", "trust", "contentment", "excitement"}


def test_llm_appraiser_clamps_and_normalizes_untrusted_fields():
    payload = {
        "desirability": 9,
        "goal_relevance": -3,
        "control": 2,
        "agency": "somebody",
        "note": "x" * 200,
    }
    a = LLMAppraiser(_chat_returning(payload), max_note_chars=12)("测试")
    assert a.desirability == 1.0
    assert a.goal_relevance == 0.0
    assert a.control == 1.0
    assert a.agency == "circumstance"
    assert len(a.note) <= 12 + len("llm:")


def test_llm_appraiser_falls_back_on_invalid_json():
    a = LLMAppraiser(lambda messages: "not json", fallback=True)("项目黄了")
    assert a.desirability < -0.20
    assert "project_loss" in a.note


def test_llm_appraiser_can_return_neutral_error_without_fallback():
    a = LLMAppraiser(lambda messages: "not json", fallback=False)("项目黄了")
    assert a.desirability == 0.0
    assert a.note.startswith("llm-error:")


def test_agent_memory_uses_one_appraiser_call_for_importance_and_affect(tmp_path):
    calls = []

    def appraiser(text):
        calls.append(text)
        return Appraisal(
            goal_relevance=0.8,
            desirability=0.7,
            certainty=0.9,
            control=0.7,
            agency="other",
            stakes=0.6,
            note="fake-llm",
        )

    memory = AgentMemory(
        str(tmp_path / "agent.json"),
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=3),
        now=START,
        appraiser=appraiser,
    )
    before = memory.snapshot()["affect"]["mood"]["valence"]
    item = memory.observe("你这次真的帮了我大忙")
    after = memory.snapshot()["affect"]["mood"]["valence"]
    assert item is not None
    assert calls == ["你这次真的帮了我大忙"]
    assert after > before


def test_small_talk_does_not_call_the_appraiser(tmp_path):
    calls = []
    memory = AgentMemory(
        str(tmp_path / "agent.json"),
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=3),
        now=START,
        appraiser=lambda text: calls.append(text) or Appraisal(),
    )
    assert memory.observe("嗯") is None
    assert calls == []


def test_supplied_appraisal_avoids_a_second_rule_pass(tmp_path, monkeypatch):
    memory = AgentMemory(
        str(tmp_path / "agent.json"),
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=3),
        now=START,
    )

    def boom(text):
        raise AssertionError("estimate_appraisal should not run twice")

    monkeypatch.setattr(agent_module, "estimate_appraisal", boom)
    item = memory.observe(
        "你真好",
        appraisal=Appraisal(
            goal_relevance=0.8,
            desirability=0.7,
            certainty=0.9,
            control=0.7,
            agency="other",
            stakes=0.6,
            note="host",
        ),
    )
    assert item is not None
