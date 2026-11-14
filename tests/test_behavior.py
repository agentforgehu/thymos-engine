import pytest

from thymos import AgentMemory, BehaviorDirective, Profile, ThymosConfig


START = 1770000000.0
PROFILE = Profile(name="测试者", location_name="上海", timezone="Asia/Shanghai")


@pytest.fixture()
def memory(tmp_path):
    return AgentMemory(
        str(tmp_path / "behavior.json"),
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=3),
        now=START,
    )


def test_high_stress_deescalates_and_shortens(memory):
    memory.mind.affect.state.stress = 0.90
    memory.mind.affect.state.dominant_emotion = "anger"
    directive = memory.behavior("我现在很生气")
    assert directive.boundary == "deescalate"
    assert directive.tone == "gentle"
    assert directive.verbosity == "short"
    assert directive.initiative == "reflect"


def test_factual_query_stays_task_first_even_with_negative_mood(memory):
    memory.mind.affect.state.mood_valence = -0.45
    directive = memory.behavior("我叫什么？", intent="factual")
    assert directive.tone == "neutral"
    assert directive.suggested_action == "answer_directly"
    assert directive.verbosity == "short"


def test_positive_affect_matches_energy(memory):
    memory.mind.affect.state.mood_valence = 0.50
    memory.mind.affect.state.dominant_emotion = "joy"
    directive = memory.behavior("今天真开心")
    assert directive.tone == "warm"
    assert directive.suggested_action.startswith("match_energy")


def test_plan_exposes_behavior_audit(memory):
    memory.observe("今天有点难过，想找人聊聊", min_importance=0)
    plan = memory.plan("今天有点难过，想找人聊聊", max_chars=500)
    assert "behavior" in plan
    assert plan["behavior"]["tone"] in {"gentle", "neutral", "warm"}
    assert memory.snapshot()["last_behavior"] == plan["behavior"]


def test_custom_mapper_is_the_extension_point(tmp_path):
    class DirectMapper:
        def map(self, affect, *, intent="mixed", emotion=None, context=None):
            return BehaviorDirective(
                tone="direct",
                verbosity="short",
                initiative="advise",
                boundary="none",
                suggested_action="give_one_concrete_step",
                confidence=0.99,
                reason="product persona",
                affect_label=emotion or "neutral",
            )

    memory = AgentMemory(
        str(tmp_path / "custom.json"),
        profile=PROFILE,
        config=ThymosConfig(offline=True, seed=3),
        now=START,
        behavior_mapper=DirectMapper(),
    )
    directive = memory.behavior("怎么办", intent="affective")
    assert directive.tone == "direct"
    assert directive.confidence == 0.99
    assert memory.plan("怎么办")["behavior"]["suggested_action"] == "give_one_concrete_step"


def test_default_mapper_is_deterministic_and_json_serializable(memory):
    first = memory.behavior("你好").to_dict()
    second = memory.behavior("你好").to_dict()
    assert first == second
    assert isinstance(first["confidence"], float)
    assert set(first) == {
        "tone",
        "verbosity",
        "initiative",
        "boundary",
        "suggested_action",
        "confidence",
        "reason",
        "affect_label",
    }
