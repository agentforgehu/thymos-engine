import json

from thymos import AgentMemory, Profile, ThymosConfig
from thymos.llm_extract import LLMFactExtractor


def test_llm_extractor_validates_schema_and_event_time():
    def chat(messages):
        return json.dumps(
            [
                {"predicate": "city", "value": "上海", "event_at": 1770604800, "confidence": 0.9},
                {"predicate": "allergy", "value": "花生", "confidence": 0.8},
                {"predicate": "evil", "value": "rm -rf", "confidence": 1.0},
                {"predicate": "name", "value": "什么", "confidence": 1.0},
            ]
        )

    facts = LLMFactExtractor(chat).extract("我下周搬去上海，对花生过敏。", 1770000000.0)
    assert [(f.predicate, f.value) for f in facts] == [("city", "上海"), ("allergy", "花生")]
    assert facts[0].event_at == 1770604800
    assert facts[0].source == "llm"


def test_llm_extractor_falls_back_on_bad_json():
    def chat(messages):
        return "not json"

    facts = LLMFactExtractor(chat).extract("我叫林墨，住在杭州。", 1770000000.0)
    assert dict((f.predicate, f.value) for f in facts)["name"] == "林墨"


def test_agent_memory_observe_llm_uses_extractor(tmp_path):
    def chat(messages):
        return '[{"predicate":"name","value":"林墨","confidence":0.95}]'

    memory = AgentMemory(
        str(tmp_path / "llm.json"),
        profile=Profile(name="林墨", location_name="杭州"),
        config=ThymosConfig(offline=True, seed=7),
        now=1770000000.0,
    )
    item = memory.observe_llm("我叫林墨。", LLMFactExtractor(chat), min_importance=0)
    assert item is not None
    assert memory.mind.facts.current("user", "name", 1770000000.0).value == "林墨"
