from thymos import AgentMemory, Profile, ThymosConfig
from thymos.api import token_matches

START = 1770000000.0


def test_token_matches_bearer_and_x_token():
    assert token_matches(None, None, None) is True
    assert token_matches(None, None, "secret") is False
    assert token_matches("Bearer secret", None, "secret") is True
    assert token_matches("bearer wrong", None, "secret") is False
    assert token_matches(None, "secret", "secret") is True


def test_agent_correct_and_delete_fact(tmp_path):
    memory = AgentMemory(
        str(tmp_path / "facts.json"),
        profile=Profile(name="林墨", location_name="杭州"),
        config=ThymosConfig(offline=True, seed=7),
        now=START,
    )
    memory.observe("我住在杭州。", min_importance=0)
    memory.correct_fact("city", "上海", old_value="杭州")
    assert memory.mind.facts.current("user", "city", START).value == "上海"
    assert memory.delete_fact(predicate="city", value="上海") == 1
    assert memory.mind.facts.current("user", "city", START) is None
