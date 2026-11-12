import io
import json

from thymos.agent import AgentMemory
from thymos.config import Profile, ThymosConfig
from thymos.mcp_server import MCPServer, run_stdio

START = 1770000000.0


def _memory(tmp_path):
    return AgentMemory(
        str(tmp_path / "mcp.json"),
        profile=Profile(name="林墨", location_name="杭州"),
        config=ThymosConfig(offline=True, seed=7),
        now=START,
    )


def test_mcp_initialize_and_tool_list(tmp_path):
    server = MCPServer(_memory(tmp_path))
    init = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init["result"]["protocolVersion"]
    listed = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert "memory_context" in names
    assert "memory_correct_fact" in names


def test_mcp_tool_calls_and_stdio(tmp_path):
    server = MCPServer(_memory(tmp_path))
    observed = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "memory_observe", "arguments": {"user_text": "我叫林墨，住在杭州。"}},
        }
    )
    assert observed["result"]["isError"] is False
    context = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "memory_context", "arguments": {"user_text": "我叫什么？", "max_chars": 600}},
        }
    )
    assert "林墨" in context["result"]["content"][0]["text"]

    stdin = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 5, "method": "tools/list"}) + "\n")
    stdout = io.StringIO()
    run_stdio(_memory(tmp_path), stdin=stdin, stdout=stdout)
    response = json.loads(stdout.getvalue().strip())
    assert response["id"] == 5
    assert response["result"]["tools"]
