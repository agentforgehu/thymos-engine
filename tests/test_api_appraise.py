import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from thymos import Profile, Thymos, ThymosConfig
from thymos.api import make_handler


def _post(port: int, payload: dict) -> dict:
    request = Request(
        f"http://127.0.0.1:{port}/api/appraise",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_appraise_endpoint_returns_dimensions_and_label(tmp_path):
    engine = Thymos(
        Profile(name="测试者", location_name="上海", timezone="Asia/Shanghai"),
        ThymosConfig(offline=True, seed=3),
        state_path=str(tmp_path / "state.json"),
        now=1770000000.0,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine, threading.Lock()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = {
            "appraisal": {
                "desirability": -0.30,
                "agency": "other",
                "control": 0.20,
                "certainty": 0.60,
                "stakes": 0.50,
                "note": "test",
            }
        }
        result = _post(server.server_port, payload)
        assert result["appraisal"]["desirability"] == -0.30
        assert result["emotion"]["label"] != "neutral"
        assert result["decision"]["strong_negative"] is True
        assert result["decision"]["neutral_reason"] == ""
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
