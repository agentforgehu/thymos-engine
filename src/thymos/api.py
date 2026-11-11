"""A tiny stdlib HTTP server for the dashboard and JSON API."""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .appraisal import estimate_appraisal
from .engine import Thymos
from .types import Appraisal


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(engine: Thymos, lock: threading.Lock):
    dashboard = Path(__file__).with_name("dashboard.html")

    class Handler(BaseHTTPRequestHandler):
        server_version = "Thymos/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # silence default logging
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path in ("/", "/index.html"):
                try:
                    body = dashboard.read_bytes()
                except OSError:
                    _json_response(self, {"error": "dashboard.html missing"}, 500)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/state":
                with lock:
                    _json_response(self, engine.snapshot())
                return
            if path == "/api/memories":
                qs = parse_qs(parsed.query)
                limit = int(qs.get("limit", ["50"])[0])
                with lock:
                    items = sorted(engine.store.all(), key=lambda m: m.created_at, reverse=True)[:limit]
                    from .forgetting import retrievability

                    payload = [m.to_public_dict(engine.now, retrievability(m.age_days(engine.now), m.stability)) for m in items]
                _json_response(self, {"items": payload, "total": len(engine.store)})
                return
            if path == "/api/export":
                with lock:
                    _json_response(self, engine.to_dict())
                return
            if path == "/api/health":
                _json_response(self, {"ok": True, "now": engine.now, "memories": len(engine.store)})
                return
            _json_response(self, {"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except ValueError:
                _json_response(self, {"error": "invalid json"}, 400)
                return

            try:
                with lock:
                    result = self._dispatch(parsed.path, body)
                _json_response(self, result)
            except KeyError as exc:
                _json_response(self, {"error": str(exc)}, 404)
            except Exception as exc:  # surface errors to the dashboard
                _json_response(self, {"error": f"{type(exc).__name__}: {exc}"}, 400)

        def _dispatch(self, path: str, body: dict[str, Any]) -> Any:
            if path == "/api/perceive":
                content = (body.get("content") or "").strip()
                if not content:
                    raise ValueError("content is required")
                appraisal = None
                if body.get("appraisal"):
                    appraisal = Appraisal.from_dict(body["appraisal"]) if hasattr(Appraisal, "from_dict") else Appraisal(**body["appraisal"])
                elif body.get("valence") is not None:
                    base = estimate_appraisal(content)
                    appraisal = Appraisal(
                        goal_relevance=base.goal_relevance,
                        desirability=float(body["valence"]),
                        novelty=base.novelty,
                        certainty=base.certainty,
                        control=base.control,
                        agency=body.get("agency", base.agency),
                        stakes=base.stakes,
                        note="dashboard",
                    )
                item = engine.perceive(
                    content,
                    appraisal=appraisal,
                    importance=body.get("importance"),
                    tags=body.get("tags") or [],
                    source=body.get("source", "dashboard"),
                )
                engine.save()
                return {"memory": item.to_public_dict(engine.now, 1.0), "state": engine.snapshot()}

            if path == "/api/recall":
                hits = engine.recall(body.get("query") or "", top_k=int(body.get("top_k", 5)))
                engine.save()
                return {"hits": [h.to_dict(engine.now) for h in hits], "state": engine.snapshot()}

            if path == "/api/tick":
                seconds = body.get("seconds")
                hours = body.get("hours")
                seconds = float(seconds) if seconds is not None else float(hours or 1.0) * 3600.0
                engine.tick(seconds, sleeping=bool(body.get("sleeping", False)))
                engine.save()
                return {"state": engine.snapshot()}

            if path == "/api/consolidate":
                result = engine.consolidate(float(body.get("hours", 7.5)))
                engine.save()
                return {"result": result, "state": engine.snapshot()}

            if path == "/api/reappraise":
                item = engine.reappraise(
                    body["id"],
                    new_valence=body.get("new_valence"),
                    delta=body.get("delta"),
                    note=body.get("note", ""),
                )
                engine.save()
                return {"memory": item.to_dict(), "state": engine.snapshot()}

            raise KeyError(f"unknown endpoint: {path}")

    return Handler


def serve(engine: Thymos, host: str = "127.0.0.1", port: int = 8787, open_browser: bool = True) -> None:
    lock = threading.Lock()
    server = ThreadingHTTPServer((host, port), make_handler(engine, lock))
    url = f"http://{host}:{port}/"
    print(f"Thymos 面板已启动：{url}")
    print("按 Ctrl+C 停止。状态会自动保存到", os.path.abspath(engine.state_path or ".thymos/state.json"))
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()
        try:
            engine.save()
        except Exception:
            pass
