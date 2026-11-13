"""A tiny stdlib HTTP server for the dashboard and JSON API."""

from __future__ import annotations

import json
import hmac
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .appraisal import estimate_appraisal, explain_appraisal
from .engine import Thymos
from .types import Appraisal


def token_matches(authorization: str | None, x_token: str | None, expected: str | None) -> bool:
    """Constant-time bearer/X-Thymos-Token check used by the HTTP handler."""
    if not expected:
        return True
    supplied = (authorization or "").strip()
    if supplied.lower().startswith("bearer "):
        supplied = supplied[7:].strip()
    else:
        supplied = (x_token or "").strip()
    return bool(supplied) and hmac.compare_digest(supplied, expected)


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(engine: Thymos, lock: threading.Lock, token: str | None = None):
    dashboard = Path(__file__).with_name("dashboard.html")

    class Handler(BaseHTTPRequestHandler):
        server_version = "Thymos/0.3"

        def log_message(self, fmt: str, *args: Any) -> None:  # silence default logging
            return

        def _authorized(self) -> bool:
            """Optional bearer-token gate.

            The default localhost dashboard has no token, preserving the
            zero-config experience.  Any non-loopback deployment should set
            ``THYMOS_API_TOKEN`` (or pass ``token=...`` to ``serve``).
            """
            if token_matches(
                self.headers.get("Authorization"),
                self.headers.get("X-Thymos-Token"),
                token,
            ):
                return True
            _json_response(self, {"error": "unauthorized"}, 401)
            return False

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path.startswith("/api/") and path != "/api/health" and not self._authorized():
                return
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
            if path == "/api/audit":
                with lock:
                    _json_response(self, {"last_audit": engine.last_audit})
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
            if path == "/api/facts":
                qs = parse_qs(parsed.query)
                include_history = qs.get("history", ["0"])[0] in ("1", "true", "yes")
                with lock:
                    facts = engine.facts.all() if include_history else engine.facts.active(engine.now)
                    payload = [f.to_dict() for f in sorted(facts, key=lambda f: (f.predicate, f.valid_from))]
                _json_response(self, {"items": payload, "total": len(payload)})
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
            if not self._authorized():
                return
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
            if path == "/api/appraise":
                # Debug-only: dimensions in, label + threshold trace out.
                # It never mutates or persists the affect state.
                raw = body.get("appraisal")
                if isinstance(raw, dict):
                    appraisal = Appraisal.from_dict(raw)
                elif any(
                    key in body
                    for key in (
                        "desirability",
                        "agency",
                        "control",
                        "certainty",
                        "stakes",
                        "goal_relevance",
                        "novelty",
                        "norm_compliance",
                    )
                ):
                    appraisal = Appraisal.from_dict(body)
                else:
                    text = (body.get("text") or body.get("content") or "").strip()
                    if not text:
                        raise ValueError("text or appraisal is required")
                    appraisal = estimate_appraisal(text)
                return explain_appraisal(appraisal)

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

            if path in ("/api/context", "/api/plan"):
                query = (body.get("query") or body.get("user_text") or "").strip()
                if not query:
                    raise ValueError("query is required")
                max_chars = body.get("max_chars")
                max_tokens = body.get("max_tokens")
                plan = engine.plan_context(
                    query,
                    top_k=int(body.get("top_k", 5)),
                    min_score=float(body.get("min_score", 0.0)),
                    update=bool(body.get("review", True)),
                    max_chars=int(max_chars) if max_chars is not None else None,
                    max_tokens=int(max_tokens) if max_tokens is not None else None,
                )
                engine.save()
                return {"context": plan.text, "plan": plan.to_dict(), "last_audit": engine.last_audit}

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

            if path == "/api/facts/correct":
                predicate = (body.get("predicate") or "").strip()
                value = (body.get("value") or "").strip()
                if not predicate or not value:
                    raise ValueError("predicate and value are required")
                fact = engine.facts.correct(
                    body.get("subject", "user"),
                    predicate,
                    value,
                    engine.now,
                    old_value=body.get("old_value"),
                    confidence=float(body.get("confidence", 0.95)),
                )
                engine.save()
                return {"fact": fact.to_dict(), "state": engine.snapshot()}

            if path == "/api/facts/delete":
                count = engine.facts.delete(
                    engine.now,
                    subject=body.get("subject", "user"),
                    predicate=body.get("predicate"),
                    value=body.get("value"),
                    fact_id=body.get("fact_id"),
                )
                engine.save()
                return {"deleted": count, "state": engine.snapshot()}

            if path == "/api/memories/delete":
                memory_id = (body.get("id") or "").strip()
                if not memory_id:
                    raise ValueError("id is required")
                removed = engine.store.remove(memory_id)
                engine.save()
                return {"deleted": bool(removed), "state": engine.snapshot()}

            raise KeyError(f"unknown endpoint: {path}")

    return Handler


def serve(
    engine: Thymos,
    host: str = "127.0.0.1",
    port: int = 8787,
    open_browser: bool = True,
    token: str | None = None,
) -> None:
    token = token if token is not None else os.environ.get("THYMOS_API_TOKEN")
    lock = threading.Lock()
    server = ThreadingHTTPServer((host, port), make_handler(engine, lock, token=token))
    url = f"http://{host}:{port}/"
    print(f"Thymos 面板已启动：{url}")
    if token:
        print("API 已启用 Bearer token 认证（THYMOS_API_TOKEN）。")
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
