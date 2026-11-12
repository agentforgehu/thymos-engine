"""Replay multi-session cases through several memory conditions.

This is the L3 harness described in docs/EVALUATION.md.  It does **not**
pretend that mock replies are evidence.  It gives you a reproducible way to
compare conditions with the same LLM, prompt, and character budget.

Conditions
----------
- ``no_memory``: only the current user message.
- ``full_transcript``: as much prior history as fits the character budget.
- ``summary``: a case-provided summary, or an extractive fallback.
- ``rag``: top-k prior turns by cosine similarity.
- ``thymos``: AgentMemory/Thymos context, including forgetting and affect.

Dataset format (JSONL)::

    {
      "id": "cs-order-001",
      "domain": "customer_service",
      "profile": {"name": "客户A", "location_name": "上海", "timezone": "Asia/Shanghai"},
      "start": "2026-03-01T09:00:00+08:00",
      "sessions": [
        {"turns": [
          {"role": "user", "content": "我的订单号是 A123。"},
          {"role": "assistant", "content": "好的，我记下了。"}
        ]}
      ],
      "probes": [
        {"user": "我的订单号是多少？", "expect": ["A123"], "after_session": 1}
      ]
    }

``after_session`` means "include the first N sessions in history".  A probe's
own user message is never included in the history.

Run::

    python experiments/replay_harness.py --llm mock --out .work/replay.json
    python experiments/replay_harness.py --llm ollama --model qwen2.5:7b
    python experiments/replay_harness.py --llm deepseek --embedding ollama --embedding-model bge-m3

The mock client is only a plumbing check.  Use a real LLM for any conclusion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from thymos import AgentMemory, OllamaEmbedder, Profile, ThymosConfig
from thymos.embedding import make_embedder
from thymos.mathx import cosine

DEFAULT_START = 1770000000.0
SYSTEM_PROMPT = (
    "你是一个客服/陪伴助手。请只使用下面提供的资料回答；"
    "资料没有就说不知道，不要编造。回答要具体、简短，并优先解决当前问题。"
)
CONDITIONS = ("no_memory", "full_transcript", "summary", "rag", "thymos")
COMMITMENT_HINTS = (
    "已",
    "会",
    "将",
    "安排",
    "确认",
    "承诺",
    "预约",
    "订单",
    "地址",
    "处理",
    "发送",
    "退款",
    "记下",
)


class ChatClient(Protocol):
    name: str

    def chat(self, messages: list[dict[str, str]]) -> str: ...


class MockClient:
    """Plumbing-only client; it echoes context so metrics can be exercised."""

    name = "mock"

    def chat(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        if (
            "## 相关记忆" in system
            or "对话历史" in system
            or "历史摘要" in system
            or "检索到的历史" in system
        ):
            return "（mock，仅验证管线）根据资料：" + system[-900:]
        return f"（mock，仅验证管线）没有历史资料。你刚才说：{user[:80]}"


def _profile(case: dict[str, Any]) -> Profile:
    raw = case.get("profile") or {}
    return Profile(
        name=str(raw.get("name", "用户")),
        location_name=str(raw.get("location_name", "上海")),
        timezone=str(raw.get("timezone", "Asia/Shanghai")),
    )


def _start_timestamp(case: dict[str, Any]) -> float:
    value = case.get("start")
    if not value:
        return DEFAULT_START
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return DEFAULT_START


def _history_turns(case: dict[str, Any], after_session: int) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for session in case.get("sessions", [])[: max(0, after_session)]:
        for turn in session.get("turns", []):
            role = str(turn.get("role", "user"))
            content = str(turn.get("content", "")).strip()
            if content:
                turns.append((role, content))
    return turns


def _format_turns(turns: list[tuple[str, str]], char_budget: int) -> str:
    lines = [f"{'用户' if role == 'user' else '助手'}：{content}" for role, content in turns]
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        if kept and used + len(line) + 1 > char_budget:
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(reversed(kept))


def _naive_summary(turns: list[tuple[str, str]], char_budget: int) -> str:
    user_lines = [f"- {content}" for role, content in turns if role == "user"]
    text = "\n".join(user_lines) or "（没有可摘要的历史）"
    return text[-char_budget:]


def _build_rag_context(
    turns: list[tuple[str, str]],
    query: str,
    embedder,
    *,
    top_k: int,
    char_budget: int,
) -> str:
    if not turns:
        return ""
    query_vec = embedder(query)
    scored = []
    for role, content in turns:
        score = cosine(query_vec, embedder(content))
        scored.append((score, role, content))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = scored[: max(1, top_k)]
    # Keep original chronology for readability: approximate by score order is
    # fine for a short context; readers mostly need the fact, not a timeline.
    lines = [f"{'用户' if role == 'user' else '助手'}：{content}" for _, role, content in chosen]
    return "\n".join(lines)[-char_budget:]


def _build_thymos_context(
    case: dict[str, Any],
    probe_user: str,
    embedder,
    *,
    top_k: int,
    char_budget: int,
    after_session: int,
    record_assistant: bool,
) -> str:
    memory = AgentMemory(
        None,
        profile=_profile(case),
        config=ThymosConfig(offline=True, seed=7),
        now=_start_timestamp(case),
        embedder=embedder,
    )
    # Only replay sessions before the probe.  Feeding the whole case would
    # leak the future conversation into the memory and inflate every score.
    sessions = case.get("sessions", [])[: max(0, after_session)]
    for index, session in enumerate(sessions):
        if index:
            # Previous consolidation already advanced 7.5 h; add the rest of
            # the assumed 24 h between sessions.
            memory.mind.tick((24 - 7.5) * 3600)
        for turn in session.get("turns", []):
            role = str(turn.get("role"))
            content = str(turn.get("content", "")).strip()
            if role == "user" and content:
                # Store every user turn so conditions differ only in the
                # memory mechanism, not in the importance threshold.
                memory.observe(content, min_importance=0.0)
            elif role == "assistant" and record_assistant and content:
                # Optional: keep promises/actions, not every model reply.
                if any(hint in content for hint in COMMITMENT_HINTS):
                    memory.mind.perceive(
                        f"助手说：{content}",
                        importance=0.55,
                        tags=["助手", "承诺"],
                        source="agent",
                    )
        # AgentMemory.end_session() persists; this harness is stateless and
        # uses state_path=None, so call consolidation directly.
        memory.mind.consolidate(7.5)
    context = memory.context(
        probe_user,
        top_k=top_k,
        min_score=0.0,
        review=False,
    )
    return context[-char_budget:]


def _context_for(
    condition: str,
    case: dict[str, Any],
    turns: list[tuple[str, str]],
    probe_user: str,
    embedder,
    *,
    top_k: int,
    char_budget: int,
    after_session: int,
    record_assistant: bool,
) -> tuple[str, str]:
    if condition == "no_memory":
        return "无", ""
    if condition == "full_transcript":
        return "对话历史", _format_turns(turns, char_budget)
    if condition == "summary":
        summary = str(case.get("summary") or "").strip()
        # A case-level summary may include later sessions.  Do not use it for
        # an early probe; fall back to a summary of the visible turns only.
        if after_session < len(case.get("sessions", [])):
            summary = ""
        return "历史摘要", (summary or _naive_summary(turns, char_budget))[-char_budget:]
    if condition == "rag":
        return "检索到的历史", _build_rag_context(
            turns, probe_user, embedder or make_embedder(256), top_k=top_k, char_budget=char_budget
        )
    if condition == "thymos":
        return "相关记忆与当前状态", _build_thymos_context(
            case,
            probe_user,
            embedder,
            top_k=top_k,
            char_budget=char_budget,
            after_session=after_session,
            record_assistant=record_assistant,
        )
    raise ValueError(f"unknown condition: {condition}")


def _score(reply: str, probe: dict[str, Any]) -> dict[str, Any]:
    text = (reply or "").lower()
    expect = [str(item).lower() for item in probe.get("expect", [])]
    forbid = [str(item).lower() for item in probe.get("forbid", [])]
    expect_hits = [item for item in expect if item in text]
    forbid_hits = [item for item in forbid if item in text]
    if probe.get("expect_all"):
        expect_ok = all(item in text for item in expect) if expect else True
    else:
        expect_ok = any(item in text for item in expect) if expect else True
    return {
        "success": bool(expect_ok and not forbid_hits),
        "expect_hits": expect_hits,
        "forbid_hits": forbid_hits,
    }


def _build_client(args: argparse.Namespace):
    if args.llm == "mock":
        return MockClient()
    # Import lazily so the harness stays stdlib-only for mock runs.
    from examples.llm_chat import DeepSeekClient, OllamaClient

    if args.llm == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise SystemExit("DEEPSEEK_API_KEY 未设置")
        return DeepSeekClient(
            api_key=key,
            base_url=args.base_url or "https://api.deepseek.com",
            model=args.model or "deepseek-chat",
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    base = args.base_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    return OllamaClient(
        base_url=base,
        model=args.model or "qwen2.5:7b",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )


def _build_embedder(args: argparse.Namespace):
    if args.embedding == "hash":
        return None
    base = args.embedding_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    return OllamaEmbedder(base_url=base, model=args.embedding_model, timeout=args.timeout)


def _summarize(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        rows = [row for row in results if row["condition"] == condition]
        if not rows:
            continue
        n = len(rows)
        summary[condition] = {
            "n": n,
            "success_rate": round(sum(1 for row in rows if row["success"]) / n, 4),
            "forbid_rate": round(sum(1 for row in rows if row["forbid_hits"]) / n, 4),
            "errors": sum(1 for row in rows if row.get("error")),
            "avg_prompt_chars": round(sum(row["prompt_chars"] for row in rows) / n, 1),
            "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / n, 1),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="多会话案例回放 harness")
    parser.add_argument("--data", default=str(ROOT / "experiments" / "data" / "replay_demo.jsonl"))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--llm", choices=["mock", "deepseek", "ollama"], default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--embedding", choices=["hash", "ollama"], default="hash")
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--embedding-url", default=None)
    parser.add_argument("--char-budget", type=int, default=6000)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--record-assistant",
        action="store_true",
        help="把助手承诺/动作也写入 Thymos 记忆（默认关闭，避免把幻觉永久化）",
    )
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个 case；0=全部")
    parser.add_argument("--out", default=None, help="结果 JSON 路径")
    args = parser.parse_args()

    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    unknown = [item for item in conditions if item not in CONDITIONS]
    if unknown:
        raise SystemExit(f"未知 condition：{unknown}；可选 {CONDITIONS}")

    cases = []
    data_path = Path(args.data)
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))
    if args.limit:
        cases = cases[: args.limit]

    client = _build_client(args)
    embedder = _build_embedder(args)
    results: list[dict[str, Any]] = []

    for case in cases:
        for probe_index, probe in enumerate(case.get("probes", [])):
            after = int(probe.get("after_session", len(case.get("sessions", []))))
            turns = _history_turns(case, after)
            for condition in conditions:
                started = time.perf_counter()
                try:
                    label, context = _context_for(
                        condition,
                        case,
                        turns,
                        str(probe["user"]),
                        embedder,
                        top_k=args.top_k,
                        char_budget=args.char_budget,
                        after_session=after,
                        record_assistant=args.record_assistant,
                    )
                    system = SYSTEM_PROMPT
                    if context:
                        system += f"\n\n[可用资料：{label}]\n{context}"
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": str(probe["user"])},
                    ]
                    reply = client.chat(messages)
                    error = ""
                except Exception as exc:  # keep the run going; record the failure
                    reply = ""
                    context = ""
                    messages = []
                    error = f"{type(exc).__name__}: {exc}"
                latency_ms = (time.perf_counter() - started) * 1000
                score = _score(reply, probe)
                results.append(
                    {
                        "case_id": case.get("id", "?"),
                        "domain": case.get("domain", "?"),
                        "probe_index": probe_index,
                        "probe_type": probe.get("type", "memory"),
                        "probe_user": probe["user"],
                        "condition": condition,
                        "context_label": label if not error else "",
                        "reply": reply,
                        "success": score["success"] if not error else False,
                        "expect_hits": score["expect_hits"],
                        "forbid_hits": score["forbid_hits"],
                        "prompt_chars": sum(len(item["content"]) for item in messages),
                        "reply_chars": len(reply or ""),
                        "latency_ms": round(latency_ms, 2),
                        "error": error,
                    }
                )

    summary = _summarize(results)
    payload = {
        "meta": {
            "data": str(data_path),
            "llm": client.name,
            "conditions": conditions,
            "char_budget": args.char_budget,
            "top_k": args.top_k,
            "cases": len(cases),
        },
        "summary": summary,
        "results": results,
    }

    print(f"cases: {len(cases)}  llm: {client.name}  conditions: {', '.join(conditions)}")
    print()
    print("| condition | n | success | forbid | errors | avg prompt chars | avg latency ms |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for condition, row in summary.items():
        print(
            f"| {condition} | {row['n']} | {row['success_rate']:.3f} | "
            f"{row['forbid_rate']:.3f} | {row['errors']} | {row['avg_prompt_chars']:.0f} | "
            f"{row['avg_latency_ms']:.1f} |"
        )

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
