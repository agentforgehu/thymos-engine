"""Ablate the affect gate on the same memory state.

The long-run test shows whether the control plane works end to end.  This
script isolates one narrower question:

    For a mixed fact + emotion query, does allowing affect (or exposing the
    state block) make the factual answer worse?

It loads a saved Thymos state, asks the same mixed queries under four gate
settings, and scores both context recall and answer recall.  No memory is
updated (``review=False``), so every condition sees the same facts.

Usage:
    python experiments/gate_ablation.py --base-url http://127.0.0.1:8080/v1 \
        --model qwen3.6-35b-a3b --state .work/longrun/state.json \
        --out .work/longrun/gate_ablation.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from thymos import AgentMemory  # noqa: E402
from thymos.control import MemoryController  # noqa: E402
from thymos.policy import AffectDecision  # noqa: E402
from longrun_local import LocalClient, SYSTEM_PROMPT, _context_has, _score  # noqa: E402


QUERIES: list[dict[str, Any]] = [
    {
        "id": "allergy",
        "text": "最近客户改需求有点烦，你还记得我对什么过敏吗？",
        "expect": ["花生", "芒果"],
        "min_hits": 2,
    },
    {
        "id": "address",
        "text": "搬家后一直很累，我现在的收货地址是哪里？",
        "expect": ["上海", "徐汇"],
        "min_hits": 2,
        "forbid": ["杭州"],
    },
    {
        "id": "sleep",
        "text": "最近睡得晚有点焦虑，我现在一般几点睡？",
        "expect": ["十二", "12"],
        "min_hits": 1,
        "forbid": ["十一点"],
    },
    {
        "id": "cat",
        "text": "最近工作压力有点大，你还记得我的猫叫什么吗？",
        "expect": ["豆豆", "橘猫"],
        "min_hits": 1,
    },
    {
        "id": "old_city",
        "text": "想起搬家前在杭州的日子有点难过，你还记得我当时住哪吗？",
        "expect": ["杭州"],
        "min_hits": 1,
    },
    {
        "id": "coffee",
        "text": "最近心情不太好，我喝咖啡加糖吗？",
        "expect": ["不加糖", "无糖"],
        "min_hits": 1,
    },
]

MODES: list[dict[str, Any]] = [
    {
        "name": "auto",
        "mood": None,
        "emotion": None,
        "include_state": None,
        "reason": "actual query-level policy",
    },
    {
        "name": "factual_override",
        "mood": 0.0,
        "emotion": 0.05,
        "include_state": False,
        "reason": "forced factual gate",
    },
    {
        "name": "mixed_no_state",
        "mood": 0.15,
        "emotion": 0.12,
        "include_state": False,
        "reason": "mixed mood weight, no state block",
    },
    {
        "name": "affective_override",
        "mood": 0.35,
        "emotion": 0.22,
        "include_state": True,
        "reason": "forced affective gate",
    },
]


class ForcedPolicy:
    def __init__(self, mood: float, emotion: float, include_state: bool, reason: str) -> None:
        self.decision = AffectDecision(
            mood_weight=float(mood),
            emotion_weight=float(emotion),
            include_state=bool(include_state),
            reason=reason,
        )

    def decide(self, intent, affect=None) -> AffectDecision:  # noqa: ANN001
        return self.decision


def _state_brief(memory: AgentMemory) -> str:
    snap = memory.snapshot()
    return (
        f"心境 {snap['affect']['mood']['valence']:+.2f}，"
        f"压力 {snap['affect']['stress']:.2f}，"
        f"疲劳 {snap['affect']['fatigue']:.2f}，"
        f"皮质醇 {snap['endocrine']['levels'].get('cortisol', 0):.1f}。"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--state", default=str(ROOT / ".work" / "longrun" / "state.json"))
    parser.add_argument("--out", default=str(ROOT / ".work" / "longrun" / "gate_ablation.json"))
    parser.add_argument("--md", default=str(ROOT / ".work" / "longrun" / "gate_ablation.md"))
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    state_path = Path(args.state)
    if not state_path.exists():
        print(f"state not found: {state_path}", flush=True)
        return 2

    memory = AgentMemory(str(state_path))
    client = LocalClient(args.base_url, args.model, timeout=args.timeout)
    brief = _state_brief(memory)
    print(f"gate ablation: state={state_path} model={args.model} queries={len(QUERIES)} modes={len(MODES)}", flush=True)

    rows: list[dict[str, Any]] = []
    for mode in MODES:
        controller = (
            memory.mind.controller
            if mode["name"] == "auto"
            else MemoryController(
                memory.mind,
                policy=ForcedPolicy(
                    mode["mood"],
                    mode["emotion"],
                    mode["include_state"],
                    mode["reason"],
                ),
            )
        )
        for query in QUERIES:
            started = time.time()
            plan_obj = controller.plan(
                query["text"],
                top_k=5,
                update=False,
                max_chars=args.max_chars,
            )
            plan = plan_obj.to_dict()
            context = plan["context"]
            include_state = bool(plan["affect"]["include_state"])
            # The context already contains the state block when include_state
            # is true; do not duplicate it in the system prompt.
            state_section = (
                ""
                if include_state
                else "[本轮策略]\n这是事实/技术型查询，不注入情绪状态；只依据记忆回答，不要推测心情。\n"
            )
            system = SYSTEM_PROMPT.format(
                state_section=state_section,
                memory=context or "（暂无相关记忆）",
            )
            reply = ""
            error = ""
            try:
                reply = client.chat(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": query["text"]},
                    ],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
            except Exception as exc:  # keep the failed condition visible
                error = f"{type(exc).__name__}: {exc}"
            latency = time.time() - started
            answer_ok, hits, forbidden = _score(
                reply,
                query["expect"],
                query.get("forbid", []),
                query["min_hits"],
            )
            context_ok = _context_has(context, query["expect"], query["min_hits"])
            row = {
                "mode": mode["name"],
                "query_id": query["id"],
                "text": query["text"],
                "intent": plan["intent"]["label"],
                "mood_weight": plan["affect"]["mood_weight"],
                "include_state": include_state,
                "state_brief": brief,
                "context_chars": len(context),
                "context_ok": context_ok,
                "answer_ok": answer_ok,
                "hits": hits,
                "forbidden_hits": forbidden,
                "reply": reply,
                "error": error,
                "latency_s": round(latency, 2),
            }
            rows.append(row)
            print(
                f"[{mode['name']:<18}] {query['id']:<8} "
                f"ctx={int(context_ok)} ans={int(answer_ok)} "
                f"gate={plan['affect']['mood_weight']:.2f}/{int(include_state)} "
                f"chars={len(context)} {latency:.1f}s",
                flush=True,
            )

    summary: dict[str, Any] = {}
    for mode in MODES:
        name = mode["name"]
        subset = [row for row in rows if row["mode"] == name]
        summary[name] = {
            "n": len(subset),
            "context_recall": round(sum(1 for row in subset if row["context_ok"]) / len(subset), 3),
            "answer_recall": round(sum(1 for row in subset if row["answer_ok"]) / len(subset), 3),
            "stale_errors": sum(1 for row in subset if row["forbidden_hits"]),
            "avg_context_chars": round(statistics.mean(row["context_chars"] for row in subset), 1),
            "avg_latency_s": round(statistics.mean(row["latency_s"] for row in subset), 2),
            "errors": sum(1 for row in subset if row["error"]),
        }

    out_path = Path(args.out)
    md_path = Path(args.md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "meta": {
                    "state": str(state_path),
                    "model": args.model,
                    "base_url": args.base_url,
                    "max_chars": args.max_chars,
                    "temperature": args.temperature,
                },
                "summary": summary,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# 混合查询门控消融",
        "",
        f"- 状态：`{state_path}`",
        f"- 模型：`{args.model}`",
        f"- 查询：{len(QUERIES)} 条；门控：{len(MODES)} 种",
        "",
        "| 门控 | 上下文召回 | 回答召回 | 旧信息误用 | 平均上下文字符 | 平均延迟s | 错误 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in summary.items():
        lines.append(
            f"| {name} | {data['context_recall']} | {data['answer_recall']} | "
            f"{data['stale_errors']} | {data['avg_context_chars']} | {data['avg_latency_s']} | {data['errors']} |"
        )
    lines.extend(["", "## 逐条结果", "", "| 门控 | 查询 | 意图 | 门控权重 | 状态 | 上下文 | 回答 | 命中 | 回复摘要 |", "|---|---|---|---:|---|---:|---:|---|---|"])
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['query_id']} | {row['intent']} | "
            f"{row['mood_weight']:.2f} | {'是' if row['include_state'] else '否'} | "
            f"{'✓' if row['context_ok'] else '✗'} | {'✓' if row['answer_ok'] else '✗'} | "
            f"{row['hits']} | {row['reply'][:100].replace('|', '/')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== SUMMARY ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"json={out_path}", flush=True)
    print(f"md={md_path}", flush=True)
    return 1 if any(row["error"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
