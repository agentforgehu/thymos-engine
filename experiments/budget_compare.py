"""A/B test under a fixed, small prompt budget.

Modes
-----
thymos
    Structured facts + budgeted episodic memories from AgentMemory.
recent
    Only the most recent user/assistant turns, clipped to the same character
    budget.  This is the "just put the chat history in the prompt" baseline.

Both modes receive the same user turns and the same model.  The only variable
is what fits into the memory block.  A probe is counted twice:

* ``context_ok``: is the expected fact present in the memory block at all?
* ``answer_ok``: did the model's answer contain it?

That separation matters: a model can fail even when the fact was supplied, and
a memory system can fail before the model ever sees the turn.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from thymos import AgentMemory, Profile, ThymosConfig  # noqa: E402
from longrun_local import LocalClient, _advance, _load_script, _score  # noqa: E402

SYSTEM_PROMPT = """你是「Thymos」驱动的陪伴型助手。下面是你能用的记忆块。
规则：
1. 只使用记忆块中的事实回答；没有就说“我不记得”，不要编造。
2. 当前事实优先于旧情景记忆；不要同时把旧值和新值都当成现在。
3. 回复自然、简短，不要逐条复述记忆，不要暴露内部字段名。

[记忆块]
{memory}
"""


def _recent_context(recent: list[tuple[str, str]], max_chars: int) -> str:
    """Clip recent turns from newest to oldest into a fixed budget."""
    blocks: list[str] = []
    used = len("## 近期对话（按时间倒序截断）")
    for user, reply in reversed(recent):
        block = f"用户：{user[:180]}\n助手：{reply[:220]}"
        if used + len(block) + 2 > max_chars:
            break
        blocks.append(block)
        used += len(block) + 2
    if not blocks:
        return ""
    return "## 近期对话（按时间倒序截断）\n" + "\n".join(blocks)


def _run_mode(
    mode: str,
    script: list[dict],
    args: argparse.Namespace,
    client: LocalClient,
) -> dict:
    state_path = ROOT / ".work" / "budget" / f"{mode}.json"
    if state_path.exists():
        state_path.unlink()
    state_path.parent.mkdir(parents=True, exist_ok=True)

    profile = Profile(name="林墨", location_name="杭州", timezone="Asia/Shanghai")
    memory = None
    if mode == "thymos":
        memory = AgentMemory(
            str(state_path),
            profile=profile,
            config=ThymosConfig(offline=True, seed=7),
            now=args.start,
        )

    recent: list[tuple[str, str]] = []
    records: list[dict] = []
    consecutive_errors = 0
    for index, item in enumerate(script, start=1):
        text = str(item["text"])
        gap = float(item.get("gap", 0))
        context = ""
        if memory is not None:
            _advance(memory.mind, gap)
            if item.get("move_to"):
                memory.mind.profile.location_name = str(item["move_to"])
                memory.mind.tick(1.0)
            context = memory.context(text, top_k=args.top_k, review=False, max_chars=args.max_chars)
        else:
            context = _recent_context(recent, args.max_chars)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(memory=context or "（空）")},
            {"role": "user", "content": text},
        ]
        started = time.time()
        error = ""
        try:
            reply = client.chat(messages, max_tokens=args.max_tokens, temperature=args.temperature)
            consecutive_errors = 0
        except Exception as exc:
            reply = ""
            error = f"{type(exc).__name__}: {exc}"
            consecutive_errors += 1
        latency = time.time() - started

        expected = list(item.get("expect") or [])
        forbid = list(item.get("forbid") or [])
        min_hits = int(item.get("min_hits", len(expected) or 0))
        is_probe = bool(item.get("probe"))
        answer_ok, hits, forbidden = _score(reply, expected, forbid, min_hits) if is_probe else (None, [], [])
        context_ok = None
        if is_probe:
            ctx = context.lower()
            context_ok = sum(1 for x in expected if x.lower() in ctx) >= min_hits

        if memory is not None and reply:
            memory.observe(text, min_importance=0, tags=item.get("tags") or ["对话"])
            memory.save()
        elif reply:
            recent.append((text, reply))
        records.append(
            {
                "turn": index,
                "text": text,
                "reply": reply,
                "error": error,
                "latency_s": round(latency, 2),
                "context_chars": len(context),
                "context": context,
                "probe": is_probe,
                "expect": expected,
                "forbid": forbid,
                "context_ok": context_ok,
                "answer_ok": answer_ok,
                "hits": hits,
                "forbidden_hits": forbidden,
            }
        )
        print(
            f"[{mode} {index:02d}/{len(script)}] {latency:5.1f}s ctx={len(context)} "
            f"probe={is_probe} ctx_ok={context_ok} ans_ok={answer_ok}",
            flush=True,
        )
        if error:
            print(f"  ERROR: {error}", flush=True)
        if consecutive_errors >= 3:
            print(f"[{mode}] stopping after 3 consecutive model errors", flush=True)
            break

    probes = [r for r in records if r["probe"]]
    answered = [r for r in probes if r["answer_ok"]]
    contextual = [r for r in probes if r["context_ok"]]
    latencies = [r["latency_s"] for r in records if not r["error"]]
    summary = {
        "mode": mode,
        "turns_completed": len(records),
        "probes": len(probes),
        "context_recall": round(len(contextual) / len(probes), 3) if probes else None,
        "answer_recall": round(len(answered) / len(probes), 3) if probes else None,
        "model_errors": sum(1 for r in records if r["error"]),
        "avg_context_chars": round(statistics.mean([r["context_chars"] for r in records]), 1) if records else 0,
        "avg_latency_s": round(statistics.mean(latencies), 2) if latencies else 0,
    }
    return {"summary": summary, "turns": records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--turns", type=int, default=50)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--modes", default="thymos,recent")
    parser.add_argument("--out", default=str(ROOT / "docs" / "results" / "budget_compare.json"))
    parser.add_argument("--md", default=str(ROOT / "docs" / "results" / "budget_compare.md"))
    parser.add_argument("--start", type=float, default=1772326800.0)
    args = parser.parse_args()

    script = _load_script()[: max(1, min(args.turns, len(_load_script())))]
    client = LocalClient(args.base_url, args.model, timeout=180.0)
    result: dict = {
        "meta": {
            "model": args.model,
            "max_chars": args.max_chars,
            "top_k": args.top_k,
            "turns": len(script),
            "modes": args.modes,
        },
        "modes": {},
    }
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        print(f"=== MODE {mode} ===", flush=True)
        result["modes"][mode] = _run_mode(mode, script, args, client)

    out = Path(args.out)
    md = Path(args.md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 固定小上下文预算 A/B 测试",
        "",
        f"- 模型：`{args.model}`",
        f"- 轮数：{len(script)}",
        f"- 记忆块预算：{args.max_chars} 字符",
        f"- 检索 Top-k：{args.top_k}",
        "",
        "| 模式 | 完成轮数 | 探针 | 上下文召回 | 回答召回 | 模型错误 | 平均记忆块字符 | 平均延迟 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, data in result["modes"].items():
        s = data["summary"]
        lines.append(
            f"| {mode} | {s['turns_completed']} | {s['probes']} | "
            f"{s['context_recall']} | {s['answer_recall']} | {s['model_errors']} | "
            f"{s['avg_context_chars']} | {s['avg_latency_s']}s |"
        )
    lines.extend(["", "## 逐探针对照", "", "| 轮 | 探针 | thymos 上下文 | thymos 回答 | recent 上下文 | recent 回答 |", "|---:|---|---:|---:|---:|---:|"])
    by_mode = result["modes"]
    turns = max((len(data["turns"]) for data in by_mode.values()), default=0)
    for i in range(turns):
        row = []
        probe = None
        for mode in ("thymos", "recent"):
            data = by_mode.get(mode)
            record = data["turns"][i] if data and i < len(data["turns"]) else None
            if record and record["probe"]:
                probe = record
                row.extend([record["context_ok"], record["answer_ok"]])
            else:
                row.extend(["", ""])
        if probe:
            lines.append(f"| {i + 1} | {probe['text']} | {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    lines.extend(["", "## 说明", "", "- `thymos` 的记忆块由结构化事实 + 预算内情景记忆组成；", "- `recent` 只放最近的用户/助手对话，截断到同一字符预算；", "- 上下文命中不等于回答命中；两者分开统计。"])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({m: d["summary"] for m, d in result["modes"].items()}, ensure_ascii=False, indent=2), flush=True)
    print(f"json={out}", flush=True)
    print(f"md={md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
