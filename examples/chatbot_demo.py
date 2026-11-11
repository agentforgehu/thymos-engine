"""A runnable "memory in a bot" demo, with a fake LLM.

The point is not the fake reply.  The point is the boundary:

    user text -> AgentMemory.context() -> LLM -> AgentMemory.observe()

Run from the repository root:

    python examples/chatbot_demo.py

The state file is written to ``.thymos/chatbot_demo.json``.  Delete it to
start over.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from thymos import AgentMemory, Profile, ThymosConfig

START = 1770000000.0  # fixed timestamp: reproducible output
STATE = Path(".thymos/chatbot_demo.json")


def fake_llm(context: str, user_text: str) -> str:
    """Stand-in for a real LLM.

    A real bot would pass ``context`` and ``user_text`` to DeepSeek/OpenAI/
    Ollama.  Here we only demonstrate what the model is allowed to see.
    """
    if "虾" in user_text or "吃" in user_text:
        return "记得，你不吃虾，过敏。我不会给你推荐海鲜。"
    if "面试" in user_text:
        return "你之前说周四面试，还有点紧张。现在感觉怎么样？"
    if "咖啡" in user_text or "喝" in user_text:
        return "冰美式，不加糖。你上次特意说过。"
    return "我听到了。要不要说说今天发生了什么？"


def main() -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    if STATE.exists():
        STATE.unlink()

    memory = AgentMemory(
        str(STATE),
        profile=Profile(name="小林", location_name="上海", timezone="Asia/Shanghai"),
        config=ThymosConfig(offline=True, seed=7),
        now=START,
    )

    print("=" * 72)
    print("DAY 1 · 用户说了几件事")
    print("=" * 72)
    day_one = [
        ("我不吃虾，过敏", 0.9),
        ("我喜欢冰美式，不加糖", 0.85),
        ("我周四面试，有点紧张", 0.8),
        ("今天中午吃了拉面", 0.2),  # trivial: stored manually to show decay
    ]
    for text, importance in day_one:
        if importance >= 0.35:
            item = memory.observe(text, importance=importance)
            print(f"  [stored] {text}  (importance={importance})")
        else:
            item = memory.mind.perceive(text, importance=importance)
            print(f"  [stored, low] {text}  (importance={importance})")
        assert item is not None

    # Small talk should not enter memory by default.
    skipped = memory.observe("嗯嗯")
    print(f"  [skipped] 嗯嗯 -> {skipped}")

    print("\n" + "=" * 72)
    print("30 DAYS LATER · 没有复习，也没有删除")
    print("=" * 72)
    memory.mind.tick(30 * 86400)
    memory.end_session(7.5)

    print("\n[当前最容易被想起的记忆]")
    for row in memory.mind.top_memories(5):
        print(
            f"  R={row['retrievability']:.2f}  S={row['stability_days']:.2f}d  "
            f"{row['content'][:30]}"
        )

    print("\n" + "=" * 72)
    print("DAY 30 · 用户提问，看看 LLM 实际收到什么")
    print("=" * 72)
    print("注意：这一轮故意不把问答本身写回记忆。")
    print("否则“我问过什么”会变成新记忆，挤掉真正的事实。")
    for question in ("我不吃什么？", "我面试是哪天？", "我喜欢喝什么？"):
        # review=False: this is a read-only probe, not a rehearsal.
        context = memory.context(question, top_k=3, min_score=0.30, review=False)
        reply = fake_llm(context, question)
        print("\nUSER:", question)
        print("-" * 72)
        print("CONTEXT SENT TO LLM:")
        print(context)
        print("-" * 72)
        print("ASSISTANT:", reply)

    # One deliberate recall: retrieval practice strengthens the trace.
    print("\n" + "=" * 72)
    print("一次真正的回忆（review=True）之后")
    print("=" * 72)
    memory.context("我不吃什么？", top_k=1, review=True)
    for row in memory.mind.top_memories(3):
        print(
            f"  R={row['retrievability']:.2f}  S={row['stability_days']:.2f}d  "
            f"{row['content'][:30]}"
        )

    memory.save()
    print("\n" + "=" * 72)
    print("状态已保存：", STATE.resolve())
    print("提示：把 fake_llm 换成真实 LLM，就是最小可用的记忆 bot。")
    print("=" * 72)


if __name__ == "__main__":
    main()
