"""A fake agent loop that shows where Thymos fits.

This does not call an LLM.  It prints the prompt a real agent would send, so you
can see exactly which memories and affect values are injected.

Run:
    python examples/agent_loop.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from thymos import Profile, Thymos, ThymosConfig


def fake_llm(system_prompt: str, user_text: str) -> str:
    # A real implementation would call OpenAI/Anthropic/local model here.
    clean = user_text.rstrip("。！？.!?")
    return f"（模拟回应）我听到你说：{clean[:40]}。我会参考上述记忆，但不会假装有真实身体。"


def main() -> None:
    mind = Thymos(
        Profile(name="用户", location_name="上海", latitude=31.23, longitude=121.47),
        ThymosConfig(offline=True, seed=11),
        state_path=".thymos/agent_demo.json",
        now=1770000000.0,
    )

    turns = [
        "上次我们说的那个评审，今天终于通过了。",
        "其实我有点担心接下来会不会又加班。",
        "谢谢你记得。",
    ]

    for text in turns:
        hits = mind.recall(text, top_k=3)
        context = mind.render_context(hits)
        system_prompt = f"""你是一个有记忆的助手。以下是你可以使用的记忆与状态。
不要声称自己有真实的激素或身体；不要用情绪操控用户。

{context}
"""
        reply = fake_llm(system_prompt, text)
        print("=" * 72)
        print("USER:", text)
        print("-" * 72)
        print("SYSTEM PROMPT (excerpt):")
        print(system_prompt)
        print("-" * 72)
        print("ASSISTANT:", reply)

        importance = 0.7 if any(k in text for k in ("评审", "担心")) else 0.4
        mind.perceive(f"用户说：{text}\n助手回应：{reply}", importance=importance, tags=["对话"])

    print("=" * 72)
    print("夜间巩固：", mind.consolidate(7.5))
    mind.save()
    print("状态已保存到", mind.state_path)


if __name__ == "__main__":
    main()
