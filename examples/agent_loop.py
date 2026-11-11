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


def render_context(mind: Thymos, hits, snapshot: dict) -> str:
    lines = ["## 相关记忆"]
    if not hits:
        lines.append("（没有找到相关记忆。不要编造。）")
    for hit in hits:
        m = hit.memory
        lines.append(
            f"- {m.content}（相关度 {hit.score:.2f}，可提取度 {hit.retrievability:.0%}，"
            f"情绪 {m.valence:+.2f}）"
        )
    affect = snapshot["affect"]
    lines.append("")
    lines.append("## 当前模拟状态")
    lines.append(
        f"心境 {affect['mood']['valence']:+.2f}；压力 {affect['stress']:.2f}；"
        f"主导情绪 {affect['emotion']['label']}。"
    )
    lines.append("这些是模拟状态，不是真实生理感受；只用于保持语气与叙事一致。")
    return "\n".join(lines)


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
        snapshot = mind.snapshot()
        context = render_context(mind, hits, snapshot)
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
