"""Minimal Thymos example: encode, feel, recall, sleep, inspect.

Run from the repository root:
    python examples/basic.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from thymos import Profile, Thymos, ThymosConfig


def main() -> None:
    profile = Profile(
        name="小林",
        location_name="上海",
        latitude=31.23,
        longitude=121.47,
        timezone="Asia/Shanghai",
    )
    mind = Thymos(profile, ThymosConfig(offline=True, seed=7), now=1770000000.0)

    events = [
        ("项目评审通过了，大家认可我的方案，我松了一口气", 0.85, ["工作"]),
        ("晚上和朋友吃饭，聊了很久，感觉很温暖", 0.7, ["朋友", "social"]),
        ("线上服务崩了，客户投诉，我有点焦虑", 0.9, ["工作"]),
    ]
    for content, importance, tags in events:
        item = mind.perceive(content, importance=importance, tags=tags)
        print(f"[encode] {content[:20]}… emotion={item.valence:+.2f} S={item.stability:.2f}d")

    print("\n[state before recall]")
    snap = mind.snapshot()
    print("  mood:", snap["affect"]["mood"])
    print("  stress:", snap["affect"]["stress"])
    print("  cortisol:", snap["endocrine"]["levels"]["cortisol"])

    print("\n[recall: 工作]")
    for hit in mind.recall("工作 压力", top_k=3):
        m = hit.memory
        print(f"  {hit.score:.2f}  R={hit.retrievability:.2f}  {m.content[:28]}")

    print("\n[advance 6h + sleep]")
    mind.tick(6 * 3600)
    print("  ", mind.consolidate(7.5))

    print("\n[top memories]")
    for m in mind.top_memories(3):
        print(f"  {m['retrievability']:.2f}  {m['content'][:30]}")


if __name__ == "__main__":
    main()
