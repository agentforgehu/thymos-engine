"""Show how the memory control plane routes facts, episodes and affect.

Run:

    python examples/control_plane.py
"""

from __future__ import annotations

import json
import os
import tempfile

from thymos import AgentMemory, Profile, ThymosConfig

START = 1770000000.0


def main() -> int:
    path = os.path.join(tempfile.gettempdir(), "thymos_control_plane_demo.json")
    memory = AgentMemory(
        path,
        profile=Profile(name="林墨", location_name="杭州", timezone="Asia/Shanghai"),
        config=ThymosConfig(offline=True, seed=7),
        now=START,
    )

    for text in [
        "我叫林墨，住在杭州。",
        "我对花生过敏，点菜要注意。",
        "我养了一只橘猫，叫豆豆。",
        "最近工作压力很大，有点焦虑。",
        "周末喜欢去爬山，天气好会去。",
    ]:
        memory.observe(text, min_importance=0)
        memory.mind.tick(3600)

    for query in ["我叫什么？", "最近压力很大，陪我聊聊", "这段 Python 代码报错怎么修"]:
        plan = memory.plan(query, max_chars=520)
        print("=" * 72)
        print("QUERY:", query)
        print(
            "intent={label}  affect={mood_weight}  state={include_state}".format(
                label=plan["intent"]["label"],
                mood_weight=plan["affect"]["mood_weight"],
                include_state=plan["affect"]["include_state"],
            )
        )
        print("operators:", [(op["name"], op["status"], op["selected"]) for op in plan["operators"]])
        print("budget:", plan["audit"]["budget"], "used:", plan["audit"]["chars"], "chars")
        print(plan["context"])

    print("=" * 72)
    print("last audit JSON:")
    print(json.dumps(memory.mind.last_audit, ensure_ascii=False, indent=2)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
