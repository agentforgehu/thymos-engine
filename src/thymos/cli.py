"""Command line interface: init, demo, perceive, recall, tick, serve."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Any

from .config import Profile, ThymosConfig
from .engine import Thymos
from .types import Appraisal

DEFAULT_STATE = os.path.join(".thymos", "state.json")
DEMO_EVENTS = [
    (1, 9.0, "晨会汇报通过了，大家认可我的方案，挺开心的", 0.85, ["工作"]),
    (1, 21.5, "加班到很晚，回家路上有点孤独", 0.55, ["工作"]),
    (2, 15.0, "收到朋友寄来的礼物，完全没想到，很惊喜", 0.8, ["朋友", "social"]),
    (3, 11.0, "线上服务崩了，客户在群里投诉，压力很大", 0.95, ["工作"]),
    (3, 20.0, "女朋友陪我散步，听我讲完，心情慢慢平复了", 0.8, ["家人", "social"]),
    (5, 18.0, "第一次跑完十公里，虽然累但很骄傲", 0.75, ["健康"]),
    (6, 10.0, "下雨取消了出行计划，有点失望", 0.45, ["天气"]),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thymos", description="Thymos — 类人情绪与记忆引擎")
    parser.add_argument("--state", default=DEFAULT_STATE, help="状态文件路径")
    parser.add_argument("--offline", action="store_true", help="不访问天气 API，使用合成天气")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="初始化一个新的状态文件")
    init.add_argument("--name", default="朋友")
    init.add_argument("--city", default=None, help="城市名，用于地理编码")
    init.add_argument("--lat", type=float, default=31.2304)
    init.add_argument("--lon", type=float, default=121.4737)
    init.add_argument("--timezone", default="Asia/Shanghai")
    init.add_argument("--wake", type=float, default=7.5, help="起床时刻，0-24")
    init.add_argument("--sleep-hours", type=float, default=7.5)

    demo = sub.add_parser("demo", help="运行 7 天演示场景")
    demo.add_argument("--days", type=int, default=7)
    demo.add_argument("--json-out", default=os.path.join(".thymos", "demo_snapshot.json"))

    perceive = sub.add_parser("perceive", help="记录一件事")
    perceive.add_argument("content")
    perceive.add_argument("--importance", type=float, default=0.5)
    perceive.add_argument("--tags", default="")
    perceive.add_argument("--valence", type=float, default=None, help="显式可取性 -1..1")
    perceive.add_argument("--arousal", type=float, default=None, help="显式唤醒 0..1")
    perceive.add_argument("--agency", default="circumstance", choices=["self", "other", "circumstance"])

    recall = sub.add_parser("recall", help="按线索提取记忆")
    recall.add_argument("query")
    recall.add_argument("--top-k", type=int, default=5)

    plan = sub.add_parser("plan", help="按控制平面生成上下文与审计轨迹")
    plan.add_argument("query")
    plan.add_argument("--top-k", type=int, default=5)
    plan.add_argument("--max-chars", type=int, default=1200)
    plan.add_argument("--json", action="store_true", dest="as_json", help="输出完整计划 JSON")

    tick = sub.add_parser("tick", help="推进模拟时间")
    tick.add_argument("--hours", type=float, default=1.0)

    sleep = sub.add_parser("consolidate", help="模拟一次睡眠巩固")
    sleep.add_argument("--hours", type=float, default=7.5)

    sub.add_parser("snapshot", help="打印当前完整状态 JSON")

    migrate = sub.add_parser("migrate", help="将旧 JSON 状态迁移到 SQLite")
    migrate.add_argument("source", help="旧 .json 状态文件")
    migrate.add_argument("target", help="新的 .sqlite/.db 状态文件")

    serve = sub.add_parser("serve", help="启动本地可视化面板")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--no-browser", action="store_true")
    serve.add_argument("--token", default=None, help="Bearer token；默认读取 THYMOS_API_TOKEN")
    return parser


def load_engine(args: argparse.Namespace) -> Thymos:
    path = args.state
    if os.path.exists(path):
        engine = Thymos(state_path=path, offline=args.offline or None)
        if args.offline:
            engine.config.offline = True
        return engine
    profile = Profile()
    config = ThymosConfig(seed=args.seed, offline=args.offline)
    return Thymos(profile, config, state_path=path)


def cmd_init(args: argparse.Namespace) -> int:
    profile = Profile(
        name=args.name,
        latitude=args.lat,
        longitude=args.lon,
        location_name=args.city or args.name,
        timezone=args.timezone,
        wake_hour=args.wake,
        sleep_hours=args.sleep_hours,
    )
    if args.city and not args.offline:
        from .weather import geocode

        hit = geocode(args.city)
        if hit:
            profile.location_name = hit["name"] or args.city
            profile.latitude = hit["latitude"]
            profile.longitude = hit["longitude"]
            profile.timezone = hit["timezone"] or profile.timezone
            print(f"已定位：{profile.location_name} ({profile.latitude:.3f}, {profile.longitude:.3f}) {profile.timezone}")
        else:
            print("地理编码失败，使用手动坐标。")

    engine = Thymos(profile, ThymosConfig(seed=args.seed, offline=args.offline), state_path=args.state)
    engine.save()
    print(f"已创建 {args.state}")
    print(f"画像：{profile.name} @ {profile.location_name}，起床 {profile.wake_hour:g} 点")
    print("下一步：python -m thymos demo 或 python -m thymos serve")
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    from .sqlite_store import migrate_json

    store = migrate_json(args.source, args.target)
    try:
        print(f"已迁移 {args.source} -> {args.target}（schema v{store.schema_version}）")
        print(f"记忆条目：{len(store)}")
    finally:
        store.close()
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    import datetime as _dt

    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("Asia/Shanghai")
    except Exception:  # pragma: no cover
        tz = _dt.timezone.utc
    start = _dt.datetime(2026, 3, 2, 8, 0, tzinfo=tz).timestamp()

    profile = Profile(name="演示用户", location_name="上海", latitude=31.23, longitude=121.47, timezone="Asia/Shanghai")
    mind = Thymos(profile, ThymosConfig(offline=True, seed=args.seed), now=start)

    print("Thymos 演示：7 天，离线合成天气\n")
    print(f"{'日期':<6}{'情绪':<12}{'心境':>7}{'压力':>7}{'皮质醇':>8}{'记忆':>6}")
    print("-" * 52)

    last_day = 0
    for day, hour, content, importance, tags in DEMO_EVENTS:
        target = start + (day - 1) * 86400 + (hour - 8.0) * 3600
        while last_day < day - 1:
            mind.tick(24 * 3600 - 8 * 3600)
            mind.consolidate(7.5)
            last_day += 1
        if target > mind.now:
            mind.tick(target - mind.now)
        mind.perceive(content, importance=importance, tags=tags)
        if day != last_day:
            s = mind.snapshot()
            print(
                f"第{day}天  {s['affect']['emotion']['label']:<10}"
                f"{s['affect']['mood']['valence']:>+7.3f}"
                f"{s['affect']['stress']:>7.3f}"
                f"{s['endocrine']['levels']['cortisol']:>8.3f}"
                f"{s['memory']['total']:>6}"
            )
            last_day = day

    mind.consolidate(7.5)
    print("\n第 8 天清晨，按线索「工作 压力」回忆：")
    for hit in mind.recall("工作 压力", top_k=4):
        m = hit.memory
        print(f"  [{hit.score:.2f}] {m.content[:32]}  稳定度={m.stability:.2f}天 情绪={m.valence:+.2f}")

    snap = mind.snapshot()
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, ensure_ascii=False, indent=2)
    print(f"\n快照已保存：{args.json_out}")
    print(
        "最终状态："
        f"心境 {snap['affect']['mood']['valence']:+.2f}，"
        f"压力 {snap['affect']['stress']:.2f}，"
        f"记忆 {snap['memory']['total']} 条，"
        f"平均可提取度 {snap['memory']['avg_retrievability']:.2f}"
    )
    return 0


def cmd_perceive(args: argparse.Namespace) -> int:
    engine = load_engine(args)
    appraisal = None
    if args.valence is not None or args.arousal is not None:
        from .appraisal import estimate_appraisal

        base = estimate_appraisal(args.content)
        appraisal = Appraisal(
            goal_relevance=base.goal_relevance,
            desirability=args.valence if args.valence is not None else base.desirability,
            novelty=base.novelty,
            certainty=base.certainty,
            control=base.control,
            agency=args.agency,
            norm_compliance=base.norm_compliance,
            stakes=base.stakes,
            note="cli",
        )
    tags = [t for t in args.tags.split(",") if t]
    item = engine.perceive(args.content, appraisal=appraisal, importance=args.importance, tags=tags)
    engine.save()
    print(f"已记录 {item.id}")
    print(f"  情绪电荷 {item.valence:+.2f} / 唤醒 {item.arousal:.2f} / 显著度 {item.salience:.2f}")
    print(f"  初始稳定度 {item.stability:.3f} 天，情境 {item.context.get('weather_bucket')} {item.context.get('temp_c')}°C")
    print(f"  当前心境 {engine.affect.state.mood_valence:+.3f}，压力 {engine.affect.state.stress:.3f}")
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    engine = load_engine(args)
    hits = engine.recall(args.query, top_k=args.top_k)
    if not hits:
        print("没有可提取的记忆。")
        return 0
    for i, hit in enumerate(hits, 1):
        m = hit.memory
        print(f"\n{i}. [{hit.score:.3f}] {m.content}")
        print(
            f"   可提取度={hit.retrievability:.2f} 激活={hit.activation:+.2f} "
            f"概率={hit.probability:.2f} 稳定度={m.stability:.2f}天 复习={m.reps}"
        )
        print(f"   情境：{m.context.get('place')} / {m.context.get('weather_bucket')} / {m.context.get('time_bucket')}")
    engine.save()
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    engine = load_engine(args)
    plan = engine.plan_context(
        args.query,
        top_k=args.top_k,
        max_chars=args.max_chars,
    )
    if args.as_json:
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(plan.text)
        print("\n--- 审计 ---")
        print(
            f"意图 {plan.intent.label} ({plan.intent.confidence:.2f})；"
            f"情绪权重 {plan.affect.mood_weight:.2f}；"
            f"状态块 {'开' if plan.affect.include_state else '关'}；"
            f"{plan.audit.chars} 字符 / {plan.audit.tokens} tokens"
        )
        for op in plan.audit.operators:
            print(f"  {op.name:<18} {op.status:<8} 选中 {op.selected}")
    engine.save()
    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    engine = load_engine(args)
    engine.tick(args.hours * 3600)
    engine.save()
    snap = engine.snapshot()
    print(
        f"时间推进 {args.hours:g} 小时 → 心境 {snap['affect']['mood']['valence']:+.3f}，"
        f"压力 {snap['affect']['stress']:.3f}，疲劳 {snap['affect']['fatigue']:.3f}"
    )
    print(f"皮质醇 {snap['endocrine']['levels']['cortisol']:.3f}，褪黑素 {snap['endocrine']['levels']['melatonin']:.3f}")
    return 0


def cmd_consolidate(args: argparse.Namespace) -> int:
    engine = load_engine(args)
    result = engine.consolidate(args.hours)
    engine.save()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    engine = load_engine(args)
    print(json.dumps(engine.snapshot(), ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .api import serve

    engine = load_engine(args)
    if not os.path.exists(args.state):
        engine.save()
    serve(
        engine,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        token=args.token,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init":
        return cmd_init(args)
    if args.command == "demo":
        return cmd_demo(args)
    if args.command == "perceive":
        return cmd_perceive(args)
    if args.command == "recall":
        return cmd_recall(args)
    if args.command == "plan":
        return cmd_plan(args)
    if args.command == "tick":
        return cmd_tick(args)
    if args.command == "consolidate":
        return cmd_consolidate(args)
    if args.command == "snapshot":
        return cmd_snapshot(args)
    if args.command == "migrate":
        return cmd_migrate(args)
    if args.command == "serve":
        return cmd_serve(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
