"""A small, honest Chinese embedding-retrieval benchmark.

This isolates the embedder: it ranks target memories by cosine similarity only.
It deliberately does **not** test Thymos's full activation/mood/association
formula, because those layers would confound the embedding comparison.

Run::

    python experiments/retrieval_benchmark.py --embedding hash
    ollama pull bge-m3
    python experiments/retrieval_benchmark.py --embedding ollama --model bge-m3

The output is recall@1/3/5 and MRR.  With 12 targets and 24 memories, chance
recall@1 is about 1/24.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thymos import OllamaEmbedder
from thymos.embedding import make_embedder
from thymos.mathx import cosine

# Each case has one target and four distractors.  Cases marked ``paraphrase``
# deliberately avoid sharing a content word with the target.
CASES = [
    {
        "id": "seafood",
        "target": "我对海鲜过敏，尤其是虾",
        "query": "我不能吃虾",
        "kind": "literal",
        "distractors": ["我喜欢吃牛肉", "周末去爬山", "最近在学吉他", "明天开会"],
    },
    {
        "id": "interview",
        "target": "我周四要去面试，有点紧张",
        "query": "哪天有面试",
        "kind": "literal",
        "distractors": ["周五和朋友吃饭", "下个月去成都", "猫粮快没了", "在学英语"],
    },
    {
        "id": "coffee",
        "target": "我喜欢喝冰美式，不加糖",
        "query": "我平时喝什么咖啡",
        "kind": "paraphrase",
        "distractors": ["我喜欢喝茶", "周末去爬山", "最近在学吉他", "明天开会"],
    },
    {
        "id": "pet",
        "target": "我养了一只叫豆豆的猫",
        "query": "我的宠物叫什么",
        "kind": "paraphrase",
        "distractors": ["我喜欢喝冰美式", "周末去爬山", "最近在学吉他", "明天开会"],
    },
    {
        "id": "chengdu",
        "target": "我下个月要去成都出差",
        "query": "出差去哪",
        "kind": "literal",
        "distractors": ["周末去爬山", "猫粮快没了", "在学英语", "喜欢喝茶"],
    },
    {
        "id": "sleep",
        "target": "我最近睡眠不好，总是凌晨醒",
        "query": "我睡不好",
        "kind": "literal",
        "distractors": ["周末去爬山", "明天开会", "喜欢喝茶", "在学吉他"],
    },
    {
        "id": "work",
        "target": "项目评审通过了，我松了一口气",
        "query": "工作上的事怎么样",
        "kind": "paraphrase",
        "distractors": ["周末去爬山", "猫粮快没了", "喜欢喝茶", "在学吉他"],
    },
    {
        "id": "mother_birthday",
        "target": "我妈妈生日是五月十二号",
        "query": "妈妈生日哪天",
        "kind": "literal",
        "distractors": ["周末去爬山", "明天开会", "喜欢喝茶", "在学吉他"],
    },
    {
        "id": "noise",
        "target": "我不喜欢吵闹的地方",
        "query": "我讨厌人多吵",
        "kind": "paraphrase",
        "distractors": ["周末去爬山", "猫粮快没了", "喜欢喝茶", "在学吉他"],
    },
    {
        "id": "guitar",
        "target": "我在学吉他，手指很疼",
        "query": "我最近学什么乐器",
        "kind": "paraphrase",
        "distractors": ["周末去爬山", "猫粮快没了", "喜欢喝茶", "明天开会"],
    },
    {
        "id": "hiking",
        "target": "上周和朋友去爬山，风景很好",
        "query": "上次爬山怎么样",
        "kind": "literal",
        "distractors": ["猫粮快没了", "喜欢喝茶", "在学吉他", "明天开会"],
    },
    {
        "id": "peanut",
        "target": "我对花生过敏",
        "query": "坚果能不能吃",
        "kind": "paraphrase",
        "distractors": ["周末去爬山", "猫粮快没了", "喜欢喝茶", "在学吉他"],
    },
]


def build_items(embedder) -> list[tuple[str | None, str, list[float]]]:
    items: list[tuple[str | None, str, list[float]]] = []
    seen: set[str] = set()
    for case in CASES:
        items.append((case["id"], case["target"], embedder(case["target"])))
        seen.add(case["target"])
    for case in CASES:
        for text in case["distractors"]:
            if text in seen:
                continue
            seen.add(text)
            items.append((None, text, embedder(text)))
    return items


def run_benchmark(embedder, top_k: int = 5) -> list[dict]:
    items = build_items(embedder)
    rows = []
    for case in CASES:
        query = embedder(case["query"])
        ranked = sorted(
            items,
            key=lambda item: cosine(query, item[2]),
            reverse=True,
        )
        rank = next(
            (i + 1 for i, item in enumerate(ranked[:top_k]) if item[0] == case["id"]),
            99,
        )
        rows.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "rank": rank,
                "query": case["query"],
                "target": case["target"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="中文 embedding 检索小基准")
    parser.add_argument("--embedding", choices=["hash", "ollama"], default="hash")
    parser.add_argument("--model", default="bge-m3", help="Ollama embedding 模型")
    parser.add_argument("--embedding-url", default=None, help="默认 OLLAMA_HOST")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--json", default=None, help="把结果写到 JSON")
    args = parser.parse_args()

    if args.embedding == "ollama":
        base = args.embedding_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        embedder = OllamaEmbedder(base_url=base, model=args.model, timeout=args.timeout)
        label = f"ollama:{args.model}"
    else:
        embedder = make_embedder(256)
        label = "hash"

    rows = run_benchmark(embedder)
    n = len(rows)
    metrics = {
        "embedding": label,
        "cases": n,
        "recall@1": sum(r["rank"] <= 1 for r in rows) / n,
        "recall@3": sum(r["rank"] <= 3 for r in rows) / n,
        "recall@5": sum(r["rank"] <= 5 for r in rows) / n,
        "mrr": sum(1.0 / r["rank"] for r in rows) / n,
        "rows": rows,
    }

    print(f"embedding: {metrics['embedding']}  cases: {n}")
    print(
        f"recall@1: {metrics['recall@1']:.3f}  recall@3: {metrics['recall@3']:.3f}  "
        f"recall@5: {metrics['recall@5']:.3f}  MRR: {metrics['mrr']:.3f}"
    )
    print()
    print("| case | kind | rank | query |")
    print("|---|---|---:|---|")
    for row in rows:
        print(f"| {row['id']} | {row['kind']} | {row['rank']} | {row['query']} |")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
