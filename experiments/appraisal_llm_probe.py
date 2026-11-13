"""Compare the lexical appraiser with an optional local LLM appraiser.

This is a small probe, not a benchmark.  It checks whether a chat model can
return the structured appraisal schema reliably and whether the resulting
labels differ from the deterministic fallback in the cases we care about:
insults, gratitude, loss, sarcasm, neutral facts and unexpected loss.

Usage:
    python experiments/appraisal_llm_probe.py --base-url http://127.0.0.1:8080/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thymos.appraisal import appraise, estimate_appraisal  # noqa: E402
from thymos.llm_appraisal import LLMAppraiser  # noqa: E402

CASES = [
    ("你怎么这么笨，闭嘴", "insult"),
    ("谢谢你，这次真的帮了我大忙", "gratitude"),
    ("项目黄了，团队可能要散", "loss"),
    ("你可真厉害啊，呵呵", "sarcasm"),
    ("我住在杭州，做软件", "neutral_fact"),
    ("我特别喜欢这个方案", "positive"),
    ("没想到项目黄了", "unexpected_loss"),
]


def make_chat(base_url: str, model: str, timeout: float):
    endpoint = base_url.rstrip("/") + "/chat/completions"

    def chat(messages):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 320,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    return chat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=len(CASES))
    args = parser.parse_args()

    appraiser = LLMAppraiser(make_chat(args.base_url, args.model, args.timeout))
    rows = []
    for text, kind in CASES[: args.limit]:
        started = time.perf_counter()
        rule = estimate_appraisal(text)
        rule_emotion = appraise(rule)
        try:
            llm = appraiser(text)
            llm_emotion = appraise(llm)
            error = ""
        except Exception as exc:  # noqa: BLE001 - probe should show failures
            llm = None
            llm_emotion = None
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "kind": kind,
                "text": text,
                "rule_label": rule_emotion.label,
                "rule_d": round(rule.desirability, 3),
                "llm_label": llm_emotion.label if llm_emotion else "ERROR",
                "llm_d": round(llm.desirability, 3) if llm else None,
                "agency": llm.agency if llm else None,
                "control": round(llm.control, 3) if llm else None,
                "stakes": round(llm.stakes, 3) if llm else None,
                "note": llm.note if llm else error,
                "seconds": round(elapsed, 2),
            }
        )

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print("\n| kind | text | rule | llm | llm d | agency | sec |")
    print("|---|---|---|---|---:|---|---:|")
    for row in rows:
        print(
            "| {kind} | {text} | {rule_label} | {llm_label} | {llm_d} | {agency} | {seconds} |".format(
                **row
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
