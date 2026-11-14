"""Evaluate the lexical appraisal baseline on a public Chinese dialogue sample.

This is intentionally a *real-data* check, not a generated happy-path demo.
The primary file is a stratified sample of the CPED test split: real Chinese
TV dialogue with human emotion/sentiment annotations.  The window file uses
real multi-turn excerpts and derived CPED labels.  It is still not a substitute
for a companion-product user study; see evals/companion_zh/README.md.

Usage:
    python experiments/eval_companion_zh.py
    python experiments/eval_companion_zh.py --out .work/companion_eval.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from thymos import AgentMemory, Profile, ThymosConfig
from thymos.appraisal import appraise, estimate_appraisal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evals" / "companion_zh" / "data"

# The engine has a smaller label set than CPED.  This mapping is fixed before
# looking at results; do not tune it per example.
EMOTION_TO_CPED = {
    "neutral": "neutral",
    "joy": "happy",
    "contentment": "relaxed",
    "excitement": "happy",
    "trust": "grateful",
    "hope": "positive-other",
    "anticipation": "positive-other",
    "surprise": "astonished",
    "sadness": "sadness",
    "anger": "anger",
    "anxiety": "worried",
    "fear": "fear",
    "disgust": "disgust",
    "depressed": "depress",
}

POSITIVE = {"joy", "contentment", "excitement", "trust", "hope", "anticipation"}
NEGATIVE = {"sadness", "anger", "anxiety", "fear", "disgust", "depressed"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def predicted_emotion(text: str) -> str:
    return appraise(estimate_appraisal(text)).label


def predicted_sentiment(label: str) -> str:
    if label in POSITIVE:
        return "positive"
    if label in NEGATIVE:
        return "negative"
    return "neutral"


def macro_f1(gold: list[str], pred: list[str], labels: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[label] = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0
    return out


def evaluate_emotion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold_emotion = [row["gold"]["emotion"] for row in rows]
    gold_sentiment = [row["gold"]["sentiment"] for row in rows]
    pred_labels = [predicted_emotion(row["text"]) for row in rows]
    pred_emotions = [EMOTION_TO_CPED.get(label, "neutral") for label in pred_labels]
    pred_sentiments = [predicted_sentiment(label) for label in pred_labels]

    emotion_labels = sorted(set(gold_emotion) | set(pred_emotions))
    sentiment_labels = ["positive", "neutral", "negative"]
    emotion_f1 = macro_f1(gold_emotion, pred_emotions, emotion_labels)
    sentiment_f1 = macro_f1(gold_sentiment, pred_sentiments, sentiment_labels)
    return {
        "n": len(rows),
        "emotion_accuracy": round(sum(g == p for g, p in zip(gold_emotion, pred_emotions)) / len(rows), 4),
        "emotion_macro_f1": round(sum(emotion_f1.values()) / len(emotion_f1), 4),
        "sentiment_accuracy": round(sum(g == p for g, p in zip(gold_sentiment, pred_sentiments)) / len(rows), 4),
        "sentiment_macro_f1": round(sum(sentiment_f1.values()) / len(sentiment_f1), 4),
        "emotion_per_label_f1": emotion_f1,
        "sentiment_per_label_f1": sentiment_f1,
        "gold_emotion_counts": dict(Counter(gold_emotion)),
        "pred_emotion_counts": dict(Counter(pred_emotions)),
        "pred_sentiment_counts": dict(Counter(pred_sentiments)),
        "examples": [
            {
                "id": row["id"],
                "text": row["text"],
                "gold_emotion": row["gold"]["emotion"],
                "pred_emotion": pred,
                "pred_label": label,
                "gold_sentiment": row["gold"]["sentiment"],
                "pred_sentiment": predicted_sentiment(label),
            }
            for row, pred, label in zip(rows, pred_emotions, pred_labels)
        ],
    }


def evaluate_windows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for row in rows:
        events = row["events"]
        with tempfile.TemporaryDirectory() as tmp:
            memory = AgentMemory(
                str(Path(tmp) / "window.json"),
                profile=Profile(name="评估", location_name="杭州", timezone="Asia/Shanghai"),
                config=ThymosConfig(offline=True, seed=7),
                now=events[0]["at"],
            )
            window = memory.observe_stream(events, window="今晚", step_seconds=900)
            context = memory.context(row["probe"], top_k=5, max_chars=900)
            behavior = memory.behavior(row["probe"]).to_dict()
        if window.valence_mean > 0.05:
            pred = "positive"
        elif window.valence_mean < -0.05:
            pred = "negative"
        else:
            pred = "neutral"
        results.append(
            {
                "id": row["id"],
                "probe": row["probe"],
                "expected_majority_sentiment": row["expected"]["majority_sentiment"],
                "predicted_sentiment": pred,
                "valence_mean": round(window.valence_mean, 4),
                "mood_end": round(window.mood_end, 4),
                "stress_max": round(window.stress_max, 4),
                "dominant_labels": window.labels,
                "summary_retrievable": "窗口" in context,
                "behavior": behavior,
            }
        )
    gold = [r["expected_majority_sentiment"] for r in results]
    pred = [r["predicted_sentiment"] for r in results]
    return {
        "n": len(results),
        "majority_sentiment_accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(results), 4),
        "summary_retrievable_rate": round(sum(r["summary_retrievable"] for r in results) / len(results), 4),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA))
    parser.add_argument("--out", default=str(ROOT / ".work" / "companion_eval.json"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    emotion_rows = read_jsonl(data_dir / "cped_emotion_sample.jsonl")
    window_rows = read_jsonl(data_dir / "companion_windows.jsonl")

    report = {
        "dataset": "CPED test split (public; see evals/companion_zh/NOTICE.md)",
        "emotion": evaluate_emotion(emotion_rows),
        "windows": evaluate_windows(window_rows),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== CPED emotion sample ===")
    for key in (
        "n",
        "emotion_accuracy",
        "emotion_macro_f1",
        "sentiment_accuracy",
        "sentiment_macro_f1",
    ):
        print(f"{key}: {report['emotion'][key]}")
    print("=== real multi-turn windows ===")
    for key in ("n", "majority_sentiment_accuracy", "summary_retrievable_rate"):
        print(f"{key}: {report['windows'][key]}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
