from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "evals" / "companion_zh" / "data"

EXPECTED_EMOTIONS = {
    "anger",
    "astonished",
    "depress",
    "disgust",
    "fear",
    "grateful",
    "happy",
    "negative-other",
    "neutral",
    "positive-other",
    "relaxed",
    "sadness",
    "worried",
}


def read_jsonl(path: Path) -> list[dict]:
    import json

    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_cped_sample_has_real_source_and_balanced_labels():
    rows = read_jsonl(DATA / "cped_emotion_sample.jsonl")
    assert len(rows) == 156
    assert {row["gold"]["emotion"] for row in rows} == EXPECTED_EMOTIONS
    assert all(row["source"]["dataset"] == "CPED" for row in rows)
    assert all(row["source"]["split"] == "test" for row in rows)
    assert all(row["text"].strip() for row in rows)
    assert all(row["source"]["utterance_id"] for row in rows)


def test_companion_windows_are_real_multiturn_and_labeled():
    rows = read_jsonl(DATA / "companion_windows.jsonl")
    assert len(rows) == 8
    for row in rows:
        assert row["source"]["dataset"] == "CPED"
        assert len(row["events"]) >= 5
        assert row["expected"]["majority_sentiment"] in {"positive", "negative", "neutral"}
        assert row["expected"]["label_source"].startswith("CPED")
        assert row["probe"]


def test_companion_eval_has_a_published_baseline():
    import json

    path = DATA.parent / "results" / "lexical_baseline.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["sample_size"] == 156
    assert report["window_count"] == 8
    assert 0.0 <= report["emotion_accuracy"] <= 1.0
    assert 0.0 <= report["sentiment_accuracy"] <= 1.0
    assert report["window_summary_retrievable_rate"] == 1.0
