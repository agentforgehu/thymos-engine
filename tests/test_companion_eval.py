from experiments.eval_companion_zh import DATA, evaluate_emotion, read_jsonl

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


def test_eval_runner_returns_finite_metrics():
    rows = read_jsonl(DATA / "cped_emotion_sample.jsonl")[:20]
    report = evaluate_emotion(rows)
    assert report["n"] == 20
    for key in ("emotion_accuracy", "emotion_macro_f1", "sentiment_accuracy", "sentiment_macro_f1"):
        assert 0.0 <= report[key] <= 1.0
    assert report["examples"][0]["text"]
