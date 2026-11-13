"""Auditable admission, deduplication and merge policy for new memories.

The engine already knows how to encode a trace; this module decides whether a
new observation deserves a *new* trace, should reinforce/merge an existing one,
or should be skipped.  The distinction matters for long chats:

* exact repeats should not inflate the store;
* a paraphrase of the same event should deepen one trace, not create twins;
* a repeated event with a different date/number may be a genuinely new episode;
* a low-value turn should not consume a memory slot just because it is recent.

The policy is intentionally conservative and observable.  It returns a
``WriteDecision`` and never deletes the original trace.
"""

from __future__ import annotations

import re
from typing import Any

from .forgetting import initial_stability
from .mathx import clamp, cosine
from .retrieval import lexical_terms
from .types import WriteDecision

_PUNCT = re.compile(r"[\s，,。！？!?；;：:、\"'“”‘’（）()\[\]【】…—\-]+")
_DIGITS = re.compile(r"\d+")


def normalize_key(text: str) -> str:
    """Normalize surface form for exact-repeat detection.

    This is not semantic normalization.  It only removes punctuation and
    whitespace so ``"项目黄了。"`` and ``"项目黄了"`` are recognized as the
    same observation.
    """
    return _PUNCT.sub("", (text or "").strip().lower())


def number_signature(text: str) -> tuple[str, ...]:
    """Numbers that make two otherwise similar events non-identical.

    ``"4 月 15 日交付"`` and ``"5 月 1 日交付"`` should not be merged just
    because their embeddings are close.  Chinese numerals are intentionally
    left to the fact store; this is a conservative guard, not a date parser.
    """
    return tuple(_DIGITS.findall(text or ""))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_merge_candidate(
    store: Any,
    text: str,
    embedding: list[float],
    now: float,
    config: Any,
    *,
    tags: list[str] | None = None,
) -> tuple[Any | None, float | None, str]:
    """Find an existing trace to reinforce or merge into.

    Returns ``(memory, similarity, reason)``.  Exact duplicates are always
    eligible, including across sessions.  Semantic merges require both a high
    similarity and a recent-enough trace; a number conflict vetoes the merge.
    """
    key = normalize_key(text)
    if not key:
        return None, None, "empty"

    memories = [m for m in store.all() if m.state != "archived"]
    for item in memories:
        item_key = item.canonical_key or normalize_key(item.content)
        if key == item_key:
            return item, 1.0, "exact_duplicate"

    if not getattr(config, "write_semantic_merge_enabled", True):
        return None, None, "no_exact_match"

    threshold = float(getattr(config, "write_merge_similarity", 0.90))
    window_hours = float(getattr(config, "write_merge_window_hours", 72.0))
    query_terms = set(lexical_terms(text))
    new_numbers = number_signature(text)
    best: tuple[float, Any] | None = None

    for item in memories:
        if item.kind not in ("episodic", "affective"):
            continue
        age_hours = max(0.0, (now - item.last_access) / 3600.0)
        if age_hours > window_hours:
            continue
        old_numbers = number_signature(item.content)
        if new_numbers and old_numbers and new_numbers != old_numbers:
            continue
        semantic = cosine(embedding, item.embedding) if embedding and item.embedding else 0.0
        lexical = _jaccard(query_terms, set(lexical_terms(item.content)))
        # A lexical overlap is weaker evidence than an embedding match, so it
        # can only lift an already-plausible candidate, not create one alone.
        similarity = max(semantic, 0.85 * lexical)
        if similarity >= threshold and (best is None or similarity > best[0]):
            best = (similarity, item)

    if best is None:
        return None, None, "no_similar_trace"
    return best[1], best[0], "semantic_merge"


def merge_memory(
    target: Any,
    text: str,
    embedding: list[float],
    *,
    now: float,
    emotion: Any,
    importance: float,
    tags: list[str] | None,
    config: Any,
) -> Any:
    """Deepen an existing trace without deleting its history.

    The new observation is treated as evidence about the same event, so
    repetition increases stability and ``merged_count`` but does not create a
    second memory.  The longer, more informative wording wins; otherwise the
    original wording is preserved.
    """
    old_key = target.canonical_key or normalize_key(target.content)
    new_key = normalize_key(text)
    if len(new_key) > len(old_key) * 1.25:
        target.content = text.strip()
        if embedding:
            target.embedding = list(embedding)

    target.canonical_key = old_key or new_key
    target.tags = sorted(set(target.tags) | set(tags or []))
    target.last_access = now
    target.access_times.append(now)
    target.reps += 1
    target.merged_count += 1
    target.last_merged_at = now
    target.salience = clamp(max(target.salience, importance * 0.8 + 0.2 * target.salience))
    target.valence = clamp(0.75 * target.valence + 0.25 * float(getattr(emotion, "valence", 0.0)))
    target.arousal = clamp(max(target.arousal, float(getattr(emotion, "arousal", 0.0))))
    target.stability = clamp(
        max(
            target.stability,
            initial_stability(
                salience=target.salience,
                valence=target.valence,
                arousal=target.arousal,
                encoding_gain=1.0,
                base_days=getattr(config, "base_stability_days", 0.55),
                emotion_gain=getattr(config, "emotion_stability_gain", 0.9),
                salience_gain=getattr(config, "salience_stability_gain", 1.2),
                minimum=getattr(config, "min_stability_days", 0.02),
                maximum=getattr(config, "max_stability_days", 365.0),
            ),
        )
        * 1.05,
        getattr(config, "min_stability_days", 0.02),
        getattr(config, "max_stability_days", 365.0),
    )
    return target


def skip_decision(reason: str, importance: float = 0.0) -> WriteDecision:
    return WriteDecision(action="skip", reason=reason, importance=importance)


def store_decision(importance: float, memory_id: str | None = None) -> WriteDecision:
    return WriteDecision(action="store", reason="admitted", importance=importance, memory_id=memory_id)


def merge_decision(
    importance: float,
    similarity: float,
    memory_id: str,
    reason: str,
) -> WriteDecision:
    action = "reinforce" if reason == "exact_duplicate" else "merge"
    return WriteDecision(
        action=action,
        reason=reason,
        importance=importance,
        similarity=similarity,
        memory_id=memory_id,
    )
