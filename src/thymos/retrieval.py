"""Retrieval: ACT-R base-level activation + context + mood congruence.

Two different quantities are reported separately:

* `retrievability` (FSRS) asks "how much of the trace is left?";
* `activation` (ACT-R style) asks "how available is it right now, given
  frequency, recency, cues, associations and mood?".

A memory can have a strong trace yet be temporarily unavailable because the
cue does not match; conflating the two is a common modelling error.
"""

from __future__ import annotations

import math
import re

from .config import ThymosConfig
from .embedding import tokenize
from .forgetting import retrievability
from .mathx import clamp, cosine, jaccard, sigmoid
from .memory import MemoryStore
from .types import AffectState, EnvironmentState, MemoryItem, RetrievalResult

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN = re.compile(r"[a-z0-9_']+")


def actr_base_activation(access_times: list[float], now: float, decay: float = 0.5) -> float:
    """B_i = ln( sum_k (now - t_k)^(-d) ), with a 1-second floor."""
    if not access_times:
        return -8.0
    total = 0.0
    for t in access_times:
        age = max(1.0, now - t)
        total += age ** (-decay)
    if total <= 0:
        return -8.0
    return math.log(total)


def context_similarity(
    memory: MemoryItem,
    query_embedding: list[float],
    query_tags: list[str],
    env: EnvironmentState | None,
    mood_valence: float,
    lexical: float = 0.0,
    *,
    semantic_weight: float = 0.45,
    lexical_weight: float = 0.22,
    tag_weight: float = 0.18,
    place_weight: float = 0.07,
    weather_weight: float = 0.05,
    time_weight: float = 0.03,
) -> tuple[float, dict[str, float]]:
    sem = max(0.0, cosine(query_embedding, memory.embedding)) if query_embedding else 0.0
    tags = jaccard(query_tags, memory.tags) if query_tags else 0.0
    place = 1.0 if env and memory.context.get("place") == env.location_name else 0.0
    weather = 1.0 if env and memory.context.get("weather_bucket") == env.weather_bucket else 0.0
    time_bucket = 1.0 if env and memory.context.get("time_bucket") == time_bucket_for(env) else 0.0

    # The lexical channel is deliberately separate from the hash embedding.
    # It keeps names, product codes and Chinese bigrams working when the
    # embedder is weak, while a real embedder can still dominate through
    # semantic evidence.  The weights are configurable and sum to 1.0.
    score = (
        semantic_weight * sem
        + tag_weight * tags
        + lexical_weight * max(0.0, min(1.0, lexical))
        + place_weight * place
        + weather_weight * weather
        + time_weight * time_bucket
    )
    mood_fit = 1.0 - abs(memory.valence - mood_valence) / 2.0
    components = {
        "semantic": sem,
        "lexical": lexical,
        "tags": tags,
        "place": place,
        "weather": weather,
        "time": time_bucket,
        "mood": mood_fit,
        "context": score,
    }
    return score, components


def lexical_overlap(query: str, content: str) -> float:
    """Query-normalised overlap of hashed n-gram tokens."""
    q = set(tokenize(query or ""))
    if not q:
        return 0.0
    c = set(tokenize(content or ""))
    return len(q & c) / max(1, len(q))


def lexical_terms(text: str, *, unigrams: bool = False) -> list[str]:
    """Terms for BM25: CJK bigrams + Latin words.

    Unigrams are intentionally excluded.  In a short Chinese query, single
    characters such as ``的`` or ``我`` create noisy matches; bigrams retain
    names and compounds while staying dependency-free.
    """
    text = (text or "").lower()
    chars = _CJK.findall(text)
    terms = [a + b for a, b in zip(chars, chars[1:])]
    if unigrams:
        terms.extend(chars)
    terms.extend(_LATIN.findall(text))
    return terms


def bm25_scores(
    query: str,
    contents: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    unigrams: bool = False,
) -> list[float]:
    """Okapi BM25 over CJK bigrams, normalized to 0..1 per query.

    This is a retrieval channel, not a claim that bigram matching understands
    Chinese.  It is the cheap, auditable half of a hybrid retriever; the
    embedding channel handles paraphrase and synonymy.
    """
    q_terms = set(lexical_terms(query, unigrams=unigrams))
    docs = [lexical_terms(content, unigrams=unigrams) for content in contents]
    if not q_terms or not docs:
        return [0.0] * len(docs)
    n_docs = len(docs)
    avg_len = sum(len(doc) for doc in docs) / max(1, n_docs)
    df = {term: sum(1 for doc in docs if term in doc) for term in q_terms}
    raw: list[float] = []
    for doc in docs:
        score = 0.0
        for term in q_terms:
            tf = doc.count(term)
            if not tf:
                continue
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf + k1 * (1.0 - b + b * len(doc) / max(1.0, avg_len))
            score += idf * tf * (k1 + 1.0) / max(1e-9, denom)
        raw.append(score)
    maximum = max(raw) if raw else 0.0
    if maximum <= 1e-12:
        return [0.0] * len(raw)
    return [min(1.0, value / maximum) for value in raw]


def exact_phrase_overlap(query: str, content: str) -> float:
    """Longest shared CJK/Latin phrase, normalized by query length.

    This is intentionally small: it gives an exact-name/number bonus without
    turning the retriever into a substring search engine.
    """
    q = (query or "").strip().lower()
    c = (content or "").strip().lower()
    if not q or not c:
        return 0.0
    # A query of one character is too ambiguous to reward.
    if len(q) < 2:
        return 0.0
    best = 0
    # Dynamic programming would be O(n*m); these memories are short.  The
    # loop is explicit and easy to audit.
    for start in range(len(q)):
        for end in range(start + 2, len(q) + 1):
            phrase = q[start:end]
            if phrase in c:
                best = max(best, len(phrase))
            else:
                break
    return min(1.0, best / max(2, len(q)))


def time_bucket_for(env: EnvironmentState, hour: float | None = None) -> str:
    if hour is None:
        from .environment import local_datetime

        hour = local_datetime(env.timestamp, env.timezone).hour + local_datetime(env.timestamp, env.timezone).minute / 60.0
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 14:
        return "noon"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 23:
        return "evening"
    return "night"


def retrieve(
    store: MemoryStore,
    *,
    now: float,
    config: ThymosConfig,
    query_embedding: list[float] | None = None,
    query_tags: list[str] | None = None,
    env: EnvironmentState | None = None,
    affect: AffectState | None = None,
    endocrine=None,
    top_k: int = 5,
    query_text: str | None = None,
    mood_weight: float | None = None,
    emotion_weight: float | None = None,
) -> list[RetrievalResult]:
    query_embedding = query_embedding or []
    query_tags = query_tags or []
    mood = affect.mood_valence if affect else 0.0
    retrieval_gain = endocrine.retrieval_gain(config) if endocrine else 1.0
    mood_weight = (
        config.mood_congruence_weight if mood_weight is None else max(0.0, min(1.0, float(mood_weight)))
    )
    emotion_weight = (
        config.emotion_boost_weight if emotion_weight is None else max(0.0, min(1.0, float(emotion_weight)))
    )
    results: list[RetrievalResult] = []
    candidates = [memory for memory in store.all() if memory.state != "archived"]
    contents = [memory.content for memory in candidates]
    bm25_bigrams = bm25_scores(query_text or "", contents)
    bm25_unigrams = bm25_scores(query_text or "", contents, unigrams=True)
    lexical_scores = [max(bi, 0.85 * uni) for bi, uni in zip(bm25_bigrams, bm25_unigrams)]

    for memory, bm25, bm25_uni in zip(candidates, lexical_scores, bm25_unigrams):
        r = retrievability(memory.age_days(now), memory.stability, config.fsrs_decay)
        if r < 0.02:
            continue

        base = actr_base_activation(memory.access_times or [memory.created_at], now, config.actr_decay)
        spread = 0.0
        for neighbor, weight in store.neighbors(memory.id):
            nr = retrievability(neighbor.age_days(now), neighbor.stability, config.fsrs_decay)
            spread += weight * nr
        spread = min(1.5, spread)

        exact = exact_phrase_overlap(query_text or "", memory.content)
        lexical = max(lexical_overlap(query_text or "", memory.content), bm25)
        lexical = min(1.0, lexical + 0.20 * exact)
        ctx, parts = context_similarity(
            memory,
            query_embedding,
            query_tags,
            env,
            mood,
            lexical=lexical,
            semantic_weight=getattr(config, "retrieval_semantic_weight", 0.45),
            lexical_weight=getattr(config, "retrieval_lexical_weight", 0.22),
            tag_weight=getattr(config, "retrieval_tag_weight", 0.18),
            place_weight=getattr(config, "retrieval_place_weight", 0.07),
            weather_weight=getattr(config, "retrieval_weather_weight", 0.05),
            time_weight=getattr(config, "retrieval_time_weight", 0.03),
        )
        mood_congruence = parts["mood"]
        emotional = 0.5 * memory.arousal + 0.5 * abs(memory.valence)

        activation = (
            config.activation_weight * (base / 6.0)
            + 1.10 * ctx
            + mood_weight * (mood_congruence - 0.5) * 2.0
            + emotion_weight * emotional
            + getattr(config, "association_spread_weight", 0.10) * spread
        )
        activation *= retrieval_gain

        probability = sigmoid((activation - config.retrieval_threshold) / max(0.05, config.retrieval_noise))
        # Do not rank by the sigmoid alone: once several candidates cross the
        # threshold, their probabilities all approach 1.0 and a salience
        # multiplier can flip the order.  Keep ``probability`` for review
        # quality/telemetry, but rank by the un-squashed activation so a
        # clearly relevant memory still wins.
        score = (
            config.retrievability_weight * r
            + (1.0 - config.retrievability_weight) * max(0.0, activation)
            + 0.05 * memory.salience
        )

        results.append(
            RetrievalResult(
                memory=memory,
                score=score,
                retrievability=r,
                activation=activation,
                probability=probability,
                components={
                    **parts,
                    "bm25": bm25,
                    "bm25_unigram": bm25_uni,
                    "exact": exact,
                    "base_activation": base / 6.0,
                    "spread": spread,
                    "emotional": emotional,
                    "retrieval_gain": retrieval_gain,
                    "mood_weight": mood_weight,
                    "emotion_weight": emotion_weight,
                },
            )
        )

    results.sort(key=lambda item: item.score, reverse=True)
    return results[: max(1, top_k)]


def review(
    result: RetrievalResult,
    *,
    now: float,
    quality: float,
    config: ThymosConfig,
    encoding_gain: float = 1.0,
) -> None:
    """Apply a successful (or failed) retrieval to the memory trace."""
    from .forgetting import update_stability

    memory = result.memory
    memory.reps += 1
    memory.access_times.append(now)
    if len(memory.access_times) > 64:
        memory.access_times = memory.access_times[-64:]
    memory.last_access = now
    memory.stability = update_stability(
        memory.stability,
        result.retrievability,
        quality=quality,
        gain=encoding_gain,
        growth=config.stability_growth if hasattr(config, "stability_growth") else 1.7,
        maximum=config.max_stability_days,
    )
    if quality < 0.35:
        memory.lapses += 1
    memory.labile_until = now + config.reconsolidation_window_min * 60.0
    if memory.state == "buffer":
        memory.state = "episodic"
