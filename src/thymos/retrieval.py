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

from .config import ThymosConfig
from .forgetting import retrievability
from .mathx import clamp, cosine, jaccard, sigmoid
from .memory import MemoryStore
from .types import AffectState, EnvironmentState, MemoryItem, RetrievalResult


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
) -> tuple[float, dict[str, float]]:
    sem = max(0.0, cosine(query_embedding, memory.embedding)) if query_embedding else 0.0
    tags = jaccard(query_tags, memory.tags) if query_tags else 0.0
    place = 1.0 if env and memory.context.get("place") == env.location_name else 0.0
    weather = 1.0 if env and memory.context.get("weather_bucket") == env.weather_bucket else 0.0
    time_bucket = 1.0 if env and memory.context.get("time_bucket") == time_bucket_for(env) else 0.0

    score = 0.55 * sem + 0.20 * tags + 0.10 * place + 0.08 * weather + 0.07 * time_bucket
    mood_fit = 1.0 - abs(memory.valence - mood_valence) / 2.0
    components = {
        "semantic": sem,
        "tags": tags,
        "place": place,
        "weather": weather,
        "time": time_bucket,
        "mood": mood_fit,
        "context": score,
    }
    return score, components


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
) -> list[RetrievalResult]:
    query_embedding = query_embedding or []
    query_tags = query_tags or []
    mood = affect.mood_valence if affect else 0.0
    retrieval_gain = endocrine.retrieval_gain(config) if endocrine else 1.0
    results: list[RetrievalResult] = []

    for memory in store.all():
        if memory.state == "archived":
            continue
        r = retrievability(memory.age_days(now), memory.stability, config.fsrs_decay)
        if r < 0.02:
            continue

        base = actr_base_activation(memory.access_times or [memory.created_at], now, config.actr_decay)
        spread = 0.0
        for neighbor, weight in store.neighbors(memory.id):
            nr = retrievability(neighbor.age_days(now), neighbor.stability, config.fsrs_decay)
            spread += weight * nr
        spread = min(1.5, spread)

        ctx, parts = context_similarity(memory, query_embedding, query_tags, env, mood)
        mood_congruence = parts["mood"]
        emotional = 0.5 * memory.arousal + 0.5 * abs(memory.valence)

        activation = (
            config.activation_weight * (base / 6.0)
            + 1.10 * ctx
            + config.mood_congruence_weight * (mood_congruence - 0.5) * 2.0
            + config.emotion_boost_weight * emotional
            + 0.25 * spread
        )
        activation *= retrieval_gain

        probability = sigmoid((activation - config.retrieval_threshold) / max(0.05, config.retrieval_noise))
        score = config.retrievability_weight * r + (1.0 - config.retrievability_weight) * probability
        score *= 1.0 + 0.12 * memory.salience

        results.append(
            RetrievalResult(
                memory=memory,
                score=score,
                retrievability=r,
                activation=activation,
                probability=probability,
                components={
                    **parts,
                    "base_activation": base / 6.0,
                    "spread": spread,
                    "emotional": emotional,
                    "retrieval_gain": retrieval_gain,
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
