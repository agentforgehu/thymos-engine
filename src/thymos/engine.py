"""The engine: encode -> appraise -> feel -> retrieve -> consolidate."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict
from typing import Any, Callable

from .affect import AffectDynamics, EnvironmentEffects
from .appraisal import appraise, estimate_appraisal
from .config import Profile, ThymosConfig
from .embedding import make_embedder
from .endocrine import EndocrineSystem, circadian_factors
from .environment import (
    build_environment,
    environment_effects,
    local_datetime,
    sleep_debt,
)
from .facts import FactStore
from .forgetting import initial_stability, retrievability
from .mathx import clamp
from .memory import MemoryStore
from .retrieval import retrieve, review, time_bucket_for
from .types import Appraisal, Emotion, EnvironmentState, MemoryItem, RetrievalResult
from .weather import WeatherClient

Embedder = Callable[[str], list[float]]


class Thymos:
    """A small, inspectable model of memory and affect.

    Parameters
    ----------
    profile:
        Who the model is calibrated for (location, sleep schedule, sensitivities).
    config:
        Model parameters.  See docs/DESIGN.md for provenance.
    state_path:
        Optional JSON file for persistence.  If it exists it is loaded.
    now:
        Initial timestamp.  Defaults to wall-clock time; pass a fixed value for
        reproducible simulations.
    offline:
        If true, never call the weather API; use deterministic synthetic weather.
    """

    VERSION = 1

    def __init__(
        self,
        profile: Profile | None = None,
        config: ThymosConfig | None = None,
        *,
        state_path: str | None = None,
        now: float | None = None,
        offline: bool | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.config = config or ThymosConfig()
        self.profile = profile or Profile()
        if offline is not None:
            self.config.offline = offline
        self.rng = random.Random(self.config.seed)
        self.embedder = embedder or make_embedder(self.config.embedding_dim)
        self.store = MemoryStore()
        self.facts = FactStore()
        self.affect = AffectDynamics(self.config, self.profile)
        self.endocrine = EndocrineSystem()
        self.now = float(now if now is not None else time.time())
        self.state_path = state_path
        self.events: list[dict[str, Any]] = []
        self.weather = WeatherClient(self.config)
        self.environment = self.weather.current(self.profile, self.now)
        self._effects = environment_effects(self.environment, self.profile)
        self._tick_count = 0

        if state_path and os.path.exists(state_path):
            self.load(state_path)

    # ------------------------------------------------------------------ time
    def local_hour(self) -> float:
        local = local_datetime(self.now, self.profile.timezone)
        return local.hour + local.minute / 60.0

    def _ensure_embedding_dim(self, dim: int) -> None:
        """Re-embed the store if a new embedder has a different dimension.

        Switching from the hash embedder to a real model (or changing models)
        otherwise makes every cosine similarity silently return 0 because the
        vectors have different lengths.  Re-embedding is cheap for a personal
        memory file and keeps the state usable after the switch.
        """
        if dim <= 0:
            return
        items = list(self.store)
        if not items:
            return
        if all(len(item.embedding) == dim for item in items):
            return
        for item in items:
            item.embedding = self.embedder(item.content)
        self._log("reembed", {"count": len(items), "dim": dim})

    def _circadian(self) -> dict[str, float]:
        return circadian_factors(self.local_hour(), self.profile.chronotype)

    def _current_sleep_debt(self) -> float:
        return sleep_debt(self.profile, self.local_hour())

    def refresh_environment(self, force: bool = False) -> EnvironmentState:
        if not self.config.weather_enabled:
            return self.environment
        age = self.now - self.environment.fetched_at
        if force or age < 0 or age > self.config.weather_ttl_seconds:
            self.environment = self.weather.current(self.profile, self.now)
            self._effects = environment_effects(self.environment, self.profile)
        return self.environment

    # ------------------------------------------------------------------ tick
    def tick(self, seconds: float, *, sleeping: bool = False) -> dict[str, Any]:
        """Advance the model by `seconds` of simulated time."""
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        self.refresh_environment()
        remaining = float(seconds)
        # Sub-step so that a 24 h jump does not erase the intraday emotion
        # that mood is supposed to integrate.  One hour is a good trade-off.
        while remaining > 1e-6:
            chunk = min(remaining, 3600.0)
            self._advance_one_chunk(chunk, sleeping=sleeping)
            remaining -= chunk
        self._tick_count += 1
        return self.snapshot()

    def _advance_one_chunk(self, seconds: float, *, sleeping: bool) -> None:
        dt_min = seconds / 60.0
        dt_hours = seconds / 3600.0
        circ = self._circadian()
        debt = self._current_sleep_debt()
        effects = self._effects
        social_tone = clamp(0.25 + 0.25 * self.affect.state.mood_valence)

        self.endocrine.step(
            dt_min,
            stress=clamp(0.18 * self.affect.state.stress + 0.05 * (1.0 - effects.energy)),
            reward=clamp(0.3 * self.affect.state.mood_valence),
            social=social_tone,
            comfort=clamp(0.4 + 0.4 * self.affect.state.mood_valence),
            light=clamp(self.environment.radiation_wm2 / 700.0) if self.environment.is_day else 0.0,
            sleep_debt=debt,
            circadian=circ,
        )
        self.affect.step(
            dt_hours,
            self.endocrine,
            effects,
            circadian=circ,
            sleeping=sleeping,
            sleep_debt=debt,
        )
        self.affect.touch(self.now)
        self.now += seconds
        if self.config.weather_enabled and self.now - self.environment.fetched_at > self.config.weather_ttl_seconds:
            self.refresh_environment()
            effects = self._effects

    # --------------------------------------------------------------- perceive
    def perceive(
        self,
        content: str,
        *,
        appraisal: Appraisal | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        source: str = "user",
        context: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> MemoryItem:
        """Encode one event, appraise it, and update affect and hormones."""
        if timestamp is not None:
            self.now = float(timestamp)
        self.refresh_environment()
        content = (content or "").strip()
        if not content:
            raise ValueError("content must not be empty")
        tags = [t.strip() for t in (tags or []) if t and t.strip()]
        a = (appraisal or estimate_appraisal(content)).clamped()
        emotion: Emotion = appraise(a)
        self.affect.apply_emotion(emotion, self.endocrine, sensitivity=self.profile.emotional_reactivity)

        # Event-driven endocrine pulse over a ten-minute appraisal window.
        stress_pulse = clamp(max(0.0, -emotion.valence) * emotion.intensity * 1.2)
        reward_pulse = clamp(emotion.valence * emotion.intensity)
        social = 1.0 if ({"social", "朋友", "家人", "同事", "社交"} & set(tags) or any(w in content for w in ("朋友", "家人", "同事", "拥抱", "一起"))) else 0.0
        self.endocrine.step(
            10.0,
            stress=stress_pulse,
            reward=reward_pulse,
            social=social,
            comfort=clamp(max(0.0, emotion.valence) * 0.7),
            light=clamp(self.environment.radiation_wm2 / 700.0) if self.environment.is_day else 0.0,
            sleep_debt=self._current_sleep_debt(),
            circadian=self._circadian(),
        )

        encoding_gain = self.endocrine.encoding_gain(self.config)
        salience = clamp(
            0.55 * (importance if importance is not None else 0.5)
            + 0.30 * emotion.intensity
            + 0.15 * a.novelty
        )
        stability = initial_stability(
            salience=salience,
            valence=emotion.valence,
            arousal=emotion.arousal,
            encoding_gain=encoding_gain,
            base_days=self.config.base_stability_days,
            emotion_gain=self.config.emotion_stability_gain,
            salience_gain=self.config.salience_stability_gain,
            minimum=self.config.min_stability_days,
            maximum=self.config.max_stability_days,
        )

        embedding = self.embedder(content)
        self._ensure_embedding_dim(len(embedding))
        ctx = self._make_context()
        if context:
            ctx.update(context)
        item = MemoryItem(
            content=content,
            kind="episodic",
            tags=tags,
            embedding=embedding,
            context=ctx,
            created_at=self.now,
            last_access=self.now,
            access_times=[self.now],
            stability=stability,
            difficulty=clamp(0.55 - 0.25 * a.goal_relevance + 0.15 * a.novelty),
            valence=emotion.valence,
            arousal=emotion.arousal,
            salience=salience,
            source=source,
        )
        self.store.add(item)
        self.facts.observe(
            content,
            at=self.now,
            source=source,
            memory_id=item.id,
        )

        for similar, similarity in self.store.most_similar(
            embedding,
            exclude=item.id,
            limit=self.config.association_top_k,
            minimum=self.config.association_min_similarity,
        ):
            self.store.link(item.id, similar.id, similarity)

        self.affect.touch(self.now)
        self._log(
            "perceive",
            {
                "memory_id": item.id,
                "content": content,
                "emotion": emotion.label,
                "intensity": round(emotion.intensity, 3),
                "stability_days": round(stability, 3),
                "salience": round(salience, 3),
            },
        )
        return item

    # ----------------------------------------------------------------- recall
    def recall(self, query: str, *, top_k: int = 5, update: bool = True) -> list[RetrievalResult]:
        """Retrieve memories and (by default) treat the top hits as a review."""
        self.refresh_environment()
        query = (query or "").strip()
        embedding = self.embedder(query)
        self._ensure_embedding_dim(len(embedding))
        results = retrieve(
            self.store,
            now=self.now,
            config=self.config,
            query_embedding=embedding,
            query_tags=self._tags_from_query(query),
            env=self.environment,
            affect=self.affect.state,
            endocrine=self.endocrine,
            top_k=top_k,
        )
        reviewed = 0
        if update:
            for hit in results:
                if hit.components.get("semantic", 0.0) <= 0.01 and hit.components.get("tags", 0.0) <= 0.0:
                    continue
                quality = clamp(0.45 + 0.55 * hit.probability, 0.25, 0.98)
                review(
                    hit,
                    now=self.now,
                    quality=quality,
                    config=self.config,
                    encoding_gain=self.endocrine.encoding_gain(self.config),
                )
                reviewed += 1
        self._log(
            "recall",
            {
                "query": query,
                "reviewed": reviewed,
                "hits": [
                    {"id": h.memory.id, "score": round(h.score, 3), "content": h.memory.content[:60]}
                    for h in results
                ],
            },
        )
        return results

    # ------------------------------------------------------------ reappraise
    def reappraise(
        self,
        memory_id: str,
        *,
        new_valence: float | None = None,
        delta: float | None = None,
        note: str = "",
    ) -> MemoryItem:
        """Change the affective charge of a memory inside its labile window.

        Outside the window the update is halved, which encodes the empirical
        finding that reconsolidation is conditional rather than automatic.
        """
        item = self.store.get(memory_id)
        if item is None:
            raise KeyError(f"unknown memory: {memory_id}")
        labile = item.labile_until is not None and self.now <= item.labile_until
        strength = 1.0 if labile else 0.5
        target = new_valence if new_valence is not None else item.valence + (delta or 0.0)
        item.valence = clamp(item.valence + (target - item.valence) * strength, -1.0, 1.0)
        item.arousal = clamp(item.arousal * (1.0 - 0.15 * strength), 0.0, 1.0)
        item.stability = clamp(item.stability * (1.0 + 0.05 * strength), 0.02, self.config.max_stability_days)
        self._log(
            "reappraise",
            {"memory_id": memory_id, "labile": labile, "valence": round(item.valence, 3), "note": note},
        )
        return item

    # ------------------------------------------------------------ consolidate
    def consolidate(self, hours: float = 7.5, quality: float | None = None) -> dict[str, Any]:
        """Model a sleep period: replay important traces and abstract schemas."""
        if hours <= 0:
            raise ValueError("hours must be positive")
        debt_before = self._current_sleep_debt()
        self.tick(hours * 3600.0, sleeping=True)
        q = quality if quality is not None else clamp(1.0 - 0.45 * debt_before - 0.25 * self.affect.state.fatigue, 0.25, 1.0)

        candidates = sorted(
            self.store.all(),
            key=lambda m: m.salience * retrievability(m.age_days(self.now), m.stability, self.config.fsrs_decay),
            reverse=True,
        )[: self.config.consolidation_replay]
        replayed = 0
        for item in candidates:
            if item.state == "archived":
                continue
            item.stability = clamp(
                item.stability * (1.0 + self.config.consolidation_gain * q),
                self.config.min_stability_days,
                self.config.max_stability_days,
            )
            item.consolidated_at = self.now
            if item.reps >= self.config.semantic_abstraction_min_reps and item.stability >= 3.0:
                item.state = "semantic"
            replayed += 1

        schemas = self._abstract_schemas(q)
        self._log("consolidate", {"hours": hours, "quality": round(q, 3), "replayed": replayed, "schemas": schemas})
        return {
            "hours": hours,
            "quality": round(q, 3),
            "replayed": replayed,
            "schemas_created": schemas,
            "sleep_debt_before": round(debt_before, 3),
            "now": self.now,
        }

    def _abstract_schemas(self, quality: float) -> int:
        groups: dict[str, list[MemoryItem]] = {}
        for item in self.store.all():
            if item.kind != "episodic" or item.reps < self.config.semantic_abstraction_min_reps:
                continue
            for tag in item.tags[:2]:
                groups.setdefault(tag, []).append(item)

        created = 0
        for tag, items in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
            if created >= self.config.semantic_abstraction_limit or len(items) < 2:
                break
            if any(m.kind == "semantic" and tag in m.tags for m in self.store.all()):
                continue
            mean_valence = sum(i.valence for i in items) / len(items)
            mean_arousal = sum(i.arousal for i in items) / len(items)
            content = f"关于「{tag}」的反复经验：{len(items)} 次相关事件，整体感受{'偏正面' if mean_valence > 0.1 else '偏负面' if mean_valence < -0.1 else '中性'}。"
            emb = self.embedder(content)
            self._ensure_embedding_dim(len(emb))
            schema = MemoryItem(
                content=content,
                kind="semantic",
                tags=[tag, "schema"],
                embedding=emb,
                context=self._make_context(),
                created_at=self.now,
                last_access=self.now,
                access_times=[self.now],
                stability=max(i.stability for i in items) * (1.0 + 0.3 * quality),
                difficulty=0.35,
                valence=mean_valence,
                arousal=mean_arousal * 0.6,
                salience=clamp(0.5 + 0.1 * len(items)),
                source="consolidation",
            )
            self.store.add(schema)
            for item in items[:6]:
                self.store.link(schema.id, item.id, 0.6)
            created += 1
        return created

    # -------------------------------------------------------------- snapshots
    def _make_context(self) -> dict[str, Any]:
        env = self.environment
        return {
            "place": env.location_name,
            "weather_bucket": env.weather_bucket,
            "condition": env.condition,
            "temp_c": env.temp_c,
            "time_bucket": time_bucket_for(env, self.local_hour()),
            "season": env.season,
            "is_day": env.is_day,
            "source": "perception",
        }

    def _tags_from_query(self, query: str) -> list[str]:
        if not query:
            return []
        tags = [t for t in ("工作", "家人", "朋友", "健康", "学习", "旅行", "天气") if t in query]
        return tags

    def top_memories(self, limit: int = 5) -> list[dict[str, Any]]:
        items = sorted(
            self.store.all(),
            key=lambda m: m.salience * retrievability(m.age_days(self.now), m.stability, self.config.fsrs_decay),
            reverse=True,
        )[:limit]
        return [
            m.to_public_dict(self.now, retrievability(m.age_days(self.now), m.stability, self.config.fsrs_decay))
            for m in items
        ]

    def render_context(
        self,
        hits: list[RetrievalResult] | None = None,
        *,
        include_endocrine: bool = False,
        query: str = "",
        max_chars: int | None = None,
    ) -> str:
        """Render facts, retrieved memories and affect as a compact prompt block.

        ``max_chars`` is a hard character budget, not a hint.  It exists because
        a local model often has a much smaller context than the memory store;
        the renderer must choose what to show instead of dumping Top-k.

        Structured facts come first: they are the smallest representation of
        "what is true now", and they are not allowed to compete with dozens of
        episodic traces for a Top-k slot.
        """
        hits = hits or []
        fact_block = self.facts.render(
            query,
            at=self.now,
            max_chars=int(max_chars * 0.45) if max_chars else None,
        )
        stale_values = self.facts.stale_values(self.now, query)

        seen: set[str] = set()
        memory_lines = ["## 相关记忆"]
        for hit in hits:
            m = hit.memory
            if any(value and value in m.content for value in stale_values):
                continue
            # Consolidation can leave a semantic copy with the same content as
            # its episodic source.  Do not show the model the same fact twice.
            key = m.content.strip()
            if key in seen:
                continue
            seen.add(key)
            if max_chars:
                memory_lines.append(f"- {m.content[:180]}")
            else:
                memory_lines.append(
                    f"- {m.content}（相关度 {hit.score:.2f}，可提取度 {hit.retrievability:.0%}，"
                    f"情绪 {m.valence:+.2f}，稳定度 {m.stability:.2f} 天）"
                )
        if len(memory_lines) == 1:
            memory_lines.append("（没有找到相关记忆。不要编造。）")

        s = self.affect.state
        state_lines = [
            "## 当前模拟状态",
            f"心境 {s.mood_valence:+.2f}；压力 {s.stress:.2f}；"
            f"疲劳 {s.fatigue:.2f}；主导情绪 {s.dominant_emotion}。",
            "这些是模拟状态，不是真实生理感受；只用于保持语气与叙事一致。",
        ]
        if include_endocrine:
            levels = ", ".join(f"{k}={v:.2f}" for k, v in self.endocrine.levels.items())
            state_lines.append(f"相对内分泌水平：{levels}（0–1 相对单位，非医学指标）")

        if not max_chars:
            parts = ([fact_block] if fact_block else []) + [
                "\n".join(memory_lines),
                "\n".join(state_lines),
            ]
            return "\n\n".join(parts)

        # Budgeted path: facts first, then as many compact memories as fit,
        # then the short state block.  Never exceed the caller's budget.
        parts: list[str] = []
        if fact_block:
            parts.append(fact_block)
        used = sum(len(part) + 2 for part in parts)
        remaining = max(0, max_chars - used - len("\n".join(state_lines)) - 2)
        selected = memory_lines[:1]
        for line in memory_lines[1:]:
            if len(line) + 1 > remaining:
                break
            selected.append(line)
            remaining -= len(line) + 1
        if len(selected) == 1 and remaining > 20:
            selected.append("（预算不足，未展开情景记忆。）")
        parts.append("\n".join(selected))
        parts.append("\n".join(state_lines))
        text = "\n\n".join(parts)
        if len(text) > max_chars:
            text = text[: max(0, max_chars)].rstrip()
        return text

    def stats(self) -> dict[str, Any]:
        items = self.store.all()
        by_state: dict[str, int] = {}
        for item in items:
            by_state[item.state] = by_state.get(item.state, 0) + 1
        avg_r = (
            sum(retrievability(m.age_days(self.now), m.stability, self.config.fsrs_decay) for m in items) / len(items)
            if items
            else 0.0
        )
        return {
            "total": len(items),
            "by_state": by_state,
            "avg_retrievability": round(avg_r, 4),
            "mean_stability_days": round(sum(m.stability for m in items) / len(items), 3) if items else 0.0,
            "ticks": self._tick_count,
        }

    def snapshot(self) -> dict[str, Any]:
        local = local_datetime(self.now, self.profile.timezone)
        return {
            "version": self.VERSION,
            "now": self.now,
            "local_time": local.isoformat(),
            "profile": self.profile.to_dict(),
            "environment": {
                **self.environment.to_dict(),
                "effects": self._effects.to_dict(),
            },
            "affect": self.affect.snapshot(),
            "endocrine": self.endocrine.snapshot(),
            "facts": self.facts.to_list(),
            "memory": self.stats(),
            "top_memories": self.top_memories(6),
            "events": self.events[-20:],
            "config": {
                "seed": self.config.seed,
                "offline": self.config.offline,
                "mood_setpoint": self.config.mood_setpoint,
            },
        }

    def _log(self, kind: str, payload: dict[str, Any]) -> None:
        self.events.append({"at": self.now, "kind": kind, **payload})
        if len(self.events) > 400:
            self.events = self.events[-400:]

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "now": self.now,
            "profile": self.profile.to_dict(),
            "config": self.config.to_dict(),
            "affect": self.affect.to_dict(),
            "endocrine": self.endocrine.to_dict(),
            "environment": self.environment.to_dict(),
            "memory": self.store.to_list(),
            "facts": self.facts.to_list(),
            "events": self.events[-200:],
            "tick_count": self._tick_count,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        state_path: str | None = None,
        offline: bool | None = None,
        embedder: Embedder | None = None,
    ) -> "Thymos":
        profile = Profile.from_dict(data.get("profile"))
        config = ThymosConfig.from_dict(data.get("config"))
        # Do not pass state_path here: __init__ would try to load the same file
        # again and recurse.  Attach it after the object is built.
        obj = cls(
            profile,
            config,
            state_path=None,
            now=data.get("now"),
            offline=offline,
            embedder=embedder,
        )
        obj.state_path = state_path
        obj.affect = AffectDynamics.from_dict(config, profile, data.get("affect"))
        obj.endocrine = EndocrineSystem.from_dict(data.get("endocrine"))
        obj.store = MemoryStore.from_list(data.get("memory"))
        obj.facts = FactStore.from_list(data.get("facts"))
        obj.environment = EnvironmentState.from_dict(data.get("environment"))
        obj._effects = environment_effects(obj.environment, profile)
        obj.events = list(data.get("events") or [])
        obj._tick_count = int(data.get("tick_count") or 0)
        return obj

    def save(self, path: str | None = None) -> str:
        target = path or self.state_path
        if not target:
            raise ValueError("no state path configured")
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        tmp = f"{target}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        os.replace(tmp, target)
        self.state_path = target
        return target

    def load(self, path: str | None = None) -> "Thymos":
        target = path or self.state_path
        if not target or not os.path.exists(target):
            raise FileNotFoundError(target)
        with open(target, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        loaded = Thymos.from_dict(data, state_path=target, embedder=self.embedder)
        self.__dict__.update(loaded.__dict__)
        return self

    def __repr__(self) -> str:
        return (
            f"Thymos(profile={self.profile.name!r}, memories={len(self.store)}, "
            f"mood={self.affect.state.mood_valence:+.2f}, now={self.now:.0f})"
        )
