"""Data structures shared across the engine."""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass(slots=True)
class Appraisal:
    """Structured appraisal of an event.

    The fields follow OCC-style appraisal (relevance, desirability, agency,
    certainty, control) rather than a free-form sentiment score.  `note`
    keeps the human-readable reason so decisions stay auditable.
    """

    goal_relevance: float = 0.5
    desirability: float = 0.0
    novelty: float = 0.2
    certainty: float = 0.6
    control: float = 0.5
    agency: str = "circumstance"  # self | other | circumstance
    norm_compliance: float = 0.0
    stakes: float = 0.5
    note: str = ""

    def clamped(self) -> "Appraisal":
        return Appraisal(
            goal_relevance=_c(self.goal_relevance),
            desirability=_c(self.desirability, -1.0, 1.0),
            novelty=_c(self.novelty),
            certainty=_c(self.certainty),
            control=_c(self.control),
            agency=self.agency if self.agency in {"self", "other", "circumstance"} else "circumstance",
            norm_compliance=_c(self.norm_compliance, -1.0, 1.0),
            stakes=_c(self.stakes),
            note=self.note,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Appraisal":
        return _from_dict(cls, data)


@dataclass(slots=True)
class Emotion:
    """A transient emotion episode produced by appraisal."""

    label: str
    intensity: float
    valence: float
    arousal: float
    dominance: float
    half_life_min: float
    appraisal: Appraisal = field(default_factory=Appraisal)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["appraisal"] = self.appraisal.to_dict()
        return data


@dataclass(slots=True)
class AffectState:
    """Fast emotion + slow mood + stress/fatigue load."""

    valence: float = 0.0
    arousal: float = 0.25
    dominance: float = 0.0
    mood_valence: float = 0.06
    mood_arousal: float = 0.35
    stress: float = 0.15
    fatigue: float = 0.2
    allostatic_load: float = 0.05
    dominant_emotion: str = "neutral"
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AffectState":
        return _from_dict(cls, data)


@dataclass
class MemoryItem:
    """A single memory trace.

    `stability` is expressed in days: the time after which retrievability
    falls to 0.9 under the FSRS-style curve.  `access_times` keeps the raw
    retrieval history so ACT-R base-level activation can be recomputed without
    lossy counters.
    """

    content: str
    kind: str = "episodic"  # episodic | semantic | affective
    tags: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    last_access: float = 0.0
    access_times: list[float] = field(default_factory=list)
    stability: float = 0.55
    difficulty: float = 0.5
    reps: int = 0
    lapses: int = 0
    valence: float = 0.0
    arousal: float = 0.2
    salience: float = 0.5
    source: str = "user"
    associations: dict[str, float] = field(default_factory=dict)
    state: str = "buffer"  # buffer | episodic | semantic | archived
    labile_until: float | None = None
    consolidated_at: float | None = None
    # Write-path audit.  A merged memory is not overwritten: the original
    # trace stays, and these fields record how many observations it absorbed.
    canonical_key: str = ""
    merged_count: int = 0
    last_merged_at: float | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryItem":
        return _from_dict(cls, data)

    def age_days(self, now: float) -> float:
        return max(0.0, (now - self.last_access) / 86400.0)

    def to_public_dict(self, now: float, retrievability: float) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "kind": self.kind,
            "tags": list(self.tags),
            "state": self.state,
            "stability_days": round(self.stability, 4),
            "retrievability": round(retrievability, 4),
            "salience": round(self.salience, 3),
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "reps": self.reps,
            "lapses": self.lapses,
            "merged_count": self.merged_count,
            "last_merged_at": _iso(self.last_merged_at) if self.last_merged_at else "",
            "created_at": _iso(self.created_at),
            "last_access": _iso(self.last_access),
            "age_days": round(self.age_days(now), 4),
            "context": self.context,
        }


@dataclass
class RetrievalResult:
    memory: MemoryItem
    score: float
    retrievability: float
    activation: float
    probability: float
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self, now: float) -> dict[str, Any]:
        return {
            "memory": self.memory.to_public_dict(now, self.retrievability),
            "score": round(self.score, 4),
            "activation": round(self.activation, 4),
            "probability": round(self.probability, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
        }


@dataclass
class WriteDecision:
    """Why a turn was stored, merged, reinforced or skipped.

    This is deliberately data, not a log string: the host can show it in a
    debug panel, and the 100-turn harness can count admission outcomes.
    """

    action: str  # store | merge | reinforce | skip
    reason: str
    importance: float = 0.0
    similarity: float | None = None
    memory_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "importance": round(self.importance, 4),
            "similarity": None if self.similarity is None else round(self.similarity, 4),
            "memory_id": self.memory_id,
        }


@dataclass
class EnvironmentState:
    """Weather + place + daylight for one moment."""

    timestamp: float = 0.0
    location_name: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    timezone: str = "UTC"
    temp_c: float = 20.0
    apparent_c: float = 20.0
    humidity: float = 60.0
    pressure_hpa: float = 1013.0
    cloud_cover: float = 40.0
    precipitation_mm: float = 0.0
    radiation_wm2: float = 300.0
    wind_kph: float = 8.0
    condition: str = "clear"
    weather_code: int = 0
    daylight_hours: float = 12.0
    is_day: bool = True
    season: str = "spring"
    weather_bucket: str = "mild"
    source: str = "synthetic"
    fetched_at: float = 0.0
    pressure_trend_3h: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EnvironmentState":
        return _from_dict(cls, data)


def _c(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = lo
    if math.isnan(v):
        return lo
    return max(lo, min(hi, v))


def _from_dict(cls, data: dict[str, Any] | None):
    if not data:
        return cls()
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in allowed})


def _iso(ts: float | None) -> str | None:
    if not ts:
        return None
    import datetime as _dt

    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()
