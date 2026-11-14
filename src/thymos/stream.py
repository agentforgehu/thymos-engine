"""Continuous event streams and window summaries.

A chat turn is not the only unit a companion needs.  People arrive with an
evening: a commute, a meeting, a small win, a bad message, a quiet stretch.
This module provides the data structures and the deterministic summary
builder; :class:`thymos.agent.AgentMemory` owns the side effects (affect,
facts, memory writes).

The summary is a *compressed episode*, not a diagnosis and not a replacement
for the individual memories.  It is stored as a semantic memory so a later
question such as "今晚怎么样" can retrieve the window without replaying every
turn.
"""

from __future__ import annotations

import datetime as _dt
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from .types import Appraisal


@dataclass
class Event:
    """One message or observation in a stream.

    ``at`` is an absolute Unix timestamp.  When omitted, :func:`normalize_events`
    assigns a monotonic timestamp using ``step_seconds``.  The host should pass
    real timestamps whenever it has them; gaps are what make an evening feel
    different from a burst of messages.
    """

    text: str
    at: float | None = None
    role: str = "user"  # user | assistant | system
    importance: float | None = None
    tags: list[str] = field(default_factory=list)
    appraisal: Appraisal | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = (self.text or "").strip()
        if not self.text:
            raise ValueError("event text must not be empty")
        self.role = (self.role or "user").lower()
        if self.role not in {"user", "assistant", "system"}:
            self.role = "user"
        if self.at is not None:
            self.at = float(self.at)
        self.tags = [str(tag).strip() for tag in (self.tags or []) if str(tag).strip()]
        if isinstance(self.appraisal, Mapping):
            self.appraisal = Appraisal.from_dict(dict(self.appraisal))
        self.metadata = dict(self.metadata or {})

    @classmethod
    def coerce(cls, value: "Event | Mapping[str, Any] | str", *, now: float, step_seconds: float) -> "Event":
        """Accept an Event, a JSON-like dict, or a plain string."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(text=value)
        if isinstance(value, Mapping):
            data = dict(value)
            text = data.get("text", data.get("content", data.get("message", "")))
            at = data.get("at", data.get("timestamp", data.get("time")))
            appraisal = data.get("appraisal")
            return cls(
                text=str(text),
                at=float(at) if at is not None else None,
                role=str(data.get("role", "user")),
                importance=data.get("importance"),
                tags=list(data.get("tags", [])),
                appraisal=appraisal,
                metadata=dict(data.get("metadata", {})),
            )
        raise TypeError(f"unsupported event value: {type(value)!r}")


def normalize_events(
    values: Iterable[Event | Mapping[str, Any] | str],
    *,
    now: float,
    step_seconds: float = 60.0,
) -> list[Event]:
    """Return a timestamped, monotonic event list.

    Explicit timestamps are preserved when possible.  A timestamp earlier than
    the stream cursor is clamped to the cursor: a live bot should never move
    the affective clock backwards just because a host sent an old message.
    """
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    out: list[Event] = []
    cursor = float(now)
    for index, value in enumerate(values):
        event = Event.coerce(value, now=cursor, step_seconds=step_seconds)
        if event.at is None:
            event.at = cursor if index == 0 else cursor + step_seconds
        event.at = max(float(event.at), cursor)
        cursor = event.at
        out.append(event)
    return out


@dataclass
class WindowSummary:
    """A compact, auditable description of one event window."""

    window: str
    start_at: float
    end_at: float
    event_count: int
    user_turns: int
    assistant_turns: int
    valence_mean: float
    valence_min: float
    valence_max: float
    mood_start: float
    mood_end: float
    stress_start: float
    stress_end: float
    stress_max: float
    labels: dict[str, int] = field(default_factory=dict)
    topics: list[str] = field(default_factory=list)
    turning_points: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    memory_id: str | None = None
    records: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def duration_minutes(self) -> float:
        return max(0.0, (self.end_at - self.start_at) / 60.0)

    def to_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        data = asdict(self)
        data["duration_minutes"] = round(self.duration_minutes, 2)
        for key in ("valence_mean", "valence_min", "valence_max", "mood_start", "mood_end", "stress_start", "stress_end", "stress_max"):
            data[key] = round(float(data[key]), 4)
        if not include_records:
            data.pop("records", None)
        return data


def build_window_summary(
    window: str,
    records: list[dict[str, Any]],
    *,
    start_at: float,
    end_at: float,
    memory_id: str | None = None,
    timezone_name: str = "UTC",
) -> WindowSummary:
    """Build a summary from per-event snapshots.

    ``records`` must contain ``at``, ``role``, ``text``, ``label``, ``valence``,
    ``mood``, ``stress`` and optionally ``tags`` and ``memory_id``.  The
    builder is pure: it does not touch the engine or the store.
    """
    if not records:
        raise ValueError("cannot summarize an empty window")
    valences = [float(r.get("valence", 0.0)) for r in records]
    moods = [float(r.get("mood", 0.0)) for r in records]
    stresses = [float(r.get("stress", 0.0)) for r in records]
    labels = Counter(str(r.get("label", "neutral")) for r in records)
    user_turns = sum(1 for r in records if r.get("role", "user") == "user")
    assistant_turns = sum(1 for r in records if r.get("role") == "assistant")

    topic_counts: Counter[str] = Counter()
    for record in records:
        for tag in record.get("tags", []) or []:
            tag = str(tag).strip()
            if tag and tag not in {"对话", "window", window, "stream"}:
                topic_counts[tag] += 1
    topics = [tag for tag, _ in topic_counts.most_common(4)]

    turning_points: list[dict[str, Any]] = []
    for index in range(1, len(records)):
        before = float(records[index - 1].get("valence", 0.0))
        after = float(records[index].get("valence", 0.0))
        delta = after - before
        if abs(delta) < 0.05:
            continue
        turning_points.append(
            {
                "index": index,
                "at": float(records[index].get("at", end_at)),
                "delta": round(delta, 4),
                "from": round(before, 4),
                "to": round(after, 4),
                "text": str(records[index].get("text", ""))[:80],
            }
        )
    turning_points.sort(key=lambda item: abs(item["delta"]), reverse=True)
    turning_points = turning_points[:3]

    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = _dt.timezone.utc

    def clock(ts: float) -> str:
        return _dt.datetime.fromtimestamp(ts, tz=tz).strftime("%H:%M")

    label_text = "、".join(f"{label} {count}" for label, count in labels.most_common())
    topic_text = "、".join(topics) if topics else "未标注"
    summary = (
        f"窗口「{window}」：{len(records)} 条（用户 {user_turns} / 助手 {assistant_turns}），"
        f"{clock(start_at)}–{clock(end_at)}。"
        f"情绪标签：{label_text}。"
        f"心境 {moods[0]:+.2f} → {moods[-1]:+.2f}；"
        f"压力 {stresses[0]:.2f} → {stresses[-1]:.2f}（峰值 {max(stresses):.2f}）。"
        f"主要话题：{topic_text}。"
    )
    if turning_points:
        point = turning_points[0]
        summary += (
            f"最大转折：第 {point['index'] + 1} 条 "
            f"{point['from']:+.2f} → {point['to']:+.2f}（{point['delta']:+.2f}）。"
        )

    return WindowSummary(
        window=window,
        start_at=float(start_at),
        end_at=float(end_at),
        event_count=len(records),
        user_turns=user_turns,
        assistant_turns=assistant_turns,
        valence_mean=sum(valences) / len(valences),
        valence_min=min(valences),
        valence_max=max(valences),
        mood_start=moods[0],
        mood_end=moods[-1],
        stress_start=stresses[0],
        stress_end=stresses[-1],
        stress_max=max(stresses),
        labels=dict(labels),
        topics=topics,
        turning_points=turning_points,
        summary=summary,
        memory_id=memory_id,
        records=records,
    )


__all__ = [
    "Event",
    "WindowSummary",
    "normalize_events",
    "build_window_summary",
]
