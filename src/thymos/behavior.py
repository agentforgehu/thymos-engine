"""Emotion-to-behavior mapping.

The dynamical core answers *how the agent feels*.  A product still has to
decide *what the agent should do with that feeling*: tone, length, whether to
ask a question, whether to de-escalate, and when to stop.  That decision is
not universal, so this module exposes a tiny replaceable interface instead of
hard-coding one company's persona.

The default mapper is deliberately conservative.  It is a product-safe prior,
not a claim about what a human would do.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from .types import AffectState


@dataclass(frozen=True)
class BehaviorDirective:
    """A host-facing suggestion derived from affect and query intent."""

    tone: str = "neutral"  # neutral | warm | gentle | direct | playful
    verbosity: str = "medium"  # short | medium | long
    initiative: str = "none"  # none | reflect | ask | advise | share
    boundary: str = "none"  # none | deescalate | safety
    suggested_action: str = "answer_directly"
    confidence: float = 0.5
    reason: str = ""
    affect_label: str = "neutral"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["confidence"] = round(float(data["confidence"]), 3)
        return data


@runtime_checkable
class BehaviorMapper(Protocol):
    """The extension point a host implements for its own persona.

    A mapper receives the current affect state plus the query intent and the
    latest emotion label.  It must return a :class:`BehaviorDirective`; it may
    use any product-specific policy (brand voice, safety rules, user tier).
    """

    def map(
        self,
        affect: AffectState,
        *,
        intent: str = "mixed",
        emotion: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> BehaviorDirective:
        ...


class DefaultBehaviorMapper:
    """A small, auditable default policy.

    The rules are ordered by safety first, then task intent, then affect.  A
    factual/technical query never gets a "be more emotional" instruction just
    because mood is low; a highly stressed or angry user gets a shorter,
    gentler reply and a reflective question instead of advice.
    """

    def map(
        self,
        affect: AffectState,
        *,
        intent: str = "mixed",
        emotion: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> BehaviorDirective:
        label = (emotion or affect.dominant_emotion or "neutral").lower()
        stress = float(affect.stress)
        mood = float(affect.mood_valence)
        intensity = abs(mood)

        # 1. Safety / de-escalation.  This is intentionally conservative: a
        #    high stress signal or an anger/fear label shortens the reply and
        #    asks before advising.  It is not a diagnosis or crisis detector.
        if stress >= 0.75 or label in {"anger", "fear", "disgust"}:
            return BehaviorDirective(
                tone="gentle",
                verbosity="short",
                initiative="reflect",
                boundary="deescalate",
                suggested_action="acknowledge_then_ask",
                confidence=0.82,
                reason="high stress or threat-related affect; reduce load first",
                affect_label=label,
            )

        # 2. Task queries stay task-first.  Affect can shape tone, but must not
        #    displace the answer or add a feelings paragraph to a fact lookup.
        if intent in {"factual", "technical", "history"}:
            return BehaviorDirective(
                tone="neutral",
                verbosity="short",
                initiative="none",
                boundary="none",
                suggested_action="answer_directly",
                confidence=0.85,
                reason=f"{intent} query; facts before affect",
                affect_label=label,
            )

        # 3. Sadness/anxiety without high stress: validate before solving.
        if label in {"sadness", "anxiety", "depressed"} or mood <= -0.30:
            return BehaviorDirective(
                tone="gentle",
                verbosity="short",
                initiative="reflect",
                boundary="none",
                suggested_action="validate_then_offer_one_step",
                confidence=0.72,
                reason="negative mood; do not lead with a checklist",
                affect_label=label,
            )

        # 4. Positive affect: match energy without overclaiming intimacy.
        if label in {"joy", "excitement", "trust", "contentment", "hope"} or mood >= 0.35:
            return BehaviorDirective(
                tone="warm",
                verbosity="medium",
                initiative="share",
                boundary="none",
                suggested_action="match_energy_then_ask_one_question",
                confidence=0.70,
                reason="positive affect; mirror energy, keep agency with user",
                affect_label=label,
            )

        # 5. Neutral / low-confidence default.
        return BehaviorDirective(
            tone="neutral",
            verbosity="medium",
            initiative="none",
            boundary="none",
            suggested_action="answer_then_check_in",
            confidence=0.50 + min(0.25, intensity * 0.25),
            reason="no strong affect or task override",
            affect_label=label,
        )


__all__ = [
    "BehaviorDirective",
    "BehaviorMapper",
    "DefaultBehaviorMapper",
]
