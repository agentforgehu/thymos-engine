"""Fast emotion, slow mood, stress and allostatic load."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Profile, ThymosConfig
from .mathx import clamp, first_order
from .types import AffectState, Emotion


@dataclass
class EnvironmentEffects:
    mood_delta: float = 0.0
    arousal_delta: float = 0.0
    serotonin_drive: float = 0.5
    melatonin_drive: float = 0.3
    energy: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {
            "mood_delta": round(self.mood_delta, 4),
            "arousal_delta": round(self.arousal_delta, 4),
            "serotonin_drive": round(self.serotonin_drive, 4),
            "melatonin_drive": round(self.melatonin_drive, 4),
            "energy": round(self.energy, 4),
        }


class AffectDynamics:
    """Two-timescale affect model.

    Emotion is an impulse that decays in roughly an hour; mood is a slow
    integrator pulled toward a homeostatic setpoint plus environmental and
    endocrine biases.  Both are continuous, so a discrete emotion label is
    only a readout, never the state itself.
    """

    def __init__(self, config: ThymosConfig, profile: Profile, state: AffectState | None = None) -> None:
        self.config = config
        self.profile = profile
        self.state = state or AffectState(
            mood_valence=config.mood_setpoint,
            updated_at=0.0,
        )
        self.state.mood_valence = clamp(self.state.mood_valence, -1.0, 1.0)

    # ----------------------------------------------------------------- input
    def apply_emotion(self, emotion: Emotion, endocrine, sensitivity: float = 1.0) -> None:
        s = self.state
        ser = endocrine.value("serotonin")
        cor = endocrine.value("cortisol")
        reactivity = (
            self.config.reactivity_gain
            * max(0.4, self.profile.emotional_reactivity)
            * (1.0 + 0.35 * (1.0 - ser) + 0.25 * max(0.0, cor - 0.35))
            * max(0.3, sensitivity)
        )
        dv = emotion.valence * emotion.intensity * 0.55 * reactivity
        da = emotion.arousal * emotion.intensity * 0.45 * reactivity
        dd = emotion.dominance * emotion.intensity * 0.40 * reactivity

        s.valence = clamp(s.valence + dv, -1.0, 1.0)
        s.arousal = clamp(s.arousal + da, 0.0, 1.0)
        s.dominance = clamp(s.dominance + dd, -1.0, 1.0)
        # Mood is slow, but not inert: a salient event moves it a little at once
        # and the rest is integrated by `step`.
        s.mood_valence = clamp(s.mood_valence + self.config.mood_reactivity * dv, -1.0, 1.0)
        # Stress is not a ratchet.  Negative events add a bounded load, while
        # positive events actively relieve it; otherwise a run of mild
        # frustrations can drift to an implausible 0.8 without any acute event.
        stress_delta = (
            max(0.0, -dv) * 0.35
            + max(0.0, da) * 0.06
            - max(0.0, dv) * 0.18
        )
        s.stress = clamp(s.stress + stress_delta, 0.0, 1.0)
        s.fatigue = clamp(s.fatigue + max(0.0, da) * 0.05, 0.0, 1.0)
        s.dominant_emotion = emotion.label

    def touch(self, now: float) -> None:
        self.state.updated_at = now

    # ------------------------------------------------------------------ step
    def step(
        self,
        dt_hours: float,
        endocrine,
        env: EnvironmentEffects | None = None,
        *,
        circadian: dict[str, float] | None = None,
        sleeping: bool = False,
        sleep_debt: float = 0.0,
    ) -> None:
        if dt_hours <= 0:
            return
        dt = min(dt_hours, 24.0 * 30)
        s = self.state
        env = env or EnvironmentEffects()
        circ = circadian or {}

        # Fast emotion decays toward neutral.
        decay = first_order(s.valence, 0.0, dt, self.config.emotion_tau_hours)
        s.valence = clamp(decay, -1.0, 1.0)
        s.arousal = clamp(first_order(s.arousal, 0.16, dt, self.config.emotion_tau_hours * 0.75), 0.0, 1.0)
        s.dominance = clamp(first_order(s.dominance, 0.0, dt, self.config.emotion_tau_hours), -1.0, 1.0)

        # Slow mood: setpoint + weather + hormones + circadian + a fraction of emotion.
        hormone_bias = endocrine.mood_bias()
        circadian_mood = 0.05 * (circ.get("serotonin", 0.5) - 0.5) - 0.04 * (circ.get("melatonin", 0.2) - 0.2)
        target = (
            self.config.mood_setpoint
            + env.mood_delta
            + hormone_bias
            + circadian_mood
            + 0.45 * s.valence
        )
        s.mood_valence = clamp(
            first_order(s.mood_valence, clamp(target, -1.0, 1.0), dt, self.config.mood_tau_hours),
            -1.0,
            1.0,
        )
        mood_arousal_target = clamp(
            0.34 + env.arousal_delta + endocrine.arousal_bias() + 0.25 * (s.arousal - 0.2),
            0.0,
            1.0,
        )
        s.mood_arousal = clamp(
            first_order(s.mood_arousal, mood_arousal_target, dt, self.config.mood_tau_hours * 0.45),
            0.0,
            1.0,
        )

        # Stress and fatigue.
        stress_target = clamp(
            0.09 + 0.24 * max(0.0, -s.mood_valence) + 0.08 * s.mood_arousal,
            0.0,
            1.0,
        )
        s.stress = clamp(first_order(s.stress, stress_target, dt, self.config.stress_tau_hours), 0.0, 1.0)
        if sleeping:
            s.fatigue = clamp(first_order(s.fatigue, 0.08, dt, 3.0), 0.0, 1.0)
        else:
            s.fatigue = clamp(s.fatigue + 0.035 * dt * (1.0 + 0.5 * sleep_debt), 0.0, 1.0)

        # Allostatic load accumulates under sustained stress and recovers slowly.
        overload = max(0.0, s.stress - 0.55)
        s.allostatic_load = clamp(
            s.allostatic_load
            + (overload * 0.020 - s.allostatic_load * 0.0010) * dt,
            0.0,
            1.0,
        )

        if s.valence < -0.45 and s.dominant_emotion not in {"fear", "anger", "sadness", "anxiety"}:
            s.dominant_emotion = "sadness" if s.arousal < 0.5 else "anxiety"
        elif s.valence > 0.35 and s.dominant_emotion in {"sadness", "anxiety", "anger", "fear"}:
            s.dominant_emotion = "contentment"

    # --------------------------------------------------------------- readout
    def snapshot(self) -> dict[str, Any]:
        s = self.state
        return {
            "emotion": {
                "label": s.dominant_emotion,
                "valence": round(s.valence, 4),
                "arousal": round(s.arousal, 4),
                "dominance": round(s.dominance, 4),
            },
            "mood": {
                "valence": round(s.mood_valence, 4),
                "arousal": round(s.mood_arousal, 4),
            },
            "stress": round(s.stress, 4),
            "fatigue": round(s.fatigue, 4),
            "allostatic_load": round(s.allostatic_load, 4),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.state.to_dict()

    @classmethod
    def from_dict(
        cls, config: ThymosConfig, profile: Profile, data: dict[str, Any] | None
    ) -> "AffectDynamics":
        state = AffectState.from_dict(data)
        return cls(config, profile, state)
