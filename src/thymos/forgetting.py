"""Forgetting curve and stability updates.

The core curve is the FSRS family:

    R(t, S) = (1 + factor * t / S) ** decay
    factor  = 0.9 ** (1 / decay) - 1

With decay = -0.5 this gives R(S, S) = 0.9, i.e. stability is the interval at
which retrievability has fallen to 90%.  `decay` is exposed as a parameter
because the literature does not support one universal exponent.

Note: Ebbinghaus' 1885 savings data is *not* free recall and has a much
shallower slope than this curve.  See docs/RESEARCH.md and
experiments/run_experiments.py for the comparison and its limits.
"""

from __future__ import annotations

import math

from .mathx import clamp

EBBINGHAUS_SAVINGS = [
    # (hours, percent savings) transcribed from the classic retention table
    (0.33, 58.2),
    (1.0, 44.2),
    (8.8, 35.8),
    (24.0, 33.7),
    (48.0, 27.8),
    (144.0, 25.4),
    (744.0, 21.1),
]


def curve_factor(decay: float = -0.5) -> float:
    """Factor that makes R(S, S) = 0.9 for the given decay exponent."""
    if decay >= 0:
        raise ValueError("decay must be negative")
    return 0.9 ** (1.0 / decay) - 1.0


def retrievability(
    elapsed_days: float,
    stability_days: float,
    decay: float = -0.5,
    factor: float | None = None,
) -> float:
    """Probability-like retrievability in [0, 1]."""
    if stability_days <= 1e-9:
        return 0.0
    t = max(0.0, elapsed_days)
    f = curve_factor(decay) if factor is None else factor
    return float((1.0 + f * t / stability_days) ** decay)


def initial_stability(
    salience: float,
    valence: float,
    arousal: float,
    encoding_gain: float = 1.0,
    base_days: float = 0.55,
    emotion_gain: float = 0.90,
    salience_gain: float = 0.80,
    minimum: float = 0.02,
    maximum: float = 3650.0,
) -> float:
    """Initial stability in days.

    Emotional arousal and personal salience both slow forgetting; the effect
    is multiplicative and deliberately bounded.  `encoding_gain` comes from
    the endocrine layer (adrenaline / cortisol / dopamine).
    """
    emotional = 1.0 + emotion_gain * abs(valence) * clamp(arousal) + 0.30 * clamp(arousal)
    important = 1.0 + salience_gain * clamp(salience)
    stability = base_days * emotional * important * max(0.35, encoding_gain)
    return float(clamp(stability, minimum, maximum))


def update_stability(
    stability_days: float,
    retrievability_now: float,
    quality: float = 0.8,
    gain: float = 1.0,
    growth: float = 1.7,
    maximum: float = 3650.0,
) -> float:
    """Update stability after a retrieval attempt.

    Successful recall at low retrievability produces the largest gain (the
    spacing effect).  A failed recall contracts stability rather than deleting
    the trace.
    """
    s = max(1e-6, stability_days)
    r = clamp(retrievability_now)
    q = clamp(quality)
    if q < 0.35:
        return float(clamp(s * (0.25 + 0.35 * r), 0.02, maximum))
    # Even a massed repetition (r close to 1) gives a small boost; the large
    # gains are reserved for successful recall when the trace is weak.
    spacing = 0.05 + 0.95 * (1.0 - r)
    new_s = s * (1.0 + growth * spacing * max(0.2, gain) * q)
    return float(clamp(new_s, 0.02, maximum))


def next_interval(stability_days: float, target: float = 0.9, decay: float = -0.5) -> float:
    """Interval in days that brings retrievability down to `target`."""
    target = clamp(target, 0.05, 0.99)
    f = curve_factor(decay)
    if abs(target - 0.9) < 1e-9:
        return max(0.0, stability_days)
    ratio = target ** (1.0 / decay) - 1.0
    return max(0.0, ratio / f * stability_days)


def forgetting_curve(
    stability_days: float,
    days: list[float],
    decay: float = -0.5,
) -> list[float]:
    return [retrievability(d, stability_days, decay) for d in days]


def stretched_exponential(elapsed_days: float, tau_days: float, beta: float = 0.35) -> float:
    """Alternative curve used only for model comparison in experiments."""
    if tau_days <= 0:
        return 0.0
    return math.exp(-((max(0.0, elapsed_days) / tau_days) ** beta))
