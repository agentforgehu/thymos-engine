"""Small numeric helpers (no numpy dependency in the core)."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value != value:  # NaN
        return lo
    return max(lo, min(hi, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * clamp(t)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def ema(previous: float, value: float, alpha: float) -> float:
    return previous + clamp(alpha) * (value - previous)


def first_order(current: float, target: float, dt: float, tau: float) -> float:
    """Exact discrete step of dH/dt = (target - H) / tau."""
    if tau <= 1e-9:
        return target
    return target + (current - target) * math.exp(-dt / tau)


def half_life_to_tau(half_life: float) -> float:
    return max(1e-9, half_life / math.log(2.0))


def weighted_mean(pairs: Sequence[tuple[float, float]]) -> float:
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return 0.0
    return sum(v * w for v, w in pairs) / total_w
