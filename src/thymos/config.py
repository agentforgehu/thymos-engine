"""Configuration objects.

Every number here is a *model parameter*, not a physiological constant. The
defaults are deliberately conservative and are documented in docs/DESIGN.md
with their provenance (literature prior, engineering estimate, or placeholder).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class Profile:
    """A person (or agent persona) that the engine is calibrated for."""

    name: str = "朋友"
    location_name: str = "上海"
    latitude: float = 31.2304
    longitude: float = 121.4737
    timezone: str = "Asia/Shanghai"
    wake_hour: float = 7.5
    sleep_hours: float = 7.5
    chronotype: str = "intermediate"  # early | intermediate | late
    weather_sensitivity: float = 1.0
    light_sensitivity: float = 1.0
    emotional_reactivity: float = 1.0
    comfort_offset_c: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Profile":
        return _from_dict(cls, data)


@dataclass
class ThymosConfig:
    """Runtime and model parameters for the engine."""

    # --- determinism -----------------------------------------------------
    seed: int = 7
    embedding_dim: int = 256

    # --- memory: encoding ------------------------------------------------
    base_stability_days: float = 0.55
    emotion_stability_gain: float = 0.90
    salience_stability_gain: float = 0.80
    max_stability_days: float = 3650.0
    min_stability_days: float = 0.02
    stability_growth: float = 1.70
    buffer_capacity: int = 40

    # --- memory: forgetting & retrieval ----------------------------------
    fsrs_decay: float = -0.5  # w20 in FSRS notation
    actr_decay: float = 0.5  # d in B_i = ln(sum t_k^-d)
    retrieval_threshold: float = -0.15
    retrieval_noise: float = 0.30
    association_top_k: int = 3
    association_min_similarity: float = 0.26
    context_weight: float = 0.55
    mood_congruence_weight: float = 0.35
    emotion_boost_weight: float = 0.22
    association_spread_weight: float = 0.10
    activation_weight: float = 0.45
    retrievability_weight: float = 0.55
    # Hybrid retrieval weights.  They sum to 1.0 and are deliberately
    # separate from the embedding choice: a weak hash embedder can lean on
    # lexical evidence, while a real Chinese embedder can take more semantic
    # weight without changing the memory model.
    retrieval_semantic_weight: float = 0.45
    retrieval_lexical_weight: float = 0.22
    retrieval_tag_weight: float = 0.18
    retrieval_place_weight: float = 0.07
    retrieval_weather_weight: float = 0.05
    retrieval_time_weight: float = 0.03
    # Write-path admission.  These are conservative defaults for a chat
    # assistant: exact repeats are always deduplicated, semantic merges only
    # happen inside a short window, and conflicting numbers/dates are kept
    # as separate episodes rather than silently merged.
    write_admission_enabled: bool = True
    write_min_importance: float = 0.35
    write_merge_similarity: float = 0.90
    write_merge_window_hours: float = 72.0
    write_semantic_merge_enabled: bool = True
    reconsolidation_window_min: float = 10.0
    # Facts are the highest-precision block, but they must not crowd out the
    # episodic evidence.  At 900 chars this leaves ~315 chars for facts.
    fact_budget_ratio: float = 0.35

    # --- affect ----------------------------------------------------------
    mood_setpoint: float = 0.06
    # 10 h is still much slower than the ~1 h emotion impulse, but fast
    # enough that a day of conversation leaves a visible trace.  These are
    # engineering parameters for chat-turn density, not physiological
    # constants; see docs/LANGUAGE_MOOD_AUDIT.md for the sensitivity sweep.
    mood_tau_hours: float = 10.0
    mood_reactivity: float = 0.40
    emotion_tau_hours: float = 1.1
    stress_tau_hours: float = 5.0
    allostatic_tau_hours: float = 240.0
    reactivity_gain: float = 1.0

    # --- endocrine -------------------------------------------------------
    cortisol_acute_gain: float = 0.55
    retrieval_cortisol_penalty: float = 0.70

    # --- environment -----------------------------------------------------
    weather_enabled: bool = True
    offline: bool = False
    weather_ttl_seconds: int = 1800
    weather_timeout_seconds: float = 8.0
    environment_cache_path: str = ".thymos/weather_cache.json"

    # --- consolidation ---------------------------------------------------
    consolidation_replay: int = 30
    consolidation_gain: float = 0.35
    semantic_abstraction_min_reps: int = 2
    semantic_abstraction_limit: int = 3

    # --- misc ------------------------------------------------------------
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ThymosConfig":
        return _from_dict(cls, data)

    def merged(self, overrides: dict[str, Any] | None) -> "ThymosConfig":
        if not overrides:
            return self
        data = self.to_dict()
        data.update(overrides)
        return ThymosConfig.from_dict(data)


def _from_dict(cls, data: dict[str, Any] | None):
    if not data:
        return cls()
    allowed = {f.name for f in fields(cls)}
    clean = {k: v for k, v in data.items() if k in allowed}
    return cls(**clean)
