"""A deliberately small endocrine layer.

Eight hormone-like variables are modelled as first-order pools with different
half-lives, drove by appraisal-derived stimuli, circadian phase and sleep debt.
Cortisol uses a three-stage HPA cascade (CRH -> ACTH -> cortisol) so that the
stress response has a realistic delay instead of jumping instantly.

Everything is normalised to [0, 1] relative units.  These are *model* levels,
not blood concentrations; converting them to pg/mL or µg/dL would require
assay-specific calibration and is out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .mathx import clamp, first_order, half_life_to_tau


@dataclass(frozen=True)
class HormoneSpec:
    name: str
    label_zh: str
    half_life_min: float
    baseline: float
    note: str


SPECS: dict[str, HormoneSpec] = {
    "adrenaline": HormoneSpec("adrenaline", "肾上腺素", 2.5, 0.12, "acute alerting; seconds to minutes"),
    "noradrenaline": HormoneSpec("noradrenaline", "去甲肾上腺素", 2.5, 0.18, "arousal and attention"),
    "cortisol": HormoneSpec("cortisol", "皮质醇", 75.0, 0.32, "HPA axis; slow stress hormone"),
    "dopamine": HormoneSpec("dopamine", "多巴胺", 25.0, 0.40, "reward prediction error proxy"),
    "serotonin": HormoneSpec("serotonin", "血清素", 1800.0, 0.50, "slow mood stability; 30 h engineering half-life"),
    "oxytocin": HormoneSpec("oxytocin", "催产素", 5.0, 0.28, "social bonding proxy"),
    "melatonin": HormoneSpec("melatonin", "褪黑素", 45.0, 0.15, "darkness / sleep pressure"),
    "endorphin": HormoneSpec("endorphin", "内啡肽", 25.0, 0.22, "comfort and pain relief proxy"),
}


def circadian_factors(hour: float, chronotype: str = "intermediate") -> dict[str, float]:
    """Diurnal multipliers in [0, 1] (peak-normalised, not absolute levels)."""
    shift = {"early": -1.0, "intermediate": 0.0, "late": 1.5}.get(chronotype, 0.0)
    h = (hour - shift) % 24.0

    def peak_at(peak: float, width: float = 1.0) -> float:
        # cos-shaped diurnal curve centred on `peak`
        return 0.5 + 0.5 * math.cos(2 * math.pi * (h - peak) / 24.0) ** width

    return {
        "cortisol": peak_at(8.0, 1.0),
        "melatonin": peak_at(3.0, 1.0),
        "serotonin": peak_at(14.0, 1.0),
        "alertness": peak_at(11.0, 1.0),
    }


class EndocrineSystem:
    def __init__(self, levels: dict[str, float] | None = None) -> None:
        self.levels: dict[str, float] = {name: spec.baseline for name, spec in SPECS.items()}
        if levels:
            for name, value in levels.items():
                if name in self.levels:
                    self.levels[name] = clamp(value)
        self.crh = 0.12
        self.acth = 0.14
        self.last_stimuli: dict[str, float] = {}
        self.last_circadian: dict[str, float] = {}

    # ------------------------------------------------------------------ step
    def step(
        self,
        dt_min: float,
        *,
        stress: float = 0.0,
        reward: float = 0.0,
        social: float = 0.0,
        comfort: float = 0.0,
        light: float = 0.5,
        sleep_debt: float = 0.0,
        circadian: dict[str, float] | None = None,
    ) -> None:
        if dt_min <= 0:
            return
        dt = min(dt_min, 60.0 * 24 * 7)
        circ = circadian or {"cortisol": 0.4, "melatonin": 0.2, "serotonin": 0.6, "alertness": 0.6}
        stress = clamp(stress)
        sleep_debt = clamp(sleep_debt)
        self.last_stimuli = {
            "stress": stress,
            "reward": reward,
            "social": social,
            "comfort": comfort,
            "light": light,
            "sleep_debt": sleep_debt,
        }
        self.last_circadian = dict(circ)

        # HPA cascade with negative feedback on CRH drive.
        feedback = 1.0 - 0.55 * clamp(self.levels["cortisol"])
        crh_target = 0.08 + 1.55 * stress * feedback + 0.10 * circ["cortisol"] + 0.12 * sleep_debt
        self.crh = first_order(self.crh, clamp(crh_target), dt, 6.0)
        acth_target = 0.06 + 1.10 * self.crh
        self.acth = first_order(self.acth, clamp(acth_target), dt, 18.0)
        cortisol_target = 0.080 + 1.15 * self.acth + 0.09 * circ["cortisol"] + 0.10 * sleep_debt
        # Adrenal release is fast; clearance is slow.  A single half-life would
        # either make acute stress invisible or let cortisol fall unrealistically
        # fast, so use an asymmetric two-timescale update.
        target_cortisol = clamp(cortisol_target)
        tau = 30.0 if target_cortisol > self.levels["cortisol"] else half_life_to_tau(SPECS["cortisol"].half_life_min)
        self.levels["cortisol"] = first_order(self.levels["cortisol"], target_cortisol, dt, tau)

        targets = {
            "adrenaline": 0.10 + 0.66 * stress + 0.22 * max(0.0, reward) + 0.10 * light,
            "noradrenaline": 0.15 + 0.55 * stress + 0.12 * abs(reward) + 0.08 * circ["alertness"],
            "dopamine": 0.34 + 0.45 * reward + 0.10 * abs(reward),
            "serotonin": 0.46 + 0.14 * light + 0.06 * social - 0.16 * sleep_debt - 0.10 * max(0.0, stress - 0.35),
            "oxytocin": 0.22 + 0.52 * social,
            "melatonin": 0.06 + 0.72 * circ["melatonin"] - 0.22 * light,
            "endorphin": 0.18 + 0.40 * comfort,
        }
        for name, target in targets.items():
            tau = half_life_to_tau(SPECS[name].half_life_min)
            self.levels[name] = first_order(self.levels[name], clamp(target), dt, tau)

    # -------------------------------------------------------------- readouts
    def value(self, name: str) -> float:
        return self.levels.get(name, 0.0)

    def cortisol_excess(self) -> float:
        """Cortisol above what the current circadian phase predicts.

        The morning cortisol awakening response is normal, not stress.  Using
        the raw level would make every morning look stressful, so modulation
        uses the excess over the phase-adjusted expectation.
        """
        circ = self.last_circadian.get("cortisol", 0.4)
        # Steady-state solution of the HPA cascade at zero stress:
        # crh=.08+.10c, acth=.06+1.1crh, cortisol=.121+.85acth+.09c
        expected = 0.250 + 0.217 * circ
        return max(0.0, self.levels["cortisol"] - expected)

    @property
    def stress_index(self) -> float:
        return clamp(
            0.40 * self.cortisol_excess()
            + 0.35 * self.levels["adrenaline"]
            + 0.25 * (1.0 - self.levels["serotonin"])
        )

    def encoding_gain(self, config=None) -> float:
        adr = self.levels["adrenaline"]
        cor = self.cortisol_excess()
        da = self.levels["dopamine"]
        ser = self.levels["serotonin"]
        gain = (
            0.70
            + 0.90 * (adr - 0.12)
            + 0.50 * cor
            + 0.45 * (da - 0.40)
            - 0.30 * max(0.0, 0.45 - ser)
        )
        return clamp(gain, 0.55, 2.10)

    def retrieval_gain(self, config=None) -> float:
        adr = self.levels["adrenaline"]
        cor = self.cortisol_excess()
        penalty = getattr(config, "retrieval_cortisol_penalty", 0.70)
        gain = 1.0 - penalty * cor + 0.20 * max(0.0, adr - 0.15)
        return clamp(gain, 0.50, 1.35)

    def mood_bias(self) -> float:
        stress_cortisol = min(0.30, self.cortisol_excess())
        bias = (
            0.80 * (self.levels["serotonin"] - 0.50)
            + 0.45 * (self.levels["dopamine"] - 0.40)
            - 0.45 * stress_cortisol
            + 0.32 * (self.levels["oxytocin"] - 0.28)
            + 0.22 * (self.levels["endorphin"] - 0.22)
        )
        return clamp(bias, -0.20, 0.20)

    def arousal_bias(self) -> float:
        bias = 0.55 * (self.levels["adrenaline"] - 0.12) + 0.35 * (self.levels["noradrenaline"] - 0.18)
        bias -= 0.40 * max(0.0, self.levels["melatonin"] - 0.30)
        return clamp(bias, -0.25, 0.40)

    def snapshot(self) -> dict[str, Any]:
        return {
            "levels": {k: round(v, 4) for k, v in self.levels.items()},
            "crh": round(self.crh, 4),
            "acth": round(self.acth, 4),
            "stress_index": round(self.stress_index, 4),
            "cortisol_excess": round(self.cortisol_excess(), 4),
            "encoding_gain": round(self.encoding_gain(), 4),
            "retrieval_gain": round(self.retrieval_gain(), 4),
            "mood_bias": round(self.mood_bias(), 4),
            "arousal_bias": round(self.arousal_bias(), 4),
            "specs": {
                k: {
                    "label_zh": v.label_zh,
                    "half_life_min": v.half_life_min,
                    "baseline": v.baseline,
                    "note": v.note,
                }
                for k, v in SPECS.items()
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {"levels": self.levels, "crh": self.crh, "acth": self.acth}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EndocrineSystem":
        obj = cls()
        if not data:
            return obj
        for k, v in (data.get("levels") or {}).items():
            if k in obj.levels:
                obj.levels[k] = clamp(v)
        obj.crh = clamp(data.get("crh", obj.crh))
        obj.acth = clamp(data.get("acth", obj.acth))
        return obj
