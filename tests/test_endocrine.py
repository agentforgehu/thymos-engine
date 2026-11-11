import math

import pytest

from thymos.endocrine import SPECS, EndocrineSystem, circadian_factors

CIRC = {"cortisol": 0.4, "melatonin": 0.2, "serotonin": 0.6, "alertness": 0.6}


def test_all_hormones_start_at_their_baseline():
    endo = EndocrineSystem()
    for name, spec in SPECS.items():
        assert endo.value(name) == pytest.approx(spec.baseline)


def test_adrenaline_rises_faster_than_cortisol():
    endo = EndocrineSystem()
    base_adr, base_cor = endo.value("adrenaline"), endo.value("cortisol")
    endo.step(5, stress=1.0, circadian=CIRC)
    adr_gain = endo.value("adrenaline") - base_adr
    cor_gain = endo.value("cortisol") - base_cor
    assert adr_gain > cor_gain * 3
    assert adr_gain > 0.1


def test_cortisol_has_a_delayed_peak():
    endo = EndocrineSystem()
    endo.step(5, stress=1.0, circadian=CIRC)
    early = endo.value("cortisol")
    endo.step(35, stress=0.0, circadian=CIRC)
    later = endo.value("cortisol")
    assert later > early


def test_cortisol_recovers_after_stress_ends():
    endo = EndocrineSystem()
    endo.step(30, stress=1.0, circadian=CIRC)
    peak = endo.value("cortisol")
    for _ in range(20):
        endo.step(30, stress=0.0, circadian=CIRC)
    assert peak > 0.32
    assert endo.value("cortisol") < 0.45


def test_circadian_melatonin_is_higher_at_night():
    night = circadian_factors(3.0)
    noon = circadian_factors(12.0)
    assert night["melatonin"] > noon["melatonin"]
    assert night["cortisol"] < noon["cortisol"]


def test_morning_cortisol_is_not_counted_as_stress():
    endo = EndocrineSystem()
    endo.step(180, stress=0.0, circadian={"cortisol": 1.0, "melatonin": 0.0, "serotonin": 0.7, "alertness": 0.8})
    assert endo.value("cortisol") > 0.38
    assert endo.cortisol_excess() < 0.15


def test_high_adrenaline_increases_encoding_gain():
    endo = EndocrineSystem()
    before = endo.encoding_gain()
    endo.step(3, stress=1.0, circadian=CIRC)
    assert endo.encoding_gain() > before


def test_encoding_gain_at_rest_is_not_stuck_at_the_floor():
    endo = EndocrineSystem()
    endo.last_circadian = {"cortisol": 0.4}
    # Baseline is a neutral state, not a maximally impaired one.
    assert 0.65 <= endo.encoding_gain() <= 0.80


def test_cortisol_excess_impairs_retrieval_gain():
    endo = EndocrineSystem()
    endo.levels["cortisol"] = 0.95
    endo.last_circadian = {"cortisol": 0.1}
    assert endo.retrieval_gain() < 1.0
    assert endo.stress_index > 0.3


def test_mood_bias_is_bounded():
    endo = EndocrineSystem()
    endo.levels = {name: 1.0 for name in endo.levels}
    endo.last_circadian = {"cortisol": 0.0}
    assert -0.2 <= endo.mood_bias() <= 0.2
    assert -0.25 <= endo.arousal_bias() <= 0.4


def test_levels_stay_normalised():
    endo = EndocrineSystem()
    for _ in range(20):
        endo.step(120, stress=1.0, reward=1.0, social=1.0, comfort=1.0, light=1.0, sleep_debt=1.0, circadian=CIRC)
    assert all(0.0 <= v <= 1.0 and not math.isnan(v) for v in endo.levels.values())
