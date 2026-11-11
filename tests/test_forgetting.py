import math

import pytest

from thymos.forgetting import (
    curve_factor,
    initial_stability,
    next_interval,
    retrievability,
    update_stability,
)


def test_retrievability_starts_at_one():
    assert retrievability(0.0, 5.0) == pytest.approx(1.0)


def test_stability_is_time_to_ninety_percent():
    for stability in (0.1, 1.0, 30.0):
        assert retrievability(stability, stability) == pytest.approx(0.9, abs=1e-9)


def test_curve_is_monotonic():
    values = [retrievability(d, 10.0) for d in (0, 1, 3, 10, 30, 100)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_curve_factor_matches_definition():
    assert curve_factor(-0.5) == pytest.approx(0.9 ** -2 - 1)


def test_initial_stability_grows_with_emotion_and_salience():
    calm = initial_stability(0.2, 0.0, 0.1, 1.0)
    charged = initial_stability(0.9, 0.9, 0.9, 1.6)
    assert charged > calm * 2


def test_update_stability_rewards_recall_at_low_retrievability():
    easy = update_stability(1.0, 0.95)
    hard = update_stability(1.0, 0.25)
    assert hard > easy > 1.0


def test_failed_recall_contracts_stability():
    assert update_stability(10.0, 0.8, quality=0.1) < 10.0


def test_next_interval_inverts_retrievability():
    stability = 7.0
    interval = next_interval(stability, 0.8)
    assert retrievability(interval, stability) == pytest.approx(0.8, abs=1e-6)


def test_next_interval_default_is_stability():
    assert next_interval(3.5) == pytest.approx(3.5)


def test_no_nan_for_extreme_inputs():
    assert not math.isnan(retrievability(1e9, 0.001))
    assert retrievability(5.0, 0.0) == 0.0
