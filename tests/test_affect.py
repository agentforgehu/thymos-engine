import pytest

from thymos.affect import AffectDynamics
from thymos.appraisal import appraise
from thymos.config import Profile, ThymosConfig
from thymos.endocrine import EndocrineSystem
from thymos.types import Appraisal


def make_affect():
    config = ThymosConfig()
    profile = Profile()
    return AffectDynamics(config, profile), EndocrineSystem()


def test_positive_emotion_raises_valence_and_mood():
    affect, endo = make_affect()
    before = affect.state.mood_valence
    affect.apply_emotion(appraise(Appraisal(desirability=0.9, stakes=0.8, goal_relevance=0.9)), endo)
    assert affect.state.valence > 0.2
    assert affect.state.mood_valence > before


def test_negative_emotion_raises_stress():
    affect, endo = make_affect()
    before = affect.state.stress
    affect.apply_emotion(appraise(Appraisal(desirability=-0.9, stakes=0.9, goal_relevance=0.9, agency="other", control=0.7)), endo)
    assert affect.state.stress > before
    assert affect.state.dominant_emotion == "anger"


def test_emotion_decays_faster_than_mood():
    affect, endo = make_affect()
    affect.apply_emotion(appraise(Appraisal(desirability=0.8, stakes=0.7)), endo)
    v0, m0 = affect.state.valence, affect.state.mood_valence
    affect.step(3.0, endo)
    v_loss = (v0 - affect.state.valence) / max(v0, 1e-9)
    m_loss = max(0.0, (m0 - affect.state.mood_valence) / max(m0, 1e-9))
    assert v_loss > 0.8
    assert m_loss < 0.5


def test_mood_returns_toward_setpoint_without_events():
    affect, endo = make_affect()
    affect.state.mood_valence = -0.6
    for _ in range(8):
        affect.step(24.0, endo)
    assert affect.state.mood_valence > -0.2
    assert abs(affect.state.mood_valence - ThymosConfig().mood_setpoint) < 0.15


def test_allostatic_load_accumulates_under_persistent_stress():
    affect, endo = make_affect()
    start = affect.state.allostatic_load
    for _ in range(240):
        affect.state.stress = 0.9
        affect.step(1.0, endo)
    assert affect.state.allostatic_load > start


def test_fatigue_recovers_during_sleep():
    affect, endo = make_affect()
    affect.state.fatigue = 0.8
    affect.step(8.0, endo, sleeping=True)
    assert affect.state.fatigue < 0.3


def test_state_clamps_stay_in_range():
    affect, endo = make_affect()
    for _ in range(5):
        affect.apply_emotion(appraise(Appraisal(desirability=-1.0, stakes=1.0, goal_relevance=1.0)), endo)
    assert -1.0 <= affect.state.valence <= 1.0
    assert 0.0 <= affect.state.arousal <= 1.0
    assert 0.0 <= affect.state.stress <= 1.0
