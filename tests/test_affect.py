import pytest

from thymos.affect import AffectDynamics
from thymos.appraisal import appraise, estimate_appraisal
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


def test_positive_event_relieves_stress():
    affect, endo = make_affect()
    affect.state.stress = 0.6
    affect.apply_emotion(
        appraise(Appraisal(desirability=0.8, stakes=0.5, goal_relevance=0.8)),
        endo,
    )
    assert affect.state.stress < 0.6


def test_neutral_turns_do_not_ratchet_stress():
    affect, endo = make_affect()
    neutral = appraise(estimate_appraisal("我叫林墨，住在杭州。"))
    for _ in range(30):
        affect.apply_emotion(neutral, endo)
        affect.step(1.0, endo)
    assert affect.state.stress < 0.25


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


def test_single_colloquial_insult_moves_mood_materially():
    affect, endo = make_affect()
    before = affect.state.mood_valence
    emotion = appraise(estimate_appraisal("你怎么这么笨，闭嘴"))
    affect.apply_emotion(emotion, endo)
    delta = affect.state.mood_valence - before
    assert emotion.label in {"anger", "disgust"}
    assert delta <= -0.03


def test_single_colloquial_praise_moves_mood_materially():
    affect, endo = make_affect()
    before = affect.state.mood_valence
    emotion = appraise(estimate_appraisal("谢谢你，辛苦了"))
    affect.apply_emotion(emotion, endo)
    assert emotion.label in {"joy", "trust"}
    assert affect.state.mood_valence - before >= 0.03


def test_mild_tired_message_moves_mood_by_more_than_one_percent():
    affect, endo = make_affect()
    before = affect.state.mood_valence
    affect.apply_emotion(appraise(estimate_appraisal("今天有点累")), endo)
    assert affect.state.mood_valence - before <= -0.015


def test_daily_chat_has_visible_mood_range():
    affect, endo = make_affect()
    texts = [
        "今天星期三", "你好", "你怎么这么笨，闭嘴", "谢谢", "我在看垃圾分类",
        "项目黄了", "你真好", "今天有点累", "我喜欢你", "呵呵",
        "我被裁了", "差不多完成了", "你做得不对", "太厉害了", "今天特别开心",
        "我们分手了", "谢谢你，辛苦了", "我在滚面团", "这个方案不行", "靠谱",
        "家人生病了", "牛啊", "别烦我", "蔬菜很新鲜", "给力",
        "就这？", "你真是个废物", "别人夸我今天的报告", "今天有点累", "晚安",
    ]
    values = []
    stresses = []
    for text in texts:
        affect.apply_emotion(appraise(estimate_appraisal(text)), endo)
        affect.step(0.5, endo)
        values.append(affect.state.mood_valence)
        stresses.append(affect.state.stress)
    assert max(values) - min(values) >= 0.08
    assert max(stresses) < 0.35
    assert affect.state.fatigue < 0.90


def test_mood_returns_toward_setpoint_after_one_day_without_events():
    affect, endo = make_affect()
    affect.apply_emotion(appraise(estimate_appraisal("你怎么这么笨，闭嘴")), endo)
    affect.step(24.0, endo)
    assert abs(affect.state.mood_valence - ThymosConfig().mood_setpoint) < 0.02
