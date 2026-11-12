from thymos.appraisal import appraise, estimate_appraisal
from thymos.types import Appraisal


def test_positive_event_maps_to_joy_or_contentment():
    e = appraise(Appraisal(desirability=0.7, certainty=0.8, stakes=0.4, goal_relevance=0.8))
    assert e.label in {"joy", "contentment", "excitement"}
    assert e.valence > 0.4


def test_low_certainty_positive_is_hope():
    e = appraise(Appraisal(desirability=0.6, certainty=0.3, stakes=0.6))
    assert e.label == "hope"


def test_other_agency_high_control_negative_is_anger():
    e = appraise(
        Appraisal(desirability=-0.7, certainty=0.8, control=0.7, agency="other", stakes=0.7)
    )
    assert e.label == "anger"


def test_uncertain_uncontrollable_negative_is_anxiety_or_fear():
    e = appraise(
        Appraisal(desirability=-0.6, certainty=0.3, control=0.2, agency="circumstance", stakes=0.8)
    )
    assert e.label in {"anxiety", "fear"}


def test_novel_neutral_event_is_surprise():
    e = appraise(Appraisal(desirability=0.1, novelty=0.95, stakes=0.3))
    assert e.label == "surprise"


def test_neutral_facts_do_not_become_anticipation():
    e = appraise(estimate_appraisal("我叫林墨，住在杭州。"))
    assert e.label == "neutral"


def test_future_decision_is_anticipation_but_uncertain_move_is_anxiety():
    decided = appraise(estimate_appraisal("我决定搬去上海了，下周搬。"))
    uncertain = appraise(estimate_appraisal("我最近在考虑搬去上海，还在犹豫。"))
    assert decided.label == "anticipation"
    assert uncertain.label == "anxiety"


def test_text_negation_flips_valence():
    positive = estimate_appraisal("今天很开心")
    negative = estimate_appraisal("今天不开心")
    assert positive.desirability > 0
    assert negative.desirability < 0


def test_unexpected_good_news_is_positive_surprise():
    a = estimate_appraisal("完全没想到会有礼物，很惊喜")
    assert a.desirability > 0.3
    assert a.novelty >= 0.7


def test_anger_words_produce_other_agency():
    a = estimate_appraisal("老板又改需求，我很生气")
    assert a.desirability < 0
    assert a.agency == "other"
    assert a.control >= 0.6


def test_fear_words_reduce_control_and_certainty():
    a = estimate_appraisal("深夜一个人走回家，有点害怕")
    assert a.control <= 0.3
    assert a.certainty <= 0.4


def test_appraisal_clamps_out_of_range_values():
    a = Appraisal(desirability=9, certainty=-3, control=5, stakes=99).clamped()
    assert a.desirability == 1.0
    assert a.certainty == 0.0
    assert 0.0 <= a.control <= 1.0
    assert a.stakes == 1.0
