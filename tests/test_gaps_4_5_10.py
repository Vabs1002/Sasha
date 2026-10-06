"""
Tests for GAP 4 (AdaptiveQuestionSelector), GAP 5 (detect_speech_stress),
and GAP 10 (load_job_description).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from interviewer_agent import AdaptiveQuestionSelector, load_job_description, BEHAVIORAL_QUESTIONS
from analyzer import detect_speech_stress


# ─── GAP 4: AdaptiveQuestionSelector ────────────────────────────────────────

def test_difficulty_seeds_from_level():
    """Fresher starts easy, senior starts hard."""
    assert AdaptiveQuestionSelector("fresher").difficulty == 0.20
    assert AdaptiveQuestionSelector("junior").difficulty == 0.40
    assert AdaptiveQuestionSelector("mid").difficulty == 0.60
    assert AdaptiveQuestionSelector("senior").difficulty == 0.80

def test_difficulty_unknown_level_defaults_to_mid():
    assert AdaptiveQuestionSelector("wizard").difficulty == 0.50

def test_difficulty_increases_when_candidate_strong():
    sel = AdaptiveQuestionSelector("junior")  # starts at 0.40
    sel.update(consistency=0.85, perplexity=80, stress_score=0.1, knowledge_gap=False)
    assert sel.difficulty == 0.55, f"Expected 0.55, got {sel.difficulty}"

def test_difficulty_decreases_when_candidate_struggling():
    sel = AdaptiveQuestionSelector("mid")  # starts at 0.60
    sel.update(consistency=0.30, perplexity=60, stress_score=0.7, knowledge_gap=True)
    assert sel.difficulty == 0.45, f"Expected 0.45, got {sel.difficulty}"

def test_difficulty_holds_on_ambiguous_signals():
    sel = AdaptiveQuestionSelector("mid")  # starts at 0.60
    sel.update(consistency=0.55, perplexity=70, stress_score=0.3, knowledge_gap=False)
    assert sel.difficulty == 0.60

def test_difficulty_clamps_at_1():
    sel = AdaptiveQuestionSelector("senior")  # 0.80
    for _ in range(10):
        sel.update(0.9, 100, 0.0, False)
    assert sel.difficulty <= 1.0

def test_difficulty_clamps_at_01():
    sel = AdaptiveQuestionSelector("fresher")  # 0.20
    for _ in range(10):
        sel.update(0.1, 20, 0.9, True)
    assert sel.difficulty >= 0.10

def test_difficulty_label_ranges():
    sel = AdaptiveQuestionSelector("fresher")
    sel.difficulty = 0.20
    assert "easy" in sel.label()
    sel.difficulty = 0.50
    assert "medium" in sel.label()
    sel.difficulty = 0.80
    assert "hard" in sel.label()

def test_difficulty_history_tracks_all_turns():
    sel = AdaptiveQuestionSelector("junior")
    sel.update(0.9, 80, 0.1, False)
    sel.update(0.2, 30, 0.8, True)
    # history[0] = initial, history[1] = after turn 1, history[2] = after turn 2
    assert len(sel.history) == 3


# ─── GAP 5: detect_speech_stress ────────────────────────────────────────────

def test_stress_low_for_confident_answer():
    answer = "I designed the caching layer using Redis with a write-through policy. We saw a 40% reduction in DB load."
    result = detect_speech_stress(answer, disfluency_rate=0.05, perplexity=80)
    assert result['stress_score'] < 0.4
    assert not result['knowledge_gap_detected']

def test_stress_high_for_uncertain_answer():
    answer = "I'm not sure exactly. I think maybe we used some kind of cache. I'm not sure how it worked actually."
    result = detect_speech_stress(answer, disfluency_rate=0.10, perplexity=90)
    assert result['stress_score'] >= 0.30
    assert result['knowledge_gap_detected']

def test_stress_high_for_very_brief_answers():
    answer = "Redis. Worked fine."
    result = detect_speech_stress(answer, disfluency_rate=0.0, perplexity=60)
    assert "very_brief_answers" in result['stress_signals']

def test_stress_empathy_prompt_only_when_high():
    # Low stress — clear confident answer, no empathy
    low = detect_speech_stress("I built X with Y and achieved Z metric improving latency by 40%.", 0.05, 80)
    assert low['empathy_prompt'] is None

    # High stress — epistemic uncertainty + very brief answers
    # "I don't really know." → brief answer (4 words) + uncertainty hit → score ≥ 0.45
    # Add backtracking to push over 0.6
    high = detect_speech_stress(
        "I'm not sure. Actually wait, no sorry, I meant something else. I don't know.",
        0.10, 90
    )
    assert high['stress_score'] > 0.6, f"Expected > 0.6, got {high['stress_score']}, signals: {high['stress_signals']}"
    assert high['empathy_prompt'] is not None

def test_non_native_speaker_not_penalized_for_fillers():
    """Filler words alone should NOT produce stress signals for non-native speakers."""
    # High disfluency but good content answer
    answer = "So like, I built a like recommendation engine, you know, using collaborative filtering."
    result_non_native = detect_speech_stress(answer, disfluency_rate=0.25, perplexity=85,
                                              native_speaker=False)
    # Should NOT have high_disfluency signal
    assert "high_disfluency" not in result_non_native['stress_signals']

    result_native = detect_speech_stress(answer, disfluency_rate=0.25, perplexity=85,
                                          native_speaker=True)
    # Native speaker SHOULD get flagged for disfluency > 0.20
    assert "high_disfluency" in result_native['stress_signals']

def test_backtracking_detected():
    answer = "We used PostgreSQL. Actually wait, no sorry, I meant MySQL. Let me start over."
    result = detect_speech_stress(answer, disfluency_rate=0.05, perplexity=80)
    assert "repeated_backtracking" in result['stress_signals']

def test_empty_transcript_returns_zero():
    result = detect_speech_stress("", 0.0, 0.0)
    assert result['stress_score'] == 0.0
    assert result['empathy_prompt'] is None


# ─── GAP 10: load_job_description ───────────────────────────────────────────

def test_jd_appended_to_resume():
    resume = "Candidate has 3 years Python experience."
    jd = "We need someone who owns distributed tracing."
    combined = load_job_description(jd, resume)
    assert "JOB DESCRIPTION REQUIREMENTS" in combined
    assert "distributed tracing" in combined
    assert "Python experience" in combined

def test_empty_jd_returns_resume_unchanged():
    resume = "Candidate has 3 years Python experience."
    combined = load_job_description("", resume)
    assert combined == resume

def test_behavioral_questions_bank_has_all_categories():
    expected = {"growth_mindset", "adaptability", "ownership_and_impact",
                "collaboration_and_conflict", "self_awareness"}
    assert set(BEHAVIORAL_QUESTIONS.keys()) == expected

def test_behavioral_questions_are_non_empty():
    for cat, questions in BEHAVIORAL_QUESTIONS.items():
        assert len(questions) >= 1, f"Category {cat} has no questions"
        for q in questions:
            assert len(q) > 20, f"Question too short in {cat}: {q}"


if __name__ == "__main__":
    # Quick manual run
    print("Running GAP 4, 5, 10 tests...")
    test_difficulty_seeds_from_level()
    test_difficulty_increases_when_candidate_strong()
    test_difficulty_decreases_when_candidate_struggling()
    test_stress_high_for_uncertain_answer()
    test_non_native_speaker_not_penalized_for_fillers()
    test_jd_appended_to_resume()
    print("All manual checks passed.")
