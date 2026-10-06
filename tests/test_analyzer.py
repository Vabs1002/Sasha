"""
Unit tests for analyzer.py
"""
import os
import sys
import pytest

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analyzer import get_perplexity, get_disfluency_rate, get_consistency


def test_get_disfluency_rate_basic():
    """Test disfluency rate calculation with basic inputs"""
    # Text with no fillers
    text1 = "Hello world this is a test"
    rate1 = get_disfluency_rate(text1)
    assert rate1 == 0.0

    # Text with fillers
    text2 = "um uh like you know hello world"
    rate2 = get_disfluency_rate(text2)
    # 4 filler terms (um, uh, like, you know) in 7 total words
    assert abs(rate2 - 4/7) < 0.0001

    # Text with mixed case fillers
    text3 = "Um Uh LIKE you know test"
    rate3 = get_disfluency_rate(text3)
    # 4 filler terms (Um, Uh, LIKE, you know) in 6 total words
    assert abs(rate3 - 4/6) < 0.0001


def test_get_disfluency_rate_edge_cases():
    """Test disfluency rate with edge cases"""
    # Empty text
    assert get_disfluency_rate("") == 0.0

    # Only fillers
    assert get_disfluency_rate("um uh like") == 1.0

    # No words
    assert get_disfluency_rate("!!!") == 0.0


def test_get_consistency_basic():
    """Test consistency calculation with basic inputs"""
    resume_text = "I have experience in Python and machine learning"
    answer_text = "I worked with Python and ML technologies"

    consistency = get_consistency(resume_text, answer_text)

    # Should return a value between 0 and 1
    assert 0.0 <= consistency <= 1.0

    # Similar texts should have high consistency
    assert consistency > 0.5


def test_get_consistency_identical_texts():
    """Test consistency with identical texts"""
    text = "This is a test sentence"
    consistency = get_consistency(text, text)
    # Should be 1.0 for identical texts (or very close)
    assert consistency >= 0.9


def test_get_consistency_different_texts():
    """Test consistency with very different texts"""
    resume_text = "I have experience in Python and machine learning"
    answer_text = "The weather is nice today"

    consistency = get_consistency(resume_text, answer_text)
    # Should be low for dissimilar texts
    assert consistency < 0.5


def test_get_perplexity_basic():
    """Test perplexity calculation"""
    # Simple English text
    text = "Hello world this is a test"
    perp = get_perplexity(text)

    # Perplexity should be a positive number
    assert perp > 0
    # Reasonable range for English text
def test_detect_ai_generated_answer():
    """Test AI script and teleprompter detection heuristics"""
    from analyzer import detect_ai_generated_answer

    # Normal spontaneous human answer: reasonable perplexity, natural fillers
    human_transcript = "Well, um, we basically used Redis for caching, and like, it reduced our database latency a lot."
    human_res = detect_ai_generated_answer(human_transcript, perplexity=85.0, disfluency_rate=0.08, consistency=0.7)
    assert not human_res["ai_script_detected"]
    assert human_res["ai_probability"] < 0.40

    # Highly structured AI script: low perplexity, zero fillers, hallmark discourse markers
    ai_script = (
        "In conclusion, the microservices architecture provides a comprehensive approach to scalability. "
        "Furthermore, it is worth noting that decoupling the database layers minimizes point-to-point network overhead "
        "while ensuring resilience across distributed deployments."
    )
    ai_res = detect_ai_generated_answer(ai_script, perplexity=18.5, disfluency_rate=0.0, consistency=0.8)
    assert ai_res["ai_script_detected"]
    assert ai_res["ai_probability"] >= 0.65
    assert "critically_low_perplexity" in ai_res["signals"]
    assert "written_llm_discourse_markers" in ai_res["signals"]
    assert ai_res["nudge_prompt"] is not None


def test_detect_conduct_violation():
    """Test profanity, insult, and hostility detection"""
    from analyzer import detect_conduct_violation

    # Professional candidate response
    clean_resp = "We deployed the microservice using Kubernetes and optimized Docker image sizes."
    res_clean = detect_conduct_violation(clean_resp)
    assert not res_clean["is_violation"]
    assert res_clean["severity"] == "NONE"

    # Abusive response (insult to AI)
    abusive_resp = "Shut up, you are stupid and this interview is bullshit."
    res_abusive = detect_conduct_violation(abusive_resp)
    assert res_abusive["is_violation"]
    assert res_abusive["severity"] == "CRITICAL"
    assert "profanity" in res_abusive["categories"] or "direct_insult" in res_abusive["categories"]


if __name__ == "__main__":
    pytest.main([__file__])