"""
Unit tests for the factual, executive HR report generator.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from report_generator import generate_report, _derive_qualitative_tier, _extract_turn_records, _build_competency_sections


def test_derive_qualitative_tier():
    assert _derive_qualitative_tier(8.5) == "Higher rubric estimate"
    assert _derive_qualitative_tier(6.0) == "Mid-range rubric estimate"
    assert _derive_qualitative_tier(4.0) == "Lower rubric estimate"


def test_generate_report_empty_history_returns_false():
    assert generate_report({}, []) is False


def test_generate_report_basic(tmp_path):
    output_pdf = str(tmp_path / "test_report.pdf")
    output_html = str(tmp_path / "test_report.html")

    profile = {
        "name": "Jane Doe",
        "role": "ML_Engineer",
        "level": "mid",
        "years_experience": 3
    }
    history = [
        {
            "question": "Tell me about your experience fine-tuning LLMs.",
            "answer": "I used LoRA and QLoRA on Llama 3 8B with Hugging Face PEFT, reducing memory usage by 65%.",
            "assessment": {
                "ownership": 8.0,
                "depth": 7.5,
                "impact": 7.0,
                "learning": 8.0,
                "communication": 8.0,
                "notes": "Clear articulation of parameter-efficient fine-tuning."
            },
            "decision": {
                "action": "move_on",
                "reasoning": "Sufficient technical depth demonstrated."
            },
            "analysis": {
                "consistency": 0.8,
                "perplexity": 75.0,
                "disfluency_rate": 0.05,
                "difficulty_level": 0.6,
                "knowledge_gap_detected": False,
                "noise_detected": False
            }
        }
    ]

    res = generate_report(profile, history, output_path=output_pdf)
    assert res is True
    assert os.path.exists(output_html)


def test_extract_turn_records_formatting():
    history = [
        {
            "question": "What is your testing strategy?",
            "answer": "We run integration tests in GitHub Actions.",
            "assessment": {"notes": "Understands CI/CD pipeline."},
            "decision": {"reasoning": "Relevant tooling mentioned."},
            "analysis": {"difficulty_level": 0.4, "consistency": 0.75}
        }
    ]
    records = _extract_turn_records(history)
    assert len(records) == 1
    assert records[0]["turn_number"] == 1
    assert "GitHub Actions" in records[0]["answer_summary"]
    assert "Understands CI/CD pipeline." in records[0]["observation"]


def test_build_competency_sections():
    history = [
        {
            "assessment": {
                "ownership": 9.0,
                "depth": 8.0,
                "impact": 7.0,
                "learning": 8.5,
                "communication": 8.0
            }
        }
    ]
    sections = _build_competency_sections(history)
    assert len(sections) == 5
    # Verify no raw numbers are in titles or tiers
    for sec in sections:
        assert "/10" not in sec["tier"]
        assert sec["tier"] in ["Higher rubric estimate", "Mid-range rubric estimate", "Lower rubric estimate"]
