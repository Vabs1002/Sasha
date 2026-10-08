"""
Unit tests for interviewer_agent.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace
import yaml
import pytest
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import interviewer_agent as interviewer
from interviewer_agent import load_competency_bank, load_competency_selector_model, extract_features


def test_load_competency_bank():
    """Test loading competency bank from YAML"""
    # Create a temporary competency bank
    sample_bank = {
        'SDE': {
            'competencies': [
                {
                    'name': 'System Design',
                    'signals': ['system design', 'apis'],
                    'follow_up_templates': ['How did you design the system?']
                }
            ]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_bank, f)
        temp_file = f.name

    try:
        bank = load_competency_bank(temp_file)
        assert bank == sample_bank
    finally:
        os.unlink(temp_file)


def test_load_competency_selector_model_nonexistent():
    """Test loading non-existent competency selector model"""
    model = load_competency_selector_model('non_existent_model.pkl')
    assert model is None


def test_extract_features_basic():
    """Test feature extraction"""
    resume_profile = {
        'years_experience': 2.5,
        'level': 'junior',
        'role': 'SDE',
        'skills': ['Python', 'JavaScript'],
        'projects': ['Web app']
    }

    history = []
    analysis = {
        'perplexity': 50.0,
        'disfluency_rate': 0.1,
        'consistency': 0.8
    }

    # This should not crash
    features = extract_features(resume_profile, history, analysis, {}, None)
    assert features is not None
    assert features.shape[0] == 1  # 1 row
    assert features.shape[1] > 0   # multiple columns


def test_extract_features_with_history():
    """Test feature extraction with history"""
    resume_profile = {
        'years_experience': 3.0,
        'level': 'mid',
        'role': 'ML Engineer',
        'skills': ['Python', 'TensorFlow'],
        'projects': ['Recommendation system']
    }

    history = [
        {
            'assessment': {
                'ownership': 8,
                'depth': 7,
                'impact': 6,
                'learning': 9,
                'communication': 8
            }
        },
        {
            'assessment': {
                'ownership': 7,
                'depth': 8,
                'impact': 7,
                'learning': 8,
                'communication': 7
            }
        }
    ]

    analysis = {
        'perplexity': 60.0,
        'disfluency_rate': 0.15,
        'consistency': 0.7
    }

    # This should not crash
    features = extract_features(resume_profile, history, analysis, {}, None)
    assert features is not None
    assert features.shape[0] == 1  # 1 row
    assert features.shape[1] > 0   # multiple columns


def test_interviewer_uses_configured_llm_and_marks_decision_source(monkeypatch):
    content = (
        '{"assessment":{"ownership":6,"depth":6,"impact":6,"learning":6,'
        '"communication":6,"notes":"Relevant example."},'
        '"decision":{"action":"move_on","reasoning":"Continue.","follow_up_topic":""},'
        '"next_question":"What trade-offs did you consider?"}'
    )
    fake_client = Mock()
    fake_client.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content=content))
    ]
    monkeypatch.setattr(interviewer, "client", fake_client)
    monkeypatch.setattr(interviewer, "default_llm_model", "gemini-test-model")
    monkeypatch.setattr(interviewer, "load_competency_bank", lambda _path: {})
    monkeypatch.setattr(interviewer, "load_competency_selector_model", lambda _path: None)
    monkeypatch.setattr(interviewer, "select_competency", lambda *args: "technical_depth")
    monkeypatch.setattr(interviewer, "format_prompt_for_agentic_rag", lambda *args, **kwargs: "prompt")

    result = interviewer.get_interviewer_decision(
        {"raw_text": "Python and distributed systems"},
        [],
        {"transcript": "I built a service."},
    )

    fake_client.chat.completions.create.assert_called_once()
    assert fake_client.chat.completions.create.call_args.kwargs["extra_body"]["reasoning_effort"] == "minimal"
    assert "Do not call tools" in fake_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert result["next_question"] == "What trade-offs did you consider?"
    assert result["decision_source"] == "llm"


def test_decode_json_string_prefix_handles_escapes_and_partial_sequences():
    assert interviewer._decode_json_string_prefix(r'Hello\nworld!') == "Hello\nworld!"
    assert interviewer._decode_json_string_prefix(r'quote: \"yes\"') == 'quote: "yes"'
    assert interviewer._decode_json_string_prefix(r'Unicode: \u263A') == "Unicode: ☺"
    assert interviewer._decode_json_string_prefix(r'trailing slash\\') == "trailing slash\\"
    assert interviewer._decode_json_string_prefix(r'incomplete \u26') == "incomplete "


def test_interviewer_streams_candidate_question_before_assessment(monkeypatch):
    content = (
        '{"decision":{"action":"follow_up","reasoning":"Probe depth.",'
        '"follow_up_topic":"trade-offs"},'
        '"next_question":"What trade-offs\\nwere most important?",'
        '"assessment":{"ownership":6,"depth":6,"impact":6,"learning":6,'
        '"communication":6,"notes":"Relevant example."}}'
    )
    pieces = [content[:120], content[120:154], content[154:176], content[176:]]
    stream = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])
        for piece in pieces
    ]
    monkeypatch.setattr(interviewer, "client", object())
    monkeypatch.setattr(interviewer, "default_llm_model", "gemini-test-model")
    monkeypatch.setattr(interviewer, "load_competency_bank", lambda _path: {})
    monkeypatch.setattr(interviewer, "load_competency_selector_model", lambda _path: None)
    monkeypatch.setattr(interviewer, "select_competency", lambda *args: "technical_depth")
    monkeypatch.setattr(interviewer, "_create_chat_completion", lambda **kwargs: stream)
    deltas = []

    result = interviewer.get_interviewer_decision(
        {"raw_text": "Python and distributed systems"},
        [],
        {"transcript": "I built a service."},
        stream_callback=deltas.append,
    )

    assert result["decision_source"] == "llm"
    assert result["next_question"] == "What trade-offs\nwere most important?"
    assert "".join(deltas) == result["next_question"]
    assert deltas


if __name__ == "__main__":
    pytest.main([__file__])
