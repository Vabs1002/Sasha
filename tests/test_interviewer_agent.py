"""
Unit tests for interviewer_agent.py
"""
import os
import sys
import tempfile
import yaml
import pytest
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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


if __name__ == "__main__":
    pytest.main([__file__])