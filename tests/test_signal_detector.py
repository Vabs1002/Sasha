"""
Unit tests for signal_detector.py
"""
import os
import sys
import tempfile
import yaml
import pytest

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from signal_detector import load_or_train_model, detect_role


def test_load_or_train_model_with_empty_signals():
    """Test model training with empty signals dictionary"""
    # Create a temporary YAML file with minimal content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({}, f)
        temp_file = f.name

    try:
        # This should not crash, even with empty signals
        clf, vec = load_or_train_model({})
        assert clf is not None
        assert vec is not None
    finally:
        os.unlink(temp_file)


def test_load_or_train_model_with_sample_data():
    """Test model training with sample role signals"""
    sample_signals = {
        'SDE': {
            'skills': ['python', 'algorithms'],
            'projects': ['web app', 'api'],
            'weight': 1.0
        },
        'Data Science': {
            'skills': ['statistics', 'machine learning'],
            'projects': ['analysis', 'modeling'],
            'weight': 1.0
        }
    }

    # This should not crash
    clf, vec = load_or_train_model(sample_signals)
    assert clf is not None
    assert vec is not None


def test_detect_role_basic():
    """Test basic role detection functionality"""
    sample_signals = {
        'SDE': {
            'skills': ['python', 'algorithms', 'software engineering'],
            'projects': ['web application', 'rest api'],
            'weight': 1.0
        }
    }

    resume_text = "I am a software engineer with experience in python and algorithms. I built web applications and REST APIs."

    result = detect_role(resume_text, sample_signals)

    # Check that we get a valid result
    assert 'role' in result
    assert 'confidence' in result
    assert 'scores' in result
    assert isinstance(result['confidence'], float)
    assert 0 <= result['confidence'] <= 1


if __name__ == "__main__":
    pytest.main([__file__])