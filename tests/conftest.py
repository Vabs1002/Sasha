"""
Shared test fixtures and configuration
"""
import pytest
import tempfile
import os
import yaml


@pytest.fixture
def sample_role_signals():
    """Sample role signals for testing"""
    return {
        'SDE': {
            'skills': ['python', 'algorithms', 'software engineering'],
            'projects': ['web application', 'rest api'],
            'weight': 1.0
        },
        'ML Engineer': {
            'skills': ['machine learning', 'python', 'tensorflow'],
            'projects': ['recommendation system', 'image classification'],
            'weight': 1.0
        }
    }


@pytest.fixture
def sample_competency_bank():
    """Sample competency bank for testing"""
    return {
        'SDE': {
            'competencies': [
                {
                    'name': 'System Design',
                    'signals': ['system design', 'apis', 'microservices'],
                    'follow_up_templates': [
                        "How did you ensure scalability in your design?",
                        "What trade-offs did you consider?"
                    ]
                },
                {
                    'name': 'Problem Solving',
                    'signals': ['algorithms', 'problem solving', 'optimization'],
                    'follow_up_templates': [
                        "Walk me through your approach to solving this problem.",
                        "What was the time complexity of your solution?"
                    ]
                }
            ]
        }
    }


@pytest.fixture
def sample_resume_profile():
    """Sample resume profile for testing"""
    return {
        'years_experience': 2.5,
        'level': 'junior',
        'role': 'SDE',
        'skills': ['Python', 'JavaScript', 'React'],
        'projects': ['Built a web application', 'Created a REST API'],
        'raw_text': 'Experienced software engineer with 2.5 years of experience building web applications.'
    }


@pytest.fixture
def sample_analysis():
    """Sample analysis results for testing"""
    return {
        'transcript': 'I built a web application using React and Node.js.',
        'perplexity': 75.5,
        'disfluency_rate': 0.12,
        'consistency': 0.8
    }


@pytest.fixture
def temp_yaml_file():
    """Create a temporary YAML file for testing"""
    def _create_temp_yaml(data):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(data, f)
            return f.name
    return _create_temp_yaml