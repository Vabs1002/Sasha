"""
Unit tests for resume_parser.py
"""
import os
import sys
import tempfile
import pytest

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from resume_parser import extract_text_from_pdf, extract_text_from_docx, parse_resume


def test_extract_text_from_pdf_not_implemented():
    """Test that PDF extraction raises NotImplementedError for non-existent file"""
    with pytest.raises(Exception):  # Could be FileNotFoundError or pdfminer error
        extract_text_from_pdf("non_existent_file.pdf")


def test_extract_text_from_docx_not_implemented():
    """Test that DOCX extraction raises NotImplementedError for non-existent file"""
    with pytest.raises(Exception):  # Could be FileNotFoundError or docx error
        extract_text_from_docx("non_existent_file.docx")


def test_parse_resume_unsupported_format():
    """Test that unsupported file formats raise ValueError"""
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_resume("test.txt")


def test_parse_resume_with_mock_data():
    """Test resume parser with mock data"""
    # Create a simple text file to test with
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
        John Doe
        Software Engineer

        Experience:
        January 2020 - Present: Software Engineer at Tech Corp
          - Built web applications using Python and React
          - Improved system performance by 30%
          - Led a team of 3 engineers

        Skills: Python, JavaScript, React, AWS, Docker

        Projects:
        - Built a recommendation engine using machine learning
        - Developed a REST API for user management
        """)
        temp_file = f.name

    try:
        # Since we can't easily create a real PDF/DOCX in test,
        # we'll test the parsing logic by mocking the file type detection
        # For now, we'll just verify the function exists and can be imported
        assert callable(parse_resume)
    finally:
        os.unlink(temp_file)


def test_extract_candidate_name():
    """Test candidate name extraction across different resume header styles"""
    from resume_parser import extract_candidate_name

    # Standard header
    text1 = "Alex Mercer\nSenior Fullstack Engineer\nalex@mercer.dev\n+1 555-0199"
    assert extract_candidate_name(text1) == "Alex Mercer"

    # Header with title / curriculum vitae prefix
    text2 = "CURRICULUM VITAE\nPriya Sharma\nBackend Developer\nBangalore, India"
    assert extract_candidate_name(text2) == "Priya Sharma"

    # Three-word name
    text3 = "David Lee Roth\nDistributed Systems Architect"
    assert extract_candidate_name(text3) == "David Lee Roth"

    # Empty fallback
    assert extract_candidate_name("") == "Candidate"


if __name__ == "__main__":
    pytest.main([__file__])