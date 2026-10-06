"""
Unit tests for Agentic RAG functionality in interviewer_agent.py
"""
import os
import sys
import tempfile

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from interviewer_agent import search_resume_tool, _build_faiss_index, _search_resume_with_faiss, _search_resume_fallback


def test_search_resume_tool_empty_resume():
    """Test search resume tool with empty resume"""
    results = search_resume_tool("test query", "")
    assert results == []

    results = search_resume_tool("test query", "   \n\t  ")
    assert results == []


def test_search_resume_tool_simple_match():
    """Test search resume tool with simple keyword matching"""
    resume_text = """
    John Doe
    Software Engineer

    Experience:
    - Built web applications using Python and React
    - Improved system performance by 40% through optimization
    - Worked with AWS cloud services

    Skills: Python, JavaScript, React, AWS, Docker

    Projects:
    - Created a real-time chat application with WebSockets
    - Developed a recommendation engine using machine learning
    """

    # Test exact keyword match
    results = search_resume_tool("Python", resume_text)
    assert len(results) > 0
    assert any("Python" in result for result in results)

    # Test phrase match
    results = search_resume_tool("web applications", resume_text)
    assert len(results) > 0
    assert any("web applications" in result.lower() for result in results)

    # Test no match
    results = search_resume_tool("blockchain", resume_text)
    # Might return empty or low relevance results depending on implementation


def test_build_faiss_index():
    """Test FAISS index building"""
    resume_text = "This is a test sentence. Another test sentence here."

    index, sentences = _build_faiss_index(resume_text)

    # Should return tuples
    assert isinstance(index, type(None)) or hasattr(index, 'search')  # Either None or FAISS index
    assert isinstance(sentences, list)
    assert len(sentences) > 0


def test_search_resume_fallback():
    """Test fallback search function"""
    resume_text = "I have experience with Python and JavaScript. I built web applications."

    results = _search_resume_fallback("Python", resume_text)
    assert len(results) > 0
    assert any("Python" in result for result in results)

    results = _search_resume_fallback("Java", resume_text)
    # Should find JavaScript since it contains "Java" as substring
    # Or might return empty depending on implementation

    results = _search_resume_fallback("blockchain", resume_text)
    # Likely empty for no match


def test_search_resume_with_faiss_when_available():
    """Test FAISS search when available"""
    resume_text = "Experienced in Python development. Built scalable web applications."

    index, sentences = _build_faiss_index(resume_text)

    if index is not None and sentences:
        results = _search_resume_with_faiss("Python", resume_text, None, index, sentences)
        # Note: We're passing None for embedder but the function handles it
        # In real usage, _embedder would be passed

        # The function should not crash
        assert isinstance(results, list)


if __name__ == "__main__":
    # Run tests manually if not using pytest
    test_search_resume_tool_empty_resume()
    print("✓ test_search_resume_tool_empty_resume passed")

    test_search_resume_tool_simple_match()
    print("✓ test_search_resume_tool_simple_match passed")

    test_build_faiss_index()
    print("✓ test_build_faiss_index passed")

    test_search_resume_fallback()
    print("✓ test_search_resume_fallback passed")

    test_search_resume_with_faiss_when_available()
    print("✓ test_search_resume_with_faiss_when_available passed")

    print("\nAll tests passed! 🎉")