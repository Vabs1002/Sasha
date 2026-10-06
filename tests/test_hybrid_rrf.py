import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from interviewer_agent import _bm25_tokenize, _search_resume_bm25, _search_resume_hybrid_rrf, search_resume_tool

RESUME_TEXT = (
    "Senior Data Engineer with 5 years experience in building distributed systems. "
    "Designed and scaled Kafka streaming pipelines handling 50,000 events per second. "
    "Optimized PostgreSQL queries, reducing latency by 45 percent. "
    "Led cross-functional teams deploying Docker and Kubernetes on AWS."
)

def test_bm25_tokenization():
    tokens = _bm25_tokenize("Kafka streaming pipelines 50,000!")
    assert "kafka" in tokens
    assert "streaming" in tokens
    assert "50" in tokens or "50000" in tokens

def test_bm25_scoring():
    sentences = [s.strip() for s in RESUME_TEXT.split('.') if len(s.strip()) > 10]
    scores = _search_resume_bm25("Kafka streaming", sentences)
    assert len(scores) > 0
    top_doc_idx = scores[0][0]
    assert "kafka" in sentences[top_doc_idx].lower()

def test_hybrid_rrf_combines_dense_and_sparse():
    results = search_resume_tool("Kafka streaming", RESUME_TEXT)
    assert len(results) > 0
    assert any("kafka" in r.lower() for r in results)

def test_hybrid_rrf_semantic_matching():
    # Query with semantic intent not sharing exact words
    results = search_resume_tool("database performance tuning", RESUME_TEXT)
    assert len(results) > 0
    assert any("postgresql" in r.lower() or "latency" in r.lower() for r in results)
