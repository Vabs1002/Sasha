"""
Integration tests for Sasha FastAPI Server (api.py).
Tests REST endpoints, lifecycle, adaptive turns, and report downloads.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from fastapi.testclient import TestClient
from api import app, sessions


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Sasha AI Interviewer API"


def test_start_interview_unsupported_format(client):
    response = client.post(
        "/api/v1/interview/start",
        files={"resume_file": ("resume.txt", b"plain text", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported resume format" in response.json()["detail"]


def test_full_interview_lifecycle_api(client, tmp_path):
    # 1. Start session by mocking a session directly in session store
    profile = {
        "name": "Jordan Lee",
        "role": "SDE",
        "level": "senior",
        "years_experience": 6,
        "raw_text": "Senior Distributed Systems Engineer with Python, Redis, and Kafka experience."
    }
    sample_jd = "Role: Senior Backend Engineer\nMust have: distributed caching, Redis, Kafka, write-through."

    session_id = sessions.create(profile, jd_text=sample_jd, jd_summary=sample_jd[:300])
    assert session_id is not None

    # 2. Check session status
    status_resp = client.get(f"/api/v1/interview/{session_id}/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["candidate_name"] == "Jordan Lee"
    assert status_data["turns_completed"] == 0

    # 3. Submit Turn 1
    turn1_payload = {
        "answer": "I built a distributed Redis caching layer with a write-through strategy backed by PostgreSQL, handling 25k RPS and cutting latency by 75%."
    }
    turn1_resp = client.post(f"/api/v1/interview/{session_id}/turn", json=turn1_payload)
    assert turn1_resp.status_code == 200
    t1_data = turn1_resp.json()
    assert t1_data["turn_number"] == 1
    assert t1_data["is_complete"] is False
    assert t1_data["next_question"] is not None

    # 4. Submit Turn 2
    turn2_payload = {
        "answer": "We handled stream processing using Kafka with cooperative sticky partition rebalancing to avoid stop-the-world lags."
    }
    turn2_resp = client.post(f"/api/v1/interview/{session_id}/turn", json=turn2_payload)
    assert turn2_resp.status_code == 200
    t2_data = turn2_resp.json()
    assert t2_data["turn_number"] == 2

    # 5. Check updated status
    status_resp = client.get(f"/api/v1/interview/{session_id}/status")
    assert status_resp.json()["turns_completed"] == 2

    # 6. Verify HTML report endpoint
    html_resp = client.get(f"/api/v1/interview/{session_id}/report/html")
    assert html_resp.status_code == 200
    assert "CONFIDENTIAL" in html_resp.text
    assert "Jordan Lee" in html_resp.text

    # 7. Verify PDF report download endpoint
    pdf_resp = client.get(f"/api/v1/interview/{session_id}/report")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"

    # Cleanup generated files
    for f in [f"interview_report_{session_id}.pdf", f"interview_report_{session_id}.html"]:
        if os.path.exists(f):
            os.remove(f)


def test_websocket_barge_in(client):
    profile = {
        "name": "Sam Taylor",
        "role": "SDE",
        "level": "mid",
        "years_experience": 3,
        "raw_text": "Software engineer working with microservices."
    }
    session_id = sessions.create(profile)

    with client.websocket_connect(f"/ws/interview/{session_id}") as websocket:
        # Initial greeting received
        data = websocket.receive_json()
        assert data["type"] == "question"
        assert "Thanks for sharing" in data["text"]

        # Test barge-in interrupt signal
        websocket.send_json({"type": "interrupt"})
        interrupt_resp = websocket.receive_json()
        assert interrupt_resp["type"] == "interrupted"
        assert "<50ms" in interrupt_resp["status"]
