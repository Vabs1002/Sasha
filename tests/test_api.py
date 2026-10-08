"""
Integration tests for Sasha FastAPI Server (api.py).
Tests REST endpoints, lifecycle, adaptive turns, and report downloads.
"""
import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from fastapi.testclient import TestClient
import api
from api import app, sessions, REPORT_DIR


@pytest.fixture
def client(monkeypatch):
    # API integration tests should stay deterministic and avoid network calls;
    # provider behavior is covered separately in interviewer-agent tests.
    monkeypatch.setattr(
        api,
        "get_interviewer_decision",
        lambda *args, **kwargs: {
            "assessment": {
                "ownership": 5,
                "depth": 5,
                "impact": 5,
                "learning": 5,
                "communication": 5,
                "notes": "Deterministic API test response.",
            },
            "decision": {
                "action": "move_on",
                "reasoning": "Continue the interview.",
                "follow_up_topic": "",
            },
            "decision_source": "llm",
            "next_question": "What trade-offs did you consider?",
        },
    )
    return TestClient(app)


def confirm_profile(client, session_id):
    response = client.post(
        f"/api/v1/interview/{session_id}/turn",
        json={"answer": "Yes, that is accurate."},
    )
    assert response.status_code == 200
    assert response.json()["profile_confirmed"] is True
    assert response.json()["turn_number"] == 0


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Sasha AI Interviewer API"


def test_cors_rejects_unconfigured_origins(client):
    response = client.options(
        "/health",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in response.headers


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
    confirm_profile(client, session_id)

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
    assert t1_data["decision_source"] == "llm"

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
    assert html_resp.headers["cache-control"] == "no-store"
    assert "CONFIDENTIAL" in html_resp.text
    assert "Jordan Lee" in html_resp.text

    # 7. Verify PDF report download endpoint
    pdf_resp = client.get(f"/api/v1/interview/{session_id}/report")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert pdf_resp.headers["cache-control"] == "no-store"

    # Cleanup generated files
    for f in [
        os.path.join(REPORT_DIR, f"interview_report_{session_id}.pdf"),
        os.path.join(REPORT_DIR, f"interview_report_{session_id}.html"),
    ]:
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
        assert "Before we start" in data["text"]
        assert "answer in your own words" in data["text"]

        websocket.send_json({"type": "answer", "text": "Yes, that is accurate."})
        assert websocket.receive_json()["type"] == "turn_processing"
        profile_result = websocket.receive_json()
        assert profile_result["type"] == "turn_result"
        assert profile_result["profile_confirmed"] is True
        assert profile_result["turn"] == 0

        # Test barge-in interrupt signal
        websocket.send_json({"type": "interrupt"})
        interrupt_resp = websocket.receive_json()
        assert interrupt_resp["type"] == "interrupted"
        assert interrupt_resp["cancelled_turn"] is False


def test_websocket_interrupt_cancels_processing_turn(client, monkeypatch):
    profile = {
        "name": "Taylor Jordan",
        "role": "SDE",
        "level": "mid",
        "years_experience": 4,
        "raw_text": "Backend engineer with Python experience.",
    }
    session_id = sessions.create(profile)
    # This test focuses on cancellation of a scored provider turn, not setup.
    sessions.get(session_id)["awaiting_profile_confirmation"] = False

    async def slow_submit_turn(*args, **kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr(api, "submit_turn", slow_submit_turn)

    with client.websocket_connect(f"/ws/interview/{session_id}") as websocket:
        assert websocket.receive_json()["type"] == "question"
        websocket.send_json({"type": "answer", "text": "A synthetic answer."})
        processing = websocket.receive_json()
        assert processing["type"] == "turn_processing"

        websocket.send_json({"type": "interrupt"})
        interrupted = websocket.receive_json()
        assert interrupted["type"] == "interrupted"
        assert interrupted["cancelled_turn"] is True
        assert sessions.get(session_id)["turn_state"] == "ready"

        # The server continues receiving messages after cancellation.
        websocket.send_json({"type": "interrupt"})
        assert websocket.receive_json()["cancelled_turn"] is False


def test_websocket_turn_reports_processing_time(client):
    profile = {
        "name": "Morgan Lee",
        "role": "SDE",
        "level": "mid",
        "years_experience": 3,
        "raw_text": "Python backend engineer.",
    }
    session_id = sessions.create(profile)

    with client.websocket_connect(f"/ws/interview/{session_id}") as websocket:
        assert websocket.receive_json()["type"] == "question"
        websocket.send_json({"type": "answer", "text": "Yes, that is accurate."})
        assert websocket.receive_json()["type"] == "turn_processing"
        assert websocket.receive_json()["profile_confirmed"] is True
        websocket.send_json({"type": "answer", "text": "I built a Python API."})
        assert websocket.receive_json()["type"] == "turn_processing"
        result = websocket.receive_json()

    assert result["type"] == "turn_result"
    assert result["decision_source"] == "llm"
    assert result["processing_ms"] >= 0
    assert sessions.get(session_id)["turn_state"] == "ready"


def test_websocket_streams_question_text_before_turn_result(client, monkeypatch):
    profile = {
        "name": "Riley Park",
        "role": "SDE",
        "level": "mid",
        "years_experience": 3,
        "raw_text": "Python backend engineer.",
    }
    session_id = sessions.create(profile)

    def streaming_decision(*args, stream_callback=None, **kwargs):
        stream_callback("How did you ")
        stream_callback("measure impact?")
        return {
            "assessment": {"ownership": 6, "depth": 6, "impact": 6, "learning": 6, "communication": 6},
            "decision": {"action": "follow_up", "reasoning": "Probe impact."},
            "decision_source": "llm",
            "next_question": "How did you measure impact?",
        }

    monkeypatch.setattr(api, "get_interviewer_decision", streaming_decision)

    with client.websocket_connect(f"/ws/interview/{session_id}") as websocket:
        assert websocket.receive_json()["type"] == "question"
        websocket.send_json({"type": "answer", "text": "Yes, that is accurate."})
        assert websocket.receive_json()["type"] == "turn_processing"
        assert websocket.receive_json()["profile_confirmed"] is True
        websocket.send_json({"type": "answer", "text": "I owned a backend API."})
        assert websocket.receive_json()["type"] == "turn_processing"
        first_delta = websocket.receive_json()
        second_delta = websocket.receive_json()
        result = websocket.receive_json()

    assert first_delta == {"type": "question_delta", "text": "How did you "}
    assert second_delta == {"type": "question_delta", "text": "measure impact?"}
    assert result["type"] == "turn_result"
    assert result["next_question"] == "How did you measure impact?"


def test_live_turn_skips_experimental_perplexity_without_consent(client, monkeypatch):
    profile = {
        "name": "Alex Kim",
        "role": "SDE",
        "level": "mid",
        "years_experience": 2,
        "raw_text": "Python engineer with database experience.",
    }
    session_id = sessions.create(profile, integrity_monitoring_consent=False)
    confirm_profile(client, session_id)
    monkeypatch.setattr(api, "get_disfluency_rate", lambda _answer: 0.0)
    monkeypatch.setattr(api, "get_consistency", lambda _resume, _answer: 0.5)
    monkeypatch.setattr(
        api,
        "get_perplexity",
        lambda _answer: (_ for _ in ()).throw(AssertionError("perplexity should be disabled")),
    )

    result = asyncio.run(api.submit_turn(session_id, api.TurnRequest(answer="I built a Python API.")))

    assert result.turn_number == 1
    assert sessions.get(session_id)["history"][0]["analysis"]["perplexity_available"] is False


def test_live_turn_runs_perplexity_with_monitoring_consent(client, monkeypatch):
    profile = {
        "name": "Jamie Chen",
        "role": "SDE",
        "level": "mid",
        "years_experience": 2,
        "raw_text": "Python engineer with database experience.",
    }
    session_id = sessions.create(profile, integrity_monitoring_consent=True)
    confirm_profile(client, session_id)
    monkeypatch.setattr(api, "get_disfluency_rate", lambda _answer: 0.0)
    monkeypatch.setattr(api, "get_consistency", lambda _resume, _answer: 0.5)
    monkeypatch.setattr(api, "get_perplexity", lambda _answer: 85.0)

    result = asyncio.run(api.submit_turn(session_id, api.TurnRequest(answer="I built a Python API.")))

    analysis = sessions.get(session_id)["history"][0]["analysis"]
    assert result.turn_number == 1
    assert analysis["perplexity_available"] is True
    assert analysis["perplexity"] == 85.0
