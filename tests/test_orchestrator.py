import os
import sys
import pytest
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from orchestrator import ConversationalStateMachine, SessionState, RealtimeOrchestrator
from analyzer import detect_background_audio_anomaly, estimate_head_eye_signals

def test_fsm_valid_transitions():
    async def _test():
        fsm = ConversationalStateMachine()
        assert fsm.current_state == SessionState.IDLE
        assert await fsm.transition_to(SessionState.LISTENING) is True
        assert await fsm.transition_to(SessionState.TOOL_CALLING) is True
        assert await fsm.transition_to(SessionState.RESPONDING) is True
        assert await fsm.transition_to(SessionState.DRAINING) is True
        assert await fsm.transition_to(SessionState.LISTENING) is True
    asyncio.run(_test())

def test_fsm_invalid_transition_rejected():
    async def _test():
        fsm = ConversationalStateMachine()
        assert await fsm.transition_to(SessionState.RESPONDING) is False
    asyncio.run(_test())

def test_orchestrator_turn_streaming():
    async def _test():
        orchestrator = RealtimeOrchestrator()
        tokens = []
        async for event in orchestrator.process_turn("I built a recommendation system using Kafka", "Led Kafka system design"):
            if event.get("type") == "token":
                tokens.append(event["text"])
        assert len(tokens) > 0
    asyncio.run(_test())

def test_orchestrator_barge_in():
    async def _test():
        orchestrator = RealtimeOrchestrator()
        orchestrator.trigger_barge_in()
        events = []
        async for event in orchestrator.process_turn("I scaled the database", "Database scaling expert"):
            events.append(event.get("type"))
        assert "interrupted" in events
    asyncio.run(_test())

def test_background_audio_anomaly():
    # Test normal speech
    res_normal = detect_background_audio_anomaly("I deployed the model to production.")
    assert res_normal["noise_detected"] is False

    # Test detected chatter / prompting
    res_prompted = detect_background_audio_anomaly("Wait, the answer is Redis, tell him that")
    assert res_prompted["noise_detected"] is True

def test_head_eye_signals():
    signals = estimate_head_eye_signals({"yaw": 12.5, "pitch": -4.2, "gaze_ratio": 0.15})
    assert signals["head_yaw"] == 12.5
    assert signals["gaze_off_center_ratio"] == 0.15

def test_video_frame_proctoring():
    from analyzer import analyze_video_frame_proctoring
    import numpy as np

    # Test empty / None frame
    res_none = analyze_video_frame_proctoring(None)
    assert res_none["face_detected"] is False

    # Test synthetic frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res_dummy = analyze_video_frame_proctoring(dummy_frame)
    assert "face_detected" in res_dummy
    assert "head_pose" in res_dummy

def test_eye_contact_score():
    from analyzer import calculate_eye_contact_score
    import numpy as np

    # Test None frame
    res_none = calculate_eye_contact_score(None)
    assert res_none["eye_contact_score"] == 1.0

    # Test dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res_dummy = calculate_eye_contact_score(dummy_frame)
    assert "eye_contact_score" in res_dummy
    assert "nervousness_detected" in res_dummy

def test_verify_audio_visual_speech_sync():
    from analyzer import verify_audio_visual_speech_sync
    import numpy as np

    # Test None frame
    res_none = verify_audio_visual_speech_sync(None, None, audio_active=True)
    assert res_none["proxy_speaker_suspected"] is False

    # Test synthetic frame pair
    prev_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    curr_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res_sync = verify_audio_visual_speech_sync(prev_frame, curr_frame, audio_active=False)
    assert "is_speaking_visually" in res_sync
    assert "mouth_movement_score" in res_sync


