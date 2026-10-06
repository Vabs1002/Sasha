"""
Unit and integration tests for Research-Grade Anti-Cheating & Proctoring Suite.
Verifies:
  1. Perspective-n-Point 3D Head Pose (Li et al. 2021, Ruiz et al. CVPR 2018)
  2. Iris-Based Gaze Estimation (MPIIGaze: Zhang et al. TPAMI 2019)
  3. Eye Aspect Ratio (EAR) & Blink Saccades (Soukupov\u00e1 & \u010cech 2016)
  4. Mouth Aspect Ratio (MAR) Lip-Sync & Proxy Speaker Detection (SyncNet: Chung & Zisserman 2016)
  5. Backwards-compatible resilient fallback handling
"""
import sys
import os
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analyzer import (
    rotation_matrix_to_euler_angles,
    estimate_head_pose_pnp,
    estimate_iris_gaze,
    calculate_mouth_aspect_ratio,
    analyze_video_frame_proctoring,
    calculate_eye_contact_score,
    verify_audio_visual_speech_sync,
    FACE_3D_MODEL_POINTS,
)


class MockLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def test_rotation_matrix_to_euler_angles_identity():
    """Identity matrix should yield (0, 0, 0) Euler angles."""
    eye_mat = np.eye(3, dtype=np.float64)
    pitch, yaw, roll = rotation_matrix_to_euler_angles(eye_mat)
    assert abs(pitch) < 0.01
    assert abs(yaw) < 0.01
    assert abs(roll) < 0.01


def test_rotation_matrix_to_euler_angles_known_yaw():
    """45 degree yaw rotation around Y axis."""
    theta = np.radians(45.0)
    # Rotation matrix around Y axis
    r_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ], dtype=np.float64)
    pitch, yaw, roll = rotation_matrix_to_euler_angles(r_y)
    assert abs(yaw - 45.0) < 0.1
    assert abs(pitch) < 0.1
    assert abs(roll) < 0.1


def test_estimate_head_pose_pnp_frontal():
    """Frontal face landmark arrangement should yield near zero Euler angles."""
    frame_w, frame_h = 640, 480
    # Create normalized mock landmarks (0.0 to 1.0)
    landmarks = [MockLandmark(0.5, 0.5)] * 478

    # Set key landmarks for frontal alignment
    landmarks[1] = MockLandmark(320.0 / frame_w, 240.0 / frame_h)      # Nose tip
    landmarks[152] = MockLandmark(320.0 / frame_w, 350.0 / frame_h)    # Chin
    landmarks[263] = MockLandmark(245.0 / frame_w, 190.0 / frame_h)    # Left eye outer
    landmarks[33] = MockLandmark(395.0 / frame_w, 190.0 / frame_h)     # Right eye outer
    landmarks[291] = MockLandmark(280.0 / frame_w, 285.0 / frame_h)    # Left mouth
    landmarks[61] = MockLandmark(360.0 / frame_w, 285.0 / frame_h)     # Right mouth

    pitch, yaw, roll = estimate_head_pose_pnp(landmarks, frame_w, frame_h)
    assert abs(yaw) < 5.0
    assert abs(pitch) < 5.0


def test_estimate_iris_gaze_center():
    """Centered iris position should yield horizontal ratio ~0.50."""
    frame_w, frame_h = 640, 480
    landmarks = [MockLandmark(0.5, 0.5)] * 478

    # Left eye: outer=240px, inner=280px, iris=260px (exact midpoint)
    landmarks[263] = MockLandmark(240.0 / frame_w, 200.0 / frame_h) # Outer
    landmarks[362] = MockLandmark(280.0 / frame_w, 200.0 / frame_h) # Inner
    landmarks[468] = MockLandmark(260.0 / frame_w, 200.0 / frame_h) # Left iris center
    landmarks[386] = MockLandmark(260.0 / frame_w, 190.0 / frame_h) # Top
    landmarks[374] = MockLandmark(260.0 / frame_w, 210.0 / frame_h) # Bottom

    # Right eye: outer=400px, inner=360px, iris=380px (exact midpoint)
    landmarks[33] = MockLandmark(400.0 / frame_w, 200.0 / frame_h)
    landmarks[133] = MockLandmark(360.0 / frame_w, 200.0 / frame_h)
    landmarks[473] = MockLandmark(380.0 / frame_w, 200.0 / frame_h)
    landmarks[159] = MockLandmark(380.0 / frame_w, 190.0 / frame_h)
    landmarks[145] = MockLandmark(380.0 / frame_w, 210.0 / frame_h)

    res = estimate_iris_gaze(landmarks, frame_w, frame_h)
    assert 0.45 <= res["horizontal_gaze_ratio"] <= 0.55
    assert res["gaze_direction"] == "CENTER"
    assert not res["is_looking_away"]


def test_estimate_iris_gaze_second_screen_left():
    """Iris shifted significantly toward left monitor should be detected as looking away."""
    frame_w, frame_h = 640, 480
    landmarks = [MockLandmark(0.5, 0.5)] * 478

    # Left eye: outer=240px, inner=300px, iris=245px (shifted far left near outer canthus)
    landmarks[263] = MockLandmark(240.0 / frame_w, 200.0 / frame_h)
    landmarks[362] = MockLandmark(300.0 / frame_w, 200.0 / frame_h)
    landmarks[468] = MockLandmark(245.0 / frame_w, 200.0 / frame_h)
    landmarks[386] = MockLandmark(245.0 / frame_w, 190.0 / frame_h)
    landmarks[374] = MockLandmark(245.0 / frame_w, 210.0 / frame_h)

    landmarks[33] = MockLandmark(400.0 / frame_w, 200.0 / frame_h)
    landmarks[133] = MockLandmark(340.0 / frame_w, 200.0 / frame_h)
    landmarks[473] = MockLandmark(345.0 / frame_w, 200.0 / frame_h)
    landmarks[159] = MockLandmark(345.0 / frame_w, 190.0 / frame_h)
    landmarks[145] = MockLandmark(345.0 / frame_w, 210.0 / frame_h)

    res = estimate_iris_gaze(landmarks, frame_w, frame_h)
    assert res["is_looking_away"]
    assert res["gaze_direction"] in ["LEFT_SCREEN", "RIGHT_SCREEN"]


def test_calculate_mouth_aspect_ratio():
    """MAR should be small when lips closed, large when mouth open."""
    frame_w, frame_h = 640, 480
    landmarks = [MockLandmark(0.5, 0.5)] * 478

    # Closed mouth: top lip y=280, bottom lip y=282 (2px opening vs 80px width)
    landmarks[13] = MockLandmark(320.0 / frame_w, 280.0 / frame_h)
    landmarks[14] = MockLandmark(320.0 / frame_w, 282.0 / frame_h)
    landmarks[291] = MockLandmark(280.0 / frame_w, 281.0 / frame_h)
    landmarks[61] = MockLandmark(360.0 / frame_w, 281.0 / frame_h)

    mar_closed = calculate_mouth_aspect_ratio(landmarks, frame_w, frame_h)
    assert mar_closed < 0.05

    # Open mouth (speaking): top lip y=270, bottom lip y=300 (30px opening vs 80px width)
    landmarks[13] = MockLandmark(320.0 / frame_w, 270.0 / frame_h)
    landmarks[14] = MockLandmark(320.0 / frame_w, 300.0 / frame_h)
    mar_open = calculate_mouth_aspect_ratio(landmarks, frame_w, frame_h)
    assert mar_open > 0.30


def test_analyze_video_frame_proctoring_none_frame():
    """None input returns safe standard dictionary without crashing."""
    res = analyze_video_frame_proctoring(None)
    assert not res["face_detected"]
    assert not res["multiple_faces"]
    assert res["head_pose"]["yaw"] == 0.0
    assert "No camera frame provided" in res["warning"]


def test_calculate_eye_contact_score_none_frame():
    """None input returns neutral 1.0 score."""
    res = calculate_eye_contact_score(None)
    assert res["eye_contact_score"] == 1.0
    assert not res["nervousness_detected"]


def test_verify_audio_visual_speech_sync_none_frame():
    """None frame returns default passthrough based on audio activity."""
    res = verify_audio_visual_speech_sync(None, None, audio_active=True)
    assert res["is_speaking_visually"] is True
    assert not res["proxy_speaker_suspected"]
