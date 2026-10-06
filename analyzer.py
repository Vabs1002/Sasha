from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import re
import torch
import numpy as np
import logging
from typing import Tuple, Optional, Union, Dict, Any, List

try:
    import faiss
except ImportError:
    faiss = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models once
try:
    _perplexity_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    _perplexity_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Successfully loaded NLP models for analysis")
except Exception as e:
    logger.error(f"Failed to load NLP models: {str(e)}")
    raise

def _build_faiss_index(text: str) -> Tuple[Optional[object], Optional[list]]:
    """
    Build a FAISS index (inner product) over sentence embeddings of the text.

    Args:
        text: Input text to build index from

    Returns:
        Tuple of (faiss_index, sentences) or (None, None) if FAISS not available
    """
    if faiss is None:
        logger.debug("FAISS not available, returning None")
        return None, None

    try:
        # Simple sentence split
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        if not sentences:
            sentences = [text]

        logger.debug(f"Building FAISS index from {len(sentences)} sentences")
        embeds = _embedder.encode(sentences, normalize_embeddings=True)  # normalized to unit length
        embeds = np.array(embeds).astype('float32')
        d = embeds.shape[1]
        index = faiss.IndexFlatIP(d)  # inner product => cosine since normalized
        index.add(embeds)
        return index, sentences
    except Exception as e:
        logger.error(f"Error building FAISS index: {str(e)}")
        return None, None

def get_perplexity(text: str) -> float:
    """
    Calculate perplexity score for text using DistilGPT-2.
    Lower perplexity indicates more predictable/AI-like text.

    Args:
        text: Input text to analyze

    Returns:
        Perplexity score (float)

    Raises:
        ValueError: If text is empty
        Exception: For model inference errors
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty for perplexity calculation")

    try:
        inputs = _perplexity_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            loss = _perplexity_model(**inputs, labels=inputs["input_ids"]).loss
        perplexity = torch.exp(loss).item()
        logger.debug(f"Calculated perplexity: {perplexity}")
        return perplexity
    except Exception as e:
        logger.error(f"Error calculating perplexity: {str(e)}")
        raise

def get_disfluency_rate(text: str) -> float:
    """
    Calculate disfluency rate based on filler words (um, uh, like, you know).

    Args:
        text: Input text to analyze

    Returns:
        Disfluency rate as ratio of fillers to total words (0.0 to 1.0)
    """
    if not text or not text.strip():
        return 0.0

    try:
        fillers = len(re.findall(r'\b(um|uh|like|you know)\b', text, re.IGNORECASE))
        words = len(text.split())
        rate = fillers / max(words, 1)
        logger.debug(f"Calculated disfluency rate: {rate} ({fillers} fillers in {words} words)")
        return rate
    except Exception as e:
        logger.error(f"Error calculating disfluency rate: {str(e)}")
        return 0.0

def get_consistency(resume_text: str, answer_text: str) -> float:
    """
    Calculate semantic consistency between resume text and answer text.
    Uses FAISS-enhanced sentence similarity if available, fallback to cosine similarity.

    Args:
        resume_text: Resume text content
        answer_text: Answer text content

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not resume_text or not resume_text.strip():
        logger.warning("Resume text is empty, returning 0.0 consistency")
        return 0.0

    if not answer_text or not answer_text.strip():
        logger.warning("Answer text is empty, returning 0.0 consistency")
        return 0.0

    try:
        # Try FAISS-based similarity; fallback to cosine
        index, sentences = _build_faiss_index(resume_text)
        if index is not None and faiss is not None:
            logger.debug("Using FAISS-based consistency calculation")
            answer_emb = _embedder.encode([answer_text], normalize_embeddings=True)
            D, I = index.search(np.array(answer_emb).astype('float32'), k=1)  # inner product
            # D[0][0] is max inner product (cosine similarity) between answer and any resume sentence
            similarity = float(D[0][0])
            # Ensure in [0,1]
            similarity = max(0.0, min(1.0, similarity))
            logger.debug(f"FAISS consistency: {similarity}")
            return similarity
        else:
            # Fallback to cosine similarity of whole texts
            logger.debug("Using fallback cosine similarity calculation")
            resume_emb = _embedder.encode([resume_text])[0]
            answer_emb = _embedder.encode([answer_text])[0]
            dot = sum(a*b for a,b in zip(resume_emb, answer_emb))
            norm_a = sum(a*a for a in resume_emb) ** 0.5
            norm_b = sum(b*b for b in answer_emb) ** 0.5
            similarity = dot / (norm_a * norm_b) if norm_a * norm_b != 0 else 0.0
            logger.debug(f"Cosine consistency: {similarity}")
            return similarity
    except Exception as e:
        logger.error(f"Error calculating consistency: {str(e)}")
        # Return 0.0 as safe fallback
        return 0.0

def detect_background_audio_anomaly(transcript: str, audio_energy_db: Optional[float] = None) -> Dict[str, Any]:
    """
    Detect whether extraneous background conversation or severe ambient noise is present.
    Returns:
        dict with 'noise_detected' (bool) and 'warning_message' (str or None)
    """
    # 1. Acoustic threshold check (if raw audio energy provided)
    if audio_energy_db is not None and audio_energy_db > -20.0:
        return {
            "noise_detected": True,
            "warning_message": "I noticed significant background audio. Could you please ensure you are in a quiet room?"
        }

    # 2. Linguistic cues for secondary speaker / background chatter
    cues = [
        r'\b(can you hear me|hey turn that down|stop talking|who are you talking to)\b',
        r'\b(alexa|siri|ok google)\b',
        r'\b(answer is|tell him|say that)\b'  # Potential prompting in room
    ]
    for pattern in cues:
        if re.search(pattern, transcript, re.IGNORECASE):
            return {
                "noise_detected": True,
                "warning_message": "I detected background conversation or secondary speech. Please ensure no one else is prompting in the room."
            }

    return {"noise_detected": False, "warning_message": None}

def detect_speech_stress(transcript: str, disfluency_rate: float, perplexity: float,
                         native_speaker: bool = True) -> Dict[str, Any]:
    """
    Detect cognitive stress / knowledge gap from text signals.

    NON-NATIVE SPEAKER BIAS FIX:
    Filler words (um, uh, like, you know) are a cultural and linguistic habit, NOT a
    stress signal on their own. Non-native English speakers routinely use fillers at
    2-3x native-speaker rates without being stressed. Similarly, non-native speakers
    produce lower perplexity due to simpler vocabulary — Stanford 2024 found >61% of
    non-native essays were wrongly flagged as AI-generated by perplexity thresholds.

    We ONLY use LANGUAGE-NEUTRAL, CONTENT-BASED signals:
      1. Epistemic uncertainty phrases ("I'm not sure", "I don't know")
      2. Answer brevity (< 8 avg words/sentence)
      3. Self-contradiction / backtracking markers ("no wait", "sorry I meant")

    Args:
        transcript: The candidate's answer text.
        disfluency_rate: Ratio of filler words to total words (0.0-1.0).
        perplexity: DistilGPT-2 perplexity score of the answer.
        native_speaker: Set False via ACCENT_FAIRNESS_MODE to skip
                        disfluency and perplexity as stress signals.

    Returns:
        dict with stress_score (0-1), stress_signals list, empathy_prompt,
        and knowledge_gap_detected bool.
    """
    if not transcript or not transcript.strip():
        return {
            "stress_score": 0.0,
            "stress_signals": [],
            "empathy_prompt": None,
            "knowledge_gap_detected": False
        }

    try:
        stress_score = 0.0
        signals = []
        words = transcript.split()
        total_words = max(len(words), 1)

        # Signal 1: Epistemic uncertainty phrases (language-neutral)
        uncertainty_patterns = [
            r"\bI('m| am) not sure\b",
            r"\bI don'?t (really )?know\b",
            r"\bnot (really )?sure (how|what|where|why|if)\b",
            r"\bI think\b.{0,40}\bI think\b",
            r"\bmaybe\b.{0,40}\bmaybe\b",
            r"\bI'?m guessing\b",
            r"\bcould be wrong\b",
        ]
        uncertainty_hits = sum(
            1 for p in uncertainty_patterns
            if re.search(p, transcript, re.IGNORECASE)
        )
        if uncertainty_hits >= 2:
            stress_score += 0.30
            signals.append("epistemic_uncertainty")
        elif uncertainty_hits == 1:
            stress_score += 0.15
            signals.append("mild_uncertainty")

        # Signal 2: Answer brevity (language-neutral)
        sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
        avg_words_per_sentence = total_words / max(len(sentences), 1)
        if avg_words_per_sentence < 6:
            stress_score += 0.30
            signals.append("very_brief_answers")
        elif avg_words_per_sentence < 10:
            stress_score += 0.15
            signals.append("brief_answers")

        # Signal 3: Backtracking / self-contradiction (language-neutral)
        backtrack_patterns = [
            r"\b(actually|wait),?\s+(no|sorry|I mean|let me)\b",
            r"\bsorry,?\s+(I meant|let me rephrase|actually)\b",
            r"\bno wait\b",
            r"\blet me start (over|again)\b",
        ]
        backtracks = sum(
            1 for p in backtrack_patterns
            if re.search(p, transcript, re.IGNORECASE)
        )
        if backtracks >= 2:
            stress_score += 0.20
            signals.append("repeated_backtracking")
        elif backtracks == 1:
            stress_score += 0.10
            signals.append("mild_backtracking")

        # Signal 4: Perplexity extremes — NATIVE SPEAKERS ONLY
        if native_speaker and perplexity > 250:
            stress_score += 0.15
            signals.append("chaotic_answer_structure")

        # Signal 5: Disfluency spike — NATIVE SPEAKERS ONLY
        if native_speaker and disfluency_rate > 0.20:
            stress_score += 0.15
            signals.append("high_disfluency")

        stress_score = min(1.0, round(stress_score, 3))
        knowledge_gap_detected = (
            "epistemic_uncertainty" in signals or
            "very_brief_answers" in signals or
            "repeated_backtracking" in signals
        )

        _empathy_pool = [
            "Take your time — there's no rush at all.",
            "Feel free to think out loud. Good thinking matters more than a fast answer.",
            "That's a nuanced area. Take a moment if you need — I'm happy to wait.",
        ]
        empathy_prompt = _empathy_pool[len(signals) % len(_empathy_pool)]

        return {
            "stress_score": stress_score,
            "stress_signals": signals,
            "empathy_prompt": empathy_prompt if stress_score > 0.6 else None,
            "knowledge_gap_detected": knowledge_gap_detected
        }

    except Exception as e:
        logger.error(f"Error in detect_speech_stress: {str(e)}")
        return {
            "stress_score": 0.0,
            "stress_signals": [],
            "empathy_prompt": None,
            "knowledge_gap_detected": False
        }

def estimate_head_eye_signals(face_landmarks: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Calculate head pose and gaze metrics from facial landmarks.
    Integrates cleanly with MediaPipe / OpenCV FaceMesh when video frames are provided.
    Returns:
        dict with 'head_yaw', 'head_pitch', 'gaze_off_center_ratio'
    """
    if not face_landmarks:
        # Default centered baseline (neutral position)
        return {"head_yaw": 0.0, "head_pitch": 0.0, "gaze_off_center_ratio": 0.0}

    yaw = float(face_landmarks.get("yaw", 0.0))
    pitch = float(face_landmarks.get("pitch", 0.0))
    gaze_ratio = float(face_landmarks.get("gaze_ratio", 0.0))
    return {
        "head_yaw": yaw,
        "head_pitch": pitch,
        "gaze_off_center_ratio": min(1.0, max(0.0, gaze_ratio))
    }

# ---------------------------------------------------------------------------
# Research-Grade Computer Vision Anti-Cheating & Proctoring Suite
#
# Primary Engine: MediaPipe 478-Landmark FaceMesh with Iris Refinement
# References:
#   1. 3D Head Pose: Perspective-n-Point (cv2.solvePnP) with Anthropometric Face Model
#      (Li et al., 2021; Ruiz et al., CVPR 2018)
#   2. Gaze Estimation: Iris-to-Canthus Horizontal & Vertical Ratios
#      (MPIIGaze: Zhang et al., TPAMI 2019; GazeCapture: Krafka et al., CVPR 2016)
#   3. Eye Aspect Ratio (EAR) & Reading Saccades:
#      (Soukupová & Čech, 2016 "Real-Time Eye Blink Detection")
#   4. Lip-Sync & Proxy Speaker: Landmark Mouth Aspect Ratio (MAR) Dynamics
#      (SyncNet: Chung & Zisserman, ACCV 2016)
# Fallback Engine: OpenCV MultiScale Haar Cascade & Frame-Delta differencing
# ---------------------------------------------------------------------------

# Canonical 3D anthropometric facial landmark model (points in mm, origin at nose tip)
# Axes: +X right, +Y down, +Z forward (matching OpenCV camera convention)
FACE_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (Landmark 1)
    (0.0, 110.0, -35.0),         # Chin bottom (Landmark 152)
    (-75.0, -50.0, -60.0),       # Left eye outer corner (Landmark 263 in image coords)
    (75.0, -50.0, -60.0),        # Right eye outer corner (Landmark 33)
    (-40.0, 45.0, -40.0),        # Left mouth corner (Landmark 291)
    (40.0, 45.0, -40.0)          # Right mouth corner (Landmark 61)
], dtype=np.float64)

_face_mesh_detector = None

def get_face_mesh_detector():
    """
    Returns a persistent singleton instance of MediaPipe FaceMesh
    with iris landmark refinement enabled.
    """
    global _face_mesh_detector
    if _face_mesh_detector is None:
        try:
            import mediapipe as mp
            _face_mesh_detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=3,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("Initialized MediaPipe FaceMesh with Iris refinement for research-grade proctoring")
        except Exception as e:
            logger.warning(f"MediaPipe FaceMesh initialization failed: {e}. Falling back to OpenCV.")
            _face_mesh_detector = False
    return _face_mesh_detector if _face_mesh_detector is not False else None


def rotation_matrix_to_euler_angles(rmat: np.ndarray) -> Tuple[float, float, float]:
    """
    Decomposes 3D rotation matrix into Euler angles (Pitch, Yaw, Roll) in degrees.
    Pitch: Up/Down (- is looking up, + is looking down)
    Yaw: Left/Right (- is looking left, + is looking right)
    Roll: Lateral head tilt
    """
    sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2]) * (180.0 / np.pi)
        yaw = np.arctan2(-rmat[2, 0], sy) * (180.0 / np.pi)
        roll = np.arctan2(rmat[1, 0], rmat[0, 0]) * (180.0 / np.pi)
    else:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1]) * (180.0 / np.pi)
        yaw = np.arctan2(-rmat[2, 0], sy) * (180.0 / np.pi)
        roll = 0.0
    return float(pitch), float(yaw), float(roll)


def estimate_head_pose_pnp(landmarks, frame_w: int, frame_h: int) -> Tuple[float, float, float]:
    """
    Computes authentic 3D head pose Euler angles (Pitch, Yaw, Roll) via Perspective-n-Point.
    Reference: Li et al. (2021) / Ruiz et al. (CVPR 2018).
    """
    import cv2
    img_pts = np.array([
        (landmarks[1].x * frame_w, landmarks[1].y * frame_h),      # Nose tip
        (landmarks[152].x * frame_w, landmarks[152].y * frame_h),  # Chin
        (landmarks[263].x * frame_w, landmarks[263].y * frame_h),  # Left eye outer
        (landmarks[33].x * frame_w, landmarks[33].y * frame_h),    # Right eye outer
        (landmarks[291].x * frame_w, landmarks[291].y * frame_h),  # Left mouth
        (landmarks[61].x * frame_w, landmarks[61].y * frame_h)     # Right mouth
    ], dtype=np.float64)

    focal_length = float(frame_w)
    cam_matrix = np.array([
        [focal_length, 0, frame_w / 2.0],
        [0, focal_length, frame_h / 2.0],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL_POINTS, img_pts, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = rotation_matrix_to_euler_angles(rmat)
    return round(pitch, 1), round(yaw, 1), round(roll, 1)


def estimate_iris_gaze(landmarks, frame_w: int, frame_h: int) -> Dict[str, Any]:
    """
    Calculates Horizontal Gaze Ratio (HGR) and Vertical Gaze Ratio (VGR)
    using iris landmarks (468, 473) relative to canthi and eyelids.
    Reference: MPIIGaze (Zhang et al. TPAMI 2019).
    """
    pts = landmarks
    # Left eye: iris 468, inner canthus 362, outer canthus 263, top eyelid 386, bottom eyelid 374
    l_iris_x = pts[468].x * frame_w
    l_min_x = min(pts[362].x, pts[263].x) * frame_w
    l_max_x = max(pts[362].x, pts[263].x) * frame_w
    h_ratio_left = (l_iris_x - l_min_x) / (l_max_x - l_min_x + 1e-6)

    # Right eye: iris 473, inner canthus 133, outer canthus 33, top eyelid 159, bottom eyelid 145
    r_iris_x = pts[473].x * frame_w
    r_min_x = min(pts[133].x, pts[33].x) * frame_w
    r_max_x = max(pts[133].x, pts[33].x) * frame_w
    h_ratio_right = (r_iris_x - r_min_x) / (r_max_x - r_min_x + 1e-6)

    h_ratio = float(np.clip((h_ratio_left + h_ratio_right) / 2.0, 0.0, 1.0))

    # Vertical gaze ratio (iris position within eyelid opening)
    l_iris_y = pts[468].y * frame_h
    l_top_y = pts[386].y * frame_h
    l_bot_y = pts[374].y * frame_h
    eye_h = abs(l_bot_y - l_top_y)
    v_ratio = float(np.clip((l_iris_y - l_top_y) / (eye_h + 1e-6), 0.0, 1.0))

    # Eye Aspect Ratio (EAR) - Soukupová & Čech (2016)
    eye_w = abs(l_max_x - l_min_x)
    ear_left = float(eye_h / (2.0 * eye_w + 1e-6))

    # Gaze classification
    direction = "CENTER"
    is_looking_away = False
    if h_ratio < 0.36:
        direction = "LEFT_SCREEN"
        is_looking_away = True
    elif h_ratio > 0.64:
        direction = "RIGHT_SCREEN"
        is_looking_away = True
    elif v_ratio > 0.72:
        direction = "DOWN_PHONE_DESK"
        is_looking_away = True
    elif v_ratio < 0.28:
        direction = "UP_CEILING"
        is_looking_away = True

    return {
        "horizontal_gaze_ratio": round(h_ratio, 2),
        "vertical_gaze_ratio": round(v_ratio, 2),
        "ear": round(ear_left, 3),
        "gaze_direction": direction,
        "is_looking_away": is_looking_away
    }


def calculate_mouth_aspect_ratio(landmarks, frame_w: int, frame_h: int) -> float:
    """
    Calculates Mouth Aspect Ratio (MAR) for speech verification and proxy speaker detection.
    Reference: SyncNet (Chung & Zisserman, ACCV 2016).
    """
    pts = landmarks
    lip_top = np.array([pts[13].x * frame_w, pts[13].y * frame_h])
    lip_bot = np.array([pts[14].x * frame_w, pts[14].y * frame_h])
    m_left = np.array([pts[291].x * frame_w, pts[291].y * frame_h])
    m_right = np.array([pts[61].x * frame_w, pts[61].y * frame_h])
    mar = float(np.linalg.norm(lip_top - lip_bot) / (np.linalg.norm(m_left - m_right) + 1e-6))
    return round(mar, 3)


def analyze_video_frame_proctoring(frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Research-grade computer vision proctoring analysis of a candidate's camera frame.
    Integrates 3D Head Pose (cv2.solvePnP), Iris Gaze Tracking (MPIIGaze), and multi-presence detection.
    
    Returns:
        {
            "face_detected": bool,
            "multiple_faces": bool,
            "head_pose": {"yaw": float, "pitch": float, "roll": float},
            "gaze_off_screen": bool,
            "warning": Optional[str],
            "gaze_details": Dict[str, Any],
            "method": str
        }
    """
    if frame is None:
        return {
            "face_detected": False,
            "multiple_faces": False,
            "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "gaze_off_screen": False,
            "warning": "No camera frame provided.",
            "gaze_details": {},
            "method": "none"
        }

    frame_h, frame_w = frame.shape[:2]

    # 1. Attempt High-Precision MediaPipe 478-Landmark FaceMesh
    detector = get_face_mesh_detector()
    if detector is not None:
        try:
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
            results = detector.process(rgb)

            if results.multi_face_landmarks:
                num_faces = len(results.multi_face_landmarks)
                if num_faces > 1:
                    return {
                        "face_detected": True,
                        "multiple_faces": True,
                        "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                        "gaze_off_screen": False,
                        "warning": "Multiple individuals detected in camera view.",
                        "gaze_details": {},
                        "method": "mediapipe_mesh"
                    }

                face_lms = results.multi_face_landmarks[0].landmark
                pitch, yaw, roll = estimate_head_pose_pnp(face_lms, frame_w, frame_h)
                gaze_data = estimate_iris_gaze(face_lms, frame_w, frame_h)

                # Off-screen deflection: either 3D head turned > 25° yaw / 20° pitch OR iris gaze deflected
                head_turned = abs(yaw) > 25.0 or abs(pitch) > 20.0
                gaze_deflected = gaze_data["is_looking_away"]
                gaze_off_screen = head_turned or gaze_deflected

                warning = None
                if gaze_off_screen:
                    if head_turned:
                        warning = f"Candidate turned head ({'left' if yaw < 0 else 'right'} yaw={abs(yaw):.1f}°)."
                    else:
                        warning = f"Candidate gaze deflected toward {gaze_data['gaze_direction']}."

                return {
                    "face_detected": True,
                    "multiple_faces": False,
                    "head_pose": {"yaw": yaw, "pitch": pitch, "roll": roll},
                    "gaze_off_screen": gaze_off_screen,
                    "warning": warning,
                    "gaze_details": gaze_data,
                    "method": "mediapipe_3d_pnp"
                }
        except Exception as mp_err:
            logger.debug(f"MediaPipe FaceMesh processing exception: {mp_err}")

    # 2. Resilient OpenCV Haar Cascade Fallback
    try:
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0:
            return {
                "face_detected": False,
                "multiple_faces": False,
                "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                "gaze_off_screen": True,
                "warning": "No face detected in camera frame. Please face the screen directly.",
                "gaze_details": {},
                "method": "haar_fallback"
            }

        if len(faces) > 1:
            return {
                "face_detected": True,
                "multiple_faces": True,
                "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                "gaze_off_screen": False,
                "warning": "Multiple individuals detected in camera view.",
                "gaze_details": {},
                "method": "haar_fallback"
            }

        x, y, w, h = faces[0]
        face_center_x = x + w / 2.0
        face_center_y = y + h / 2.0
        yaw = float(((face_center_x - (frame_w / 2.0)) / (frame_w / 2.0)) * 45.0)
        pitch = float(((face_center_y - (frame_h / 2.0)) / (frame_h / 2.0)) * 30.0)
        gaze_off_screen = abs(yaw) > 25.0 or abs(pitch) > 20.0
        warning = "Candidate looking away from screen." if gaze_off_screen else None

        return {
            "face_detected": True,
            "multiple_faces": False,
            "head_pose": {"yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": 0.0},
            "gaze_off_screen": gaze_off_screen,
            "warning": warning,
            "gaze_details": {},
            "method": "haar_fallback"
        }
    except Exception as e:
        logger.warning(f"Vision proctoring error: {e}")
        return {
            "face_detected": True,
            "multiple_faces": False,
            "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "gaze_off_screen": False,
            "warning": None,
            "gaze_details": {},
            "method": "error_default"
        }


def calculate_eye_contact_score(frame: Optional[np.ndarray], target_box: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
    """
    Measures candidate's eye-contact alignment with on-screen target (AI Eyes / Camera).
    Combines 3D Head Pose vector with Iris Gaze Ratio.
    Detects nervousness vs. reading from a second monitor or phone.
    
    Returns:
        {
            "eye_contact_score": float (0.0 to 1.0),
            "nervousness_detected": bool,
            "coaching_prompt": Optional[str],
            "details": Dict[str, Any]
        }
    """
    if frame is None:
        return {
            "eye_contact_score": 1.0,
            "nervousness_detected": False,
            "coaching_prompt": None,
            "details": {}
        }

    proctor = analyze_video_frame_proctoring(frame)
    if not proctor["face_detected"]:
        return {
            "eye_contact_score": 0.0,
            "nervousness_detected": False,
            "coaching_prompt": "Please look directly into the camera so we can maintain eye contact.",
            "details": {}
        }

    yaw = abs(proctor["head_pose"]["yaw"])
    pitch = abs(proctor["head_pose"]["pitch"])
    head_penalty = (min(yaw, 45.0) / 45.0) * 0.55 + (min(pitch, 30.0) / 30.0) * 0.25

    gaze_details = proctor.get("gaze_details", {})
    gaze_penalty = 0.0
    if "horizontal_gaze_ratio" in gaze_details:
        h_ratio = gaze_details["horizontal_gaze_ratio"]
        v_ratio = gaze_details["vertical_gaze_ratio"]
        gaze_penalty = abs(h_ratio - 0.50) * 0.70 + max(0.0, v_ratio - 0.65) * 0.40

    total_deflection = min(1.0, head_penalty + gaze_penalty)
    eye_contact_score = max(0.0, min(1.0, 1.0 - total_deflection))

    nervousness_detected = eye_contact_score < 0.55
    coaching_prompt = None
    if nervousness_detected:
        coaching_prompt = "I notice you might be a bit nervous—take a deep breath! Look right here at me, take your time, and tell me in your own words."

    return {
        "eye_contact_score": round(eye_contact_score, 2),
        "nervousness_detected": nervousness_detected,
        "coaching_prompt": coaching_prompt,
        "details": {
            "head_penalty": round(head_penalty, 2),
            "gaze_penalty": round(gaze_penalty, 2),
            "gaze_direction": gaze_details.get("gaze_direction", "CENTER")
        }
    }


def verify_audio_visual_speech_sync(
    prev_frame: Optional[np.ndarray],
    curr_frame: Optional[np.ndarray],
    audio_active: bool
) -> Dict[str, Any]:
    """
    Verifies that the candidate's mouth is actively moving in synchrony with speech audio.
    Combines landmark-based Mouth Aspect Ratio (MAR) with inter-frame optical differencing.
    Reference: SyncNet (Chung & Zisserman, ACCV 2016).
    
    Returns:
        {
            "is_speaking_visually": bool,
            "mouth_movement_score": float,
            "mouth_aspect_ratio": float,
            "proxy_speaker_suspected": bool,
            "warning": Optional[str]
        }
    """
    if curr_frame is None:
        return {
            "is_speaking_visually": audio_active,
            "mouth_movement_score": 1.0 if audio_active else 0.0,
            "mouth_aspect_ratio": 0.20 if audio_active else 0.0,
            "proxy_speaker_suspected": False,
            "warning": None
        }

    frame_h, frame_w = curr_frame.shape[:2]
    detector = get_face_mesh_detector()

    # 1. Primary: Landmark Mouth Aspect Ratio (MAR) via MediaPipe
    if detector is not None:
        try:
            import cv2
            rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB) if len(curr_frame.shape) == 3 else curr_frame
            results = detector.process(rgb)

            if results.multi_face_landmarks:
                pts = results.multi_face_landmarks[0].landmark
                mar = calculate_mouth_aspect_ratio(pts, frame_w, frame_h)

                # Previous frame comparison for dynamic mouth motion
                prev_mar = None
                if prev_frame is not None:
                    prev_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB) if len(prev_frame.shape) == 3 else prev_frame
                    prev_res = detector.process(prev_rgb)
                    if prev_res.multi_face_landmarks:
                        prev_mar = calculate_mouth_aspect_ratio(prev_res.multi_face_landmarks[0].landmark, frame_w, frame_h)

                delta_mar = abs(mar - prev_mar) if prev_mar is not None else 0.0
                movement_score = float(mar * 10.0 + delta_mar * 20.0)

                # Visual speaking: mouth is open (MAR > 0.12) OR dynamically moving (delta_mar > 0.03)
                is_speaking_visually = mar > 0.12 or delta_mar > 0.03

                # Proxy speaker attack: speech audio is clearly active, but candidate's mouth is clamped shut
                proxy_speaker_suspected = audio_active and (mar < 0.07 and delta_mar < 0.02)
                warning = None
                if proxy_speaker_suspected:
                    warning = "I can hear audio, but I am not seeing your lips move. Please ensure you are answering directly."

                return {
                    "is_speaking_visually": is_speaking_visually,
                    "mouth_movement_score": round(movement_score, 2),
                    "mouth_aspect_ratio": mar,
                    "proxy_speaker_suspected": proxy_speaker_suspected,
                    "warning": warning
                }
        except Exception as sync_err:
            logger.debug(f"Landmark MAR analysis error: {sync_err}")

    # 2. Resilient OpenCV Optical Differencing Fallback
    try:
        import cv2
        gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0 or prev_frame is None:
            return {
                "is_speaking_visually": audio_active,
                "mouth_movement_score": 0.5 if audio_active else 0.0,
                "mouth_aspect_ratio": 0.15 if audio_active else 0.0,
                "proxy_speaker_suspected": False,
                "warning": None
            }

        x, y, w, h = faces[0]
        mouth_y = int(y + 0.65 * h)
        mouth_h = int(0.35 * h)
        mouth_curr = gray[mouth_y:mouth_y+mouth_h, x:x+w]

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
        mouth_prev = prev_gray[mouth_y:mouth_y+mouth_h, x:x+w]

        if mouth_curr.shape != mouth_prev.shape:
            mouth_prev = cv2.resize(mouth_prev, (mouth_curr.shape[1], mouth_curr.shape[0]))

        diff = cv2.absdiff(mouth_curr, mouth_prev)
        movement_score = float(np.mean(diff))

        is_speaking_visually = movement_score > 2.5
        proxy_speaker_suspected = audio_active and not is_speaking_visually and movement_score < 1.0
        warning = "I'm picking up speech, but I don't see you speaking." if proxy_speaker_suspected else None

        return {
            "is_speaking_visually": is_speaking_visually,
            "mouth_movement_score": round(movement_score, 2),
            "mouth_aspect_ratio": round(min(0.5, movement_score / 10.0), 3),
            "proxy_speaker_suspected": proxy_speaker_suspected,
            "warning": warning
        }
    except Exception as e:
        logger.warning(f"Lip-sync analysis error: {e}")
        return {
            "is_speaking_visually": audio_active,
            "mouth_movement_score": 0.5,
            "mouth_aspect_ratio": 0.15,
            "proxy_speaker_suspected": False,
            "warning": None
        }

def extract_voice_acoustic_fingerprint(audio_pcm: Optional[np.ndarray], sample_rate: int = 16000) -> Dict[str, float]:
    """
    Extracts acoustic biometric signature (fundamental pitch and spectral centroid)
    to detect if a different person starts speaking mid-interview.
    """
    if audio_pcm is None or len(audio_pcm) == 0:
        return {"pitch_mean": 0.0, "spectral_energy": 0.0, "zero_crossing_rate": 0.0}
    
    try:
        audio = audio_pcm.astype(np.float32)
        # Zero-Crossing Rate (correlates with voice timbre & fricatives)
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0)
        
        # Root Mean Square Energy
        rms = float(np.sqrt(np.mean(audio**2)))
        
        # Approximate fundamental pitch using Auto-Correlation
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        d = np.diff(autocorr)
        start = np.where(d > 0)[0]
        if len(start) > 0:
            peak = np.argmax(autocorr[start[0]:]) + start[0]
            pitch = float(sample_rate / peak) if peak > 0 else 0.0
        else:
            pitch = 0.0
            
        return {
            "pitch_mean": round(pitch, 1),
            "spectral_energy": round(rms, 3),
            "zero_crossing_rate": round(zcr, 3)
        }
    except Exception as e:
        logger.warning(f"Voiceprint extraction error: {e}")
        return {"pitch_mean": 0.0, "spectral_energy": 0.0, "zero_crossing_rate": 0.0}

def detect_voice_drift(baseline_print: Dict[str, float], current_print: Dict[str, float]) -> Dict[str, Any]:
    """
    Compares the current turn's acoustic biometric against candidate baseline.
    Flags sudden voice timbre / pitch shifts (e.g. proxy speaker answering while candidate lip-syncs).
    """
    if not baseline_print or not current_print:
        return {"voice_match": True, "voice_drift_score": 0.0, "warning": None}
        
    p_base = baseline_print.get("pitch_mean", 0.0)
    p_curr = current_print.get("pitch_mean", 0.0)
    z_base = baseline_print.get("zero_crossing_rate", 0.0)
    z_curr = current_print.get("zero_crossing_rate", 0.0)
    
    if p_base <= 0.0 or p_curr <= 0.0:
        return {"voice_match": True, "voice_drift_score": 0.0, "warning": None}
        
    pitch_diff_ratio = abs(p_curr - p_base) / max(p_base, 1.0)
    zcr_diff_ratio = abs(z_curr - z_base) / max(z_base, 1e-4)
    
    drift_score = (pitch_diff_ratio * 0.7) + (min(1.0, zcr_diff_ratio) * 0.3)
    # Significant voice shift (e.g., pitch shifts > 45% or timbre changes drastically)
    voice_changed = drift_score > 0.45
    
    warning = None
    if voice_changed:
        warning = "Acoustic signature mismatch detected: The speaking voice significantly differs from your initial voiceprint."
        
    return {
        "voice_match": not voice_changed,
        "voice_drift_score": round(drift_score, 3),
        "warning": warning
    }

def detect_ai_generated_answer(
    transcript: str,
    perplexity: float,
    disfluency_rate: float,
    consistency: float,
    prev_history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Detects whether a candidate is reciting an AI-generated answer (e.g., from ChatGPT,
    Copilot, or a dual-monitor live prompter) rather than speaking organically.

    Key Signals:
    1. Critically low token perplexity (< 32.0; typical LLM output is 15-28, spoken English is 60-150+)
    2. Zero disfluency in a sustained answer (> 40 words with 0 fillers)
    3. Hallmark written LLM discourse markers ("in summary", "furthermore", "it is worth noting")
    4. Sudden conversational leap vs earlier turns
    """
    if not transcript or not transcript.strip():
        return {
            "ai_script_detected": False,
            "ai_probability": 0.0,
            "signals": [],
            "nudge_prompt": None,
            "incident_reason": None
        }

    words = transcript.strip().split()
    word_count = len(words)
    signals = []
    confidence_points = 0.0

    # 1. Critically Low Perplexity Check (Strongest LLM token predictor)
    if perplexity > 0:
        if perplexity < 22.0 and word_count >= 25:
            confidence_points += 0.45
            signals.append("critically_low_perplexity")
        elif perplexity < 32.0 and word_count >= 30:
            confidence_points += 0.30
            signals.append("low_perplexity")

    # 2. Unnatural Oral Fluency (Reading written text aloud)
    if word_count >= 45 and disfluency_rate == 0.0:
        confidence_points += 0.30
        signals.append("unnaturally_fluent_script_reading")
    elif word_count >= 70 and disfluency_rate < 0.015:
        confidence_points += 0.20
        signals.append("minimal_spontaneous_disfluency")

    # 3. LLM Hallmark Discourse Markers (common in ChatGPT/Claude explanations)
    llm_markers = [
        r"\bin conclusion\b",
        r"\bin summary\b",
        r"\bfurthermore\b",
        r"\bmoreover\b",
        r"\bit is worth noting that\b",
        r"\bit is important to consider\b",
        r"\bdelving into\b",
        r"\ba testament to\b",
        r"\bfirstly,?\s+.{10,60}\bsecondly\b",
        r"\bcomprehensive approach\b"
    ]
    marker_hits = sum(1 for pat in llm_markers if re.search(pat, transcript, re.IGNORECASE))
    if marker_hits >= 2:
        confidence_points += 0.35
        signals.append("written_llm_discourse_markers")
    elif marker_hits == 1 and word_count >= 35:
        confidence_points += 0.15
        signals.append("formal_literary_syntax")

    # 4. Sudden Turn-over-Turn Quality Leap
    if prev_history and len(prev_history) >= 1:
        prev_analysis = prev_history[-1].get("analysis", {})
        prev_perp = prev_analysis.get("perplexity", 100.0)
        prev_disflu = prev_analysis.get("disfluency_rate", 0.1)

        # Candidate was stumbling or high-perplexity previously, suddenly textbook perfect
        if prev_disflu > 0.10 and disfluency_rate == 0.0 and (prev_perp - perplexity) > 50.0:
            confidence_points += 0.25
            signals.append("sudden_turn_over_turn_fluency_leap")

    ai_prob = min(1.0, round(confidence_points, 2))
    detected = ai_prob >= 0.65

    nudge = None
    incident = None
    if detected:
        nudge = (
            "That was a very structured explanation! To better understand your direct experience, "
            "could you tell me in your own words about a specific challenge you personally faced while building it?"
        )
        incident = (
            f"AI script-reading pattern detected (p={ai_prob}). "
            f"Signals: {', '.join(signals)}. Perplexity: {round(perplexity, 1)}, Word Count: {word_count}."
        )

    return {
        "ai_script_detected": detected,
        "ai_probability": ai_prob,
        "signals": signals,
        "nudge_prompt": nudge,
        "incident_reason": incident
    }

def detect_conduct_violation(transcript: str) -> Dict[str, Any]:
    """
    Detects profanity, hostility, insults, or abusive conduct toward Sasha.
    Returns:
        {
            "is_violation": bool,
            "severity": "CRITICAL" | "HIGH" | "NONE",
            "categories": List[str],
            "flagged_snippet": Optional[str]
        }
    """
    if not transcript or not transcript.strip():
        return {
            "is_violation": False,
            "severity": "NONE",
            "categories": [],
            "flagged_snippet": None
        }

    abusive_patterns = [
        (r"\b(shut\s*up|fuck\s*off|fuck\s*you|fuck\s*this|fucking)\b", "profanity"),
        (r"\b(you('re| are)\s+(stupid|dumb|useless|an idiot|a joke|retarded|a piece of shit))\b", "direct_insult"),
        (r"\b(bitch|bastard|asshole|bullshit|cunt|dickhead|moron)\b", "derogatory_abuse"),
        (r"\b(hate\s+you|waste\s+of\s+my\s+time|go\s+to\s+hell)\b", "hostility"),
    ]

    matched_categories = []
    snippet = None
    for pattern, cat in abusive_patterns:
        match = re.search(pattern, transcript, re.IGNORECASE)
        if match:
            matched_categories.append(cat)
            if not snippet:
                snippet = match.group(0)

    is_violation = len(matched_categories) > 0
    severity = "CRITICAL" if ("derogatory_abuse" in matched_categories or "direct_insult" in matched_categories) else ("HIGH" if is_violation else "NONE")

    return {
        "is_violation": is_violation,
        "severity": severity,
        "categories": list(set(matched_categories)),
        "flagged_snippet": snippet
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python analyzer.py <resume_text> <answer_text>")
        sys.exit(1)
    resume_text = sys.argv[1]
    answer_text = sys.argv[2]
    perp = get_perplexity(answer_text)
    disflu = get_disfluency_rate(answer_text)
    consist = get_consistency(resume_text, answer_text)
    import json
    print(json.dumps({
        "perplexity": round(perp, 3),
        "disfluency_rate": round(disflu, 3),
        "consistency": round(consist, 3)
    }, indent=2))