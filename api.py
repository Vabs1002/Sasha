from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import sys
import uuid
import secrets
import shutil
import tempfile
import logging
import base64
import asyncio
import json
import re
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
import yaml
import numpy as np
import uvicorn

# Resolve application assets and generated reports independently of the shell's
# current working directory (for example, when launched by a service manager).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import Sasha core modules
from resume_parser import parse_resume
from signal_detector import detect_role
from analyzer import (
    get_perplexity,
    get_disfluency_rate,
    get_consistency,
    detect_background_audio_anomaly,
    detect_speech_stress,
    analyze_video_frame_proctoring,
    calculate_eye_contact_score,
    verify_audio_visual_speech_sync,
    detect_ai_generated_answer,
    detect_explicit_answer_request,
    detect_conduct_violation,
)
from interviewer_agent import (
    get_interviewer_decision,
    AdaptiveQuestionSelector,
    load_job_description,
    load_competency_bank,
    BEHAVIORAL_QUESTIONS
)
from report_generator import generate_report


# ---------------------------------------------------------------------------
# Proctoring State — per session, tracks consistent suspicious signals
# ---------------------------------------------------------------------------
class ProctorState:
    """
    Tracks consecutive suspicious frames per signal type across a session.
    No hardcoded time threshold — fires when N consecutive frames confirm the same anomaly.
    This is purely frame-consistency based: a single bad frame is noise, repeated frames
    across multiple samples is a real behavioral pattern.

    Thresholds chosen to be strict enough to avoid false positives (nervousness ≠ cheating)
    but sensitive enough to catch real deflection patterns.
    """
    # How many consecutive positive detections before we consider it "consistent"
    GAZE_AWAY_THRESHOLD = 4        # 4 consecutive frames looking away
    MULTIPLE_FACES_THRESHOLD = 3   # 3 consecutive frames with multiple faces (harder to fluke)
    PROXY_SPEAKER_THRESHOLD = 3    # 3 consecutive frames: audio active but lips not moving
    NO_FACE_THRESHOLD = 5          # 5 frames with no face at all

    # Positive Sasha nudges — warm, never accusatory, never repetitive
    NUDGES = {
        "gaze_away": [
            "Just a reminder — try to keep your focus on the screen. Take your time with your answer.",
            "No worries at all, just checking in — looking at the screen helps me follow along with you.",
            "Whenever you are ready, just speak toward the camera and I am right here with you.",
        ],
        "no_face": [
            "I seem to have lost you for a moment — please make sure your camera is pointing toward you.",
            "Could you adjust your camera slightly? I want to make sure I can see you clearly.",
        ],
        "multiple_faces": [
            "It looks like there might be someone else in the frame. For the integrity of the session, please make sure you are in a private space.",
            "I noticed another person in the camera view. Please ensure you are conducting the interview in a private, quiet area.",
        ],
        "proxy_speaker": [
            "I can hear audio, but I am not quite seeing you speak. Please make sure you are answering directly — I am here to listen to you.",
            "Just checking — please ensure your responses are coming from you directly. I want to hear your perspective.",
        ],
    }

    def __init__(self):
        # Consecutive frame counters
        self._gaze_counter: int = 0
        self._no_face_counter: int = 0
        self._multi_face_counter: int = 0
        self._proxy_counter: int = 0

        # Previous frame reference for lip-sync delta
        self._prev_frame: Optional[np.ndarray] = None

        # How many times each nudge type has been delivered (to rotate phrasing, avoid spam)
        self._nudge_counts: Dict[str, int] = {
            "gaze_away": 0,
            "no_face": 0,
            "multiple_faces": 0,
            "proxy_speaker": 0,
        }

        # Total incidents per type, written to HR report at session end
        self.integrity_log: List[Dict[str, Any]] = []

    def _get_nudge(self, signal_type: str) -> str:
        """Rotate through nudge phrases so Sasha never says the same thing twice."""
        phrases = self.NUDGES.get(signal_type, ["Please ensure you are focused on the interview."])
        count = self._nudge_counts.get(signal_type, 0)
        phrase = phrases[count % len(phrases)]
        self._nudge_counts[signal_type] = count + 1
        return phrase

    def _log_incident(self, signal_type: str, detail: str):
        """Append to HR integrity log with timestamp."""
        self.integrity_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": signal_type,
            "detail": detail,
        })

    def process_frame(
        self,
        frame: np.ndarray,
        audio_active: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze one webcam frame. Returns a nudge dict if a consistent suspicious
        pattern has been confirmed, else None.

        Return shape when triggered:
            {
                "type": "sasha_nudge",
                "signal": str,            # internal signal key
                "text": str,              # what Sasha says to candidate
                "logged_to_hr": True      # HR always gets told
            }
        """
        result = analyze_video_frame_proctoring(frame)
        eye_result = calculate_eye_contact_score(frame)
        sync_result = verify_audio_visual_speech_sync(self._prev_frame, frame, audio_active)
        self._prev_frame = frame

        nudge = None

        # --- Signal 1: No face detected ---
        if not result["face_detected"]:
            self._gaze_counter = 0
            self._multi_face_counter = 0
            self._proxy_counter = 0
            self._no_face_counter += 1
            if self._no_face_counter >= self.NO_FACE_THRESHOLD:
                self._log_incident("no_face_detected", "Candidate not visible in camera for extended period.")
                nudge = {
                    "type": "sasha_nudge",
                    "signal": "no_face",
                    "text": self._get_nudge("no_face"),
                    "logged_to_hr": True,
                }
                self._no_face_counter = 0  # reset to avoid spam
            return nudge

        self._no_face_counter = 0

        # --- Signal 2: Multiple faces detected ---
        if result["multiple_faces"]:
            self._gaze_counter = 0
            self._proxy_counter = 0
            self._multi_face_counter += 1
            if self._multi_face_counter >= self.MULTIPLE_FACES_THRESHOLD:
                self._log_incident("multiple_faces", "Multiple individuals detected in camera frame.")
                nudge = {
                    "type": "sasha_nudge",
                    "signal": "multiple_faces",
                    "text": self._get_nudge("multiple_faces"),
                    "logged_to_hr": True,
                }
                self._multi_face_counter = 0
            return nudge

        self._multi_face_counter = 0

        # --- Signal 3: Gaze consistently off screen ---
        if result["gaze_off_screen"]:
            self._gaze_counter += 1
            if self._gaze_counter >= self.GAZE_AWAY_THRESHOLD:
                yaw = result["head_pose"]["yaw"]
                pitch = result["head_pose"]["pitch"]
                self._log_incident(
                    "gaze_deflection",
                    f"Candidate gaze off-screen. Head yaw={yaw}°, pitch={pitch}°. "
                    f"Eye contact score={eye_result['eye_contact_score']}."
                )
                nudge = {
                    "type": "sasha_nudge",
                    "signal": "gaze_away",
                    "text": self._get_nudge("gaze_away"),
                    "logged_to_hr": True,
                }
                self._gaze_counter = 0
        else:
            self._gaze_counter = 0

        if nudge:
            return nudge

        # --- Signal 4: Proxy speaker (audio active, lips not moving) ---
        if sync_result["proxy_speaker_suspected"]:
            self._proxy_counter += 1
            if self._proxy_counter >= self.PROXY_SPEAKER_THRESHOLD:
                self._log_incident(
                    "proxy_speaker_suspected",
                    f"Audio active but no lip movement detected. Mouth movement score={sync_result['mouth_movement_score']}."
                )
                nudge = {
                    "type": "sasha_nudge",
                    "signal": "proxy_speaker",
                    "text": self._get_nudge("proxy_speaker"),
                    "logged_to_hr": True,
                }
                self._proxy_counter = 0
        else:
            self._proxy_counter = 0

        return nudge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sasha_api")

app = FastAPI(
    title="Sasha AI Interviewer API",
    version="1.0.0",
    description="Experimental resume-aware interview prototype with browser speech recognition, WebSocket turn control, and human-review signals."
)

# Use an explicit origin allowlist. Credentialed wildcard CORS is unsafe and
# browsers reject it inconsistently; deployments must configure their frontend.
_default_cors_origins = "http://localhost:5173,http://127.0.0.1:5173"
CORS_ALLOW_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv("CORS_ALLOW_ORIGINS", _default_cors_origins).split(",")
    if origin.strip() and origin.strip() != "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-Memory Session Manager
# ---------------------------------------------------------------------------
class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create(
        self,
        resume_profile: Dict[str, Any],
        jd_text: str = "",
        jd_summary: str = "",
        integrity_monitoring_consent: bool = False,
        evidence_capture_consent: bool = False,
    ) -> str:
        # The session ID is also a bearer capability for private evidence URLs.
        session_id = secrets.token_urlsafe(32)
        difficulty_selector = AdaptiveQuestionSelector(resume_profile.get("level", "mid"))
        browser_interaction_events: List[Dict[str, Any]] = []
        resume_profile["integrity_monitoring_enabled"] = integrity_monitoring_consent
        resume_profile["browser_interaction_events"] = browser_interaction_events
        resume_profile["evidence_records"] = []

        cand_name = resume_profile.get("name")
        name_greeting = f", {cand_name}" if cand_name and cand_name.lower() != "candidate" else ""
        role_label = str(resume_profile.get("role", "Software Engineer")).replace("_", " ")
        experience_years = resume_profile.get("years_experience", 0)
        jd_context = " A job description was shared, and I will use it to guide the questions." if jd_summary.strip() else ""
        first_question = (
            f"Welcome{name_greeting}. Before we start, I read your resume as about "
            f"{experience_years} years of experience at the {resume_profile.get('level', 'mid')} level, "
            f"with a focus on {role_label}.{jd_context} Does that sound right, or would you like to correct anything? "
            "This is a supportive practice interview: take your time, think aloud, and ask me to repeat or clarify a question. "
            "For a fair practice session, answer in your own words; I can clarify a question but will not provide its solution. "
            "Any enabled monitoring is limited to imperfect review signals, not proof of misconduct."
        )
        first_competency_question = (
            "Thanks for confirming. To begin, could you walk me through the project from your resume "
            "that best shows your fit for this role, including what you personally built?"
        )

        self._sessions[session_id] = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "resume_profile": resume_profile,
            "jd_text": jd_text,
            "jd_summary": jd_summary,
            "integrity_monitoring_consent": integrity_monitoring_consent,
            "evidence_capture_consent": evidence_capture_consent,
            "browser_interaction_events": browser_interaction_events,
            "difficulty_selector": difficulty_selector,
            "history": [],
            "current_question": first_question,
            "first_competency_question": first_competency_question,
            "awaiting_profile_confirmation": True,
            "confirmed_profile_context": "",
            "behavioral_asked": 0,
            "behavioral_turn_counter": 0,
            "is_complete": False,
            "turn_state": "ready",
            "max_turns": 8,
            "report_paths": {},
            "proctor": ProctorState(),   # live proctoring engine for this session
            "conduct_strikes": 0,        # tracks profanity/abusive language offenses
        }
        return session_id

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)


sessions = SessionStore()

# Evidence and reports live outside the frontend's static directory and are
# never mounted as public file trees. Session URLs remain bearer capabilities;
# deployments still need authenticated recruiter access and encrypted storage.
PRIVATE_DATA_DIR = os.path.join(BASE_DIR, "data", "private")
EVIDENCE_DIR = os.path.join(PRIVATE_DATA_DIR, "interview_evidence")
REPORT_DIR = os.path.join(PRIVATE_DATA_DIR, "reports")
EVIDENCE_RETENTION_DAYS = 30
PRIVATE_REPORT_RETENTION_DAYS = 30
EVIDENCE_SIGNALS = {"gaze_away", "no_face", "multiple_faces", "proxy_speaker"}
EVIDENCE_INCIDENT_TYPES = {
    "gaze_away": "gaze_deflection",
    "no_face": "no_face_detected",
    "multiple_faces": "multiple_faces",
    "proxy_speaker": "proxy_speaker_suspected",
}


def _evidence_manifest_path(session_id: str) -> str:
    # Session IDs are generated with token_urlsafe, so reject path separators
    # before using one as a directory component.
    if not session_id or not re.fullmatch(r"[A-Za-z0-9_-]{32,64}", session_id):
        raise HTTPException(status_code=404, detail="Evidence not found.")
    return os.path.join(EVIDENCE_DIR, session_id, "manifest.json")


def _read_evidence_manifest(session_id: str) -> Optional[Dict[str, Any]]:
    path = _evidence_manifest_path(session_id)
    try:
        with open(path, "r", encoding="utf-8") as manifest_file:
            return json.load(manifest_file)
    except (OSError, ValueError):
        return None


def _write_evidence_manifest(session_id: str, records: List[Dict[str, Any]]) -> None:
    path = _evidence_manifest_path(session_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.name != "nt":
        os.chmod(os.path.dirname(path), 0o700)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as manifest_file:
        json.dump({"enabled": True, "records": records}, manifest_file)
    if os.name != "nt":
        os.chmod(temporary_path, 0o600)
    os.replace(temporary_path, path)


def _remove_expired_evidence() -> None:
    """Delete private evidence and reports after the disclosed retention window."""
    now = datetime.now(timezone.utc).timestamp()
    for directory, retention_days in (
        (EVIDENCE_DIR, EVIDENCE_RETENTION_DAYS),
        (REPORT_DIR, PRIVATE_REPORT_RETENTION_DAYS),
    ):
        cutoff = now - retention_days * 86400
        if not os.path.isdir(directory):
            continue
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                path = os.path.join(root, name)
                try:
                    if os.path.getmtime(path) <= cutoff:
                        os.remove(path)
                except OSError:
                    logger.warning("Could not remove expired private interview file")
            for name in dirs:
                path = os.path.join(root, name)
                try:
                    os.rmdir(path)
                except OSError:
                    pass


async def _evidence_cleanup_loop() -> None:
    while True:
        await asyncio.to_thread(_remove_expired_evidence)
        await asyncio.sleep(6 * 60 * 60)


@app.on_event("startup")
async def start_evidence_cleanup() -> None:
    app.state.evidence_cleanup_task = asyncio.create_task(_evidence_cleanup_loop())


@app.on_event("shutdown")
async def stop_evidence_cleanup() -> None:
    task = getattr(app.state, "evidence_cleanup_task", None)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class TurnRequest(BaseModel):
    answer: str = Field(..., min_length=1, description="Candidate's spoken or typed answer to the current question")

class StartInterviewResponse(BaseModel):
    session_id: str
    candidate_name: str
    role: str
    experience_level: str
    years_experience: float
    starting_difficulty: str
    first_question: str
    jd_loaded: bool
    integrity_monitoring_consent: bool = False
    evidence_capture_consent: bool = False
    profile_confirmation_required: bool = True

class TurnResponse(BaseModel):
    session_id: str
    turn_number: int
    is_complete: bool
    next_question: Optional[str] = None
    empathy_prompt: Optional[str] = None
    action: str
    reasoning: str
    difficulty_level: float
    difficulty_label: str
    stress_detected: bool
    stress_score: float
    knowledge_gap_detected: bool
    noise_detected: bool
    ai_script_detected: bool = False
    ai_signal_types: List[str] = Field(default_factory=list)
    assistance_request_detected: bool = False
    decision_source: str = "unknown"
    profile_confirmed: bool = False


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health_check():
    """Service health and readiness check."""
    return {
        "status": "healthy",
        "service": "Sasha AI Interviewer API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/tts", tags=["Voice"])
def get_tts_audio(text: str):
    """
    Synthesize speech audio using Rumik Silk TTS.
    Returns audio/wav bytes stream for instant playback.
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text required")
    try:
        import voice_services
        audio_bytes = voice_services.text_to_speech(text.strip())
        from fastapi.responses import Response
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        logger.warning(f"TTS synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/interview/start", response_model=StartInterviewResponse, tags=["Interview Lifecycle"])
async def start_interview(
    resume_file: UploadFile = File(..., description="Resume document (.pdf or .docx)"),
    jd_text: Optional[str] = Form(None, description="Job Description text"),
    jd_file: Optional[UploadFile] = File(None, description="Job Description file (.txt)"),
    integrity_monitoring_consent: bool = Form(False),
    evidence_capture_consent: bool = Form(False),
):
    """
    Initialize a new autonomous interview session.
    Parses resume, detects target role, binds Job Description for Hybrid RAG,
    and returns the personalized warm-up question.
    """
    # 1. Save uploaded resume to temp file
    filename = resume_file.filename or ""
    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in [".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail="Unsupported resume format. Please upload .pdf or .docx.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_resume:
        shutil.copyfileobj(resume_file.file, tmp_resume)
        tmp_resume_path = tmp_resume.name

    try:
        # Parse resume and turn malformed documents into a client error rather
        # than an opaque 500 response.
        try:
            resume_profile_raw = parse_resume(tmp_resume_path)
        except Exception as exc:
            logger.info("Rejected unreadable resume upload %r: %s", filename, exc)
            raise HTTPException(status_code=400, detail="Could not read the uploaded resume. Please upload a valid PDF or DOCX file.") from exc
    finally:
        if os.path.exists(tmp_resume_path):
            os.remove(tmp_resume_path)

    if not resume_profile_raw.get("raw_text", "").strip():
        raise HTTPException(status_code=400, detail="The uploaded resume contains no extractable text. Please upload a text-based PDF or DOCX file.")

    # 2. Detect role
    role_signals_path = os.path.join(BASE_DIR, "role_signals.yaml")
    signals = {}
    if os.path.exists(role_signals_path):
        with open(role_signals_path, "r", encoding="utf-8") as f:
            signals = yaml.safe_load(f) or {}

    role_result = detect_role(resume_profile_raw.get("raw_text", ""), signals)
    resume_profile = {
        **resume_profile_raw,
        "role": role_result.get("role", "SDE"),
        "role_confidence": role_result.get("confidence", 0.0)
    }

    # 3. Handle Job Description if provided
    final_jd = jd_text or ""
    if jd_file and not final_jd:
        content = await jd_file.read()
        final_jd = content.decode("utf-8", errors="ignore")

    jd_summary = ""
    if final_jd.strip():
        resume_profile["raw_text"] = load_job_description(final_jd, resume_profile_raw.get("raw_text", ""))
        resume_profile["jd_text"] = final_jd
        jd_summary = final_jd[:500].strip()

    # 4. Create in-memory session
    session_id = sessions.create(
        resume_profile,
        jd_text=final_jd,
        jd_summary=jd_summary,
        integrity_monitoring_consent=integrity_monitoring_consent,
        evidence_capture_consent=evidence_capture_consent and integrity_monitoring_consent,
    )
    session = sessions.get(session_id)

    return StartInterviewResponse(
        session_id=session_id,
        candidate_name=resume_profile.get("name") or "Candidate",
        role=resume_profile.get("role", "SDE"),
        experience_level=resume_profile.get("level", "mid"),
        years_experience=float(resume_profile.get("years_experience", 0)),
        starting_difficulty=session["difficulty_selector"].label(),
        first_question=session["current_question"],
        jd_loaded=bool(final_jd.strip()),
        integrity_monitoring_consent=integrity_monitoring_consent,
        evidence_capture_consent=session["evidence_capture_consent"],
    )


@app.post("/api/v1/interview/{session_id}/turn", response_model=TurnResponse, tags=["Interview Lifecycle"])
async def submit_turn(session_id: str, payload: TurnRequest):
    """
    Submit a candidate's answer for the current interview turn.
    Performs real-time linguistic, stress, and integrity analysis, updates
    adaptive question difficulty, and returns the next probe.
    """
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found or expired.")

    turn_owner = session.get("_ws_turn_task")
    if session.get("turn_state") == "processing" and turn_owner is not asyncio.current_task():
        raise HTTPException(status_code=409, detail="An interview turn is already processing.")

    if session["is_complete"]:
        raise HTTPException(status_code=400, detail="Interview has already concluded.")

    resume_profile = session["resume_profile"]
    selector: AdaptiveQuestionSelector = session["difficulty_selector"]
    history = session["history"]
    turn_idx = len(history)

    transcript = payload.answer.strip()
    raw_resume = resume_profile.get("raw_text", "")

    # The opening profile check is conversational setup, not a scored answer.
    # Save the candidate's correction so the LLM can use their stated context.
    if session.get("awaiting_profile_confirmation"):
        session["awaiting_profile_confirmation"] = False
        session["confirmed_profile_context"] = transcript[:500]
        session["current_question"] = session["first_competency_question"]
        return TurnResponse(
            session_id=session_id,
            turn_number=0,
            is_complete=False,
            next_question=session["current_question"],
            action="profile_confirmed",
            reasoning="Candidate confirmed or corrected the interview profile.",
            difficulty_level=selector.difficulty,
            difficulty_label=selector.label(),
            stress_detected=False,
            stress_score=0.0,
            knowledge_gap_detected=False,
            noise_detected=False,
            decision_source="setup",
            profile_confirmed=True,
        )

    # ---------------------------------------------------------------------------
    # Calibrated Conduct & De-Escalation Protocol
    # Step 1: First Offense -> Professional Boundary Setting (De-escalation)
    # Step 2: Repeated Abuse -> Immediate Clean Session Termination
    # Step 3: Evidentiary Logging -> Verbatim preserved for HR report
    # ---------------------------------------------------------------------------
    conduct_check = detect_conduct_violation(transcript)
    if conduct_check["is_violation"]:
        strikes = session.get("conduct_strikes", 0) + 1
        session["conduct_strikes"] = strikes
        current_q = session["current_question"] or "the problem we were discussing"

        if strikes == 1:
            boundary_prompt = (
                f"I am here to ensure a respectful and objective evaluation of your technical skills. "
                f"Let's please keep our discussion professional. Turning back to the problem: {current_q}"
            )
            session["resume_profile"].setdefault("proctoring_incidents", []).append({
                "timestamp": datetime.now().isoformat(),
                "type": "conduct_violation_warning",
                "detail": f"First Offense (De-escalation issued): Flagged '{conduct_check['flagged_snippet']}' in response: \"{transcript[:140]}\""
            })
            history.append({
                "question": current_q,
                "answer": transcript,
                "analysis": {"conduct_violation": conduct_check, "strike": 1},
                "assessment": {"communication": 2},
                "decision_source": "rule_based",
                "decision": {"action": "follow_up", "reasoning": "Professional conduct warning issued."}
            })
            session["current_question"] = boundary_prompt
            return TurnResponse(
                session_id=session_id,
                turn_number=turn_idx + 1,
                is_complete=False,
                next_question=boundary_prompt,
                empathy_prompt=None,
                action="follow_up",
                reasoning="First offense conduct boundary set.",
                difficulty_level=selector.difficulty,
                difficulty_label=selector.label(),
                stress_detected=False,
                stress_score=0.0,
                knowledge_gap_detected=False,
                noise_detected=False,
                ai_script_detected=False,
                decision_source="rule_based",
            )
        else:
            term_prompt = (
                "Because this evaluation requires professional conduct, I am going to conclude our interview here. "
                "Thank you for your time today; your session has been closed."
            )
            session["resume_profile"].setdefault("proctoring_incidents", []).append({
                "timestamp": datetime.now().isoformat(),
                "type": "conduct_violation_terminated",
                "detail": f"Repeated Abuse (Immediate Termination): Flagged '{conduct_check['flagged_snippet']}' in response: \"{transcript[:140]}\""
            })
            history.append({
                "question": current_q,
                "answer": transcript,
                "analysis": {"conduct_violation": conduct_check, "strike": strikes},
                "assessment": {"communication": 0},
                "decision_source": "rule_based",
                "decision": {"action": "end_early", "reasoning": "Session terminated due to repeated abusive conduct."}
            })
            session["is_complete"] = True
            session["current_question"] = None
            return TurnResponse(
                session_id=session_id,
                turn_number=turn_idx + 1,
                is_complete=True,
                next_question=term_prompt,
                empathy_prompt=None,
                action="end_early",
                reasoning="Session terminated due to repeated abusive conduct.",
                difficulty_level=selector.difficulty,
                difficulty_label=selector.label(),
                stress_detected=False,
                stress_score=0.0,
                knowledge_gap_detected=False,
                noise_detected=False,
                ai_script_detected=False,
                decision_source="rule_based",
            )

    # 1. Telemetry & Linguistics
    # Perplexity is a slow, experimental review signal. Run it automatically
    # for opted-in integrity monitoring; never run it after a candidate declines.
    integrity_monitoring_enabled = session["integrity_monitoring_consent"]
    perplexity_enabled = integrity_monitoring_enabled
    native_speaker = os.getenv("ACCENT_FAIRNESS_MODE", "false").lower() != "true"
    analysis_tasks = [
        asyncio.to_thread(get_disfluency_rate, transcript),
        asyncio.to_thread(get_consistency, raw_resume, transcript),
    ]
    if perplexity_enabled:
        analysis_tasks.append(asyncio.to_thread(get_perplexity, transcript))
    analysis_results = await asyncio.gather(*analysis_tasks)
    disflu, consist = analysis_results[:2]
    # Preserve the adaptive selector's neutral baseline without claiming that
    # perplexity was actually measured.
    perp = analysis_results[2] if perplexity_enabled else 50.0

    # 2. Stress & Empathy
    stress_result = detect_speech_stress(transcript, disflu, perp, native_speaker=native_speaker)
    stress_score = stress_result["stress_score"]
    knowledge_gap = stress_result["knowledge_gap_detected"]

    # 3. Optional integrity signals. These checks are only run after the
    # candidate opts in; their output is context for human review, never proof.
    audio_check = (
        detect_background_audio_anomaly(transcript)
        if integrity_monitoring_enabled
        else {"noise_detected": False, "warning_message": None}
    )
    ai_check = (
        detect_ai_generated_answer(transcript, perp, disflu, consist, history)
        if integrity_monitoring_enabled
        else {
            "ai_script_detected": False,
            "ai_probability": 0.0,
            "signals": [],
            "nudge_prompt": None,
            "incident_reason": None,
        }
    )
    assistance_request_detected = (
        detect_explicit_answer_request(transcript)
        if integrity_monitoring_enabled
        else False
    )

    if ai_check.get("ai_script_detected"):
        session["resume_profile"].setdefault("proctoring_incidents", []).append({
            "timestamp": datetime.now().isoformat(),
            "type": "ai_script_reading",
            "detail": ai_check["incident_reason"]
        })

    active_nudge = stress_result.get("empathy_prompt") or (
        ai_check.get("nudge_prompt") if ai_check.get("ai_script_detected") else None
    )

    analysis = {
        "transcript": transcript,
        "perplexity": perp,
        "perplexity_available": perplexity_enabled,
        "disfluency_rate": disflu,
        "consistency": consist,
        "noise_detected": audio_check.get("noise_detected", False),
        "ai_script_detected": ai_check.get("ai_script_detected", False),
        "ai_probability": ai_check.get("ai_probability", 0.0),
        "assistance_request_detected": assistance_request_detected,
        "stress_score": stress_score,
        "stress_signals": stress_result["stress_signals"],
        "knowledge_gap_detected": knowledge_gap,
        "difficulty_level": selector.difficulty,
        "empathy_triggered": bool(active_nudge)
    }

    # 4. LLM Decision with Adaptive Difficulty
    decision = await asyncio.to_thread(
        get_interviewer_decision,
        resume_profile, history, analysis,
        competency_bank_path=os.path.join(BASE_DIR, "competency_bank.yaml"),
        model_path=os.path.join(BASE_DIR, "competency_selector.pkl"),
        difficulty_label=selector.label(),
        jd_summary=(
            session["jd_summary"]
            + (
                "\nCandidate-confirmed or corrected context: "
                + session["confirmed_profile_context"]
                if session.get("confirmed_profile_context")
                else ""
            )
        ),
        stream_callback=session.get("_ws_question_stream_callback"),
    )

    # 5. Update IRT Difficulty for next turn
    selector.update(consist, perp, stress_score, knowledge_gap)

    # 6. Log turn
    current_q = session["current_question"]
    history.append({
        "timestamp": datetime.now().isoformat(),
        "question": current_q,
        "answer": transcript,
        "analysis": analysis,
        "assessment": decision.get("assessment", {}),
        "decision_source": decision.get("decision_source", "unknown"),
        "decision": decision.get("decision", {})
    })

    # 7. Check if interview should end
    action = decision.get("decision", {}).get("action", "move_on")
    reasoning = decision.get("decision", {}).get("reasoning", "")
    next_q = decision.get("next_question")

    if action == "end_early" or len(history) >= session["max_turns"]:
        session["is_complete"] = True
        session["current_question"] = None
        return TurnResponse(
            session_id=session_id,
            turn_number=turn_idx + 1,
            is_complete=True,
            next_question=None,
            empathy_prompt=active_nudge,
            action=action,
            reasoning=reasoning or "Interview evaluation completed.",
            difficulty_level=selector.difficulty,
            difficulty_label=selector.label(),
            stress_detected=stress_score > 0.6,
            stress_score=stress_score,
            knowledge_gap_detected=knowledge_gap,
            noise_detected=analysis["noise_detected"],
            ai_script_detected=ai_check.get("ai_script_detected", False),
            ai_signal_types=ai_check.get("signals", []),
            assistance_request_detected=assistance_request_detected,
            decision_source=decision.get("decision_source", "unknown"),
        )

    # 8. Behavioral Question Injection (Every 3 turns)
    session["behavioral_turn_counter"] += 1
    behavioral_cats = list(BEHAVIORAL_QUESTIONS.keys())
    if session["behavioral_turn_counter"] >= 3 and session["behavioral_asked"] < len(behavioral_cats):
        cat = behavioral_cats[session["behavioral_asked"] % len(behavioral_cats)]
        import random
        behavioral_q = random.choice(BEHAVIORAL_QUESTIONS[cat])
        next_q = behavioral_q
        session["behavioral_asked"] += 1
        session["behavioral_turn_counter"] = 0

    session["current_question"] = next_q

    return TurnResponse(
        session_id=session_id,
        turn_number=turn_idx + 1,
        is_complete=False,
        next_question=next_q,
        empathy_prompt=active_nudge,
        action=action,
        reasoning=reasoning,
        difficulty_level=selector.difficulty,
        difficulty_label=selector.label(),
        stress_detected=stress_score > 0.6,
        stress_score=stress_score,
        knowledge_gap_detected=knowledge_gap,
        noise_detected=analysis["noise_detected"],
        ai_script_detected=ai_check.get("ai_script_detected", False),
        ai_signal_types=ai_check.get("signals", []),
        assistance_request_detected=assistance_request_detected,
        decision_source=decision.get("decision_source", "unknown"),
    )


@app.get("/api/v1/interview/{session_id}/status", tags=["Interview Lifecycle"])
def get_session_status(session_id: str):
    """Get current session metadata, progress, and turn history."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found.")

    return {
        "session_id": session_id,
        "candidate_name": session["resume_profile"].get("name") or "Candidate",
        "role": session["resume_profile"].get("role"),
        "level": session["resume_profile"].get("level"),
        "turns_completed": len(session["history"]),
        "turn_state": session.get("turn_state", "ready"),
        "max_turns": session["max_turns"],
        "is_complete": session["is_complete"],
        "current_difficulty": session["difficulty_selector"].label()
    }


@app.post("/api/v1/interview/{session_id}/evidence", tags=["Interview Evidence"])
async def save_interview_evidence(
    session_id: str,
    signal: str = Form(...),
    snapshot: Optional[UploadFile] = File(None),
    clip: Optional[UploadFile] = File(None),
):
    """Store a still and/or short camera-only clip after a camera alert."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found or expired.")
    if not session.get("integrity_monitoring_consent") or not session.get("evidence_capture_consent"):
        raise HTTPException(status_code=403, detail="Evidence capture was not enabled for this session.")
    if signal not in EVIDENCE_SIGNALS:
        raise HTTPException(status_code=400, detail="Unsupported camera signal.")
    if snapshot is None and clip is None:
        raise HTTPException(status_code=400, detail="No evidence file was provided.")

    # Accept evidence only for a camera alert actually emitted by this session's
    # server-side proctor. Browser-supplied signal labels alone are not trusted.
    incidents = session["proctor"].integrity_log
    used_incidents = {record.get("incident_timestamp") for record in session["resume_profile"].get("evidence_records", [])}
    incident = next((
        item for item in incidents
        if item.get("type") == EVIDENCE_INCIDENT_TYPES[signal]
        and item.get("timestamp") not in used_incidents
    ), None)
    if incident is None:
        raise HTTPException(status_code=409, detail="No matching server-generated camera alert is available for evidence.")

    capture_id = str(uuid.uuid4())
    capture_dir = os.path.join(EVIDENCE_DIR, session_id, capture_id)
    os.makedirs(capture_dir, exist_ok=True)
    if os.name != "nt":
        os.chmod(capture_dir, 0o700)
    stored: Dict[str, str] = {}

    async def save_upload(upload: UploadFile, asset: str, accepted_types: Dict[str, str], max_bytes: int):
        content_type = (upload.content_type or "").split(";")[0].lower()
        extension = accepted_types.get(content_type)
        if not extension:
            raise HTTPException(status_code=400, detail=f"Unsupported {asset} media type.")
        data = await upload.read(max_bytes + 1)
        if not data or len(data) > max_bytes:
            raise HTTPException(status_code=413, detail=f"{asset.capitalize()} exceeds the size limit.")
        path = os.path.join(capture_dir, f"{asset}.{extension}")
        with open(path, "wb") as evidence_file:
            evidence_file.write(data)
        if os.name != "nt":
            os.chmod(path, 0o600)
        stored[asset] = path

    try:
        if snapshot is not None:
            await save_upload(snapshot, "snapshot", {"image/jpeg": "jpg"}, 1_000_000)
        if clip is not None:
            await save_upload(clip, "clip", {"video/webm": "webm", "video/mp4": "mp4"}, 8_000_000)
    except Exception:
        shutil.rmtree(capture_dir, ignore_errors=True)
        raise

    captured_at = datetime.now(timezone.utc)
    record = {
        "capture_id": capture_id,
        "signal": signal,
        "alert_at": incident["timestamp"],
        "incident_timestamp": incident["timestamp"],
        "captured_at": captured_at.isoformat(),
        "expires_at": (captured_at + timedelta(days=EVIDENCE_RETENTION_DAYS)).isoformat(),
        "files": stored,
    }
    session["resume_profile"].setdefault("evidence_records", []).append(record)
    _write_evidence_manifest(session_id, session["resume_profile"]["evidence_records"])
    return {
        "capture_id": capture_id,
        "signal": signal,
        "alert_at": record["alert_at"],
        "captured_at": record["captured_at"],
        "expires_at": record["expires_at"],
        "assets": {asset: f"/api/v1/interview/{session_id}/evidence/{capture_id}/{asset}" for asset in stored},
    }


@app.get("/api/v1/interview/{session_id}/evidence", tags=["Interview Evidence"])
def list_interview_evidence(session_id: str):
    session = sessions.get(session_id)
    if session:
        enabled = session.get("evidence_capture_consent", False)
        records = session["resume_profile"].get("evidence_records", [])
    else:
        manifest = _read_evidence_manifest(session_id)
        if not manifest:
            raise HTTPException(status_code=404, detail="Interview session not found or expired.")
        enabled = manifest.get("enabled", False)
        records = manifest.get("records", [])
    if not enabled:
        raise HTTPException(status_code=404, detail="Interview session not found or expired.")
    now = datetime.now(timezone.utc)
    result = []
    for record in records:
        if datetime.fromisoformat(record["expires_at"]) <= now:
            continue
        assets = {
            asset: f"/api/v1/interview/{session_id}/evidence/{record['capture_id']}/{asset}"
            for asset, path in record["files"].items() if os.path.isfile(path)
        }
        result.append({key: record[key] for key in ("capture_id", "signal", "alert_at", "captured_at", "expires_at")} | {"assets": assets})
    return {"enabled": enabled, "records": result}


@app.get("/api/v1/interview/{session_id}/evidence/{capture_id}/{asset}", tags=["Interview Evidence"])
def get_interview_evidence_asset(session_id: str, capture_id: str, asset: str):
    session = sessions.get(session_id)
    manifest = None if session else _read_evidence_manifest(session_id)
    enabled = session.get("evidence_capture_consent", False) if session else bool(manifest and manifest.get("enabled"))
    if not enabled:
        raise HTTPException(status_code=404, detail="Evidence not found.")
    if asset not in {"snapshot", "clip"}:
        raise HTTPException(status_code=404, detail="Evidence not found.")
    records = session["resume_profile"].get("evidence_records", []) if session else manifest.get("records", [])
    record = next((r for r in records if r["capture_id"] == capture_id), None)
    if not record or datetime.fromisoformat(record["expires_at"]) <= datetime.now(timezone.utc):
        raise HTTPException(status_code=404, detail="Evidence not found or expired.")
    path = record["files"].get(asset)
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Evidence not found.")
    media_type = "image/jpeg" if asset == "snapshot" else ("video/mp4" if path.endswith(".mp4") else "video/webm")
    return FileResponse(path, media_type=media_type, headers={"Content-Disposition": "inline", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


@app.delete("/api/v1/interview/{session_id}/evidence/{capture_id}", tags=["Interview Evidence"])
def delete_interview_evidence(session_id: str, capture_id: str):
    session = sessions.get(session_id)
    manifest = None if session else _read_evidence_manifest(session_id)
    enabled = session.get("evidence_capture_consent", False) if session else bool(manifest and manifest.get("enabled"))
    if not enabled:
        raise HTTPException(status_code=404, detail="Evidence not found.")
    records = session["resume_profile"].get("evidence_records", []) if session else manifest.get("records", [])
    record = next((r for r in records if r["capture_id"] == capture_id), None)
    if not record:
        raise HTTPException(status_code=404, detail="Evidence not found.")
    for path in record["files"].values():
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    shutil.rmtree(os.path.dirname(next(iter(record["files"].values()))), ignore_errors=True)
    records.remove(record)
    if session:
        _write_evidence_manifest(session_id, records)
    elif records:
        _write_evidence_manifest(session_id, records)
    else:
        try:
            os.remove(_evidence_manifest_path(session_id))
        except FileNotFoundError:
            pass
    return {"deleted": True}


@app.get("/api/v1/interview/{session_id}/report", tags=["Evaluation & Reports"])
def download_pdf_report(session_id: str):
    """
    Download the executive factual HR interview debrief PDF.
    Contains qualitative observations and job-description checklist alignment. The report does not certify legal compliance.
    """
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found.")

    if not session["history"]:
        raise HTTPException(status_code=400, detail="Cannot generate report for an interview with zero turns.")

    os.makedirs(REPORT_DIR, exist_ok=True)
    if os.name != "nt":
        os.chmod(REPORT_DIR, 0o700)
    pdf_output_path = os.path.join(REPORT_DIR, f"interview_report_{session_id}.pdf")
    success = generate_report(session["resume_profile"], session["history"], output_path=pdf_output_path)

    if not success or not os.path.exists(pdf_output_path):
        raise HTTPException(status_code=500, detail="Failed to compile PDF report.")

    return FileResponse(
        pdf_output_path,
        media_type="application/pdf",
        filename=f"executive_debrief_{session_id}.pdf",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/api/v1/interview/{session_id}/report/html", tags=["Evaluation & Reports"])
def view_html_report(session_id: str):
    """
    View the executive debrief report as a standalone HTML page.
    Ideal for embedding in browser dashboards or iframes.
    """
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found.")

    os.makedirs(REPORT_DIR, exist_ok=True)
    if os.name != "nt":
        os.chmod(REPORT_DIR, 0o700)
    html_path = os.path.join(REPORT_DIR, f"interview_report_{session_id}.html")
    if not os.path.exists(html_path):
        # Generate it if not already built
        generate_report(
            session["resume_profile"],
            session["history"],
            output_path=os.path.join(REPORT_DIR, f"interview_report_{session_id}.pdf"),
        )

    if not os.path.exists(html_path):
        raise HTTPException(status_code=500, detail="HTML report unavailable.")

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(
        content=html_content,
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


# ---------------------------------------------------------------------------
# WebSocket for Real-Time Streaming
# ---------------------------------------------------------------------------
@app.websocket("/ws/interview/{session_id}")
async def websocket_interview_endpoint(websocket: WebSocket, session_id: str):
    """
    Bi-directional WebSocket for real-time conversational voice/text turns.
    Receives JSON messages: {"type": "answer", "text": "..."} or {"type": "interrupt"}
    Streams back analysis and next questions.
    """
    await websocket.accept()
    session = sessions.get(session_id)
    if not session:
        await websocket.send_json({"error": "Session not found."})
        await websocket.close()
        return

    proctor: ProctorState = session["proctor"]
    send_lock = asyncio.Lock()
    active_turn_task: Optional[asyncio.Task] = None

    async def send_event(event: Dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(event)

    async def process_answer_turn(answer_text: str) -> None:
        owner_task = asyncio.current_task()
        session["_ws_turn_task"] = owner_task
        session["turn_state"] = "processing"
        started_at = time.perf_counter()
        event_loop = asyncio.get_running_loop()

        def send_question_delta(text: str) -> None:
            future = asyncio.run_coroutine_threadsafe(
                send_event({"type": "question_delta", "text": text}), event_loop
            )
            try:
                future.result(timeout=2.0)
            except Exception:
                future.cancel()

        session["_ws_question_stream_callback"] = send_question_delta
        try:
            turn_resp = await submit_turn(session_id, TurnRequest(answer=answer_text))
            session["turn_state"] = "completed" if turn_resp.is_complete else "ready"
            if turn_resp.empathy_prompt:
                await send_event({"type": "empathy", "text": turn_resp.empathy_prompt})

            await send_event({
                "type": "turn_result",
                "turn": turn_resp.turn_number,
                "profile_confirmed": turn_resp.profile_confirmed,
                "is_complete": turn_resp.is_complete,
                "next_question": turn_resp.next_question,
                "difficulty_label": turn_resp.difficulty_label,
                "stress_detected": turn_resp.stress_detected,
                "noise_detected": turn_resp.noise_detected,
                "ai_script_detected": turn_resp.ai_script_detected,
                "ai_signal_types": turn_resp.ai_signal_types,
                "assistance_request_detected": turn_resp.assistance_request_detected,
                "decision_source": turn_resp.decision_source,
                "processing_ms": round((time.perf_counter() - started_at) * 1000),
            })

            if turn_resp.is_complete:
                session["resume_profile"]["proctoring_incidents"] = proctor.integrity_log
                async with send_lock:
                    await websocket.send_json({"type": "completed", "message": "Interview session finished."})
                    await websocket.close(code=1000)
        except asyncio.CancelledError:
            # Interrupt/disconnect cancels this coroutine so stale results never
            # reach the candidate. Worker threads already in progress may finish
            # in the background because Python cannot forcibly stop a thread.
            session["turn_state"] = "ready"
            raise
        except Exception as turn_error:
            session["turn_state"] = "ready"
            logger.exception("Interview turn failed for session %s: %s", session_id, turn_error)
            try:
                await send_event({"type": "turn_error", "message": "The interviewer could not complete this turn. Please try again."})
            except Exception:
                pass
        finally:
            if session.get("_ws_question_stream_callback") is send_question_delta:
                session.pop("_ws_question_stream_callback", None)
            if session.get("_ws_turn_task") is owner_task:
                session.pop("_ws_turn_task", None)

    # Send initial greeting
    await send_event({
        "type": "question",
        "turn": 0,
        "text": session["current_question"],
        "difficulty": session["difficulty_selector"].label()
    })

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            # ------------------------------------------------------------------
            # Frame message: frontend streams a base64-encoded webcam frame.
            # We decode it, run the proctoring engine, and if a consistent
            # suspicious pattern is detected, immediately send Sasha's nudge back.
            # The candidate always gets a warm, positive message — never an accusation.
            # The incident is silently logged for the HR report regardless.
            # ------------------------------------------------------------------
            if msg_type == "frame":
                if not session["integrity_monitoring_consent"]:
                    continue
                raw_b64 = data.get("data", "")
                audio_active = bool(data.get("audio_active", False))
                if raw_b64:
                    try:
                        import cv2
                        img_bytes = base64.b64decode(raw_b64)
                        arr = np.frombuffer(img_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            nudge = await asyncio.to_thread(proctor.process_frame, frame, audio_active)
                            if nudge:
                                # Sasha speaks warmly to the candidate
                                await send_event(nudge)
                                # Also write to session's resume_profile so report_generator
                                # can access proctoring incidents at report time
                                session["resume_profile"].setdefault(
                                    "proctoring_incidents", []
                                ).extend(proctor.integrity_log[-1:])
                    except Exception as frame_err:
                        logger.debug(f"Frame decode error session {session_id}: {frame_err}")
                continue

            if msg_type == "integrity_event":
                # These are transparent, client-reported context events. They
                # are intentionally kept separate from answers, scores, and
                # proctoring incidents, and are never treated as proof.
                allowed_events = {
                    "page_hidden",
                    "page_visible",
                    "window_blur",
                    "window_focus",
                    "answer_paste",
                }
                event_name = data.get("event")
                events = session["browser_interaction_events"]
                if (
                    session["integrity_monitoring_consent"]
                    and event_name in allowed_events
                    and len(events) < 200
                ):
                    events.append({
                        "event": event_name,
                        "timestamp": datetime.now().isoformat(),
                        "turn": len(session["history"]) + 1,
                    })
                continue

            if msg_type == "interrupt":
                # Cancel the active turn coroutine while keeping the receive loop
                # available for the next answer and camera/context events.
                cancelled_turn = bool(active_turn_task and not active_turn_task.done())
                if cancelled_turn:
                    active_turn_task.cancel()
                    active_turn_task = None
                    session["turn_state"] = "ready"
                    session.pop("_ws_turn_task", None)
                await send_event({
                    "type": "interrupted",
                    "status": "Turn cancelled." if cancelled_turn else "Speech playback stopped.",
                    "cancelled_turn": cancelled_turn,
                })
                continue

            if msg_type == "answer":
                answer_text = data.get("text", "")
                if not answer_text.strip():
                    continue

                if (active_turn_task and not active_turn_task.done()) or session.get("turn_state") == "processing":
                    await send_event({"type": "turn_busy", "message": "A turn is already being processed."})
                    continue

                await send_event({"type": "turn_processing", "turn": len(session["history"]) + 1})
                active_turn_task = asyncio.create_task(process_answer_turn(answer_text))

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for session {session_id}")
        if active_turn_task and not active_turn_task.done():
            active_turn_task.cancel()
        if session.get("turn_state") == "processing":
            session["turn_state"] = "ready"
        session.pop("_ws_turn_task", None)
        # Still attach proctor log so a partial report is accurate
        session["resume_profile"]["proctoring_incidents"] = proctor.integrity_log
    except Exception as e:
        logger.error(f"WebSocket error in session {session_id}: {e}")
        if active_turn_task and not active_turn_task.done():
            active_turn_task.cancel()
        if session.get("turn_state") == "processing":
            session["turn_state"] = "ready"
        session.pop("_ws_turn_task", None)
        await websocket.close()


# Mount compiled React frontend if present
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
