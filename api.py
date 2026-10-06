from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import sys
import uuid
import shutil
import tempfile
import logging
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml
import numpy as np
import uvicorn

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
    description="Autonomous Full-Duplex AI Technical Interviewer & Executive Evaluation Platform"
)

# Enable CORS for local and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-Memory Session Manager
# ---------------------------------------------------------------------------
class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create(self, resume_profile: Dict[str, Any], jd_text: str = "", jd_summary: str = "") -> str:
        session_id = str(uuid.uuid4())[:8]
        difficulty_selector = AdaptiveQuestionSelector(resume_profile.get("level", "mid"))

        cand_name = resume_profile.get("name")
        name_greeting = f", {cand_name}" if cand_name and cand_name.lower() != "candidate" else ""
        first_question = (
            f"Thanks for sharing your background{name_greeting}! I see you have "
            f"{resume_profile.get('years_experience', 0)} years of experience as a "
            f"{resume_profile.get('level', 'mid')} engineer specializing in "
            f"{resume_profile.get('role', 'SDE').replace('_', ' ')}. "
            f"To start, could you tell me about the project you're most proud of from your resume?"
        )

        self._sessions[session_id] = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "resume_profile": resume_profile,
            "jd_text": jd_text,
            "jd_summary": jd_summary,
            "difficulty_selector": difficulty_selector,
            "history": [],
            "current_question": first_question,
            "behavioral_asked": 0,
            "behavioral_turn_counter": 0,
            "is_complete": False,
            "max_turns": 8,
            "report_paths": {},
            "proctor": ProctorState(),   # live proctoring engine for this session
            "conduct_strikes": 0,        # tracks profanity/abusive language offenses
        }
        return session_id

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)


sessions = SessionStore()


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
    jd_file: Optional[UploadFile] = File(None, description="Job Description file (.txt)")
):
    """
    Initialize a new autonomous interview session.
    Parses resume, detects target role, binds Job Description for Hybrid RAG,
    and returns the personalized warm-up question.
    """
    # 1. Save uploaded resume to temp file
    suffix = os.path.splitext(resume_file.filename)[1].lower()
    if suffix not in [".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail="Unsupported resume format. Please upload .pdf or .docx.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_resume:
        shutil.copyfileobj(resume_file.file, tmp_resume)
        tmp_resume_path = tmp_resume.name

    try:
        # Parse resume
        resume_profile_raw = parse_resume(tmp_resume_path)
    finally:
        if os.path.exists(tmp_resume_path):
            os.remove(tmp_resume_path)

    # 2. Detect role
    role_signals_path = "role_signals.yaml"
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
    session_id = sessions.create(resume_profile, jd_text=final_jd, jd_summary=jd_summary)
    session = sessions.get(session_id)

    return StartInterviewResponse(
        session_id=session_id,
        candidate_name=resume_profile.get("name") or "Candidate",
        role=resume_profile.get("role", "SDE"),
        experience_level=resume_profile.get("level", "mid"),
        years_experience=float(resume_profile.get("years_experience", 0)),
        starting_difficulty=session["difficulty_selector"].label(),
        first_question=session["current_question"],
        jd_loaded=bool(final_jd.strip())
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

    if session["is_complete"]:
        raise HTTPException(status_code=400, detail="Interview has already concluded.")

    resume_profile = session["resume_profile"]
    selector: AdaptiveQuestionSelector = session["difficulty_selector"]
    history = session["history"]
    turn_idx = len(history)

    transcript = payload.answer.strip()
    raw_resume = resume_profile.get("raw_text", "")

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
                ai_script_detected=False
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
                ai_script_detected=False
            )

    # 1. Telemetry & Linguistics
    native_speaker = os.getenv("ACCENT_FAIRNESS_MODE", "false").lower() != "true"
    perp = get_perplexity(transcript)
    disflu = get_disfluency_rate(transcript)
    consist = get_consistency(raw_resume, transcript)

    # 2. Stress & Empathy
    stress_result = detect_speech_stress(transcript, disflu, perp, native_speaker=native_speaker)
    stress_score = stress_result["stress_score"]
    knowledge_gap = stress_result["knowledge_gap_detected"]

    # 3. Audio & Room Integrity & AI Script Detection
    audio_check = detect_background_audio_anomaly(transcript)
    ai_check = detect_ai_generated_answer(transcript, perp, disflu, consist, history)

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
        "disfluency_rate": disflu,
        "consistency": consist,
        "noise_detected": audio_check.get("noise_detected", False),
        "ai_script_detected": ai_check.get("ai_script_detected", False),
        "ai_probability": ai_check.get("ai_probability", 0.0),
        "stress_score": stress_score,
        "stress_signals": stress_result["stress_signals"],
        "knowledge_gap_detected": knowledge_gap,
        "difficulty_level": selector.difficulty,
        "empathy_triggered": bool(active_nudge)
    }

    # 4. LLM Decision with Adaptive Difficulty
    decision = get_interviewer_decision(
        resume_profile, history, analysis,
        competency_bank_path="competency_bank.yaml",
        difficulty_label=selector.label(),
        jd_summary=session["jd_summary"]
    )

    # 5. Update IRT Difficulty for next turn
    selector.update(consist, perp, stress_score, knowledge_gap)

    # 6. Log turn
    current_q = session["current_question"]
    history.append({
        "question": current_q,
        "answer": transcript,
        "analysis": analysis,
        "assessment": decision.get("assessment", {}),
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
            ai_script_detected=ai_check.get("ai_script_detected", False)
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
        ai_script_detected=ai_check.get("ai_script_detected", False)
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
        "max_turns": session["max_turns"],
        "is_complete": session["is_complete"],
        "current_difficulty": session["difficulty_selector"].label()
    }


@app.get("/api/v1/interview/{session_id}/report", tags=["Evaluation & Reports"])
def download_pdf_report(session_id: str):
    """
    Download the executive factual HR interview debrief PDF.
    Contains qualitative observations, JD checklist alignment, and NYC Law 144 compliance statement.
    """
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found.")

    if not session["history"]:
        raise HTTPException(status_code=400, detail="Cannot generate report for an interview with zero turns.")

    pdf_output_path = f"interview_report_{session_id}.pdf"
    success = generate_report(session["resume_profile"], session["history"], output_path=pdf_output_path)

    if not success or not os.path.exists(pdf_output_path):
        raise HTTPException(status_code=500, detail="Failed to compile PDF report.")

    return FileResponse(
        pdf_output_path,
        media_type="application/pdf",
        filename=f"executive_debrief_{session_id}.pdf"
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

    html_path = f"interview_report_{session_id}.html"
    if not os.path.exists(html_path):
        # Generate it if not already built
        generate_report(session["resume_profile"], session["history"], output_path=f"interview_report_{session_id}.pdf")

    if not os.path.exists(html_path):
        raise HTTPException(status_code=500, detail="HTML report unavailable.")

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content)


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

    # Send initial greeting
    await websocket.send_json({
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
                raw_b64 = data.get("data", "")
                audio_active = bool(data.get("audio_active", False))
                if raw_b64:
                    try:
                        import cv2
                        img_bytes = base64.b64decode(raw_b64)
                        arr = np.frombuffer(img_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            nudge = proctor.process_frame(frame, audio_active=audio_active)
                            if nudge:
                                # Sasha speaks warmly to the candidate
                                await websocket.send_json(nudge)
                                # Also write to session's resume_profile so report_generator
                                # can access proctoring incidents at report time
                                session["resume_profile"].setdefault(
                                    "proctoring_incidents", []
                                ).extend(proctor.integrity_log[-1:])
                    except Exception as frame_err:
                        logger.debug(f"Frame decode error session {session_id}: {frame_err}")
                continue

            if msg_type == "interrupt":
                # Instant conversational barge-in signal from client
                await websocket.send_json({"type": "interrupted", "status": "Audio stream cut immediately (<50ms)"})
                continue

            if msg_type == "answer":
                answer_text = data.get("text", "")
                if not answer_text.strip():
                    continue

                # Process turn
                turn_resp = await submit_turn(session_id, TurnRequest(answer=answer_text))

                # Empathy prompt if triggered
                if turn_resp.empathy_prompt:
                    await websocket.send_json({
                        "type": "empathy",
                        "text": turn_resp.empathy_prompt
                    })

                # Stream next question or conclusion
                await websocket.send_json({
                    "type": "turn_result",
                    "turn": turn_resp.turn_number,
                    "is_complete": turn_resp.is_complete,
                    "next_question": turn_resp.next_question,
                    "difficulty_label": turn_resp.difficulty_label,
                    "stress_detected": turn_resp.stress_detected,
                    "noise_detected": turn_resp.noise_detected
                })

                if turn_resp.is_complete:
                    # Attach full proctoring log to session for HR report
                    session["resume_profile"]["proctoring_incidents"] = proctor.integrity_log
                    await websocket.send_json({"type": "completed", "message": "Interview session finished."})
                    break

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for session {session_id}")
        # Still attach proctor log so a partial report is accurate
        session["resume_profile"]["proctoring_incidents"] = proctor.integrity_log
    except Exception as e:
        logger.error(f"WebSocket error in session {session_id}: {e}")
        await websocket.close()


# Mount compiled React frontend if present
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
