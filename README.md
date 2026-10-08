# Sasha — Resume-Informed Technical Interview Prototype

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/react-19-61dafb.svg)](https://react.dev)
[![Vite](https://img.shields.io/badge/vite-8.3-646cff.svg)](https://vitejs.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-007FFF.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11-5C3EE8.svg)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sasha is an experimental prototype for resume-informed, voice-based technical interviews. A candidate can provide a resume and optional job description, answer interview questions by voice or text, and receive a generated transcript and rubric summary.

This project is not independently validated for employment decisions. It does not run coding tasks, verify resume claims, identify browser extensions or phones, or determine whether a candidate cheated. Camera evidence capture is a separate, optional consent: after a repeated camera alert, it can save a reduced-size still and up to five seconds of camera-only video. Evidence is stored privately for 30 days and is intended only for human review. Optional camera and browser signals can be wrong and are not proof of misconduct.

---

# 🏛️ Complete System Architecture

```
                                  ┌────────────────────────────────────────┐
                                  │      1. MULTIMODAL INGESTION LAYER      │
                                  └───────────────────┬────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
           [ Candidate Resume (.pdf/.docx) ]                             [ Target Job Description ]
                       │                                                             │
         Resume text extraction and heuristics                         Optional job-description text
       (Fields depend on document quality)                              (Context for interview generation)
                       │                                                             │
         Logistic Regression Role Classifier                                         │
        (SDE, ML Eng, Backend, DevOps, Data)                                        │
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │
                                                      ▼
                                  ┌────────────────────────────────────────┐
                                  │   2. REAL-TIME SESSION ORCHESTRATOR     │
                                  │       (FastAPI WebSocket Event Loop)   │
                                  └───────────────────┬────────────────────┘
                                                      │
         ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
         ▼                                            ▼                                            ▼
┌─────────────────────────────────┐      ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│   OPTIONAL CAMERA SIGNALS       │      │     INTERVIEW UI & API          │      │     TEXT ANALYSIS               │
│    (Opt-in, experimental)       │      │     (React + FastAPI)           │      │   (Unvalidated heuristics)      │
├─────────────────────────────────┤      ├─────────────────────────────────┤      ├─────────────────────────────────┤
│ • Experimental face signals    │      │ • Interview API / WebSocket     │      │ • Resume-answer similarity     │
│ • Periodic camera frames        │      │ • Interview transcript          │      │ • Statistical answer signals    │
│ • Head/gaze direction estimate  │      │ • Browser speech/text input     │      │ • Cannot identify AI source    │
│ • No phone/object recognition   │      │ • Optional interaction events │      │ • No AI-use attribution        │
│ • Opt-in event evidence      │      │ • Opt-in event evidence     │      │ • Not a misconduct finding     │
└────────────────┬────────────────┘      └────────────────┬────────────────┘      └────────────────┬────────────────┘
                 │                                        │                                        │
                 └────────────────────────────────────────┼────────────────────────────────────────┘
                                                          │
                                                          ▼
                                  ┌────────────────────────────────────────┐
                                  │   3. ADAPTIVE INTERVIEW FLOW           │
                                  │      (Not psychometrically validated)  │
                                  └───────────────────┬────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
         [ Adaptive Question Selection ]                               [ Conversation Boundary Prompts ]
         Uses the configured question/competency bank                  May stop a session after repeated conduct flags
         and interview context                                           Flags are not a hiring decision
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │
                                                      ▼
                                  ┌────────────────────────────────────────┐
                                  │     4. REPORT GENERATION               │
                                  │        (Human Review Required)          │
                                  └───────────────────┬────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
           [ Interview Summary PDF/HTML ]                               [ Optional Signal Summary ]
         • Rubric estimates with response excerpts                    • Some event timestamps and descriptions
         • Turn-by-turn transcript                                     • Opt-in alert evidence links, if saved
         • Optional follow-up prompts                                  • Signals are unverified context
```

---

## 🔬 Subsystem Deep-Dive

### 1. Ingestion & Dossier Parsing Layer
* **`resume_parser.py`**: Extracts candidate identity, contacts, raw text, and accomplishment bullets using spaCy Named Entity Recognition (`en_core_web_sm`) and action-verb grammar filters. Accurately parses multi-year tenures across 4-digit date patterns (`2021 – Present`, `2018 - 2022`).
* **`signal_detector.py`**: A TF-IDF vectorizer paired with a Logistic Regression classifier (`role_detector_model.pkl`) that maps candidate text to target job archetypes (`frontend`, `backend`, `machine_learning`, `devops`, `fullstack`).
* **`competency_bank.yaml`**: Hierarchical skill tree defining core competencies, depth indicators, and probing trajectories across junior, mid, senior, and staff engineering tiers.

---

### 2. Experimental Camera Signals (`analyzer.py`)

When the candidate opts in to integrity monitoring, the browser sends reduced-size camera frames for face presence, multiple-face, head/gaze direction, and possible audio/video speaking-mismatch checks. A second, unchecked-by-default consent is required to save evidence. After a repeated server-generated camera alert, the browser captures a 320×180 still and up to five seconds of 320×180, 10 fps camera-only video beginning at the alert; no microphone audio is recorded. Files are kept outside the public frontend directory, expire after 30 days, and can be reviewed or deleted from the debrief. Capture depends on browser camera and recording support. The app has no phone/object detector. These checks do not identify people or establish cheating; treat an alert as an unverified prompt for human review.

---

### 3. Adaptive Interview Flow (`interviewer_agent.py`)

The opening exchange first summarizes the role and experience inferred from the resume, discloses the supportive practice format, and asks the candidate to confirm or correct the profile. This setup response is not scored as an interview turn. Sasha then asks about a resume project and selects follow-ups using interview context, the configured competency bank, and the candidate's responses. When a job description is provided, its summary is supplied to question generation. This behavior has not been validated as psychometric measurement or calibrated against job performance.

---

### 4. Optional Browser Interaction Context

With the candidate's opt-in, Sasha can record page visibility/focus changes and paste events in the answer field. These browser-reported events do not identify an extension, website, clipboard source, or use of another device. Transcript checks can flag a small set of direct requests for Sasha to provide an answer and statistical answer patterns; a dismissible in-interview notice explains the reason. The report may note when different camera, audio, and answer-text signal types occur within 60 seconds. That timing is context only, not a combined score or confirmation. These checks cannot determine whether a response came from an AI tool or establish cheating.

---

### 5. Acoustic Isolation & Frontend Studio (`InterviewRoom.jsx`, `SashaPresence.jsx`)

* **Acoustic Echo Isolation (`isSpeakingRef`)**:
  When Sasha speaks, microphone speech recognition is actively gated to prevent acoustic feedback loops where the system transcribes its own voice. A **400ms echo cooldown buffer** allows room reverberation to settle before listening resumes.
* **Tactile Character Aesthetics**:
  Inspired by plush character design principles, Sasha features soft velvet gradient bodies, ambient ground floor underglow, signature deep obsidian oval eyes with **dual specular reflection glints** (large key highlight + sparkle), organic asymmetric blinking, and real-time cursor gaze parallax.
* **Curated Personas**:
  * **Pearl**: Soft ivory plush, warm amber underglow, balanced staff interviewer persona.
  * **Indigo**: Deep violet velvet with over-ear studio DJ headphones and audio-reactive soundwave crown.
  * **Onyx**: Charcoal-slate minimal visor bar with horizontal scanner eyes for executive evaluation.
* **Conversational Barge-In**: Candidates can interrupt the interviewer during speech.

---

### 6. Conversation Boundary Handling

The interviewer can respond to detected abusive language with a boundary prompt and may end a session after repeated flags. Speech-to-text and conduct heuristics can be wrong; a session ending or conduct flag is not an employment decision.

---

### 7. Interview Reports (`report_generator.py`)

Generates PDF and HTML summaries with turn-by-turn answers, automated rubric estimates, and optional unverified integrity signals. Rubric and job-description matches are review aids, not factual verification, audit evidence, or hiring recommendations. Each rubric estimate points reviewers to a candidate response excerpt.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
* **Python**: 3.10, 3.11, or 3.12
* **Node.js**: v18+ or v20+
* **Browser**: Chrome or Edge (for Web Speech API and MediaPipe camera streaming)

### 2. Installation

```bash
git clone https://github.com/Vabs1002/Sasha.git
cd Sasha

# Install Python backend dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Install and build React frontend
cd frontend
npm install
npm run build
cd ..
```

### 3. Environment Configuration

Create a `.env` file in the root directory:
```env
# OpenAI-compatible provider (Gemini example)
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-3.5-flash-lite
LLM_TIMEOUT_SECONDS=15
LLM_MAX_RETRIES=0
LLM_REASONING_EFFORT=minimal
ENABLE_AGENTIC_RESUME_SEARCH=false

# Or configure one of the supported providers instead
# GROQ_API_KEY=your_groq_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here     # Supports gpt-4o / gpt-4o-mini
# GROK_API_KEY=your_grok_api_key_here         # Supports grok-beta
# OLLAMA_BASE_URL=http://localhost:11434/v1   # 100% offline local model

# Comma-separated browser origins allowed to call the API
CORS_ALLOW_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# Feature Flags
ENABLE_VOICE_OUTPUT=true
```

Keep `.env` local and never commit provider keys. The experimental perplexity signal runs automatically when the candidate opts into integrity monitoring. It does not run if the candidate declines; it is still only a review signal, not proof of AI use.

### 4. Running the Application

#### Option A: Unified Full-Stack Server (Recommended)
FastAPI serves the compiled React production application, WebSocket streaming, and REST APIs from a single port:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to: **`http://localhost:8000`**

#### Option B: Hot-Reload Frontend Development
Run backend API and Vite dev server concurrently:
```bash
# Terminal 1: Backend API
uvicorn api:app --port 8000 --reload

# Terminal 2: Frontend Dev Server
cd frontend
npm run dev
```
Navigate to: **`http://localhost:5173`**

#### Option C: CLI Terminal Mode
Run an autonomous interview simulation directly in your shell:
```bash
python main.py samples/sample_resume.pdf
```

---

## 🧪 Tests

Run the current test suite locally with:

```bash
python -m pytest tests/ -v
```

No test result is claimed here; results depend on the checked-out revision and local environment.

---

## 📁 Repository Structure

```
Sasha/
├── analyzer.py                 # Experimental camera/audio and text analysis
├── api.py                      # FastAPI REST & WebSocket orchestrator with ProctorState
├── interviewer_agent.py        # Adaptive interviewer and LLM configuration
├── report_generator.py         # PDF/HTML interview summary generator
├── resume_parser.py            # spaCy candidate extraction & 4-digit date tenures
├── signal_detector.py          # TF-IDF & Logistic Regression role & seniority classification
├── competency_bank.yaml        # Structured technical competencies across SDE, MLE, Devops
├── role_signals.yaml           # Industry competency signals & probing templates
├── main.py                     # CLI standalone interview entry point
├── requirements.txt            # Python dependencies (pinned protobuf==4.25.3 for MediaPipe)
├── frontend/                   # Modern React 19 studio application
│   ├── src/
│   │   ├── components/
│   │   │   ├── SashaPresence.jsx   # Tactile character entity (Pearl, Indigo, Onyx)
│   │   │   ├── InterviewRoom.jsx   # Browser speech input/output, camera signals, VU meter
│   │   │   ├── LobbyView.jsx       # Resume dropzone & target JD configuration
│   │   │   └── ReportView.jsx      # Interactive debrief viewer & PDF download
│   │   ├── api.js                  # WebSocket & REST client
│   │   └── App.jsx
│   ├── dist/                       # Compiled production bundle
│   └── package.json
└── tests/                      # Automated tests
    ├── test_research_proctoring.py # Experimental camera-signal tests
    ├── test_api.py                 # REST & WebSocket integration tests
    ├── test_report_generator.py    # PDF rendering & integrity incident tests
    └── ...
```

---

## ⚠️ Use Limitations

Sasha is a prototype and has not been independently validated or audited for employment decisions. Interview sessions are held in server memory and are not durable across server restarts. Employers must assess their own legal, privacy, accessibility, and security obligations before using interview data in a hiring process. Do not use automated scores or integrity signals as the sole basis for an employment decision.

Interview reports and consented camera evidence are kept in a private, non-static directory with no-store download headers and 30-day cleanup. This is prototype hardening, not production security: the app still needs authenticated recruiter authorization, TLS, encrypted storage, and auditable access controls before handling real candidate data.

### Realtime status and next steps

Sasha uses browser speech recognition for an incremental transcript and sends the completed answer over a WebSocket. Candidates can optionally enable auto-send after 1.8 seconds without a transcript update; it is off by default. The server acknowledges processing, rejects overlapping WebSocket turns, reports per-turn processing time, and accepts an interrupt while the turn task is awaiting work. In the default LLM path, the server streams partial next-question text to the UI while it collects the remaining assessment JSON; question speech starts only after the full decision arrives. The browser reports last-turn latency and session p50/p95 over up to 20 scored turns. These are in-session diagnostics, not a controlled performance benchmark.

The app still does not stream raw microphone audio to the server or stream synthesized reply audio. Only candidate-facing question text is progressively displayed; assessments and speech playback wait for the complete interviewer decision. Agentic resume-search mode also uses a non-streaming path. Cancelling a Python task cannot stop a provider request already running in a worker thread. Browser speech recognition and speech synthesis support vary by browser and operating system.

Latency work in the current path avoids loading DistilGPT-2 at startup and runs that signal only for candidates who opt into monitoring. For those sessions, the model loads on the first opted-in answer and its inference adds processing time. Repeated resume embeddings are cached with a bounded cache. Perplexity remains an experimental review signal and is not proof of AI use.

The next real-time milestones are server-side or provider-backed streaming ASR with a replaceable adapter; incremental interviewer text and audio playback; cancellation that stops provider generation and drains playback; and repeatable TTFR/p50/p95 benchmarks under warm and cold conditions. Gesture realism stays a separate layer after those milestones: add a small number of avatar expressions and gestures synchronized to speaking/listening/acknowledgment states. Keep avatar animation separate from candidate emotion inference.

The [Real-Time Conversational AI Commentator](https://github.com/sushant-mishra-dtu/Real-Time-Conversational-AI-Commentator) is an architecture reference for server-owned state, VAD, audio queues, interruption handling, model adapters, and race testing. Its performance claims have not been independently measured as part of this project.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author & Maintainer

Created by **Vaibhav Mittal** ([@Vabs1002](https://github.com/Vabs1002)).  
Contributions, issues, and feature requests are welcome. Feel free to open a pull request or star the repository!
