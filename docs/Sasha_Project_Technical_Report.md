# Sasha Project Technical Report

**Purpose:** Explain what the Sasha prototype does, how its technical flow works, what the current evidence signals can and cannot say, and what remains before production use.  
**Status date:** October 2026

## Executive summary

Sasha is a web-based practice interviewer that uses a candidate's resume and optional job description to guide a technical interview. Candidates can answer by voice or text. A FastAPI service manages sessions and turns, an OpenAI-compatible language-model endpoint generates follow-up questions and rubric estimates, and the app can produce a PDF/HTML debrief.

The candidate experience now opens with a separate, unscored profile confirmation. Sasha summarizes the inferred role and experience, explains the supportive practice format, and lets candidates correct the profile. For the default LLM path, the next question's text is progressively shown while the rest of the decision is generated. Candidates can interrupt speech or cancel a pending WebSocket turn.

This is a functional prototype, not a validated hiring instrument or a full-duplex voice agent. The browser does not stream raw microphone audio to the server; the app waits for a complete answer before model reasoning. Only next-question text is progressively displayed; speech playback waits for the complete decision. Optional integrity checks are imperfect review cues. They do not prove cheating or identify use of ChatGPT, Parakeet AI, another extension, a phone, or a second device.

## Product behavior and candidate journey

1. Candidate uploads a PDF or DOCX resume and may provide a job description.
2. Candidate separately opts into limited integrity monitoring and, if desired, camera evidence capture.
3. Sasha presents the resume-derived role and experience, explains that the interview is supportive practice, invites correction, and asks the candidate to confirm or correct the profile. This setup reply is not scored as an interview turn.
4. Sasha asks about a resume project as the first scored prompt. Later questions use interview history, the competency bank, answer analysis, candidate-confirmed context, and the supplied job-description summary.
5. Candidate answers through browser speech recognition or text input. They can submit manually or turn on optional auto-send after a pause. They can stop speech or cancel a pending turn.
6. Sasha displays the result and may show a limited, dismissible review notice. The debrief contains turn-by-turn answers, generated rubric estimates, and available signal context.

The opening says candidates can take their time, think aloud, and ask for a question to be repeated or clarified. It says they should answer in their own words and that Sasha can clarify a question but will not provide its solution. It also explains that enabled monitoring is made of imperfect review signals, not proof. This does not replace an employer's published interview policy or accessibility process.

## Architecture

| Layer | Main implementation | Responsibility |
|---|---|---|
| Candidate interface | React 19, Vite, browser Web Speech API and MediaRecorder | Resume/job input, consent, interview controls, webcam preview, transcript and response display |
| HTTP and event API | FastAPI, Pydantic, WebSocket | Session lifecycle, answer submission, event routing, evidence and report endpoints |
| Turn orchestration | `api.py`, asyncio and worker threads | Turn state, concurrent analysis, model request, cancellation boundary, report data |
| Resume and role context | `resume_parser.py`, `signal_detector.py`, `analyzer.py` | Resume extraction, broad role inference, answer/resume comparison and cached embeddings |
| Question and assessment | `interviewer_agent.py`, `competency_bank.yaml` | Competency choice, LLM decision, follow-up question and rubric estimates |
| Review signals | `signal_detector.py`, `analyzer.py`, `ProctorState` | Optional camera, browser, speech and answer-pattern cues |
| Debrief | `report_generator.py` | PDF/HTML summary with answers, generated estimates and signal context |

The main browser app is served by FastAPI in the unified configuration. The standalone terminal `realtime_session.py` demo is separate from the browser interview path.

### Main scored-turn flow

1. Browser speech recognition incrementally creates a transcript. The completed answer is submitted through the WebSocket.
2. Server immediately reports processing and rejects an overlapping turn.
3. Speech/text checks and resume-consistency work run in worker threads. If the candidate opted into integrity monitoring, the server also computes the experimental DistilGPT-2 perplexity signal.
4. The default LLM path streams its response internally. Sasha forwards only decoded pieces of the `next_question` string while collecting the full JSON decision and assessment.
5. The complete JSON is parsed, the answer and decision provenance are saved, and the turn result is sent to the browser. Browser speech synthesis starts after this complete result.
6. On interrupt, the server cancels the WebSocket coroutine and suppresses stale UI results. A provider request already running in a Python worker thread may continue and consume provider resources.

Candidate profile corrections are stored in the session and appended to later LLM context. The raw opening response is not treated as an assessed interview answer. Subsequent scored prompts receive the optional job-description summary.

## Real-time status

| Area | Current behavior | Not currently supported |
|---|---|---|
| Speech input | Browser speech recognition produces an incremental transcript; completed answers go to the API. | Streaming raw microphone audio to FastAPI or Gemini |
| Interviewer text | The default LLM path progressively displays the next-question text. | Progressive assessment JSON; streaming is disabled in agentic resume-search mode |
| Voice output | Browser speech synthesis begins after the complete decision arrives. | Model-generated streaming audio |
| Interruption | Candidate can stop speech or cancel the pending WebSocket coroutine. | Hard cancellation of a provider request already executing in a worker thread |
| Latency | UI displays last response time and session p50/p95 over up to 20 scored turns. | A controlled benchmark, performance service-level objective, or guarantee |

The accurate description is **turn-based voice input with progressive question text**, not full-duplex real-time conversation. Browser speech recognition and synthesis support varies by browser, OS, language, and microphone setup.

## Integrity signals and evidence

Signals are presented as unverified context. Candidate opt-in controls monitoring; a separate opt-in controls camera evidence. No single signal should be treated as a cheating finding or used as the sole basis for a hiring outcome.

| Signal | What the prototype may observe | What it cannot establish |
|---|---|---|
| Camera | Face not visible, multiple faces in view, rough head/gaze direction, possible audio/mouth-motion mismatch | What the candidate looked at, identity, intent, whether a phone is present, or whether someone helped |
| Browser context | Page hidden/visible, window blur/focus, paste into the answer field | Which app/site was opened, extension activity, clipboard source, another device, or AI use |
| Speech/text patterns | Some direct requests for Sasha to provide an answer, plus experimental phrasing/disfluency patterns | Whether an answer was AI-written, copied, memorized, rehearsed, or unaided |
| Signal timing | Different signal types occurring close together can be shown as context | A validated combined probability or stronger proof |
| Evidence files | With separate consent, repeated camera alerts can save a reduced-size still and a short camera-only clip when browser support permits | A causal record of cheating; the camera may not contain relevant context |

The prototype has no phone/object detector and no reliable way to detect third-party extensions such as Parakeet AI. A browser page-focus event does not reveal which application gained focus. Camera cues do not identify a second speaker or person.

### DistilGPT-2 perplexity behavior

DistilGPT-2 is not loaded at service startup. When a candidate opts into integrity monitoring, the first scored answer triggers lazy model loading and every opted-in scored answer receives a perplexity calculation. This can add cold-start and per-answer latency. If monitoring is declined, this model is not invoked. The signal has not been validated as an AI-use detector and does not prove that AI was used.

## Data handling and security posture

| Data | Current use and protection | Production gap |
|---|---|---|
| Resume and job description | Parsed and held in session memory as generation context | Sessions are lost at server restart; limit collected/stored fields and access |
| Answers and assessments | Saved in in-memory session history and used for question selection and report output | Need authenticated access, encryption, audited access, deletion, and documented retention |
| Browser/camera signals | Collected only after monitoring consent; evidence requires separate consent | Need deployment-specific privacy, accessibility, consent and legal review |
| Evidence/reports | Written outside the static frontend tree; no-store headers and 30-day cleanup are configured | Session IDs are bearer capabilities; no full recruiter identity authorization or encrypted storage |
| LLM requests | Resume/role context and answer are sent to the configured provider | Data handling depends on provider/account configuration and must be reviewed before real candidate use |

Private storage and cleanup are prototype hardening, not production security. Before processing real candidate data, add identity-based access control, TLS, encryption at rest, audited report/evidence access, a deletion workflow, rate/abuse controls, and clear retention/consent policies. Do not use real candidate information until those controls and provider data terms have been reviewed.

## Verification and performance

| Check | Result | What it tells us |
|---|---|---|
| Full Python test suite | **85 passed**, 13 warnings, 83.32 seconds | Deterministic behavior and integration paths pass; this does not validate detector accuracy or interview fairness |
| Targeted API/interviewer tests | **17 passed**, 38.90 seconds | Includes profile setup, mock model streaming, WebSocket behavior, consent-gated perplexity and cancellation |
| Frontend production build | **Passed**, Vite transformed 1,901 modules | Frontend compiles; does not verify browser media permissions on different devices |
| Python compilation | **Passed** for API, interviewer, analyzer and CLI session modules | Syntax/bytecode compilation only |
| `git diff --check` | **Passed** | Whitespace and conflict-marker check only |
| Live synthetic Gemini streaming smoke | **Passed**; first question text about 1.49 s, complete decision about 1.80 s | One call including cold local embedding-model initialization; not a benchmark or latency guarantee |

The Gemini timing came from a single synthetic call using the currently configured model. Earlier one-off observations ranged from roughly 6 to 11 seconds under different conditions; a previous warm mocked run was roughly 72 ms. These numbers are not directly comparable because the test setup differed. A repeatable benchmark should separate cold model loading, provider first token, full decision time, and speech playback start.

Warnings from the full suite include existing upstream deprecations and test/runtime warnings. They did not fail the suite.

## Known gaps and next priorities

1. **Measure baseline reproducibly.** Track first question text, full decision, voice start, p50/p95, cold/warm local-model startup, and provider errors. Separate consent-on and consent-off sessions.
2. **Add real audio streaming.** Use provider-backed streaming ASR or an adapter with server-side VAD and incremental output. Preserve swappable hosted/local model options.
3. **Improve cancellation.** Use provider-native stream cancellation and drain audio playback; test disconnect/interrupt races and stale responses.
4. **Evaluate question quality.** Build a role/JD test set and human rubric review for relevance, grounding, correction handling, accessibility, and consistency.
5. **Study signal validity.** Measure false positives across accents, disabilities, lighting, camera placement, and answer styles. Keep integrity signals separate from scored competency ratings.
6. **Harden deployment security.** Add authentication, authorization, encryption, audited access, retention/deletion enforcement, threat modeling, and privacy review.
7. **Add gesture realism later.** Synchronize a small set of avatar speaking/listening/acknowledgment gestures after audio timing is stable. Do not infer candidate emotion from facial movement.

## Accurate resume description

> Built Sasha, a resume- and job-description-informed interview practice prototype with a React/FastAPI WebSocket flow, adaptive LLM follow-up questions, progressive question-text display, interruptible turn handling, session latency diagnostics, separately consented experimental review signals, and PDF/HTML debriefs. Validity, fairness, and production security remain unvalidated.

Avoid describing the project as a proven anti-cheating system, autonomous hiring evaluator, full-duplex real-time voice agent, extension detector, emotion reader, or production-secure proctoring platform. The current code does not support those claims.

## Key implementation files

| File | Responsibility |
|---|---|
| `api.py` | Session setup, REST/WebSocket APIs, turn processing, consent, evidence and report endpoints |
| `interviewer_agent.py` | Prompt construction, competency choice, LLM streaming, fallback and decision source |
| `analyzer.py` | Embeddings, linguistic checks and experimental perplexity computation |
| `signal_detector.py` | Broad role classification and answer-pattern review cues |
| `frontend/src/components/LobbyView.jsx` | Resume/job input and consent controls |
| `frontend/src/components/InterviewRoom.jsx` | Interview UI, browser media, WebSocket events, evidence upload and latency display |
| `report_generator.py` | PDF/HTML debrief generation |
| `README.md` | Setup instructions, limitations, realtime status and roadmap |
