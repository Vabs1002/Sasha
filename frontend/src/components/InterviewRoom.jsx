import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Mic,
  MicOff,
  Video,
  VideoOff,
  PhoneOff,
  ShieldCheck,
  Zap,
  TrendingUp,
  Volume2,
  AlertTriangle,
  Send,
  CornerDownLeft,
  Sparkles,
  X
} from 'lucide-react';
import SashaPresence from './SashaPresence';
import { createInterviewWebSocket, submitTurn, uploadInterviewEvidence } from '../api';

export default function InterviewRoom({ sessionData, onCompleteSession }) {
  const { session_id, candidate_name, role, experience_level, starting_difficulty, first_question } = sessionData;
  const integrityMonitoringEnabled = sessionData.integrity_monitoring_consent === true;
  const evidenceCaptureEnabled = sessionData.evidence_capture_consent === true;
  const [profileCheckPending, setProfileCheckPending] = useState(sessionData.profile_confirmation_required === true);

  // Conversational State
  const [sashaState, setSashaState] = useState('speaking'); // 'idle' | 'listening' | 'thinking' | 'speaking'
  const [currentQuestion, setCurrentQuestion] = useState(first_question || 'Welcome to your interview. Let us begin.');
  const [streamedQuestionDraft, setStreamedQuestionDraft] = useState('');
  const [turnNumber, setTurnNumber] = useState(1);
  const [maxTurns] = useState(8);
  const [difficultyLabel, setDifficultyLabel] = useState(starting_difficulty || 'Standard');
  const [nudgeMessage, setNudgeMessage] = useState(null);
  const [reviewSignal, setReviewSignal] = useState(null);
  const [latencyMetrics, setLatencyMetrics] = useState(null);

  // Candidate Controls
  const [isMicMuted, setIsMicMuted] = useState(false);
  const [isCameraOff, setIsCameraOff] = useState(false);
  const [manualAnswerText, setManualAnswerText] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [autoSendAfterPause, setAutoSendAfterPause] = useState(false);

  // Audio & Video Refs
  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const [vuLevels, setVuLevels] = useState(() => new Array(28).fill(4));
  const wsRef = useRef(null);
  const frameIntervalRef = useRef(null);
  const recognitionRef = useRef(null);
  const recognitionRestartTimerRef = useRef(null);
  const recognitionDisposedRef = useRef(false);
  const canvasRef = useRef(null);
  const pendingIntegrityEventsRef = useRef([]);
  const pendingEvidenceUploadsRef = useRef(new Set());
  const sessionCompletingRef = useRef(false);
  const recentReviewSignalsRef = useRef([]);
  const turnStartedAtRef = useRef(null);
  const latencySamplesRef = useRef([]);
  const activeSubmissionRef = useRef(false);

  const cameraSignalDetails = {
    gaze_away: ['Camera check: gaze direction', 'Repeated frames estimated an off-center gaze or head direction. Looking away can have ordinary causes; the camera cannot tell what you were looking at.'],
    no_face: ['Camera check: face not visible', 'Repeated frames could not detect a face. This can happen because of lighting, framing, or camera quality.'],
    multiple_faces: ['Camera check: more than one face', 'Repeated frames detected more than one face in view. This does not identify anyone or show that anyone helped.'],
    proxy_speaker: ['Camera/audio check: speaking mismatch', 'The check saw a possible mismatch between audio activity and mouth movement. It can be wrong and does not identify another speaker.'],
  };
  const answerSignalLabels = {
    critically_low_perplexity: 'unusually predictable word patterns',
    low_perplexity: 'predictable word patterns',
    unnaturally_fluent_script_reading: 'few speech disfluencies in a long answer',
    minimal_spontaneous_disfluency: 'few speech disfluencies',
    written_llm_discourse_markers: 'formal, written-style phrases',
    formal_literary_syntax: 'formal, written-style phrasing',
    sudden_turn_over_turn_fluency_leap: 'a change in measured fluency from an earlier answer',
  };
  const makeAnswerReviewSignal = (isAssistanceRequest, signalTypes = []) => {
    if (isAssistanceRequest) {
      return {
        title: 'Review signal: direct request for an answer',
        detail: 'The transcript matched wording that asks Sasha to provide an answer or solution. Speech recognition can be mistaken, and this does not establish that outside help was used. You can clarify your intent with Sasha.',
      };
    }
    const labels = [...new Set(signalTypes.map((type) => answerSignalLabels[type]).filter(Boolean))];
    return {
      title: 'Review signal: answer-text pattern',
      detail: `${labels.length ? `This response matched: ${labels.join('; ')}. ` : ''}These text and speech patterns cannot identify an AI tool or prove that AI was used. A structured or prepared answer may also trigger a flag.`,
    };
  };

  const publishReviewSignal = (source, signal) => {
    const now = Date.now();
    const relatedPrior = recentReviewSignalsRef.current.filter((item) => now - item.at <= 60_000 && item.source !== source);
    const current = { source, signal, at: now };
    if (relatedPrior.length) {
      const related = [...relatedPrior, current];
      recentReviewSignalsRef.current = related;
      setReviewSignal({
        title: 'Related review signals occurred close together',
        detail: `${related.map((item) => `${item.signal.title} — ${item.signal.detail}`).join(' ')} This timing is context only; it is not a combined cheating score or proof.`,
      });
    } else {
      recentReviewSignalsRef.current = [current];
      setReviewSignal(signal);
    }
  };

  const completeAfterEvidence = async () => {
    if (sessionCompletingRef.current) return;
    sessionCompletingRef.current = true;
    await Promise.allSettled(Array.from(pendingEvidenceUploadsRef.current));
    onCompleteSession(session_id);
  };

  const captureEvidence = (signal) => {
    if (!evidenceCaptureEnabled || !['gaze_away', 'no_face', 'multiple_faces', 'proxy_speaker'].includes(signal)) return;
    const video = videoRef.current;
    if (!video || isCameraOffRef.current || video.readyState < 2) return;

    const snapshotPromise = new Promise((resolve) => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = 320;
        canvas.height = 180;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(resolve, 'image/jpeg', 0.65);
      } catch (_) { resolve(null); }
    });

    const task = (async () => {
      let clipBlob = null;
      const videoTracks = mediaStreamRef.current?.getVideoTracks().filter((track) => track.readyState === 'live' && track.enabled) || [];
      const MediaRecorderClass = window.MediaRecorder;
      if (MediaRecorderClass && videoTracks.length) {
        let frameTimer = null;
        let evidenceStream = null;
        try {
          const evidenceCanvas = document.createElement('canvas');
          evidenceCanvas.width = 320;
          evidenceCanvas.height = 180;
          const evidenceContext = evidenceCanvas.getContext('2d');
          const drawEvidenceFrame = () => {
            if (video.readyState >= 2) evidenceContext.drawImage(video, 0, 0, 320, 180);
          };
          drawEvidenceFrame();
          frameTimer = window.setInterval(drawEvidenceFrame, 100);
          evidenceStream = evidenceCanvas.captureStream(10);
          const mimeType = ['video/webm;codecs=vp8', 'video/webm', 'video/mp4'].find((type) => MediaRecorderClass.isTypeSupported(type));
          const recorder = mimeType
            ? new MediaRecorderClass(evidenceStream, { mimeType })
            : new MediaRecorderClass(evidenceStream);
          const chunks = [];
          clipBlob = await new Promise((resolve) => {
            recorder.ondataavailable = (event) => { if (event.data?.size) chunks.push(event.data); };
            recorder.onerror = () => {
              window.clearInterval(frameTimer);
              evidenceStream.getTracks().forEach((track) => track.stop());
              resolve(null);
            };
            recorder.onstop = () => {
              window.clearInterval(frameTimer);
              evidenceStream.getTracks().forEach((track) => track.stop());
              resolve(chunks.length ? new Blob(chunks, { type: recorder.mimeType || 'video/webm' }) : null);
            };
            recorder.start();
            window.setTimeout(() => {
              if (recorder.state !== 'inactive') recorder.stop();
            }, 5000);
          });
        } catch (error) {
          if (frameTimer) window.clearInterval(frameTimer);
          evidenceStream?.getTracks().forEach((track) => track.stop());
          console.warn('Short camera evidence recording was unavailable:', error);
        }
      }
      const snapshot = await snapshotPromise;
      if (!snapshot && !clipBlob) return;
      try {
        await uploadInterviewEvidence(session_id, signal, snapshot, clipBlob);
      } catch (error) {
        console.warn('Could not upload opted-in camera evidence:', error);
      }
    })();
    pendingEvidenceUploadsRef.current.add(task);
    task.finally(() => pendingEvidenceUploadsRef.current.delete(task));
  };

  const recordIntegrityEvent = useCallback((eventName) => {
    if (!integrityMonitoringEnabled) return;
    const ws = wsRef.current;
    const message = JSON.stringify({ type: 'integrity_event', event: eventName });
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(message);
    } else if (ws?.readyState === WebSocket.CONNECTING && pendingIntegrityEventsRef.current.length < 50) {
      pendingIntegrityEventsRef.current.push(message);
    }
  }, [integrityMonitoringEnabled]);

  // Synchronization refs to eliminate WebSocket reconnection churn
  const isSpeakingRef = useRef(false);
  const isCameraOffRef = useRef(isCameraOff);
  const isMicMutedRef = useRef(isMicMuted);
  const interimTranscriptRef = useRef(interimTranscript);
  const sashaStateRef = useRef(sashaState);

  useEffect(() => { isCameraOffRef.current = isCameraOff; }, [isCameraOff]);
  useEffect(() => { isMicMutedRef.current = isMicMuted; }, [isMicMuted]);
  useEffect(() => { interimTranscriptRef.current = interimTranscript; }, [interimTranscript]);
  useEffect(() => { sashaStateRef.current = sashaState; }, [sashaState]);

  // Browser speech playback with a short echo guard before candidate listening resumes.
  const speakText = (textToSpeak) => {
    if (!textToSpeak || !textToSpeak.trim()) return;

    // Immediately stop any active speech
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }

    // Set speaking flag to mute STT microphone input from transcribing Sasha's own voice
    isSpeakingRef.current = true;
    setSashaState('speaking');

    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(textToSpeak.trim());
      utterance.rate = 1.0;
      utterance.pitch = 1.04;
      const voices = window.speechSynthesis.getVoices();
      const preferredVoice = voices.find(
        (v) => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha') || v.name.includes('Zira') || v.name.includes('Jenny'))
      );
      if (preferredVoice) utterance.voice = preferredVoice;

      utterance.onstart = () => {
        isSpeakingRef.current = true;
        setSashaState('speaking');
      };

      const handleSpeechComplete = () => {
        // Echo buffer: wait 400ms for speaker audio reverb to clear before accepting mic input
        setTimeout(() => {
          isSpeakingRef.current = false;
          setSashaState('listening');
        }, 400);
      };

      utterance.onend = handleSpeechComplete;
      utterance.onerror = handleSpeechComplete;
      window.speechSynthesis.speak(utterance);
    } else {
      isSpeakingRef.current = false;
      setSashaState('listening');
    }
  };

  // Speak initial question on load
  useEffect(() => {
    if (first_question) {
      const timer = setTimeout(() => {
        speakText(first_question);
      }, 700);
      return () => clearTimeout(timer);
    }
  }, []);

  // 1. Initialize Webcam & Audio Stream
  useEffect(() => {
    let stream = null;
    let disposed = false;
    recognitionDisposedRef.current = false;
    async function setupMedia() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 360, frameRate: 15 },
          audio: true,
        });
        if (disposed) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        mediaStreamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

        // Web Audio API Analyzer for VU Waveform
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (AudioCtx) {
          const audioCtx = new AudioCtx();
          audioContextRef.current = audioCtx;
          const source = audioCtx.createMediaStreamSource(stream);
          const analyser = audioCtx.createAnalyser();
          analyser.fftSize = 64;
          source.connect(analyser);
          analyserRef.current = analyser;

          const dataArray = new Uint8Array(analyser.frequencyBinCount);
          const updateAudioVisualizer = () => {
            if (analyserRef.current && !isMicMuted) {
              analyserRef.current.getByteFrequencyData(dataArray);
              const bars = Array.from({ length: 28 }, (_, i) => {
                const val = dataArray[i % dataArray.length] || 0;
                return Math.max(3, Math.round((val / 255) * 32));
              });
              setVuLevels(bars);
            }
            animationFrameRef.current = requestAnimationFrame(updateAudioVisualizer);
          };
          updateAudioVisualizer();
        }

        // Setup Speech Recognition (Browser Native STT)
        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRec) {
          const rec = new SpeechRec();
          rec.continuous = true;
          rec.interimResults = true;
          rec.lang = 'en-US';

          rec.onresult = (event) => {
            // Discard microphone audio while Sasha is speaking to prevent acoustic feedback/self-transcription
            if (isSpeakingRef.current) {
              return;
            }

            let interim = '';
            let final = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
              const transcript = event.results[i][0].transcript;
              if (event.results[i].isFinal) {
                final += transcript + ' ';
              } else {
                interim += transcript;
              }
            }
            if (final) {
              setManualAnswerText((prev) => (prev ? `${prev} ${final.trim()}` : final.trim()));
            }
            setInterimTranscript(interim);
          };

          rec.onerror = (e) => console.debug('Speech recognition event:', e);
          rec.onend = () => {
            if (recognitionDisposedRef.current || isMicMutedRef.current) return;
            window.clearTimeout(recognitionRestartTimerRef.current);
            recognitionRestartTimerRef.current = window.setTimeout(() => {
              if (recognitionDisposedRef.current || isMicMutedRef.current) return;
              try { rec.start(); } catch (_) { /* already running */ }
            }, 250);
          };
          try {
            rec.start();
            recognitionRef.current = rec;
          } catch (e) {
            console.debug('Recognition already running');
          }
        }
      } catch (err) {
        console.warn('Camera/Microphone access was denied or unavailable:', err);
      }
    }

    setupMedia();

    return () => {
      disposed = true;
      recognitionDisposedRef.current = true;
      window.clearTimeout(recognitionRestartTimerRef.current);
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      if (audioContextRef.current) {
        audioContextRef.current.close().catch(() => {});
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (recognitionRef.current) {
        try { recognitionRef.current.stop(); } catch (e) {}
      }
    };
  }, []);

  // 2. Setup WebSocket Connection with FastAPI (Mounted once per session)
  useEffect(() => {
    const ws = createInterviewWebSocket(session_id);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Sasha WebSocket live connected:', session_id);
      pendingIntegrityEventsRef.current.splice(0).forEach((message) => ws.send(message));
    };

    ws.onmessage = async (event) => {
      try {
        const data = JSON.parse(event.data);

        // A. Immediate Barge-in acknowledgement
        if (data.type === 'interrupted') {
          activeSubmissionRef.current = false;
          setStreamedQuestionDraft('');
          turnStartedAtRef.current = null;
          isSpeakingRef.current = false;
          setSashaState('listening');
        }

        if (data.type === 'turn_processing') {
          setSashaState('thinking');
        }

        if (data.type === 'turn_error' || data.type === 'turn_busy') {
          activeSubmissionRef.current = false;
          setStreamedQuestionDraft('');
          turnStartedAtRef.current = null;
          setAutoSendAfterPause(false);
          setSashaState('listening');
          setNudgeMessage(data.message || 'Please wait for the current response to finish.');
        }

        if (data.type === 'question_delta' && typeof data.text === 'string') {
          setStreamedQuestionDraft((current) => current + data.text);
        }

        // B. Sasha Nudge (Integrity, Gaze, AI Script or Empathy)
        if (data.type === 'sasha_nudge' || data.type === 'empathy') {
          if (data.type === 'sasha_nudge' && data.signal && cameraSignalDetails[data.signal]) {
            captureEvidence(data.signal);
            publishReviewSignal('camera', { title: cameraSignalDetails[data.signal][0], detail: cameraSignalDetails[data.signal][1] });
          }
          const text = data.text;
          setNudgeMessage(text);
          speakText(text);
          // Auto clear nudge toast after 6 seconds
          setTimeout(() => setNudgeMessage(null), 6000);
        }

        // C. Next Turn Result received
        if (data.type === 'turn_result') {
          activeSubmissionRef.current = false;
          setStreamedQuestionDraft('');
          if (data.profile_confirmed) setProfileCheckPending(false);
          if (!data.profile_confirmed && turnStartedAtRef.current !== null) {
            const roundTripMs = Math.round(performance.now() - turnStartedAtRef.current);
            turnStartedAtRef.current = null;
            const samples = [...latencySamplesRef.current, roundTripMs].slice(-20);
            latencySamplesRef.current = samples;
            const ordered = [...samples].sort((a, b) => a - b);
            const percentile = (fraction) => ordered[Math.max(0, Math.ceil(ordered.length * fraction) - 1)];
            setLatencyMetrics({
              lastMs: roundTripMs,
              serverMs: data.processing_ms,
              p50Ms: percentile(0.5),
              p95Ms: percentile(0.95),
              count: ordered.length,
            });
          }
          if (data.assistance_request_detected || data.ai_script_detected) {
            publishReviewSignal('answer', makeAnswerReviewSignal(data.assistance_request_detected, data.ai_signal_types || []));
          }
          if (data.is_complete) {
            await completeAfterEvidence();
            return;
          }
          setCurrentQuestion(data.next_question);
          if (data.turn > 0) setTurnNumber(data.turn);
          if (data.difficulty_label) {
            setDifficultyLabel(data.difficulty_label);
          }
          speakText(data.next_question);
          setManualAnswerText('');
          setInterimTranscript('');
        }

        if (data.type === 'completed') {
          await completeAfterEvidence();
        }
      } catch (err) {
        console.error('Error handling WebSocket message:', err);
      }
    };

    ws.onclose = () => console.log('WebSocket connection closed');

    // 3. Periodic Webcam Frame Capture (Reuses canvas, accesses synced refs without reconnecting WS)
    frameIntervalRef.current = setInterval(() => {
      if (integrityMonitoringEnabled && ws.readyState === WebSocket.OPEN && videoRef.current && !isCameraOffRef.current) {
        try {
          if (!canvasRef.current) {
            canvasRef.current = document.createElement('canvas');
            canvasRef.current.width = 320;
            canvasRef.current.height = 180;
          }
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
          const base64Data = canvas.toDataURL('image/jpeg', 0.6).split(',')[1];

          if (base64Data) {
            ws.send(
              JSON.stringify({
                type: 'frame',
                data: base64Data,
                audio_active: !isMicMutedRef.current && interimTranscriptRef.current.length > 0,
              })
            );
          }
        } catch (e) {
          // ignore frame capture error
        }
      }
    }, 1500);

    return () => {
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
      if (ws) ws.close();
    };
  }, [session_id, integrityMonitoringEnabled, evidenceCaptureEnabled]);

  useEffect(() => {
    if (!integrityMonitoringEnabled) return undefined;

    const handleVisibilityChange = () => {
      recordIntegrityEvent(document.hidden ? 'page_hidden' : 'page_visible');
    };
    const handleBlur = () => recordIntegrityEvent('window_blur');
    const handleFocus = () => recordIntegrityEvent('window_focus');

    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('blur', handleBlur);
    window.addEventListener('focus', handleFocus);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('blur', handleBlur);
      window.removeEventListener('focus', handleFocus);
    };
  }, [session_id, integrityMonitoringEnabled, recordIntegrityEvent]);

  // 4. Handle Submitting Answer Turn
  const handleAnswerSubmit = async () => {
    const answer = (manualAnswerText + ' ' + interimTranscript).trim();
    if (!answer || activeSubmissionRef.current) return;
    activeSubmissionRef.current = true;

    setSashaState('thinking');
    setInterimTranscript('');

    try {
      // Send via WebSocket if open, fallback to HTTP REST
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        turnStartedAtRef.current = performance.now();
        wsRef.current.send(JSON.stringify({ type: 'answer', text: answer }));
      } else {
        const requestStartedAt = performance.now();
        const result = await submitTurn(session_id, answer);
        activeSubmissionRef.current = false;
        const roundTripMs = Math.round(performance.now() - requestStartedAt);
        const samples = [...latencySamplesRef.current, roundTripMs].slice(-20);
        latencySamplesRef.current = samples;
        const ordered = [...samples].sort((a, b) => a - b);
        const percentile = (fraction) => ordered[Math.max(0, Math.ceil(ordered.length * fraction) - 1)];
        setLatencyMetrics({ lastMs: roundTripMs, p50Ms: percentile(0.5), p95Ms: percentile(0.95), count: ordered.length });
        if (result.assistance_request_detected || result.ai_script_detected) {
          publishReviewSignal('answer', makeAnswerReviewSignal(result.assistance_request_detected, result.ai_signal_types || []));
        }
        if (result.profile_confirmed) setProfileCheckPending(false);
        if (result.is_complete) {
          await completeAfterEvidence();
          return;
        }
        setCurrentQuestion(result.next_question);
        if (result.turn_number > 0) setTurnNumber(result.turn_number);
        setDifficultyLabel(result.difficulty_label);
        setSashaState('speaking');
        speakText(result.next_question);
        setManualAnswerText('');
        setInterimTranscript('');
      }
    } catch (err) {
      console.error('Submit turn error:', err);
      activeSubmissionRef.current = false;
      turnStartedAtRef.current = null;
      setAutoSendAfterPause(false);
      setSashaState('listening');
    }
  };

  // Optional hands-free turn submission. Keep it off by default so candidates
  // stay in control of when their answer is sent.
  useEffect(() => {
    if (
      !autoSendAfterPause ||
      sashaState !== 'listening' ||
      (!manualAnswerText.trim() && !interimTranscript.trim())
    ) return undefined;

    const timer = window.setTimeout(() => {
      if (!isSpeakingRef.current) handleAnswerSubmit();
    }, 1800);
    return () => window.clearTimeout(timer);
  }, [autoSendAfterPause, manualAnswerText, interimTranscript, sashaState]);

  // 5. Trigger Instant Barge-In
  const triggerBargeIn = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    isSpeakingRef.current = false;
    activeSubmissionRef.current = false;
    turnStartedAtRef.current = null;
    setSashaState('listening');
    setManualAnswerText('');
    setInterimTranscript('');
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'interrupt' }));
    }
  };

  // Mic Toggle
  const toggleMic = () => {
    if (mediaStreamRef.current) {
      const nextMuted = !isMicMuted;
      mediaStreamRef.current.getAudioTracks().forEach((track) => (track.enabled = !nextMuted));
      setIsMicMuted(nextMuted);
      if (nextMuted) {
        window.clearTimeout(recognitionRestartTimerRef.current);
        try { recognitionRef.current?.stop(); } catch (_) { /* recognition already stopped */ }
      } else {
        try { recognitionRef.current?.start(); } catch (_) { /* recognition already running */ }
      }
    }
  };

  // Camera Toggle
  const toggleCamera = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getVideoTracks().forEach((t) => (t.enabled = isCameraOff));
      setIsCameraOff(!isCameraOff);
    }
  };

  return (
    <div className="h-screen w-screen flex flex-col justify-between bg-[#06080d] text-slate-100 overflow-hidden font-['Inter',sans-serif]">
      {/* Studio Header Bar */}
      <header className="h-14 border-b border-white/[0.07] px-6 flex items-center justify-between bg-[#0a0d15]/80 backdrop-blur-md shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="font-mono text-xs uppercase tracking-wider text-slate-300">Live Evaluation</span>
          </div>
          <span className="text-slate-600">|</span>
          <span className="text-xs text-slate-400 font-mono">Session ID: {session_id}</span>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-white/[0.04] border border-white/[0.07] text-xs font-mono">
            <TrendingUp className="w-3.5 h-3.5 text-violet-400" />
            <span className="text-slate-400">IRT Calibrated:</span>
            <span className="text-violet-300 font-medium uppercase">{difficultyLabel}</span>
          </div>

          <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-mono ${integrityMonitoringEnabled ? 'bg-amber-500/10 border-amber-500/20 text-amber-300' : 'bg-slate-500/10 border-slate-500/20 text-slate-400'}`}>
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>{integrityMonitoringEnabled ? 'Opt-in review signals' : 'Monitoring off'}</span>
          </div>
        </div>
      </header>

      {/* Main Studio 3-Column Grid */}
      <div className="flex-1 grid grid-cols-12 gap-5 p-5 min-h-0 overflow-hidden">
        {/* Left Column: Candidate & Telemetry Dossier (3 cols) */}
        <aside className="col-span-3 flex flex-col gap-4 overflow-y-auto pr-1">
          {/* Candidate Card */}
          <div className="glass-panel rounded-2xl p-4 border border-white/[0.07]">
            <p className="text-[10px] uppercase font-mono tracking-widest text-slate-400">Candidate Profile</p>
            <h2 className="text-lg font-semibold text-white mt-1">{candidate_name || 'Candidate'}</h2>
            <p className="text-xs text-sky-400 font-medium mt-0.5">{role || 'Software Engineer'}</p>
            
            <div className="mt-4 pt-3 border-t border-white/[0.06] flex items-center justify-between text-xs font-mono text-slate-400">
              <span>Experience Tier</span>
              <span className="text-slate-200 capitalize">{experience_level || 'Mid-Level'}</span>
            </div>
          </div>

          {/* Progress Tracker Card */}
          <div className="glass-panel rounded-2xl p-4 border border-white/[0.07]">
            <div className="flex items-center justify-between text-xs font-mono mb-2">
              <span className="text-slate-400">Turn Progress</span>
              <span className="text-sky-400 font-semibold">{profileCheckPending ? 'Profile check' : `${turnNumber} / ${maxTurns}`}</span>
            </div>
            
            <div className="w-full h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
              <div 
                className="h-full bg-sky-400 transition-all duration-500 rounded-full"
                style={{ width: `${(profileCheckPending ? 0 : (turnNumber / maxTurns) * 100)}%` }}
              />
            </div>

            <p className="text-[11px] text-slate-500 mt-3 font-mono leading-relaxed">
              Sasha adapts technical depth live. Every 3 turns, she shifts to high-impact behavioral scenarios.
            </p>
          </div>

          {/* Barge-In Instant Interrupt Card */}
          <div className="glass-panel-subtle rounded-2xl p-4 border border-white/[0.06] text-left">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              <span>Live Turn Control</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
              Stop Sasha while she is speaking or processing, then continue with another answer.
            </p>
            {latencyMetrics && (
              <p className="mt-2 text-[10px] text-slate-500 font-mono" aria-live="polite">
                Last response {latencyMetrics.lastMs} ms
                {Number.isFinite(latencyMetrics.serverMs) ? ` (server ${latencyMetrics.serverMs} ms)` : ''}
                {' · '}session p50 {latencyMetrics.p50Ms} ms / p95 {latencyMetrics.p95Ms} ms ({latencyMetrics.count} turns)
              </p>
            )}
            {(sashaState === 'speaking' || sashaState === 'thinking') && (
              <button
                onClick={triggerBargeIn}
                className="mt-3 w-full py-1.5 px-3 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border border-amber-500/30 text-xs font-mono transition-colors flex items-center justify-center gap-1.5"
              >
                <Zap className="w-3 h-3" />
                <span>{sashaState === 'thinking' ? 'Cancel Current Turn' : 'Stop Sasha Speaking'}</span>
              </button>
            )}
          </div>
        </aside>

        {/* Center Column: Sasha's Visual Presence & Question (6 cols) */}
        <main className="col-span-6 flex flex-col justify-between items-center glass-panel rounded-2xl p-6 relative overflow-hidden border border-white/[0.07]">
          {/* Top Stage Tag */}
          <div className="w-full flex items-center justify-between border-b border-white/[0.05] pb-3 text-xs font-mono">
            <span className="text-slate-400 tracking-wider">AGENT PRESENCE</span>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${
                sashaState === 'speaking' ? 'bg-sky-400 animate-pulse' :
                sashaState === 'thinking' ? 'bg-violet-400 animate-spin' :
                'bg-emerald-400'
              }`} />
              <span className="text-slate-300 uppercase tracking-wider text-[11px]">
                {sashaState === 'speaking' ? 'Sasha Speaking' :
                 sashaState === 'thinking' ? 'Thinking & Analyzing' :
                 sashaState === 'listening' ? 'Listening...' : 'Active'}
              </span>
            </div>
          </div>

          {/* Floating Constructive Nudge Banner (If any) */}
          {reviewSignal ? (
            <div role="status" aria-live="polite" className="absolute top-14 left-6 right-6 z-20 flex items-start gap-2.5 p-2.5 rounded-lg bg-rose-950/95 border border-rose-500/50 text-rose-100 text-[11px] shadow-lg animate-in fade-in slide-in-from-top-2 duration-200">
              <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400 mt-0.5" />
              <div className="min-w-0 flex-1">
                <p className="font-semibold">{reviewSignal.title}</p>
                <p className="mt-0.5 leading-relaxed text-rose-100/90">{reviewSignal.detail}</p>
                <p className="mt-1 text-rose-200/70">A review signal is not a finding of cheating.</p>
              </div>
              <button type="button" aria-label="Dismiss review signal" title="Dismiss" onClick={() => { setReviewSignal(null); setNudgeMessage(null); }} className="shrink-0 p-1 rounded hover:bg-rose-800/70 text-rose-200">
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          ) : nudgeMessage && (
            <div className="absolute top-14 left-6 right-6 z-20 flex items-center gap-3 p-3.5 rounded-xl bg-amber-500/15 border border-amber-400/30 text-amber-200 text-xs shadow-lg animate-in fade-in slide-in-from-top-3 duration-300">
              <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400" />
              <span className="leading-relaxed">{nudgeMessage}</span>
            </div>
          )}

          {/* The Hero Entity: Sasha */}
          <div className="my-auto flex flex-col items-center">
            <SashaPresence state={sashaState} size={210} />
          </div>

          {/* Spoken Question Box */}
          <div className="w-full max-w-xl text-center space-y-4">
            <div className="p-4 rounded-xl bg-white/[0.02] border border-white/[0.05]">
              <p className="text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-1">
                {profileCheckPending ? 'Quick profile check' : `Turn ${turnNumber} Technical Probe`}
              </p>
              <h3 className="text-base sm:text-lg font-medium text-slate-100 leading-relaxed">
                "{currentQuestion}"
              </h3>
            </div>
            {streamedQuestionDraft && (
              <div className="p-3 rounded-xl bg-violet-500/[0.06] border border-violet-400/[0.12] text-left" role="status" aria-live="polite">
                <p className="text-[10px] font-mono uppercase tracking-widest text-violet-300/70 mb-1">
                  Sasha is forming the next question
                </p>
                <p className="text-sm text-violet-100/90 leading-relaxed">{streamedQuestionDraft}</p>
              </div>
            )}

            {/* Answer Typing / Speech Box */}
            <div className="relative flex items-center">
              <input
                type="text"
                value={manualAnswerText}
                onPaste={() => recordIntegrityEvent('answer_paste')}
                onChange={(e) => setManualAnswerText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleAnswerSubmit();
                }}
                placeholder={
                  interimTranscript 
                    ? `Heard: "${interimTranscript}"...` 
                    : sashaState === 'thinking'
                    ? 'Sasha is synthesizing next probe...'
                    : 'Speak via microphone or type your technical answer here...'
                }
                disabled={sashaState === 'thinking'}
                className="w-full bg-white/[0.04] border border-white/[0.08] focus:border-sky-400/50 rounded-xl px-4 py-3 text-xs sm:text-sm text-slate-200 placeholder-slate-500 focus:outline-none transition-colors pr-24"
              />
              <button
                type="button"
                onClick={handleAnswerSubmit}
                disabled={sashaState === 'thinking' || (!manualAnswerText.trim() && !interimTranscript.trim())}
                className="absolute right-2 px-3 py-1.5 rounded-lg bg-sky-400 hover:bg-sky-300 text-slate-950 text-xs font-semibold disabled:opacity-30 disabled:cursor-not-allowed transition-all flex items-center gap-1 shadow-sm"
              >
                <span>Submit</span>
                <CornerDownLeft className="w-3.5 h-3.5" />
              </button>
            </div>
            <label className="mt-2 flex items-center justify-center gap-2 text-[11px] text-slate-400 select-none">
              <input
                type="checkbox"
                checked={autoSendAfterPause}
                onChange={(event) => setAutoSendAfterPause(event.target.checked)}
                disabled={sashaState === 'thinking'}
                className="accent-sky-400"
              />
              Auto-send after 1.8 seconds of silence
            </label>
          </div>
        </main>

        {/* Right Column: Candidate Feed, VU Meter & Transcript (3 cols) */}
        <aside className="col-span-3 flex flex-col gap-4 overflow-y-auto pl-1">
          {/* Webcam Card */}
          <div className="glass-panel rounded-2xl p-3 border border-white/[0.07] relative">
            <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-2 px-1">
              <span>Candidate Feed</span>
              <span className="text-slate-400 text-[10px] uppercase">{integrityMonitoringEnabled ? 'Opt-in camera signals' : 'Preview only'}</span>
            </div>

            <div className="relative rounded-xl overflow-hidden aspect-video bg-black/50 border border-white/[0.08]">
              {isCameraOff ? (
                <div className="w-full h-full flex flex-col items-center justify-center text-slate-500 gap-2">
                  <VideoOff className="w-6 h-6" />
                  <span className="text-[11px] font-mono">Camera Disabled</span>
                </div>
              ) : (
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover transform -scale-x-100"
                />
              )}

              {/* Live Status Overlay */}
              <div className="absolute top-2 left-2 flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-black/60 backdrop-blur-sm text-[10px] font-mono text-slate-300">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                <span>Live preview</span>
              </div>
            </div>
          </div>

          {/* Live Mic Waveform Card */}
          <div className="glass-panel rounded-2xl p-4 border border-white/[0.07]">
            <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-3">
              <span className="flex items-center gap-1.5">
                <Volume2 className="w-3.5 h-3.5 text-sky-400" />
                Microphone Input
              </span>
              <span className="text-[10px] text-slate-500">{isMicMuted ? 'Muted' : 'Active'}</span>
            </div>

            {/* Waveform Bars */}
            <div className="flex items-end justify-between h-9 px-1 gap-1">
              {vuLevels.map((height, i) => (
                <div
                  key={i}
                  className="w-1.5 bg-sky-400/80 rounded-full transition-all duration-75"
                  style={{
                    height: `${height}px`,
                    opacity: isMicMuted ? 0.2 : 0.6 + (height / 32) * 0.4,
                  }}
                />
              ))}
            </div>
          </div>

          {/* Live Real-time STT Transcript */}
          <div className="glass-panel rounded-2xl p-4 border border-white/[0.07] flex-1 flex flex-col min-h-[140px]">
            <p className="text-[10px] font-mono uppercase tracking-widest text-slate-400 mb-2">Live Transcript</p>
            <div className="flex-1 overflow-y-auto text-xs text-slate-300 leading-relaxed font-mono space-y-1">
              {manualAnswerText && (
                <p className="text-slate-200">{manualAnswerText}</p>
              )}
              {interimTranscript && (
                <p className="text-sky-400/90 italic">{interimTranscript} ...</p>
              )}
              {!manualAnswerText && !interimTranscript && (
                <p className="text-slate-600 italic">Waiting for speech input...</p>
              )}
            </div>
          </div>
        </aside>
      </div>

      {/* Bottom Bar Studio Controls */}
      <footer className="h-16 border-t border-white/[0.07] px-6 flex items-center justify-between bg-[#0a0d15]/80 backdrop-blur-md shrink-0">
        <div className="text-xs font-mono text-slate-500 hidden sm:block">
          Practice interview • Review signals {integrityMonitoringEnabled ? 'On' : 'Off'}
        </div>

        <div className="flex items-center gap-3 mx-auto sm:mx-0">
          {/* Mute Mic Button */}
          <button
            onClick={toggleMic}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium border transition-colors ${
              isMicMuted
                ? 'bg-rose-500/20 border-rose-500/40 text-rose-300'
                : 'glass-panel hover:bg-white/[0.05] text-slate-300'
            }`}
          >
            {isMicMuted ? <MicOff className="w-4 h-4 text-rose-400" /> : <Mic className="w-4 h-4" />}
            <span>{isMicMuted ? 'Unmute Mic' : 'Mute Mic'}</span>
          </button>

          {/* Toggle Camera Button */}
          <button
            onClick={toggleCamera}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium border transition-colors ${
              isCameraOff
                ? 'bg-rose-500/20 border-rose-500/40 text-rose-300'
                : 'glass-panel hover:bg-white/[0.05] text-slate-300'
            }`}
          >
            {isCameraOff ? <VideoOff className="w-4 h-4 text-rose-400" /> : <Video className="w-4 h-4" />}
            <span>{isCameraOff ? 'Enable Camera' : 'Disable Camera'}</span>
          </button>

          {/* End Session Button */}
          <button
            onClick={completeAfterEvidence}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/30 text-rose-300 transition-colors"
          >
            <PhoneOff className="w-4 h-4" />
            <span>End Interview</span>
          </button>
        </div>

        <div className="text-xs font-mono text-slate-500 hidden sm:block">
          Experimental prototype
        </div>
      </footer>
    </div>
  );
}
