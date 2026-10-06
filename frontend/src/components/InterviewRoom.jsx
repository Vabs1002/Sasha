import React, { useState, useEffect, useRef } from 'react';
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
  Sparkles
} from 'lucide-react';
import SashaPresence from './SashaPresence';
import { createInterviewWebSocket, submitTurn } from '../api';

export default function InterviewRoom({ sessionData, onCompleteSession }) {
  const { session_id, candidate_name, role, experience_level, starting_difficulty, first_question } = sessionData;

  // Conversational State
  const [sashaState, setSashaState] = useState('speaking'); // 'idle' | 'listening' | 'thinking' | 'speaking'
  const [currentQuestion, setCurrentQuestion] = useState(first_question || 'Welcome to your interview. Let us begin.');
  const [turnNumber, setTurnNumber] = useState(1);
  const [maxTurns] = useState(8);
  const [difficultyLabel, setDifficultyLabel] = useState(starting_difficulty || 'Standard');
  const [nudgeMessage, setNudgeMessage] = useState(null);

  // Candidate Controls
  const [isMicMuted, setIsMicMuted] = useState(false);
  const [isCameraOff, setIsCameraOff] = useState(false);
  const [manualAnswerText, setManualAnswerText] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');

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
  const canvasRef = useRef(null);

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

  // Audio Playback Function (Zero-latency Web Speech with acoustic echo prevention)
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
    async function setupMedia() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 360, frameRate: 15 },
          audio: true,
        });
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
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // A. Immediate Barge-in acknowledgement
        if (data.type === 'interrupted') {
          isSpeakingRef.current = false;
          setSashaState('listening');
        }

        // B. Sasha Nudge (Integrity, Gaze, AI Script or Empathy)
        if (data.type === 'sasha_nudge' || data.type === 'empathy') {
          const text = data.text;
          setNudgeMessage(text);
          speakText(text);
          // Auto clear nudge toast after 6 seconds
          setTimeout(() => setNudgeMessage(null), 6000);
        }

        // C. Next Turn Result received
        if (data.type === 'turn_result') {
          if (data.is_complete) {
            onCompleteSession(session_id);
            return;
          }
          setCurrentQuestion(data.next_question);
          setTurnNumber(data.turn);
          if (data.difficulty_label) {
            setDifficultyLabel(data.difficulty_label);
          }
          speakText(data.next_question);
          setManualAnswerText('');
          setInterimTranscript('');
        }

        if (data.type === 'completed') {
          onCompleteSession(session_id);
        }
      } catch (err) {
        console.error('Error handling WebSocket message:', err);
      }
    };

    ws.onclose = () => console.log('WebSocket connection closed');

    // 3. Periodic Webcam Frame Capture (Reuses canvas, accesses synced refs without reconnecting WS)
    frameIntervalRef.current = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN && videoRef.current && !isCameraOffRef.current) {
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
  }, [session_id]);

  // 4. Handle Submitting Answer Turn
  const handleAnswerSubmit = async () => {
    const answer = (manualAnswerText + ' ' + interimTranscript).trim();
    if (!answer) return;

    setSashaState('thinking');
    setInterimTranscript('');

    try {
      // Send via WebSocket if open, fallback to HTTP REST
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'answer', text: answer }));
      } else {
        const result = await submitTurn(session_id, answer);
        if (result.is_complete) {
          onCompleteSession(session_id);
          return;
        }
        setCurrentQuestion(result.next_question);
        setTurnNumber(result.turn_number);
        setDifficultyLabel(result.difficulty_label);
        setSashaState('speaking');
        setManualAnswerText('');
      }
    } catch (err) {
      console.error('Submit turn error:', err);
      setSashaState('listening');
    }
  };

  // 5. Trigger Instant Barge-In
  const triggerBargeIn = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    isSpeakingRef.current = false;
    setSashaState('listening');
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'interrupt' }));
    }
  };

  // Mic Toggle
  const toggleMic = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getAudioTracks().forEach((t) => (t.enabled = isMicMuted));
      setIsMicMuted(!isMicMuted);
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

          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-xs font-mono text-emerald-400">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Anti-Cheat Proctored</span>
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
              <span className="text-sky-400 font-semibold">{turnNumber} / {maxTurns}</span>
            </div>
            
            <div className="w-full h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
              <div 
                className="h-full bg-sky-400 transition-all duration-500 rounded-full"
                style={{ width: `${(turnNumber / maxTurns) * 100}%` }}
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
              <span>Full-Duplex Barge-In</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
              You can speak or interrupt Sasha at any moment. The system detects your voice and yields immediately (&lt;50ms).
            </p>
            {sashaState === 'speaking' && (
              <button
                onClick={triggerBargeIn}
                className="mt-3 w-full py-1.5 px-3 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border border-amber-500/30 text-xs font-mono transition-colors flex items-center justify-center gap-1.5"
              >
                <Zap className="w-3 h-3" />
                <span>Interrupt Sasha & Answer</span>
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
          {nudgeMessage && (
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
                Turn {turnNumber} Technical Probe
              </p>
              <h3 className="text-base sm:text-lg font-medium text-slate-100 leading-relaxed">
                "{currentQuestion}"
              </h3>
            </div>

            {/* Answer Typing / Speech Box */}
            <div className="relative flex items-center">
              <input
                type="text"
                value={manualAnswerText}
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
          </div>
        </main>

        {/* Right Column: Candidate Feed, VU Meter & Transcript (3 cols) */}
        <aside className="col-span-3 flex flex-col gap-4 overflow-y-auto pl-1">
          {/* Webcam Card */}
          <div className="glass-panel rounded-2xl p-3 border border-white/[0.07] relative">
            <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-2 px-1">
              <span>Candidate Feed</span>
              <span className="text-emerald-400 text-[10px] uppercase">Gaze Sync Active</span>
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
                <span>30 FPS</span>
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
          Sasha Autonomous Session • Biometric Anti-Cheat Loop Active
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
            onClick={() => onCompleteSession(session_id)}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/30 text-rose-300 transition-colors"
          >
            <PhoneOff className="w-4 h-4" />
            <span>End Interview</span>
          </button>
        </div>

        <div className="text-xs font-mono text-slate-500 hidden sm:block">
          NYC Local Law 144
        </div>
      </footer>
    </div>
  );
}
