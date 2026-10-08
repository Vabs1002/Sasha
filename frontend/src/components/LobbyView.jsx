import React, { useState, useRef } from 'react';
import { 
  Upload, 
  FileText, 
  CheckCircle2, 
  Sparkles, 
  ChevronDown, 
  ChevronUp, 
  ArrowRight,
  ShieldCheck,
  Cpu,
  Layers,
  AlertCircle
} from 'lucide-react';
import SashaPresence from './SashaPresence';

export default function LobbyView({ onStartSession, isStarting = false, serverOnline = true }) {
  const [resumeFile, setResumeFile] = useState(null);
  const [jdText, setJdText] = useState('');
  const [integrityMonitoringConsent, setIntegrityMonitoringConsent] = useState(false);
  const [evidenceCaptureConsent, setEvidenceCaptureConsent] = useState(false);
  const [showJd, setShowJd] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    setErrorMessage(null);

    const file = e.dataTransfer.files[0];
    validateAndSetFile(file);
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      validateAndSetFile(file);
    }
  };

  const validateAndSetFile = (file) => {
    if (!file) return;
    const ext = file.name.split('.').pop().toLowerCase();
    if (ext !== 'pdf' && ext !== 'docx') {
      setErrorMessage('Please upload a PDF or DOCX resume document.');
      return;
    }
    setResumeFile(file);
    setErrorMessage(null);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!resumeFile) {
      setErrorMessage('Please upload your resume to begin.');
      return;
    }
    onStartSession({ resumeFile, jdText, integrityMonitoringConsent, evidenceCaptureConsent });
  };

  return (
    <div className="min-h-screen flex flex-col justify-between py-12 px-6 max-w-5xl mx-auto">
      {/* Top Bar Navigation */}
      <header className="flex items-center justify-between border-b border-white/[0.07] pb-6">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-sky-500/10 border border-sky-400/30 flex items-center justify-center">
            <span className="font-mono text-xs font-bold text-sky-400">S</span>
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-wide text-slate-100 flex items-center gap-2">
              SASHA
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-white/[0.06] text-slate-400">
                v2.0
              </span>
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.03] border border-white/[0.06]">
            <span className={`w-2 h-2 rounded-full ${serverOnline ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
            <span className="text-slate-300">{serverOnline ? 'BACKEND READY' : 'BACKEND OFFLINE (PORT 8000)'}</span>
          </div>
          <span className="text-slate-500 hidden sm:inline">INTERVIEW PROTOTYPE</span>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="my-auto py-10 flex flex-col items-center text-center">
        {/* Subtle Sasha Aura in Lobby */}
        <div className="mb-8 cursor-default">
          <SashaPresence state="idle" size={170} />
        </div>

        {/* Headline */}
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-400/20 text-sky-400 text-xs font-mono tracking-wider uppercase mb-5">
          <Sparkles className="w-3.5 h-3.5" />
          Autonomous Technical Interviewer
        </div>

        <h2 className="text-4xl sm:text-5xl font-semibold tracking-tight text-white max-w-2xl leading-[1.15]">
          Direct technical signal, zero performance anxiety.
        </h2>

        <p className="mt-4 text-slate-400 max-w-lg text-sm sm:text-base leading-relaxed">
          Sasha analyzes your background, retrieves your real project experiences via Hybrid RAG, 
          and calibrates technical probes live to your seniority.
        </p>

        {/* Upload Form Container */}
        <form onSubmit={handleSubmit} className="w-full max-w-md mt-10 space-y-4">
          {/* Drag & Drop Zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleFileDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`cursor-pointer relative rounded-2xl p-7 transition-all duration-300 text-center flex flex-col items-center justify-center border ${
              resumeFile
                ? 'bg-sky-500/[0.04] border-sky-400/50 shadow-[0_0_30px_rgba(56,189,248,0.1)]'
                : isDragging
                ? 'bg-sky-500/[0.08] border-sky-400/70 scale-[1.01]'
                : 'glass-panel hover:border-white/20'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx"
              onChange={handleFileSelect}
              className="hidden"
            />

            {resumeFile ? (
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center mb-3">
                  <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                </div>
                <p className="text-sm font-medium text-white max-w-[280px] truncate">{resumeFile.name}</p>
                <p className="text-xs text-slate-400 mt-1 font-mono">
                  {(resumeFile.size / 1024).toFixed(1)} KB • Click to change
                </p>
              </div>
            ) : (
              <div className="flex flex-col items-center">
                <div className="w-12 h-12 rounded-xl bg-white/[0.04] border border-white/[0.08] flex items-center justify-center mb-3 text-slate-400 group-hover:text-sky-400 transition-colors">
                  <Upload className="w-5 h-5 text-sky-400/80" />
                </div>
                <p className="text-sm font-medium text-slate-200">
                  Drop your resume here, or <span className="text-sky-400 underline underline-offset-4">browse</span>
                </p>
                <p className="text-xs text-slate-500 mt-1 font-mono">PDF or DOCX (Single or Multi-column supported)</p>
              </div>
            )}
          </div>

          {/* Optional JD Expandable */}
          <div className="text-left">
            <button
              type="button"
              onClick={() => setShowJd(!showJd)}
              className="text-xs font-mono text-slate-400 hover:text-slate-200 transition-colors flex items-center gap-1.5 py-1 px-1"
            >
              {showJd ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              {showJd ? 'Hide Job Description' : '+ Add Job Description (Optional for custom matching)'}
            </button>

            {showJd && (
              <div className="mt-2 transition-all">
                <textarea
                  value={jdText}
                  onChange={(e) => setJdText(e.target.value)}
                  placeholder="Paste target role requirements or job description here to have Sasha probe specific stack expectations..."
                  rows={4}
                  className="w-full text-xs font-mono rounded-xl p-3 bg-white/[0.03] border border-white/[0.08] text-slate-200 placeholder-slate-600 focus:outline-none focus:border-sky-400/50 transition-colors resize-none"
                />
              </div>
            )}
          </div>

          <div className="rounded-xl border border-white/[0.08] bg-white/[0.025] p-4 text-left">
            <label className="flex items-start gap-3 text-xs text-slate-200">
              <input
                type="checkbox"
                checked={integrityMonitoringConsent}
                onChange={(e) => {
                  setIntegrityMonitoringConsent(e.target.checked);
                  if (!e.target.checked) setEvidenceCaptureConsent(false);
                }}
                className="mt-0.5 accent-sky-400"
              />
              <span>
                <span className="font-medium">Opt in to limited integrity monitoring (optional)</span>
                <span className="mt-1 block leading-relaxed text-slate-400">
                  If enabled, Sasha records timestamps for when this page is hidden or loses focus, and when text is pasted into the answer box. It also sends reduced-size camera frames for face/gaze/mouth-movement checks and applies statistical heuristics to answer text for possible scripted responses. These checks can be wrong. Sasha cannot identify which website or extension was used, read other tabs or URLs, clipboard contents, screen contents, or activity on another device. Browser page/focus/paste events are never scored; camera and answer-text flags are unverified and may only prompt follow-up or human review. Leave this unchecked to disable these optional integrity checks.
                </span>
              </span>
            </label>
            <label className={`flex items-start gap-3 text-xs mt-4 ${integrityMonitoringConsent ? 'text-slate-200' : 'text-slate-500'}`}>
              <input
                type="checkbox"
                checked={evidenceCaptureConsent}
                disabled={!integrityMonitoringConsent}
                onChange={(e) => setEvidenceCaptureConsent(e.target.checked)}
                className="mt-0.5 accent-sky-400"
              />
              <span>
                <span className="font-medium">Separately opt in to saving camera evidence (optional)</span>
                <span className="mt-1 block leading-relaxed text-slate-400">
                  Only after a repeated camera alert, Sasha saves one reduced-size still and up to five seconds of camera-only video beginning at that alert. No audio is recorded. Evidence is stored privately for 30 days, then automatically deleted. Looking away, another face, or a camera mismatch can have innocent causes; evidence is for human review and does not establish cheating. No evidence is saved when this box is unchecked.
                </span>
              </span>
            </label>
          </div>

          {/* Error Banner if any */}
          {errorMessage && (
            <div className="flex items-center gap-2 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs text-left">
              <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
              <span>{errorMessage}</span>
            </div>
          )}

          {/* Action Button */}
          <button
            type="submit"
            disabled={!resumeFile || isStarting || !serverOnline}
            className={`w-full h-13 rounded-xl font-medium text-sm transition-all duration-200 flex items-center justify-center gap-2 ${
              !resumeFile || !serverOnline
                ? 'bg-white/[0.04] text-slate-500 cursor-not-allowed border border-white/[0.05]'
                : isStarting
                ? 'bg-sky-500 text-slate-950 font-semibold animate-pulse'
                : 'bg-sky-400 hover:bg-sky-300 text-slate-950 font-semibold shadow-[0_0_25px_rgba(56,189,248,0.25)] hover:shadow-[0_0_35px_rgba(56,189,248,0.4)] scale-[1.0] hover:scale-[1.01] active:scale-[0.99]'
            }`}
          >
            {isStarting ? (
              <span>Initializing Interview Brain...</span>
            ) : (
              <>
                <span>Enter Interview with Sasha</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </form>
      </main>

      {/* Footer Feature Pillars */}
      <footer className="pt-8 border-t border-white/[0.07] grid grid-cols-1 md:grid-cols-3 gap-6 text-left">
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg bg-white/[0.03] border border-white/[0.06] text-sky-400 shrink-0">
            <Layers className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-semibold text-slate-200 uppercase font-mono tracking-wider">Hybrid RAG Grounding</h3>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Questions are anchored directly to your verified projects using in-memory FAISS & BM25 retrieval.
            </p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg bg-white/[0.03] border border-white/[0.06] text-violet-400 shrink-0">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-semibold text-slate-200 uppercase font-mono tracking-wider">Adaptive IRT Calibration</h3>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Difficulty recalibrates ±0.15 dynamically based on architectural depth and spontaneous consistency.
            </p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg bg-white/[0.03] border border-white/[0.06] text-emerald-400 shrink-0">
            <ShieldCheck className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-semibold text-slate-200 uppercase font-mono tracking-wider">Experimental review signals</h3>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Camera and answer-pattern checks can be wrong. They do not prove cheating or determine a hiring decision.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
