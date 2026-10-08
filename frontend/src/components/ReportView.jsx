import React, { useState, useEffect } from 'react';
import {
  FileText,
  Download,
  RotateCcw,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  Award,
  Layers,
  Clock,
  ExternalLink
} from 'lucide-react';
import { deleteInterviewEvidence, getInterviewEvidence, getReportDownloadUrl, getSessionStatus, resolveApiUrl } from '../api';

export default function ReportView({ sessionId, onRestart }) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [evidence, setEvidence] = useState([]);
  const [evidenceEnabled, setEvidenceEnabled] = useState(false);

  useEffect(() => {
    async function loadStatus() {
      const [statusResult, evidenceResult] = await Promise.allSettled([
        getSessionStatus(sessionId),
        getInterviewEvidence(sessionId),
      ]);
      if (statusResult.status === 'fulfilled') setStatus(statusResult.value);
      else console.error('Failed to load session status:', statusResult.reason);
      if (evidenceResult.status === 'fulfilled') {
        setEvidence(evidenceResult.value.records || []);
        setEvidenceEnabled(evidenceResult.value.enabled === true);
      }
      setLoading(false);
    }
    loadStatus();
  }, [sessionId]);

  const pdfUrl = getReportDownloadUrl(sessionId);

  const removeEvidence = async (captureId) => {
    try {
      await deleteInterviewEvidence(sessionId, captureId);
      setEvidence((items) => items.filter((item) => item.capture_id !== captureId));
    } catch (err) {
      console.error('Failed to delete evidence:', err);
    }
  };

  return (
    <div className="min-h-screen py-12 px-6 max-w-4xl mx-auto flex flex-col justify-between">
      {/* Top Header */}
      <header className="flex items-center justify-between border-b border-white/[0.07] pb-6">
        <div>
          <span className="text-[10px] uppercase font-mono tracking-widest text-sky-400">
            Executive Debrief Complete
          </span>
          <h1 className="text-2xl sm:text-3xl font-semibold text-white mt-1">
            Technical Evaluation Debrief
          </h1>
        </div>

        <button
          onClick={onRestart}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-mono bg-white/[0.04] hover:bg-white/[0.08] text-slate-300 border border-white/[0.08] transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>New Session</span>
        </button>
      </header>

      {/* Main Content */}
      <main className="my-8 space-y-6">
        {/* Candidate Dossier Overview Banner */}
        <div className="glass-panel rounded-2xl p-6 border border-white/[0.07] grid grid-cols-1 md:grid-cols-4 gap-6">
          <div>
            <p className="text-[10px] uppercase font-mono tracking-wider text-slate-400">Candidate</p>
            <p className="text-base font-semibold text-white mt-1">{status?.candidate_name || 'Candidate'}</p>
          </div>

          <div>
            <p className="text-[10px] uppercase font-mono tracking-wider text-slate-400">Evaluated Role</p>
            <p className="text-base font-semibold text-sky-400 mt-1">{status?.role || 'SDE'}</p>
          </div>

          <div>
            <p className="text-[10px] uppercase font-mono tracking-wider text-slate-400">Turns Completed</p>
            <p className="text-base font-semibold text-slate-200 mt-1">
              {status?.turns_completed || 8} of {status?.max_turns || 8}
            </p>
          </div>

          <div>
            <p className="text-[10px] uppercase font-mono tracking-wider text-slate-400">Peak IRT Difficulty</p>
            <span className="inline-block px-2.5 py-0.5 mt-1 rounded-md bg-violet-500/10 border border-violet-500/30 text-violet-300 font-mono text-xs font-medium uppercase">
              {status?.current_difficulty || 'Advanced'}
            </span>
          </div>
        </div>

        {/* Executive Signal Callout */}
        <div className="rounded-2xl p-6 bg-gradient-to-r from-sky-500/[0.07] to-violet-500/[0.07] border border-sky-400/20 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-5">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-xl bg-sky-500/10 border border-sky-400/30 flex items-center justify-center shrink-0 text-sky-400">
              <Award className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-white">Full Debrief Compiled</h3>
              <p className="text-xs text-slate-300 mt-1 leading-relaxed max-w-lg">
                The comprehensive PDF report includes qualitative domain breakdowns, 
                adaptive difficulty calibration curves, and verbatim turn-by-turn evidence logs.
              </p>
            </div>
          </div>

          <a
            href={pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-5 py-3 rounded-xl bg-sky-400 hover:bg-sky-300 text-slate-950 text-xs font-semibold shadow-[0_0_25px_rgba(56,189,248,0.3)] transition-all shrink-0"
          >
            <Download className="w-4 h-4" />
            <span>Download PDF Debrief</span>
          </a>
        </div>

        {evidenceEnabled && (
          <section className="glass-panel rounded-2xl p-6 border border-amber-400/20 space-y-4">
            <div>
              <h2 className="text-base font-semibold text-white">Camera event evidence</h2>
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                {evidence.length > 0
                  ? 'These files were saved after repeated camera alerts with candidate opt-in. They show limited camera context and may have innocent explanations; they do not establish cheating. Review alongside the interview and give the candidate an opportunity to respond. Files expire 30 days after capture.'
                  : 'No camera evidence was saved during this session. If a repeated camera alert occurs while the camera is available, the opted-in capture stores a still and a short clip.'}
              </p>
            </div>
            <div className="space-y-5">
              {evidence.map((item) => (
                <article key={item.capture_id} className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-4">
                  <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                    <span className="text-sm font-medium text-amber-200">{item.signal.replaceAll('_', ' ')}</span>
                    <div className="flex items-center gap-3">
                      <time className="text-[11px] font-mono text-slate-500">Alert: {new Date(item.alert_at).toLocaleString()}</time>
                      <button onClick={() => removeEvidence(item.capture_id)} className="text-[11px] text-rose-300 hover:text-rose-200">Delete evidence</button>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {item.assets.snapshot && (
                      <a href={resolveApiUrl(item.assets.snapshot)} target="_blank" rel="noopener noreferrer" className="block">
                        <img src={resolveApiUrl(item.assets.snapshot)} alt={`Camera still for ${item.signal.replaceAll('_', ' ')} alert`} className="w-full rounded-lg border border-white/10" />
                        <span className="block mt-2 text-xs text-sky-300">Open still image</span>
                      </a>
                    )}
                    {item.assets.clip && (
                      <div>
                        <video src={resolveApiUrl(item.assets.clip)} controls preload="metadata" className="w-full rounded-lg border border-white/10" />
                        <a href={resolveApiUrl(item.assets.clip)} download className="block mt-2 text-xs text-sky-300">Download 5-second camera-only clip</a>
                      </div>
                    )}
                  </div>
                </article>
              ))}
            </div>
          </section>
        )}

        {/* Methodological Details Card */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] space-y-3">
            <div className="flex items-center gap-2 text-xs font-mono text-slate-200">
              <Layers className="w-4 h-4 text-sky-400" />
              <span className="font-semibold uppercase tracking-wider">Evaluation Methodology</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Every score in this debrief is grounded in concrete statements extracted from the candidate's answers. 
              Sasha does not evaluate personality, native accent, or superficial fluency.
            </p>
          </div>

          <div className="glass-panel rounded-2xl p-5 border border-white/[0.07] space-y-3">
            <div className="flex items-center gap-2 text-xs font-mono text-slate-200">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span className="font-semibold uppercase tracking-wider">Human review</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Camera alerts are imperfect signals, not findings of misconduct. A reviewer should consider the full context, avoid making an adverse decision from an automated alert alone, and allow the candidate to explain.
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/[0.07] pt-6 flex items-center justify-between text-xs font-mono text-slate-500">
        <span>Sasha Autonomous Evaluation Engine v2.0</span>
        <span>Executive Debrief Session ID: {sessionId}</span>
      </footer>
    </div>
  );
}
