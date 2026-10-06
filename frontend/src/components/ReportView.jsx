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
import { getReportDownloadUrl, getSessionStatus } from '../api';

export default function ReportView({ sessionId, onRestart }) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadStatus() {
      try {
        const data = await getSessionStatus(sessionId);
        setStatus(data);
      } catch (err) {
        console.error('Failed to load session status:', err);
      } finally {
        setLoading(false);
      }
    }
    loadStatus();
  }, [sessionId]);

  const pdfUrl = getReportDownloadUrl(sessionId);

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
              <span className="font-semibold uppercase tracking-wider">NYC Local Law 144 Disclosures</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              In accordance with Automated Employment Decision Tools (AEDT) requirements, this assessment 
              is audited for bias neutrality. Raw video/audio is processed ephemerally and not used for emotional profiling.
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
