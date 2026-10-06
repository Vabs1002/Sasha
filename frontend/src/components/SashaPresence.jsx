import React, { useEffect, useRef, useState } from 'react';

/**
 * SashaPresence — Original design inspired by tactile plush character aesthetics.
 * Clean, professional, built from scratch. No emojis. No slop.
 *
 * Three carefully crafted personas:
 * 1. Pearl   — Soft ivory plush, the default warm interviewer look
 * 2. Indigo  — Deep cool-violet with studio headphones, focused technical mode
 * 3. Onyx    — Charcoal-slate with minimal accent, executive authority mode
 */

const PERSONAS = [
  {
    id: 'pearl',
    name: 'Pearl',
    swatch: '#f1f5f9',        // color of the selector dot
    bodyStops: ['#ffffff', '#f1f5f9', '#dde3ec', '#c4cdd8', '#8fa0b2'],
    glow: (state) =>
      state === 'thinking' ? 'rgba(139, 92, 246, 0.38)' :
      state === 'listening' ? 'rgba(56, 189, 248, 0.5)' :
      state === 'speaking' ? 'rgba(244, 114, 182, 0.44)' :
      'rgba(251, 191, 36, 0.35)',
    hasHeadphones: false,
    hasVisor: false,
  },
  {
    id: 'indigo',
    name: 'Indigo',
    swatch: '#6d28d9',
    bodyStops: ['#c4b5fd', '#8b5cf6', '#6d28d9', '#4c1d95', '#2e1065'],
    glow: (state) =>
      state === 'thinking' ? 'rgba(139, 92, 246, 0.55)' :
      state === 'listening' ? 'rgba(99, 102, 241, 0.55)' :
      state === 'speaking' ? 'rgba(167, 139, 250, 0.5)' :
      'rgba(109, 40, 217, 0.4)',
    hasHeadphones: true,
    hasVisor: false,
  },
  {
    id: 'onyx',
    name: 'Onyx',
    swatch: '#334155',
    bodyStops: ['#64748b', '#475569', '#334155', '#1e293b', '#0f172a'],
    glow: (state) =>
      state === 'thinking' ? 'rgba(100, 116, 139, 0.45)' :
      state === 'listening' ? 'rgba(148, 163, 184, 0.4)' :
      state === 'speaking' ? 'rgba(203, 213, 225, 0.38)' :
      'rgba(71, 85, 105, 0.38)',
    hasHeadphones: false,
    hasVisor: true,
  },
];

export default function SashaPresence({
  state = 'idle',       // 'idle' | 'listening' | 'thinking' | 'speaking'
  isSpeakingAudio = false,
  size = 240,
  initialPersona = 'pearl',
  onPersonaChange,
}) {
  const containerRef = useRef(null);
  const [gazeOffset, setGazeOffset] = useState({ x: 0, y: 0 });
  const [isBlinking, setIsBlinking] = useState(false);
  const [audioBars, setAudioBars] = useState(() => new Array(24).fill(4));
  const [persona, setPersona] = useState(initialPersona);
  const [isSquishing, setIsSquishing] = useState(false);

  const isListening = state === 'listening';
  const isThinking = state === 'thinking';
  const isSpeaking = state === 'speaking' || isSpeakingAudio;

  const p = PERSONAS.find((x) => x.id === persona) || PERSONAS[0];

  // 1. Cursor gaze parallax
  useEffect(() => {
    const onMouseMove = (e) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dx = (e.clientX - cx) / (window.innerWidth / 2);
      const dy = (e.clientY - cy) / (window.innerHeight / 2);
      setGazeOffset({
        x: Math.max(-1, Math.min(1, dx)) * 11,
        y: Math.max(-1, Math.min(1, dy)) * 7,
      });
    };
    window.addEventListener('mousemove', onMouseMove, { passive: true });
    return () => window.removeEventListener('mousemove', onMouseMove);
  }, []);

  // 2. Organic randomised blinking
  useEffect(() => {
    let t;
    const blink = () => {
      setIsBlinking(true);
      setTimeout(() => setIsBlinking(false), 120);
      t = setTimeout(blink, 2800 + Math.random() * 2600);
    };
    t = setTimeout(blink, 3000);
    return () => clearTimeout(t);
  }, []);

  // 3. Audio-reactive waveform bars
  useEffect(() => {
    if (!isSpeaking) { setAudioBars(new Array(24).fill(4)); return; }
    let raf;
    const animate = () => {
      const t = Date.now() * 0.007;
      setAudioBars(Array.from({ length: 24 }, (_, i) => {
        const w = Math.sin(t + i * 0.45) * Math.cos(t * 0.85 - i * 0.3);
        return Math.max(3, Math.round(5 + Math.abs(w) * 18));
      }));
      raf = requestAnimationFrame(animate);
    };
    raf = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(raf);
  }, [isSpeaking]);

  const handleClick = () => {
    setIsSquishing(true);
    setTimeout(() => setIsSquishing(false), 320);
  };

  const handlePersona = (id) => {
    setPersona(id);
    if (onPersonaChange) onPersonaChange(id);
  };

  const groundGlowColor = p.glow(state);

  // Eye morphing per state
  let eyeScaleY = 1.0;
  let eyeScaleX = 1.0;
  if (isBlinking)      { eyeScaleY = 0.05; }
  else if (isThinking) { eyeScaleY = 0.42; eyeScaleX = 0.92; }
  else if (isListening){ eyeScaleY = 1.14; eyeScaleX = 1.06; }
  else if (isSpeaking) { eyeScaleY = 0.84; }

  const pupilX = isThinking ? gazeOffset.x * 0.5 + 4 : gazeOffset.x;
  const pupilY = isThinking ? -8 : gazeOffset.y;

  // Color stop ids are scoped to avoid collision when multiple instances exist
  const uid = persona;

  return (
    <div
      ref={containerRef}
      onClick={handleClick}
      className="relative flex flex-col items-center justify-center select-none cursor-pointer group"
      style={{ width: size, height: size }}
    >
      {/* Ground underglow */}
      <div
        className="absolute bottom-2 rounded-full pointer-events-none transition-all duration-700"
        style={{
          width: size * 0.82,
          height: size * 0.22,
          background: groundGlowColor,
          filter: 'blur(20px)',
          transform: isListening ? 'scale(1.18)' : 'scale(1)',
        }}
      />

      {/* Audio-reactive waveform halo — only when speaking */}
      {isSpeaking && (
        <div className="absolute inset-[-16px] pointer-events-none">
          <svg className="w-full h-full" viewBox="0 0 280 280">
            {audioBars.map((h, i) => {
              const angle = (i / 24) * 2 * Math.PI - Math.PI / 2;
              const ri = 120;
              const ro = ri + h;
              return (
                <line
                  key={i}
                  x1={140 + ri * Math.cos(angle)} y1={140 + ri * Math.sin(angle)}
                  x2={140 + ro * Math.cos(angle)} y2={140 + ro * Math.sin(angle)}
                  stroke={persona === 'indigo' ? '#a78bfa' : '#38bdf8'}
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  opacity={0.82}
                />
              );
            })}
          </svg>
        </div>
      )}

      {/* Thinking orbit ring */}
      {isThinking && (
        <div
          className="absolute inset-[-10px] rounded-full pointer-events-none animate-spin"
          style={{
            border: '2px solid transparent',
            borderTopColor: persona === 'indigo' ? '#a78bfa' : '#64748b',
            borderRightColor: persona === 'indigo' ? 'rgba(167,139,250,0.3)' : 'rgba(100,116,139,0.3)',
            animationDuration: '1.4s',
          }}
        />
      )}

      {/* Body */}
      <div
        className="relative w-full h-full flex items-center justify-center transition-transform duration-300 ease-out"
        style={{
          transform: isSquishing
            ? 'scale(1.11, 0.89) translateY(6px)'
            : isListening
            ? 'scale(1.02) translateY(-2px)'
            : 'scale(1)',
        }}
      >
        <svg viewBox="0 0 200 200" className="w-full h-full" style={{ overflow: 'visible', filter: 'drop-shadow(0 12px 24px rgba(0,0,0,0.45))' }}>
          <defs>
            {/* Body gradient — different per persona */}
            <radialGradient id={`body-${uid}`} cx="36%" cy="30%" r="65%">
              {p.bodyStops.map((c, i) => (
                <stop key={i} offset={`${[0, 30, 65, 88, 100][i]}%`} stopColor={c} />
              ))}
            </radialGradient>

            {/* Belly light bounce from floor */}
            <radialGradient id={`belly-${uid}`} cx="50%" cy="88%" r="42%">
              <stop offset="0%" stopColor={groundGlowColor} stopOpacity="0.55" />
              <stop offset="100%" stopColor="transparent" stopOpacity="0" />
            </radialGradient>

            {/* Glossy obsidian eye */}
            <radialGradient id="eyeGlass" cx="34%" cy="28%" r="70%">
              <stop offset="0%" stopColor="#1e293b" />
              <stop offset="58%" stopColor="#0a0f1e" />
              <stop offset="100%" stopColor="#020408" />
            </radialGradient>

            {/* Headphone dark metal */}
            <linearGradient id="phoneBand" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#0c0f1a" />
              <stop offset="50%" stopColor="#161b2e" />
              <stop offset="100%" stopColor="#0c0f1a" />
            </linearGradient>

            {/* Visor glass */}
            <linearGradient id="visorGlass" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#1e293b" />
              <stop offset="100%" stopColor="#0f172a" />
            </linearGradient>
          </defs>

          {/* ── Body silhouette — gentle plush blob squashed at base ── */}
          <path
            d="M 100,20
               C 157,20 190,62 190,115
               C 190,160 156,184 100,184
               C 44,184 10,160 10,115
               C 10,62 43,20 100,20 Z"
            fill={`url(#body-${uid})`}
          />

          {/* Belly floor-bounce light */}
          <path
            d="M 100,20 C 157,20 190,62 190,115 C 190,160 156,184 100,184 C 44,184 10,160 10,115 C 10,62 43,20 100,20 Z"
            fill={`url(#belly-${uid})`}
          />

          {/* Top specular sheen — gives the plush 3-D roundness */}
          <ellipse
            cx="74" cy="47" rx="33" ry="16"
            fill="#ffffff"
            opacity={persona === 'pearl' ? 0.46 : persona === 'onyx' ? 0.10 : 0.18}
            transform="rotate(-16 74 47)"
          />

          {/* ══════════════════ ACCESSORY LAYER ══════════════════ */}

          {/* INDIGO — Studio over-ear headphones */}
          {persona === 'indigo' && (
            <g>
              {/* Headband arch */}
              <path
                d="M 20,108 C 18,36 182,36 180,108"
                fill="none"
                stroke="url(#phoneBand)"
                strokeWidth="13"
                strokeLinecap="round"
              />
              {/* Subtle chrome highlight on band */}
              <path
                d="M 24,106 C 23,43 177,43 176,106"
                fill="none"
                stroke="#4c1d95"
                strokeWidth="2"
                opacity="0.65"
              />
              {/* LEFT earcup */}
              <g transform="translate(7, 92)">
                <rect x="0" y="0" width="19" height="36" rx="9.5" fill="#0f172a" />
                <rect x="1" y="1" width="17" height="34" rx="8.5" fill="none" stroke="#6d28d9" strokeWidth="1.5" />
                {/* Status LED */}
                <circle
                  cx="9.5" cy="18" r="4"
                  fill={isSpeaking ? '#38bdf8' : isListening ? '#34d399' : '#8b5cf6'}
                  opacity={isSpeaking || isListening ? 1 : 0.8}
                  className={isSpeaking || isListening ? 'animate-pulse' : ''}
                />
              </g>
              {/* RIGHT earcup */}
              <g transform="translate(174, 92)">
                <rect x="0" y="0" width="19" height="36" rx="9.5" fill="#0f172a" />
                <rect x="1" y="1" width="17" height="34" rx="8.5" fill="none" stroke="#6d28d9" strokeWidth="1.5" />
                <circle
                  cx="9.5" cy="18" r="4"
                  fill={isSpeaking ? '#38bdf8' : isListening ? '#34d399' : '#8b5cf6'}
                  opacity={isSpeaking || isListening ? 1 : 0.8}
                  className={isSpeaking || isListening ? 'animate-pulse' : ''}
                />
              </g>
            </g>
          )}

          {/* ONYX — Slim minimal visor bar (authority / executive look) */}
          {persona === 'onyx' && (
            <g>
              <rect
                x="33" y="96" width="134" height="38" rx="14"
                fill="url(#visorGlass)"
                stroke="#334155"
                strokeWidth="1.5"
              />
              {/* Two thin horizontal specular lines on visor glass */}
              <line x1="44" y1="107" x2="156" y2="107" stroke="rgba(255,255,255,0.08)" strokeWidth="2.5" strokeLinecap="round" />
              <line x1="52" y1="118" x2="148" y2="118" stroke="rgba(255,255,255,0.04)" strokeWidth="1.5" strokeLinecap="round" />
              {/* Small glare streak top-left corner */}
              <path d="M 42,100 L 54,125" stroke="rgba(255,255,255,0.22)" strokeWidth="3.5" strokeLinecap="round" />
            </g>
          )}

          {/* ══════════════════ EYES ══════════════════
               Rendered for Pearl & Indigo normally.
               Onyx uses the visor lens layer instead. */}
          {persona !== 'onyx' && (
            <g
              transform={`translate(${pupilX}, ${pupilY})`}
              className="transition-transform duration-100 ease-out"
            >
              {/* LEFT EYE */}
              <g
                transform={`translate(68, 115) scale(${eyeScaleX}, ${eyeScaleY})`}
                className="transition-transform duration-150 ease-out origin-center"
              >
                <ellipse cx="0" cy="0" rx="17" ry="22" fill="url(#eyeGlass)" />
                {/* Primary specular — upper-left key light */}
                <circle cx="-5.5" cy="-7" r="5.5" fill="#ffffff" />
                {/* Secondary sparkle — lower-right */}
                <circle cx="5.5" cy="7" r="2.2" fill="#ffffff" opacity="0.88" />
                {/* Faint tertiary depth glint */}
                <circle cx="-6" cy="7" r="1.3" fill="#ffffff" opacity="0.5" />
              </g>

              {/* RIGHT EYE */}
              <g
                transform={`translate(132, 115) scale(${eyeScaleX}, ${eyeScaleY})`}
                className="transition-transform duration-150 ease-out origin-center"
              >
                <ellipse cx="0" cy="0" rx="17" ry="22" fill="url(#eyeGlass)" />
                <circle cx="-5.5" cy="-7" r="5.5" fill="#ffffff" />
                <circle cx="5.5" cy="7" r="2.2" fill="#ffffff" opacity="0.88" />
                <circle cx="-6" cy="7" r="1.3" fill="#ffffff" opacity="0.5" />
              </g>
            </g>
          )}

          {/* Onyx visor: two glowing horizontal scanner eyes instead */}
          {persona === 'onyx' && (
            <g transform={`translate(${pupilX * 0.5}, ${pupilY * 0.5})`}
              className="transition-transform duration-120 ease-out">
              {/* Left scanner strip */}
              <rect
                x="46" y={106 + (isBlinking ? 8 : 0)}
                width="40" height={isBlinking ? 2 : 20}
                rx="10"
                fill={isSpeaking ? '#38bdf8' : isListening ? '#34d399' : '#94a3b8'}
                opacity={0.9}
                className={isSpeaking || isListening ? 'animate-pulse' : ''}
              />
              {/* Right scanner strip */}
              <rect
                x="114" y={106 + (isBlinking ? 8 : 0)}
                width="40" height={isBlinking ? 2 : 20}
                rx="10"
                fill={isSpeaking ? '#38bdf8' : isListening ? '#34d399' : '#94a3b8'}
                opacity={0.9}
                className={isSpeaking || isListening ? 'animate-pulse' : ''}
              />
            </g>
          )}

          {/* ══════════════════ MOUTH ══════════════════ */}
          {persona !== 'onyx' && (
            <g transform={`translate(${pupilX * 0.4}, ${pupilY * 0.35})`}>
              {isSpeaking ? (
                <ellipse
                  cx="100" cy="148"
                  rx={Math.max(4, (audioBars[0] || 6) * 0.72)}
                  ry={Math.max(5, (audioBars[1] || 8) * 0.9)}
                  fill="#0a0f1e"
                  className="transition-all duration-75"
                />
              ) : isListening ? (
                <ellipse cx="100" cy="146" rx="3.5" ry="4" fill="#1e293b" opacity="0.82" />
              ) : isThinking ? (
                <circle cx="100" cy="146" r="2.2" fill="#334155" opacity="0.7" />
              ) : (
                <path
                  d="M 93,143 Q 100,149 107,143"
                  fill="none"
                  stroke="#1e293b"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  opacity="0.78"
                />
              )}
            </g>
          )}

          {/* Soft cheek blush — speaking or listening */}
          {persona !== 'onyx' && (isSpeaking || isListening) && (
            <g opacity="0.28" className="transition-opacity duration-500">
              <ellipse cx="44" cy="133" rx="9" ry="5.5"
                fill={persona === 'pearl' ? '#f43f5e' : '#c084fc'} />
              <ellipse cx="156" cy="133" rx="9" ry="5.5"
                fill={persona === 'pearl' ? '#f43f5e' : '#c084fc'} />
            </g>
          )}
        </svg>
      </div>

      {/* ── Persona Switcher — color swatch dots, professional ── */}
      <div className="absolute -bottom-7 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-all duration-250">
        {PERSONAS.map((q) => (
          <button
            key={q.id}
            onClick={(e) => { e.stopPropagation(); handlePersona(q.id); }}
            title={q.name}
            style={{ background: q.swatch }}
            className={`w-4 h-4 rounded-full border-2 transition-all duration-200 shadow-sm ${
              persona === q.id
                ? 'border-white scale-125 ring-1 ring-white/30'
                : 'border-transparent opacity-60 hover:opacity-100 hover:scale-110'
            }`}
          />
        ))}
        <span className="ml-1 text-[10px] font-mono text-slate-400 tracking-wider uppercase">
          {p.name}
        </span>
      </div>
    </div>
  );
}
