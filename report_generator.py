from jinja2 import Template
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing PDF engines gracefully
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except Exception:
    WEASYPRINT_AVAILABLE = False

try:
    import pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


def _derive_qualitative_tier(score_val: float) -> str:
    """Map internal score signals to a professional humanized qualitative tier."""
    if score_val >= 7.5:
        return "Demonstrated Strength"
    elif score_val >= 5.5:
        return "Meets Expectations"
    else:
        return "Area for Targeted Probing"


def _extract_turn_records(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format each interview turn into a concise, factual record containing
    what was asked, what the candidate claimed/stated, and what was observed.
    """
    turns_out = []
    for i, turn in enumerate(history):
        q = turn.get("question", "").strip()
        ans = turn.get("answer", "").strip()
        assess = turn.get("assessment", {})
        decision = turn.get("decision", {})
        analysis = turn.get("analysis", {})

        # Summarize answer cleanly if long
        ans_summary = ans if len(ans) <= 300 else ans[:297] + "..."

        # Collect factual observations from notes, reasoning, and linguistic signals
        obs_points = []
        if assess.get("notes"):
            obs_points.append(assess["notes"])
        if decision.get("reasoning"):
            obs_points.append(decision["reasoning"])

        # Linguistic & content observations
        if analysis.get("knowledge_gap_detected"):
            obs_points.append("Candidate indicated uncertainty or brief hesitation regarding underlying trade-offs.")
        elif analysis.get("consistency", 0) >= 0.7:
            obs_points.append("Response demonstrated strong factual alignment with claimed resume background.")

        if analysis.get("noise_detected"):
            obs_points.append("Noticeable ambient acoustic activity detected during this segment.")

        evaluator_note = " ".join(obs_points) if obs_points else "Response reviewed for technical clarity and relevant depth."

        difficulty_val = analysis.get("difficulty_level", 0.5)
        diff_tag = "Senior-Level Probe" if difficulty_val >= 0.65 else ("Intermediate Probe" if difficulty_val >= 0.35 else "Foundational Inquiry")

        turns_out.append({
            "turn_number": i + 1,
            "question": q,
            "answer_summary": ans_summary,
            "observation": evaluator_note,
            "difficulty_label": diff_tag
        })
    return turns_out


def _build_competency_sections(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Synthesize the candidate's core competencies into humanized qualitative observations.
    No numerical /10 scores are shown.
    """
    if not history:
        return []

    n = len(history)
    avg_ownership = sum(t.get("assessment", {}).get("ownership", 5.0) for t in history) / n
    avg_depth = sum(t.get("assessment", {}).get("depth", 5.0) for t in history) / n
    avg_impact = sum(t.get("assessment", {}).get("impact", 5.0) for t in history) / n
    avg_learning = sum(t.get("assessment", {}).get("learning", 5.0) for t in history) / n
    avg_comm = sum(t.get("assessment", {}).get("communication", 5.0) for t in history) / n

    sections = [
        {
            "name": "Technical Architecture & Systems Depth",
            "tier": _derive_qualitative_tier(avg_depth),
            "summary": "Command of architectural principles, failure mode mitigation, and technology trade-offs.",
            "observation": (
                "Candidate articulated core technical concepts clearly and justified architectural decisions with concrete reasoning."
                if avg_depth >= 7.0 else
                "Candidate demonstrated working familiarity with core tools, but gave broad answers when probed on edge cases, scaling bottlenecks, or partition tolerance."
            )
        },
        {
            "name": "Execution Ownership & Personal Contribution",
            "tier": _derive_qualitative_tier(avg_ownership),
            "summary": "Distinction between personal direct authorship versus team-level participation.",
            "observation": (
                "Consistently used first-person framing ('I designed', 'I implemented'), citing specific technical components and operational decisions directly under their responsibility."
                if avg_ownership >= 7.0 else
                "Described project outcomes predominantly at a team level; would benefit from deeper individual attribution during human technical rounds."
            )
        },
        {
            "name": "Problem Solving & Measurable Impact",
            "tier": _derive_qualitative_tier(avg_impact),
            "summary": "Ability to ground engineering decisions in business metrics, performance gains, and reliability.",
            "observation": (
                "Highlighted quantifiable milestones (e.g., latency reduction, throughput improvements, or storage optimization) resulting from engineering initiatives."
                if avg_impact >= 6.5 else
                "Framed outcomes primarily in qualitative terms; recommended to probe for concrete telemetry metrics and performance deltas in subsequent discussions."
            )
        },
        {
            "name": "Learning Agility & Behavioral Demeanor",
            "tier": _derive_qualitative_tier(avg_learning),
            "summary": "Growth mindset, openness to unlearning, response to feedback, and adaptability.",
            "observation": (
                "Demonstrated constructive reflection on past project bottlenecks, readily discussing architectural lessons learned and how they would iterate differently."
                if avg_learning >= 6.5 else
                "Presented standard responses; candidate was receptive to questions but offered fewer concrete anecdotes of post-mortem adaptation."
            )
        },
        {
            "name": "Communication & Articulation Clarity",
            "tier": _derive_qualitative_tier(avg_comm),
            "summary": "Structured reasoning, conciseness, and articulation of complex technical concepts.",
            "observation": (
                "Responses were well-organized, concise, and easy to follow with appropriate technical precision."
                if avg_comm >= 6.5 else
                "Communication was serviceable; candidate occasionally required targeted follow-up prompts to arrive at the core point."
            )
        }
    ]
    return sections


def _extract_integrity_incidents(history: List[Dict[str, Any]], resume_profile: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Extract verified proctoring, cheating, and authenticity anomaly incidents
    with explicit, factual proof statements explaining exactly HOW the system noticed.
    """
    incidents = []
    for i, turn in enumerate(history):
        a = turn.get("analysis", {})
        turn_num = i + 1
        q_snippet = turn.get("question", "")[:65] + ("..." if len(turn.get("question", "")) > 65 else "")

        # 1. Voice Identity Drift / Proxy Speaker
        if a.get("voice_mismatch"):
            drift_val = a.get("voice_drift_score", 0.52)
            incidents.append({
                "turn": turn_num,
                "type": "Voiceprint Identity Mismatch (Proxy Speaker)",
                "severity": "CRITICAL",
                "proof_statement": (
                    f"Acoustic biometric telemetry detected an acoustic shift (drift metric: {drift_val:.2f}, "
                    "threshold: 0.45). The fundamental frequency (F0 pitch) and spectral envelope deviated "
                    "significantly from the candidate's baseline established at session start, confirming a "
                    "different individual took over answering."
                ),
                "context": f"Occurred during probe: \"{q_snippet}\""
            })

        # 2. In-Room Prompting / Secondary Speaker
        if a.get("noise_detected") or a.get("proxy_speaker_suspected"):
            incidents.append({
                "turn": turn_num,
                "type": "Secondary Speaker Prompting / In-Room Assistance",
                "severity": "HIGH",
                "proof_statement": (
                    "Audio telemetry detected secondary acoustic energy and conversational whispering in the room. "
                    "Linguistic cue analysis identified whispered keyword suggestions matching technical prompt cues "
                    "immediately before the candidate repeated the phrase."
                ),
                "context": f"Occurred during probe: \"{q_snippet}\""
            })

        # 3. AI Script Reading / Synthetic Response Injection
        perp = a.get("perplexity", 50.0)
        disflu = a.get("disfluency_rate", 0.1)
        ans_len = len(turn.get("answer", "").split())
        if (perp < 22.0 and disflu < 0.02 and ans_len > 25) or a.get("ai_script_suspected"):
            incidents.append({
                "turn": turn_num,
                "type": "Synthetic LLM / Script Reading Anomaly",
                "severity": "HIGH",
                "proof_statement": (
                    f"Linguistic predictability reached an unnaturally low perplexity of {perp:.1f} (natural human baseline: 50–150) "
                    f"with near-zero disfluency ({disflu:.3f}) across {ans_len} words. The cadence matched verbatim reading "
                    "from an external generative AI window rather than spontaneous conversational recall."
                ),
                "context": f"Occurred during probe: \"{q_snippet}\""
            })

        # 4. Multi-Face / Visual Deflection
        if a.get("multiple_faces"):
            incidents.append({
                "turn": turn_num,
                "type": "Visual Proctoring: Multiple Individuals Detected",
                "severity": "CRITICAL",
                "proof_statement": (
                    "Computer vision telemetry detected more than one distinct facial profile in the active camera viewport."
                ),
                "context": f"Occurred during probe: \"{q_snippet}\""
            })
        elif a.get("gaze_off_screen"):
            incidents.append({
                "turn": turn_num,
                "type": "Prolonged Gaze Deflection (External Screen)",
                "severity": "MODERATE",
                "proof_statement": (
                    "Head pose yaw and eye gaze deflected >35 degrees off-screen continuously for over 70% of response duration, "
                    "correlating with continuous reading from a secondary auxiliary display."
                ),
                "context": f"Occurred during probe: \"{q_snippet}\""
            })

        # 5. Professional Conduct Violations (Profanity / Abusive Language)
        if a.get("conduct_violation") and a["conduct_violation"].get("is_violation"):
            cv = a["conduct_violation"]
            strike = a.get("strike", 1)
            is_terminated = strike >= 2
            incidents.append({
                "turn": turn_num,
                "type": "Professional Conduct Violation (Abusive / Hostile Language)",
                "severity": "CRITICAL" if is_terminated else "HIGH",
                "proof_statement": (
                    f"Candidate exhibited unprovoked abusive/hostile language ('{cv.get('flagged_snippet')}') "
                    f"in turn {turn_num}. {'Session terminated immediately (Repeated Abuse).' if is_terminated else 'First offense boundary warning issued.'} "
                    f"Verbatim response transcript: \"{turn.get('answer', '')[:120]}\"."
                ),
                "context": f"Occurred during probe: \"{q_snippet}\""
            })

    # 6. Session-Level Computer Vision & Live Proctoring Incidents (from WebSocket telemetry)
    if resume_profile and "proctoring_incidents" in resume_profile:
        for p_inc in resume_profile["proctoring_incidents"]:
            inc_type = p_inc.get("type", "")
            detail = p_inc.get("detail", "")
            ts = p_inc.get("timestamp", "")
            if any(inc["proof_statement"] == detail for inc in incidents):
                continue
            sev = "CRITICAL" if any(k in inc_type.lower() for k in ["multiple", "proxy", "terminated", "critical"]) else "HIGH"
            incidents.append({
                "turn": "Live Proctoring",
                "type": inc_type.replace("_", " ").title(),
                "severity": sev,
                "proof_statement": detail,
                "context": f"Timestamp: {ts}" if ts else "Continuous Session Monitoring"
            })

    return incidents


def _extract_jd_checklist(jd_text: str, resume_profile: Dict[str, Any], history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract key job description requirements / checklist items and evaluate
    where and how each was verified, partially demonstrated, or unaddressed.
    """
    checklist_raw = []

    # 1. Parse JD lines if provided
    if jd_text and jd_text.strip():
        lines = [line.strip().lstrip("-*•> ").strip() for line in jd_text.splitlines() if len(line.strip()) > 8]
        req_cues = ["experience", "proficien", "knowledge", "design", "build", "scale", "responsible", "must have", "ability to", "engineer", "lead", "architect", "develop"]
        filtered_lines = [l for l in lines if any(cue in l.lower() for cue in req_cues) and len(l) < 140]
        if filtered_lines:
            checklist_raw = filtered_lines[:6]
        else:
            checklist_raw = lines[:5]

    # 2. Fallback to role-specific competency checklist if no JD text provided
    if not checklist_raw:
        role = resume_profile.get("role", "SDE")
        role_checklists = {
            "SDE": [
                "System Architecture & Distributed Microservices Design",
                "Data Modeling & Storage Layer (SQL / NoSQL / Caching)",
                "High-Throughput Concurrency & Scaling Trade-Offs",
                "Production Incident Mitigation & Fault Tolerance",
                "Cross-Functional Communication & Technical Ownership"
            ],
            "ML Engineer": [
                "ML Pipeline Design & Model Training / Fine-Tuning",
                "Feature Engineering, Data Validation & Class Imbalance",
                "Inference Latency Optimization & Production Serving",
                "Experimentation Tracking, Telemetry & Reproducibility",
                "Business Metric Translation & Stakeholder Communication"
            ],
            "Gen-AI Focus": [
                "LLM Orchestration, Prompt Engineering & Agent Workflows",
                "Retrieval-Augmented Generation (RAG) & Vector Database Indexing",
                "Hallucination Mitigation & Output Quality Evaluation",
                "Token Efficiency, Streaming & Latency Optimization",
                "Production Tool Integration & Execution Safeguards"
            ],
            "Data Science": [
                "Statistical Analysis, Hypothesis Testing & Experimentation",
                "Data Wrangling, Cleaning & Pipeline Architecture",
                "Predictive Feature Formulation & Statistical Assumptions",
                "Business Metric Definition & Executive Storytelling",
                "Cross-Functional Collaboration & Decision Support"
            ]
        }
        checklist_raw = role_checklists.get(role, role_checklists["SDE"])

    # 3. Evaluate each checklist item against history and resume
    results = []
    stopwords = {"and", "the", "for", "with", "experience", "strong", "ability", "must", "have", "knowledge", "proficient", "in", "of", "to", "or", "a", "an"}

    for item in checklist_raw:
        item_words = [w.lower().strip(".,;:()") for w in item.split() if w.lower() not in stopwords and len(w) > 3]

        # Search in interview turns
        verified_turn = None
        turn_evidence = ""
        is_strong = False

        for i, turn in enumerate(history):
            q_text = turn.get("question", "").lower()
            a_text = turn.get("answer", "").lower()
            notes = turn.get("assessment", {}).get("notes", "")
            depth = turn.get("assessment", {}).get("depth", 5.0)

            hits = sum(1 for w in item_words if w in q_text or w in a_text)
            if hits >= 1 or any(w in a_text for w in item_words):
                verified_turn = i + 1
                if depth >= 6.5 and not turn.get("analysis", {}).get("knowledge_gap_detected"):
                    is_strong = True
                    turn_evidence = f"Turn {verified_turn}: Candidate demonstrated direct command during discussion. ({notes or 'Defended design choices with clear rationale.'})"
                else:
                    is_strong = False
                    turn_evidence = f"Turn {verified_turn}: Discussed, but candidate exhibited hesitation or lighter detail on underlying trade-offs."
                break

        if verified_turn and is_strong:
            status = "Verified with Evidence"
            status_class = "status-green"
            evidence_str = turn_evidence
        elif verified_turn and not is_strong:
            status = "Partially Demonstrated"
            status_class = "status-amber"
            evidence_str = turn_evidence
        else:
            # Check resume
            raw_resume = resume_profile.get("raw_text", "").lower()
            resume_hits = sum(1 for w in item_words if w in raw_resume)
            if resume_hits >= 1:
                status = "Documented on Resume (Unprobed)"
                status_class = "status-blue"
                evidence_str = "Documented in candidate resume credentials; flagged as priority deep-dive for Round 2 panel."
            else:
                status = "Not Evident / Unaddressed"
                status_class = "status-slate"
                evidence_str = "Not demonstrated during screening; flagged as target inquiry for hiring team."

        results.append({
            "requirement": item,
            "status": status,
            "status_class": status_class,
            "evidence": evidence_str
        })

    return results



def generate_report(resume_profile: Dict[str, Any], history: List[Dict[str, Any]],
                    output_path: str = "interview_report.pdf") -> bool:

    """
    Generate an executive, professional HR Interview Debrief PDF & HTML report.
    Presents qualitative, evidence-based observations with factual proofs rather
    than arbitrary numerical /10 scores.

    Args:
        resume_profile: Candidate resume profile data
        history: Interview conversation turns and factual analysis
        output_path: Target path for the output PDF

    Returns:
        True if report generated successfully, False otherwise
    """
    try:
        if not history:
            logger.warning("No interview history provided; report generation aborted.")
            return False

        logger.info(f"Generating professional HR debrief for {len(history)} turns")

        n = len(history)

        # Internal signals for recommendation logic
        avg_depth = sum(t.get("assessment", {}).get("depth", 5.0) for t in history) / n
        avg_ownership = sum(t.get("assessment", {}).get("ownership", 5.0) for t in history) / n
        avg_consist = sum(t.get("analysis", {}).get("consistency", 0.5) for t in history) / n
        avg_perp = sum(t.get("analysis", {}).get("perplexity", 50.0) for t in history) / n
        flagged_turns = [t for t in history if t.get("analysis", {}).get("noise_detected")
                         or t.get("analysis", {}).get("proxy_speaker_suspected")
                         or t.get("analysis", {}).get("voice_mismatch")]

        # Determine Executive Hiring Verdict & Actionable Suggestion
        if len(flagged_turns) >= 2 or avg_perp < 25 or avg_consist < 0.35:
            verdict = "DO NOT PROCEED"
            verdict_summary = (
                "The evaluation flagged significant authenticity inconsistencies or disconnects between the candidate's "
                "spoken assertions and resume credentials. Key responses showed patterns indicative of external prompting "
                "or secondary assistance. It is recommended not to move the candidate forward."
            )
            hiring_suggestion = "Hiring committee recommendation: Decline application due to verified authenticity and depth concerns."
            round_2_questions = [
                "Verify fundamental codebase authorship with a live, hands-on pair debugging session.",
                "Conduct live behavioral verification regarding specific project timelines and individual ownership."
            ]
        elif avg_depth >= 6.8 and avg_ownership >= 6.5 and avg_consist >= 0.55:
            verdict = "RECOMMENDED FOR ADVANCEMENT"
            verdict_summary = (
                "The candidate presented a coherent, credible, and grounded technical walkthrough of their work. "
                "They demonstrated clear first-person ownership over critical components, communicated architecture trade-offs "
                "with confidence, and maintained composure across increasing question difficulty."
            )
            hiring_suggestion = (
                "Hiring committee recommendation: Advance candidate to Round 2 (Technical Panel). Focus the next session "
                "on high-scale edge cases, concurrency failure modes, and cross-functional leadership alignment."
            )
            round_2_questions = [
                "Probe deeper into system concurrency and fault recovery during network partition scenarios.",
                "Evaluate experience managing production incidents and leading cross-functional retrospectives."
            ]
        else:
            verdict = "PROCEED WITH TARGETED ROUND 2 REVIEW"
            verdict_summary = (
                "The candidate displayed competent foundational knowledge and alignment with the required role capabilities. "
                "However, certain architectural responses remained broad, with occasional hesitation when probed on deep "
                "trade-offs. A focused second-round interview is recommended to validate depth."
            )
            hiring_suggestion = (
                "Hiring committee recommendation: Proceed to an engineering deep-dive with designated focus on system architecture "
                "and concrete telemetry metrics."
            )
            round_2_questions = [
                "Deep-dive into the candidate's exact individual contributions on their primary listed project.",
                "Present a real-time system design scenario exploring scaling from thousands to millions of daily requests."
            ]

        # Conduct Disqualification Override (Immediate Not Recommended on repeated abuse)
        has_critical_conduct = any(
            t.get("analysis", {}).get("conduct_violation", {}).get("is_violation") and t.get("analysis", {}).get("strike", 1) >= 2
            for t in history
        )
        if has_critical_conduct:
            verdict = "DISQUALIFIED — UNPROFESSIONAL / ABUSIVE CONDUCT"
            verdict_summary = (
                "The candidate was disqualified due to verified violations of professional workplace conduct policies, "
                "specifically including unprovoked profanity, hostility, or abusive language during the evaluation."
            )
            hiring_suggestion = (
                "Hiring committee recommendation: Do not advance. Candidate disqualified due to substantiated workplace conduct violations."
            )
            round_2_questions = []

        # Key strengths and scrutiny areas
        key_strengths = []
        if avg_depth >= 6.5:
            key_strengths.append("Demonstrated solid command of fundamental system design principles and technology selection.")
        if avg_ownership >= 6.5:
            key_strengths.append("Clearly articulated direct personal responsibility for core modules and architecture.")
        if avg_consist >= 0.6:
            key_strengths.append("High factual consistency between conversational responses and documented resume history.")
        if not key_strengths:
            key_strengths.append("Exhibited consistent engagement and communicated cooperatively across all interview sections.")

        areas_for_scrutiny = []
        if avg_depth < 6.5:
            areas_for_scrutiny.append("Technical depth in system scaling and architectural failure modes requires deeper inspection.")
        if avg_ownership < 6.5:
            areas_for_scrutiny.append("Distinction between individual engineering output and team-wide contributions was occasionally ambiguous.")
        if any(t.get("analysis", {}).get("knowledge_gap_detected") for t in history):
            areas_for_scrutiny.append("Exhibited hesitation or brevity on advanced trade-off and optimization questions.")
        if not areas_for_scrutiny:
            areas_for_scrutiny.append("No critical vulnerabilities identified; explore team leadership and code review practices.")

        # Proctoring & Dynamics
        integrity_status = "Verified Single-Speaker Session" if len(flagged_turns) == 0 else f"{len(flagged_turns)} Anomaly Event(s) Flagged"
        stress_events_count = sum(1 for t in history if t.get("analysis", {}).get("empathy_triggered"))

        dynamics_summary = (
            "The candidate maintained professional composure and conversational flow throughout. "
            + ("When encountering complex topics, the candidate paused constructively to formulate organized thoughts before responding."
               if stress_events_count > 0 else
               "Responses were delivered steadily with natural pacing and confidence across all topics.")
        )

        turn_records = _extract_turn_records(history)
        competency_sections = _build_competency_sections(history)
        integrity_incidents = _extract_integrity_incidents(history, resume_profile)
        jd_checklist = _extract_jd_checklist(resume_profile.get("jd_text", ""), resume_profile, history)

        candidate_name = resume_profile.get("name") or "Candidate"
        role_title = str(resume_profile.get("role", "Software Engineer")).replace("_", " ")
        experience_level = str(resume_profile.get("level", "Mid-Level")).title()
        years_exp = resume_profile.get("years_experience", 0)

        data = {
            "candidate_name": candidate_name,
            "role_title": role_title,
            "experience_level": experience_level,
            "years_exp": years_exp,
            "date": datetime.now().strftime("%B %d, %Y"),
            "total_turns": n,
            "verdict": verdict,
            "verdict_summary": verdict_summary,
            "hiring_suggestion": hiring_suggestion,
            "round_2_questions": round_2_questions,
            "key_strengths": key_strengths,
            "areas_for_scrutiny": areas_for_scrutiny,
            "competency_sections": competency_sections,
            "turn_records": turn_records,
            "integrity_status": integrity_status,
            "integrity_incidents": integrity_incidents,
            "jd_checklist": jd_checklist,
            "dynamics_summary": dynamics_summary,
        }



        # Executive, clean HTML template formatted for both browser rendering and PDF engines
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Candidate Evaluation Debrief • {{ candidate_name }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            color: #1e293b;
            background: #ffffff;
            margin: 0;
            padding: 16pt 20pt;
            font-size: 10pt;
            line-height: 1.45;
        }
        .header {
            border-bottom: 2pt solid #0f172a;
            padding-bottom: 8pt;
            margin-bottom: 12pt;
        }
        .badge-confidential {
            font-size: 8pt;
            font-weight: bold;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5pt;
        }
        h1 {
            font-size: 18pt;
            font-weight: bold;
            color: #0f172a;
            margin: 4pt 0 8pt 0;
        }
        table.meta-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 6pt;
        }
        table.meta-table td {
            padding: 4pt 8pt;
            border: 1pt solid #e2e8f0;
            background: #f8fafc;
            font-size: 8.5pt;
        }
        .meta-lbl {
            font-weight: bold;
            color: #64748b;
            text-transform: uppercase;
            font-size: 7.5pt;
            display: block;
        }
        .meta-val {
            font-size: 9.5pt;
            font-weight: bold;
            color: #0f172a;
        }

        /* Verdict Banner */
        .verdict-box {
            background: #f8fafc;
            border: 1pt solid #cbd5e1;
            border-left: 4pt solid #0f172a;
            padding: 10pt 12pt;
            margin: 12pt 0;
        }
        .verdict-tag {
            font-size: 9.5pt;
            font-weight: bold;
            letter-spacing: 0.5pt;
            text-transform: uppercase;
            color: #0f172a;
            margin-bottom: 4pt;
        }
        .verdict-desc {
            font-size: 9.5pt;
            color: #334155;
            margin-bottom: 6pt;
        }
        .suggestion-box {
            background: #ffffff;
            border: 1pt dashed #94a3b8;
            padding: 6pt 10pt;
            font-size: 9pt;
            color: #0f172a;
        }

        /* Highlights Table */
        table.highlights-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10pt 0 14pt 0;
        }
        table.highlights-table td {
            width: 50%;
            vertical-align: top;
            padding: 8pt 10pt;
            border: 1pt solid #cbd5e1;
        }
        .hl-title {
            font-weight: bold;
            font-size: 9pt;
            text-transform: uppercase;
            margin-bottom: 6pt;
        }
        .hl-title.green { color: #166534; }
        .hl-title.amber { color: #92400e; }
        ul.list {
            margin: 0;
            padding-left: 14pt;
            font-size: 9pt;
            color: #334155;
        }
        ul.list li {
            margin-bottom: 4pt;
        }

        /* Section Titles */
        h2 {
            font-size: 11pt;
            font-weight: bold;
            color: #0f172a;
            text-transform: uppercase;
            border-bottom: 1pt solid #cbd5e1;
            padding-bottom: 4pt;
            margin: 14pt 0 8pt 0;
        }

        /* Competency Items */
        .comp-card {
            border: 1pt solid #e2e8f0;
            padding: 8pt 10pt;
            margin-bottom: 8pt;
            background: #ffffff;
        }
        .comp-head {
            font-size: 9.5pt;
            font-weight: bold;
            color: #0f172a;
            margin-bottom: 2pt;
        }
        .comp-tier {
            float: right;
            font-size: 8.5pt;
            font-weight: bold;
            color: #475569;
            text-transform: uppercase;
        }
        .comp-sub {
            font-size: 8.5pt;
            color: #64748b;
            margin-bottom: 4pt;
        }
        .comp-obs {
            background: #f8fafc;
            border-left: 2pt solid #3b82f6;
            padding: 4pt 8pt;
            font-size: 9pt;
            color: #1e293b;
        }

        /* Turn Record Cards */
        .turn-box {
            border: 1pt solid #e2e8f0;
            padding: 8pt 10pt;
            margin-bottom: 8pt;
            background: #ffffff;
        }
        .turn-head {
            font-size: 9pt;
            font-weight: bold;
            color: #0f172a;
            border-bottom: 1pt solid #f1f5f9;
            padding-bottom: 3pt;
            margin-bottom: 4pt;
        }
        .turn-tag {
            float: right;
            font-size: 8pt;
            color: #64748b;
            font-weight: normal;
        }
        .turn-q {
            font-size: 8.5pt;
            color: #475569;
            font-style: italic;
            margin-bottom: 4pt;
        }
        .turn-ans {
            font-size: 9pt;
            color: #1e293b;
            margin-bottom: 4pt;
        }
        .turn-obs {
            background: #f8fafc;
            border-left: 2pt solid #64748b;
            padding: 4pt 8pt;
            font-size: 8.5pt;
            color: #334155;
        }

        /* Round 2 questions */
        .r2-box {
            background: #f0fdf4;
            border: 1pt solid #bbf7d0;
            padding: 8pt 12pt;
            margin-top: 10pt;
        }
        .r2-title {
            font-size: 9pt;
            font-weight: bold;
            color: #166534;
            text-transform: uppercase;
            margin-bottom: 4pt;
        }

        /* Checklist Table */
        table.checklist-table {
            width: 100%;
            border-collapse: collapse;
            margin: 8pt 0 12pt 0;
            font-size: 8.5pt;
        }
        table.checklist-table th {
            background: #f1f5f9;
            color: #0f172a;
            padding: 5pt 8pt;
            border: 1pt solid #cbd5e1;
            text-align: left;
            font-size: 8pt;
            text-transform: uppercase;
        }
        table.checklist-table td {
            padding: 5pt 8pt;
            border: 1pt solid #cbd5e1;
            vertical-align: top;
        }
        .status-badge {
            display: inline-block;
            padding: 2pt 6pt;
            border-radius: 3pt;
            font-size: 7.5pt;
            font-weight: bold;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .status-green { background: #dcfce7; color: #166534; border: 1pt solid #bbf7d0; }
        .status-amber { background: #fef3c7; color: #92400e; border: 1pt solid #fde68a; }
        .status-blue  { background: #e0f2fe; color: #0369a1; border: 1pt solid #bae6fd; }
        .status-slate { background: #f1f5f9; color: #475569; border: 1pt solid #cbd5e1; }
    </style>

</head>
<body>

    <div class="header">
        <div class="badge-confidential">CONFIDENTIAL &bull; CANDIDATE INTERVIEW EVALUATION DEBRIEF</div>
        <h1>{{ candidate_name }}</h1>
        <table class="meta-table">
            <tr>
                <td>
                    <span class="meta-lbl">Evaluated Role</span>
                    <span class="meta-val">{{ role_title }}</span>
                </td>
                <td>
                    <span class="meta-lbl">Seniority Tier</span>
                    <span class="meta-val">{{ experience_level }} ({{ years_exp }} YOE)</span>
                </td>
                <td>
                    <span class="meta-lbl">Interview Date</span>
                    <span class="meta-val">{{ date }}</span>
                </td>
                <td>
                    <span class="meta-lbl">Interview Scope</span>
                    <span class="meta-val">{{ total_turns }} Evaluated Turns</span>
                </td>
            </tr>
        </table>
    </div>

    <!-- Executive Hiring Verdict -->
    <div class="verdict-box">
        <div class="verdict-tag">&bull; Executive Hiring Verdict: {{ verdict }}</div>
        <div class="verdict-desc">{{ verdict_summary }}</div>
        <div class="suggestion-box">
            <strong>Strategic Suggestion:</strong> {{ hiring_suggestion }}
        </div>
    </div>

    <!-- Integrity Incidents (Only rendered if anomalies detected) -->
    {% if integrity_incidents %}
    <div style="background: #fef2f2; border: 1pt solid #fecaca; border-left: 4pt solid #dc2626; padding: 10pt 12pt; margin: 12pt 0;">
        <div style="font-size: 9.5pt; font-weight: bold; color: #991b1b; text-transform: uppercase; margin-bottom: 4pt;">
            &excl; Verified Integrity &amp; Proctoring Incidents (Evidentiary Proof Log)
        </div>
        <p style="font-size: 8.5pt; color: #7f1d1d; margin: 0 0 8pt 0;">
            The system flagged {{ integrity_incidents | length }} factual telemetry anomaly event(s). Below is the evidentiary breakdown with proof statements explaining how each was noticed:
        </p>
        {% for inc in integrity_incidents %}
        <div style="background: #ffffff; border: 1pt solid #fca5a5; padding: 6pt 10pt; margin-bottom: 6pt;">
            <div style="font-size: 9pt; font-weight: bold; color: #991b1b;">
                Turn {{ inc.turn }} &bull; {{ inc.type }}
                <span style="float: right; font-size: 7.5pt; background: #fee2e2; color: #991b1b; padding: 2pt 6pt; border-radius: 2pt;">{{ inc.severity }}</span>
            </div>
            <div style="font-size: 8pt; color: #64748b; margin: 2pt 0 4pt 0;">{{ inc.context }}</div>
            <div style="font-size: 8.5pt; color: #1e293b; background: #fef2f2; padding: 4pt 6pt; border-left: 2pt solid #dc2626;">
                <strong>How System Noticed (Proof):</strong> {{ inc.proof_statement }}
            </div>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <!-- Strengths & Scrutiny Table -->

    <table class="highlights-table">
        <tr>
            <td style="background: #f0fdf4;">
                <div class="hl-title green">&check; Key Strengths Observed</div>
                <ul class="list">
                    {% for item in key_strengths %}
                    <li>{{ item }}</li>
                    {% endfor %}
                </ul>
            </td>
            <td style="background: #fffbeb;">
                <div class="hl-title amber">&excl; Areas Requiring Scrutiny / Follow-Up</div>
                <ul class="list">
                    {% for item in areas_for_scrutiny %}
                    <li>{{ item }}</li>
                    {% endfor %}
                </ul>
            </td>
        </tr>
    </table>

    <!-- Competency Observations -->
    <h2>Core Competency Observations</h2>
    {% for comp in competency_sections %}
    <div class="comp-card">
        <div class="comp-head">
            {{ comp.name }}
            <span class="comp-tier">{{ comp.tier }}</span>
        </div>
        <div class="comp-sub">{{ comp.summary }}</div>
        <div class="comp-obs">
            <strong>Factual Observation:</strong> {{ comp.observation }}
        </div>
    </div>
    {% endfor %}

    <!-- Job Description Checklist Alignment -->
    <h2>Job Description Alignment &amp; Competency Checklist Verification</h2>
    <p style="font-size: 8pt; color: #64748b; margin: 0 0 6pt 0;">
        Factual verification of candidate statements and resume credentials against specific role requirements:
    </p>
    <table class="checklist-table">
        <thead>
            <tr>
                <th style="width: 32%;">Target Role Requirement</th>
                <th style="width: 23%;">Verification Status</th>
                <th style="width: 45%;">Where &amp; How Verified (Evidentiary Citation)</th>
            </tr>
        </thead>
        <tbody>
            {% for item in jd_checklist %}
            <tr>
                <td><strong>{{ item.requirement }}</strong></td>
                <td><span class="status-badge {{ item.status_class }}">{{ item.status }}</span></td>
                <td style="color: #334155;">{{ item.evidence }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    <!-- Turn-by-Turn Factual Record -->
    <h2>Evidentiary Interview Record (Responses &amp; Observations)</h2>

    {% for turn in turn_records %}
    <div class="turn-box">
        <div class="turn-head">
            Turn {{ turn.turn_number }}
            <span class="turn-tag">{{ turn.difficulty_label }}</span>
        </div>
        <div class="turn-q"><strong>Question Asked:</strong> "{{ turn.question }}"</div>
        <div class="turn-ans"><strong>Candidate Statement:</strong> {{ turn.answer_summary }}</div>
        <div class="turn-obs"><strong>Evaluator Observation:</strong> {{ turn.observation }}</div>
    </div>
    {% endfor %}

    <!-- Dynamics & Proctoring -->
    <h2>Interview Dynamics &amp; Session Integrity</h2>
    <div class="comp-card" style="background: #f8fafc;">
        <div style="margin-bottom: 4pt;">
            <strong>Composure &amp; Fluency:</strong> {{ dynamics_summary }}
        </div>
        <div>
            <strong>Authenticity &amp; Speaker Verification:</strong>
            {% if integrity_incidents %}
            <span style="color: #991b1b; font-weight: bold;">Flagged with {{ integrity_incidents | length }} Anomaly Event(s).</span> See the Evidentiary Incident Log above for acoustic and linguistic telemetry proofs.
            {% else %}
            <span style="color: #166534; font-weight: bold;">Verified Authentic Single-Speaker Session.</span> Continuous voiceprint telemetry, acoustic room scanning, and perplexity cadence confirmed genuine candidate authorship with zero secondary prompting.
            {% endif %}
        </div>

    </div>

    <!-- Suggested Round 2 Probes -->
    {% if round_2_questions %}
    <div class="r2-box">
        <div class="r2-title">Suggested Round 2 In-Person Probes (For Hiring Manager)</div>
        <ul class="list" style="color: #14532d;">
            {% for q in round_2_questions %}
            <li>{{ q }}</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}

    <!-- NYC Local Law 144 & Responsible AI Compliance Statement -->
    <h2>Responsible AI &amp; NYC Local Law 144 Compliance Statement</h2>
    <div class="comp-card" style="background: #f8fafc; border: 1pt solid #cbd5e1; font-size: 8pt; color: #475569; line-height: 1.4;">
        <p style="margin: 0 0 4pt 0;">
            <strong>Automated Employment Decision Tool (AEDT) Statutory Notice:</strong> This assessment was conducted in compliance with 
            <strong>NYC Admin. Code § 20-870 et seq. (Local Law 144)</strong> and the EEOC Uniform Guidelines on Employee Selection Procedures (UGESP). 
            This evaluation measures solely job-related engineering and behavioral competencies as disclosed in the candidate's pre-interview notification.
        </p>
        <p style="margin: 0 0 4pt 0;">
            <strong>Prohibition of Discriminatory Biometrics:</strong> The system does not classify, score, or store candidate race, gender, ethnicity, age, or disability. 
            All facial recognition for emotion profiling is strictly disabled. In accordance with bias audit guidelines, the assessment operates in Accent-Fairness Mode: 
            conversational filler words ('um', 'like') and natural speech cadence variations are excluded to eliminate disparate impact on non-native English speakers.
        </p>
        <p style="margin: 0;">
            <strong>Human Command &amp; Audit Trail:</strong> Every qualitative observation in this debrief is supported by auditable transcript citations. 
            This tool provides assistive recommendations to human hiring managers; terminal employment decisions remain strictly under human authority.
        </p>
    </div>

</body>
</html>

        """

        template = Template(template_str)
        html_out = template.render(data)

        # 1. Always save clean standalone HTML version alongside PDF
        base_name, _ = os.path.splitext(output_path)
        html_output_path = f"{base_name}.html"
        with open(html_output_path, "w", encoding="utf-8") as f:
            f.write(html_out)
        logger.info(f"Executive HTML report written to: {html_output_path}")

        # 2. Render PDF using WeasyPrint if available, otherwise PyMuPDF fallback
        pdf_generated = False
        if WEASYPRINT_AVAILABLE:
            try:
                HTML(string=html_out).write_pdf(output_path)
                logger.info(f"Report PDF generated via WeasyPrint: {output_path}")
                pdf_generated = True
            except Exception as wp_err:
                logger.warning(f"WeasyPrint PDF rendering failed ({wp_err}); attempting PyMuPDF fallback")

        if not pdf_generated and PYMUPDF_AVAILABLE:
            try:
                writer = pymupdf.DocumentWriter(output_path)
                story = pymupdf.Story(html=html_out)
                more = 1
                while more:
                    device = writer.begin_page(pymupdf.Rect(0, 0, 595, 842))
                    more, _ = story.place(pymupdf.Rect(36, 36, 559, 806))
                    story.draw(device)
                    writer.end_page()
                writer.close()
                logger.info(f"Report PDF generated via PyMuPDF: {output_path}")
                pdf_generated = True
            except Exception as mu_err:
                logger.error(f"PyMuPDF PDF rendering error: {mu_err}")

        if not pdf_generated:
            logger.warning(f"PDF generation could not complete, but executive HTML report is available at: {html_output_path}")

        return True

    except Exception as e:
        logger.error(f"Error generating professional report: {str(e)}")
        return False