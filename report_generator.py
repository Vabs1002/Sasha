from jinja2 import Template
import os
import logging
import re
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
    """Describe an unvalidated rubric estimate without presenting it as a finding."""
    if score_val >= 7.5:
        return "Higher rubric estimate"
    elif score_val >= 5.5:
        return "Mid-range rubric estimate"
    else:
        return "Lower rubric estimate"


def _summarize_browser_interaction_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Summarize consented, client-reported context events without treating them as findings."""
    labels = {
        "page_hidden": "Interview page became hidden",
        "page_visible": "Interview page became visible",
        "window_blur": "Interview window lost focus",
        "window_focus": "Interview window regained focus",
        "answer_paste": "Text was pasted into the answer field",
    }
    summaries: Dict[str, Dict[str, Any]] = {}
    for item in events or []:
        event_name = item.get("event")
        if event_name not in labels:
            continue
        summary = summaries.setdefault(event_name, {"label": labels[event_name], "count": 0, "first_seen": "", "last_seen": ""})
        summary["count"] += 1
        timestamp = str(item.get("timestamp", ""))
        if timestamp and (not summary["first_seen"] or timestamp < summary["first_seen"]):
            summary["first_seen"] = timestamp
        if timestamp > summary["last_seen"]:
            summary["last_seen"] = timestamp
    return list(summaries.values())


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
        decision_source = turn.get("decision_source")
        decision_source_labels = {
            "llm": "Configured language model",
            "fallback": "Fallback interviewer logic",
            "rule_based": "Rule-based conduct response",
        }

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
            obs_points.append("An automated knowledge-gap heuristic was flagged; review the answer and follow-up in context.")
        elif analysis.get("consistency", 0) >= 0.7:
            obs_points.append("Automated text comparison found overlap with resume text; this does not verify the underlying claim.")

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
            "difficulty_label": diff_tag,
            "decision_source_label": decision_source_labels.get(decision_source, ""),
        })
    return turns_out


def _build_competency_sections(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Pair each automated rubric summary with an exact response excerpt for review.
    """
    if not history:
        return []

    dimensions = [
        ("Technical Architecture & Systems Depth", "depth", "Architecture, failure modes, and technical trade-offs."),
        ("Execution Ownership & Personal Contribution", "ownership", "Personal contribution and ownership described in the interview."),
        ("Problem Solving & Measurable Impact", "impact", "Problem-solving approach and impact described in the interview."),
        ("Learning Agility", "learning", "Reflection and adaptation described in the interview."),
        ("Communication", "communication", "Organization and clarity of the interview response."),
    ]
    sections = []
    for name, metric, summary in dimensions:
        scored_turns = [
            (turn.get("assessment", {}).get(metric, 5.0), index, turn)
            for index, turn in enumerate(history, start=1)
            if str(turn.get("answer", "")).strip()
        ]
        average = sum(turn.get("assessment", {}).get(metric, 5.0) for turn in history) / len(history)
        if scored_turns:
            _, evidence_turn, evidence_record = max(scored_turns, key=lambda item: item[0])
            evidence_quote = str(evidence_record.get("answer", "")).strip()[:360]
        else:
            evidence_turn = None
            evidence_quote = ""
        sections.append({
            "name": name,
            "tier": _derive_qualitative_tier(average),
            "summary": summary,
            "observation": "Automated rubric estimate; review the cited response before drawing a conclusion.",
            "evidence_turn": evidence_turn,
            "evidence_quote": evidence_quote,
        })
    return sections


def _extract_integrity_incidents(history: List[Dict[str, Any]], resume_profile: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Extract unverified integrity signals as context for human review.
    """
    incidents = []
    integrity_monitoring_enabled = (
        resume_profile.get("integrity_monitoring_enabled", False)
        if resume_profile
        else False
    )
    for i, turn in enumerate(history):
        a = turn.get("analysis", {})
        turn_num = i + 1
        q_snippet = turn.get("question", "")[:65] + ("..." if len(turn.get("question", "")) > 65 else "")
        answer_snippet = str(turn.get("answer", "")).strip()[:240]
        turn_timestamp = turn.get("timestamp", "")
        turn_context = (f"Timestamp: {turn_timestamp} | " if turn_timestamp else "") + f'Probe: "{q_snippet}"'
        if answer_snippet:
            turn_context += f' | Response excerpt: "{answer_snippet}"'

        # 1. Voice Identity Drift / Proxy Speaker
        if integrity_monitoring_enabled and a.get("voice_mismatch"):
            drift_val = a.get("voice_drift_score", 0.52)
            incidents.append({
                "turn": turn_num,
                "type": "Automated Voice-Consistency Signal",
                "severity": "Review signal",
                "proof_statement": (
                    f"The automated voice-consistency signal was flagged (reported drift metric: {drift_val:.2f}). "
                    "This signal cannot establish identity or confirm that another person answered."
                ),
                "context": turn_context
            })

        # 2. In-Room Prompting / Secondary Speaker
        if integrity_monitoring_enabled and (a.get("noise_detected") or a.get("proxy_speaker_suspected")):
            incidents.append({
                "turn": turn_num,
                "type": "Background Audio / Possible Speaking Mismatch",
                "severity": "Review signal",
                "proof_statement": (
                    "The audio analysis flagged background sound or a possible audio/video speaking mismatch. "
                    "It does not identify another speaker or establish that anyone provided assistance."
                ),
                "context": turn_context
            })

        # 3. AI Script Reading / Synthetic Response Injection
        perp = a.get("perplexity", 50.0)
        disflu = a.get("disfluency_rate", 0.1)
        ans_len = len(turn.get("answer", "").split())
        if integrity_monitoring_enabled and ((perp < 22.0 and disflu < 0.02 and ans_len > 25) or a.get("ai_script_detected")):
            incidents.append({
                "turn": turn_num,
                "type": "Answer-Text Statistical Signal",
                "severity": "Review signal",
                "proof_statement": (
                    f"Answer-text heuristics flagged this response (perplexity metric: {perp:.1f}, "
                    f"disfluency metric: {disflu:.3f}, answer length: {ans_len} words). "
                    "These measures cannot identify the source of the text or determine whether AI was used."
                ),
                "context": turn_context
            })

        if integrity_monitoring_enabled and a.get("assistance_request_detected"):
            incidents.append({
                "turn": turn_num,
                "type": "Explicit Request for an Answer or Solution",
                "severity": "Review signal",
                "proof_statement": (
                    "The transcript matched a phrase pattern that directly asks the interviewer to provide an answer or solution. "
                    "This is a transcript-based flag, may be misrecognized, and does not establish that outside assistance was used."
                ),
                "context": turn_context
            })

        # 4. Multi-Face / Visual Deflection
        if integrity_monitoring_enabled and a.get("multiple_faces"):
            incidents.append({
                "turn": turn_num,
                "type": "Camera Flag: More Than One Face",
                "severity": "Review signal",
                "proof_statement": (
                    "Camera-frame analysis flagged more than one face in the frame. It does not identify the people or "
                    "show whether anyone assisted with the interview."
                ),
                "context": turn_context
            })
        elif integrity_monitoring_enabled and a.get("gaze_off_screen"):
            incidents.append({
                "turn": turn_num,
                "type": "Camera Flag: Off-Center Gaze or Head Pose",
                "severity": "Review signal",
                "proof_statement": (
                    "Camera-frame analysis flagged off-center gaze or head pose in sampled frames during this response. "
                    "The system does not measure how long the candidate looked away or determine what they were looking at."
                ),
                "context": turn_context
            })

        # 5. Professional Conduct Violations (Profanity / Abusive Language)
        if a.get("conduct_violation") and a["conduct_violation"].get("is_violation"):
            cv = a["conduct_violation"]
            incidents.append({
                "turn": turn_num,
                "type": "Automated Conduct Classifier Flag",
                "severity": "Review signal",
                "proof_statement": (
                    f"An automated conduct classifier flagged this transcript excerpt: '{cv.get('flagged_snippet')}'. "
                    f"Turn {turn_num} response: \"{turn.get('answer', '')[:240]}\". "
                    "Review the full exchange; this flag is not a policy finding."
                ),
                "context": turn_context
            })

    # Attach each answer-derived flag to its turn timestamp for temporal context.
    for turn_index, turn in enumerate(history, start=1):
        turn_timestamp = turn.get("timestamp")
        if turn_timestamp:
            for incident in incidents:
                if incident.get("turn") == turn_index:
                    incident.setdefault("timestamp", turn_timestamp)

    # 6. Session-Level Computer Vision & Live Proctoring Incidents (from WebSocket telemetry)
    if integrity_monitoring_enabled and resume_profile and "proctoring_incidents" in resume_profile:
        for p_inc in resume_profile["proctoring_incidents"]:
            inc_type = p_inc.get("type", "")
            detail = p_inc.get("detail", "")
            ts = p_inc.get("timestamp", "")
            if any(inc["proof_statement"] == detail for inc in incidents):
                continue
            incidents.append({
                "turn": "Live Proctoring",
                "type": inc_type.replace("_", " ").title(),
                "severity": "Review signal",
                "proof_statement": detail,
                "context": f"Timestamp: {ts}" if ts else "Continuous Session Monitoring",
                "timestamp": ts,
            })

    # Correlate only distinct signal sources that occurred close in time.
    # Head pose and eye gaze are already one camera signal in ProctorState.
    def source_family(incident_type: str) -> Optional[str]:
        label = incident_type.lower()
        if any(term in label for term in ("answer-text", "explicit request", "ai script")):
            return "answer"
        if any(term in label for term in ("camera", "gaze", "face")):
            return "camera"
        if any(term in label for term in ("audio", "voice", "speaker")):
            return "audio"
        return None

    def parse_timestamp(value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
        except (TypeError, ValueError):
            return None

    parsed_events = [(item, source_family(item.get("type", "")), parse_timestamp(item.get("timestamp"))) for item in incidents]
    for incident, family, occurred_at in parsed_events:
        if not family or not occurred_at:
            continue
        related = []
        for other, other_family, other_at in parsed_events:
            if other is incident or not other_family or other_family == family or not other_at:
                continue
            if abs((occurred_at - other_at).total_seconds()) <= 60:
                label = other.get("type", "Review signal")
                if label not in related:
                    related.append(label)
        if related:
            incident["related_signals"] = related
            incident["coordination_note"] = (
                "Different signal sources occurred within 60 seconds: " + ", ".join(related) + ". "
                "Timing is context only; the signals are not calibrated as independent confirmation and do not establish misconduct."
            )

    return incidents


def _extract_jd_checklist(jd_text: str, resume_profile: Dict[str, Any], history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compare requirement keywords with answers and show matching text without
    treating keyword overlap as proof of skill.
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

        answer_matches = []
        for i, turn in enumerate(history, start=1):
            answer = str(turn.get("answer", "")).strip()
            answer_words = set(re.findall(r"[a-z0-9+#.-]+", answer.lower()))
            hits = sum(1 for word in item_words if word in answer_words)
            if hits:
                answer_matches.append((hits, i, answer))

        if answer_matches:
            _, matched_turn, answer = max(answer_matches, key=lambda item: item[0])
            status = "Related terms in answer"
            status_class = "status-amber"
            evidence_str = f'Turn {matched_turn}: “{answer[:360]}”'
        else:
            raw_resume = str(resume_profile.get("raw_text", "")).lower()
            resume_words = set(re.findall(r"[a-z0-9+#.-]+", raw_resume))
            resume_hits = sum(1 for word in item_words if word in resume_words)
            if resume_hits:
                status = "Resume keyword match only"
                status_class = "status-blue"
                evidence_str = "Matching terms appear in the resume; no related interview answer was found. This is not skill verification."
            else:
                status = "Not covered in interview"
                status_class = "status-slate"
                evidence_str = "No matching answer text found for this requirement."

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
    Generate a human-review interview summary as PDF and HTML.
    Automated rubric estimates are paired with response excerpts and are not
    hiring recommendations or findings of misconduct.

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

        logger.info(f"Generating interview review summary for {len(history)} turns")

        n = len(history)

        # Rubric aggregates are shown only as review context, never as an
        # automated hiring, rejection, or misconduct decision.
        verdict = "HUMAN REVIEW REQUIRED"
        verdict_summary = (
            "This report summarizes automated rubric estimates and interview content. It does not recommend hiring, "
            "rejection, or a finding of misconduct. Review the cited responses and apply the employer's structured rubric."
        )
        review_guidance = "Compare the cited answer excerpts with the job-related rubric; record any decision and rationale separately."
        round_2_questions = [
            "Ask the candidate to explain the design trade-offs in one cited project response.",
            "Use a short job-related work sample to clarify any competency the interview did not cover."
        ]

        # Integrity telemetry remains separate from assessment and hiring review.
        integrity_incidents = _extract_integrity_incidents(history, resume_profile)
        if not resume_profile.get("integrity_monitoring_enabled", False):
            integrity_status = "Optional integrity monitoring disabled"
        elif not integrity_incidents:
            integrity_status = "No automated signals flagged (not identity verification)"
        else:
            integrity_status = f"{len(integrity_incidents)} unverified signal(s); human review only"

        dynamics_summary = "No automated conclusion about confidence, composure, or honesty is made from speech or camera behavior. See the cited transcript responses below."

        turn_records = _extract_turn_records(history)
        competency_sections = _build_competency_sections(history)
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
            "review_guidance": review_guidance,
            "round_2_questions": round_2_questions,
            "competency_sections": competency_sections,
            "turn_records": turn_records,
            "integrity_status": integrity_status,
            "integrity_incidents": integrity_incidents,
            "browser_monitoring_enabled": bool(resume_profile.get("integrity_monitoring_enabled", False)),
            "browser_event_summary": _summarize_browser_interaction_events(
                resume_profile.get("browser_interaction_events", [])
            ),
            "jd_checklist": jd_checklist,
            "dynamics_summary": dynamics_summary,
        }



        # Standalone HTML template formatted for browser rendering and PDF engines
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Interview Review Summary • {{ candidate_name }}</title>
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
        <div class="badge-confidential">CONFIDENTIAL &bull; INTERVIEW REVIEW SUMMARY</div>
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

    <!-- Non-decisional review summary -->
    <div class="verdict-box">
        <div class="verdict-tag">&bull; Interview Review Status: {{ verdict }}</div>
        <div class="verdict-desc">{{ verdict_summary }}</div>
        <div class="suggestion-box">
            <strong>Reviewer guidance:</strong> {{ review_guidance }}
        </div>
    </div>

    <!-- Integrity Incidents (Only rendered if anomalies detected) -->
    {% if integrity_incidents %}
    <div style="background: #fef2f2; border: 1pt solid #fecaca; border-left: 4pt solid #dc2626; padding: 10pt 12pt; margin: 12pt 0;">
        <div style="font-size: 9.5pt; font-weight: bold; color: #991b1b; text-transform: uppercase; margin-bottom: 4pt;">
            &excl; Automated Integrity Signals (Requires Human Review)
        </div>
        <p style="font-size: 8.5pt; color: #7f1d1d; margin: 0 0 8pt 0;">
            The system flagged {{ integrity_incidents | length }} signal event(s). These automated signals may be inaccurate, do not prove misconduct, and should not be used alone to make a hiring decision.
        </p>
        {% for inc in integrity_incidents %}
        <div style="background: #ffffff; border: 1pt solid #fca5a5; padding: 6pt 10pt; margin-bottom: 6pt;">
            <div style="font-size: 9pt; font-weight: bold; color: #991b1b;">
                Turn {{ inc.turn }} &bull; {{ inc.type }}
                <span style="float: right; font-size: 7.5pt; background: #fee2e2; color: #991b1b; padding: 2pt 6pt; border-radius: 2pt;">{{ inc.severity }}</span>
            </div>
            <div style="font-size: 8pt; color: #64748b; margin: 2pt 0 4pt 0;">{{ inc.context }}</div>
            {% if inc.coordination_note %}
            <div style="font-size: 8pt; color: #7c2d12; background: #fff7ed; padding: 4pt 6pt; margin-bottom: 4pt; border-left: 2pt solid #fb923c;">
                <strong>Related timing context:</strong> {{ inc.coordination_note }}
            </div>
            {% endif %}
            <div style="font-size: 8.5pt; color: #1e293b; background: #fef2f2; padding: 4pt 6pt; border-left: 2pt solid #dc2626;">
                <strong>Signal details (unverified):</strong> {{ inc.proof_statement }}
            </div>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <!-- Optional Browser Interaction Context -->
    <div style="background: #f8fafc; border: 1pt solid #cbd5e1; padding: 10pt 12pt; margin: 12pt 0;">
        <div style="font-size: 9.5pt; font-weight: bold; color: #334155; text-transform: uppercase; margin-bottom: 4pt;">
            Optional Browser Interaction Context
        </div>
        {% if browser_monitoring_enabled %}
            <p style="font-size: 8.5pt; color: #475569; margin: 0 0 7pt 0;">
                The candidate opted in to limited interaction monitoring. Browser-reported page/focus/paste events do not identify a website, browser extension, copied text source, or activity on another device, and are never scored. Separate camera/answer-text flags are unverified context, not proof of tool use, and require human review.
            </p>
            {% if browser_event_summary %}
                {% for event in browser_event_summary %}
                <div style="font-size: 8.5pt; color: #334155; padding: 3pt 0;">
                    {{ event.label }}: {{ event.count }} event(s)
                    {% if event.first_seen %}(first: {{ event.first_seen }}; last: {{ event.last_seen }}){% endif %}
                </div>
                {% endfor %}
            {% else %}
                <p style="font-size: 8.5pt; color: #475569; margin: 0;">No browser interaction events were recorded.</p>
            {% endif %}
        {% else %}
            <p style="font-size: 8.5pt; color: #475569; margin: 0;">Optional integrity monitoring was not enabled; no browser interaction events, camera frames, or AI-script flags were collected for integrity review.</p>
        {% endif %}
    </div>

    <!-- Competency Observations -->
    <h2>Rubric Estimates With Interview Evidence</h2>
    {% for comp in competency_sections %}
    <div class="comp-card">
        <div class="comp-head">
            {{ comp.name }}
            <span class="comp-tier">{{ comp.tier }}</span>
        </div>
        <div class="comp-sub">{{ comp.summary }}</div>
        <div class="comp-obs">
            <strong>Automated rubric note:</strong> {{ comp.observation }}
        </div>
        <div class="comp-obs">
            <strong>Response for review:</strong>
            {% if comp.evidence_turn %}Turn {{ comp.evidence_turn }}: “{{ comp.evidence_quote }}”{% else %}No answer transcript is available for this dimension.{% endif %}
        </div>
    </div>
    {% endfor %}

    <!-- Job Description Checklist Alignment -->
    <h2>Role Requirement Coverage &amp; Transcript References</h2>
    <p style="font-size: 8pt; color: #64748b; margin: 0 0 6pt 0;">
        Keyword overlap is a navigation aid only; it does not verify skill or experience. Review the cited response in context.
    </p>
    <table class="checklist-table">
        <thead>
            <tr>
                <th style="width: 32%;">Target Role Requirement</th>
                <th style="width: 23%;">Discussion Status</th>
                <th style="width: 45%;">Interview Discussion Evidence</th>
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

    <!-- Turn-by-Turn Transcript Record -->
    <h2>Interview Transcript &amp; Automated Notes</h2>

    {% for turn in turn_records %}
    <div class="turn-box">
        <div class="turn-head">
            Turn {{ turn.turn_number }}
            <span class="turn-tag">{{ turn.difficulty_label }}</span>
        </div>
        <div class="turn-q"><strong>Question Asked:</strong> "{{ turn.question }}"</div>
        <div class="turn-ans"><strong>Candidate Statement:</strong> {{ turn.answer_summary }}</div>
        {% if turn.decision_source_label %}<div class="turn-obs"><strong>Interviewer response source:</strong> {{ turn.decision_source_label }}</div>{% endif %}
        <div class="turn-obs"><strong>Automated note (review alongside the response):</strong> {{ turn.observation }}</div>
    </div>
    {% endfor %}

    <!-- Interview context -->
    <h2>Interview Context &amp; Optional Integrity Signals</h2>
    <div class="comp-card" style="background: #f8fafc;">
        <div style="margin-bottom: 4pt;">
            <strong>Interpretation limit:</strong> {{ dynamics_summary }}
        </div>
        <div>
            <strong>Automated Integrity Signals:</strong>
            {% if integrity_incidents %}
            <span style="color: #991b1b; font-weight: bold;">{{ integrity_incidents | length }} signal event(s) need human review.</span> These are not proof of misconduct.
            {% else %}
            <span style="color: #166534; font-weight: bold;">No automated integrity signals were flagged.</span> This does not verify identity or prove that no external tools were used.
            {% endif %}
        </div>

    </div>

    <!-- Suggested follow-up questions -->
    {% if round_2_questions %}
    <div class="r2-box">
        <div class="r2-title">Optional Follow-Up Questions for Human Review</div>
        <ul class="list" style="color: #14532d;">
            {% for q in round_2_questions %}
            <li>{{ q }}</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}

    <!-- Limits of use -->
    <h2>Assessment Limits</h2>
    <div class="comp-card" style="background: #f8fafc; border: 1pt solid #cbd5e1; font-size: 8pt; color: #475569; line-height: 1.4;">
        <p style="margin: 0 0 4pt 0;">
            This report contains automated summaries and rubric estimates from an interview prototype. It has not been independently validated for employment decisions and is not a determination of a candidate's ability, honesty, or eligibility.
        </p>
        <p style="margin: 0 0 4pt 0;">
            A human reviewer should examine the cited responses, account for context and accommodation needs, and make any employment decision independently. Integrity signals are unverified and must not be used alone to make a decision.
        </p>
    </div>

</body>
</html>

        """

        # Resume fields, job descriptions, and interview answers are user input.
        # Escape them in the standalone HTML report to prevent markup/script
        # injection when the report is opened in a browser.
        template = Template(template_str, autoescape=True)
        html_out = template.render(data)

        # 1. Always save clean standalone HTML version alongside PDF
        base_name, _ = os.path.splitext(output_path)
        html_output_path = f"{base_name}.html"
        with open(html_output_path, "w", encoding="utf-8") as f:
            f.write(html_out)
        logger.info(f"Interview summary HTML written to: {html_output_path}")

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
