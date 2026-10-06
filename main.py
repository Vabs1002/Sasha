import sys
import os
import json
from datetime import datetime
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# local imports
from resume_parser import parse_resume
from signal_detector import detect_role
# We'll import analyzer functions directly
from analyzer import (
    get_perplexity,
    get_disfluency_rate,
    get_consistency,
    detect_background_audio_anomaly,
    detect_speech_stress,
    analyze_video_frame_proctoring,
    calculate_eye_contact_score,
    verify_audio_visual_speech_sync
)
from interviewer_agent import (
    get_interviewer_decision,
    search_resume_tool,
    AdaptiveQuestionSelector,
    load_job_description,
    BEHAVIORAL_QUESTIONS,
)


# Voice services
try:
    from voice_services import speech_to_text, text_to_speech
    VOICE_SERVICES_AVAILABLE = True
except ImportError:
    VOICE_SERVICES_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Voice services not available - continuing in text-only mode")

def load_competency_bank(path="competency_bank.yaml"):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_audio_input():
    """
    Get audio input from microphone and convert to text using RUMIK STT.
    Falls back to text input if voice services unavailable or disabled.
    """
    # Check if voice input is enabled
    voice_enabled = os.getenv('ENABLE_VOICE_INPUT', 'false').lower() == 'true'
    services_available = VOICE_SERVICES_AVAILABLE and bool(os.getenv("RUMIK_API_KEY"))

    if voice_enabled and services_available:
        try:
            print("🎤 Listening... (speak now)")
            # In a real implementation, we would:
            # 1. Capture audio from microphone using pyaudio/sounddevice
            # 2. Package it as WAV bytes
            # 3. Call speech_to_text() on the audio data

            # For this implementation, we'll simulate by checking if there's
            # pre-recorded audio or fall back to text input with a notice
            print("Note: Voice input simulation - please type your answer below")
            print("(In full implementation, this would use microphone input)")
            audio_data = None  # Placeholder for actual audio capture

            if audio_data:
                return speech_to_text(audio_data)
            else:
                # Fall back to text input for now
                return input("Your answer: ").strip()

        except Exception as e:
            print(f"Voice input error: {e}")
            print("Falling back to text input...")
            return input("Your answer: ").strip()
    else:
        # Text input mode (default)
        return input("Your answer: ").strip()

def speak_text(text: str):
    """
    Convert text to speech using RUMIK TTS and play it.
    Falls back to text display if voice services unavailable or disabled.
    """
    # Check if voice output is enabled
    voice_enabled = os.getenv('ENABLE_VOICE_OUTPUT', 'false').lower() == 'true'
    services_available = VOICE_SERVICES_AVAILABLE and bool(os.getenv("RUMIK_API_KEY"))

    if voice_enabled and services_available and text and text.strip():
        try:
            print("🔊 Speaking...")
            # In a real implementation, we would:
            # 1. Call text_to_speech(text) to get audio bytes
            # 2. Play the audio bytes using pyaudio/sounddevice or similar

            # For this implementation, we'll simulate by showing what would be spoken
            audio_bytes = text_to_speech(text)
            if sys.platform == "win32":
                try:
                    import winsound
                    winsound.PlaySound(audio_bytes, winsound.SND_MEMORY)
                except Exception as snd_err:
                    pass

            # Also show the text for accessibility/debugging
            print(f"[TTS] {text}")

        except Exception as e:
            print(f"Voice output error: {e}")
            print(f"[TTS Fallback] {text}")
    else:
        # Text output mode (default) - just show the text
        print(f"[TTS] {text}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <resume.pdf|resume.docx> [job_description.txt]")
        sys.exit(1)
    resume_path = sys.argv[1]
    if not os.path.exists(resume_path):
        print(f"Resume file not found: {resume_path}")
        sys.exit(1)

    # Show voice status
    voice_input_enabled = os.getenv('ENABLE_VOICE_INPUT', 'false').lower() == 'true'
    voice_output_enabled = os.getenv('ENABLE_VOICE_OUTPUT', 'false').lower() == 'true'
    services_available = VOICE_SERVICES_AVAILABLE and bool(os.getenv("RUMIK_API_KEY"))

    if voice_input_enabled or voice_output_enabled:
        if services_available:
            print("🎤 Voice features: ENABLED")
            if voice_input_enabled:
                print("   - Speech-to-Text: ON")
            if voice_output_enabled:
                print("   - Text-to-Speech: ON")
        else:
            print("⚠️  Voice features: DISABLED (missing RUMIK_API_KEY or voice services)")
            print("    Set RUMIK_API_KEY environment variable to enable voice features")
    else:
        print("💬 Voice features: DISABLED (text-only mode)")

    # GAP 11: Accent fairness mode — disables disfluency + perplexity as stress signals
    # for non-native English speakers, preventing research-documented bias.
    native_speaker = os.getenv('ACCENT_FAIRNESS_MODE', 'false').lower() != 'true'
    if not native_speaker:
        print("🌐 Accent Fairness Mode: ON  (disfluency/perplexity not used as stress signals)")

    print("-" * 50)

    # 1. Parse resume
    print("Parsing resume...")
    resume_profile_raw = parse_resume(resume_path)

    # 2. Detect role
    print("Detecting role...")
    with open("role_signals.yaml", 'r') as f:
        import yaml
        signals = yaml.safe_load(f)
    role_result = detect_role(resume_profile_raw['raw_text'], signals)
    resume_profile = {
        **resume_profile_raw,
        "role": role_result['role'],
        "role_confidence": role_result['confidence']
    }
    print(f"Detected role: {resume_profile['role']} (confidence {resume_profile['confidence']:.3f})")
    print(f"Experience: {resume_profile['years_experience']} years ({resume_profile['level']})")
    print()

    # GAP 10: Load Job Description if provided as second CLI argument
    jd_text = ""
    jd_summary = ""
    if len(sys.argv) >= 3:
        jd_path = sys.argv[2]
        if os.path.exists(jd_path):
            with open(jd_path, 'r', encoding='utf-8') as f:
                jd_text = f.read()
            # Merge JD into resume context for hybrid RAG retrieval
            resume_profile['raw_text'] = load_job_description(jd_text, resume_profile_raw['raw_text'])
            resume_profile['jd_text'] = jd_text
            # Keep a short summary for prompt injection (first 500 chars)
            jd_summary = jd_text[:500].strip()
            print(f"📋 Job Description loaded ({len(jd_text)} chars) — Sasha will probe JD requirements.")
        else:
            print(f"⚠️  JD file not found: {jd_path} — continuing without JD context.")

    # 3. Initialize interview state
    history = []
    competency_bank = load_competency_bank("competency_bank.yaml")

    # GAP 4: Adaptive difficulty — seeds from resume level, adjusts live each turn
    difficulty_selector = AdaptiveQuestionSelector(resume_profile['level'])
    print(f"🎯 Starting difficulty: {difficulty_selector.label()}")

    # Behavioral question rotation state (ask one behavioral Q every 3 technical turns)
    behavioral_categories = list(BEHAVIORAL_QUESTIONS.keys())
    behavioral_asked = 0  # how many behavioral questions we've asked
    _behavioral_turn_counter = 0  # counts technical turns since last behavioral

    # 4. Warm-up question based on resume
    cand_name = resume_profile.get('name')
    name_greeting = f", {cand_name}" if cand_name and cand_name.lower() != "candidate" else ""
    first_question = (
        f"Thanks for sharing your background{name_greeting}! I see you have "
        f"{resume_profile['years_experience']} years of experience as a "
        f"{resume_profile['level']} engineer specializing in "
        f"{resume_profile['role'].replace('_', ' ')}. "
        f"To start, could you tell me about the project you're most proud of from your resume?"
    )
    speak_text(first_question)
    print(f"\nInterviewer: {first_question}")

    next_question = None
    max_turns = 8  # safety limit

    for turn in range(max_turns):
        # Get answer (voice or text input)
        answer = get_audio_input()
        if answer.lower() in ['exit', 'quit', 'stop']:
            print("Interview ended by candidate.")
            break

        # 5. Analyze answer
        transcript = answer
        perp = get_perplexity(transcript)
        disflu = get_disfluency_rate(transcript)
        consist = get_consistency(resume_profile_raw['raw_text'], transcript)

        # GAP 5: Stress / knowledge-gap detection (bias-corrected for non-native speakers)
        stress_result = detect_speech_stress(transcript, disflu, perp, native_speaker=native_speaker)
        stress_score = stress_result['stress_score']
        knowledge_gap = stress_result['knowledge_gap_detected']

        # Proctoring: Check background noise / second speaker prompting
        audio_check = detect_background_audio_anomaly(transcript)
        if audio_check.get("noise_detected") and audio_check.get("warning_message"):
            warning_msg = audio_check["warning_message"]
            print(f"\n⚠️  [Integrity Alert]: {warning_msg}")
            speak_text(warning_msg)

        analysis = {
            'transcript': transcript,
            'perplexity': perp,
            'disfluency_rate': disflu,
            'consistency': consist,
            'noise_detected': audio_check.get("noise_detected", False),
            'stress_score': stress_score,
            'stress_signals': stress_result['stress_signals'],
            'knowledge_gap_detected': knowledge_gap,
            'difficulty_level': difficulty_selector.difficulty,  # log for HR report
        }
        print(
            f"[Analysis] Perplexity: {perp:.1f} | Disfluency: {disflu:.3f} | "
            f"Consistency: {consist:.3f} | Stress: {stress_score:.2f} | "
            f"Difficulty: {difficulty_selector.label().split('—')[0].strip()}"
        )

        # GAP 5: Empathy trigger — Sasha speaks before asking next question
        empathy_triggered = False
        if stress_result.get('empathy_prompt'):
            speak_text(stress_result['empathy_prompt'])
            print(f"[Empathy] {stress_result['empathy_prompt']}")
            empathy_triggered = True
        analysis['empathy_triggered'] = empathy_triggered

        # 6. Get interviewer decision (with adaptive difficulty + JD context)
        decision = get_interviewer_decision(
            resume_profile, history, analysis,
            "competency_bank.yaml",
            difficulty_label=difficulty_selector.label(),
            jd_summary=jd_summary,
        )

        # GAP 4: Update difficulty for next turn based on this turn's signals
        difficulty_selector.update(consist, perp, stress_score, knowledge_gap)

        # 7. Log turn
        history.append({
            'question': first_question if turn == 0 else next_question,
            'answer': answer,
            'analysis': analysis,
            'assessment': decision['assessment'],
            'decision': decision['decision']
        })

        # 8. Next question logic
        next_question = decision.get('next_question')
        action = decision['decision']['action']
        reasoning = decision['decision']['reasoning']
        print(f"\n[Interviewer Decision] Action: {action}, Reasoning: {reasoning}")

        if action == 'end_early':
            speak_text("Thank you for your time. We'll be in touch.")
            print("Interviewer: Thank you for your time. We'll be in touch.")
            break

        # Inject a behavioral question every 3 technical turns
        _behavioral_turn_counter += 1
        if _behavioral_turn_counter >= 3 and behavioral_asked < len(behavioral_categories):
            cat = behavioral_categories[behavioral_asked % len(behavioral_categories)]
            import random
            behavioral_q = random.choice(BEHAVIORAL_QUESTIONS[cat])
            next_question = behavioral_q
            behavioral_asked += 1
            _behavioral_turn_counter = 0
            print(f"\n[Behavioral Turn — {cat.replace('_', ' ').title()}]")

        if action in ['follow_up', 'move_on'] and next_question:
            speak_text(next_question)
            print(f"\nInterviewer: {next_question}")

        if not next_question:
            print("No next question generated; ending interview.")
            break

    # 9. Generate HR report
    print("\nGenerating report...")
    from report_generator import generate_report
    generate_report(resume_profile, history)
    print("Report generated: interview_report.pdf")

if __name__ == "__main__":
    main()