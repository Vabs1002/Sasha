import os
import json
import yaml
import numpy as np
from openai import OpenAI
import pickle
import logging
from typing import Dict, Any, Tuple, Optional, Union, List
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import faiss
except ImportError:
    faiss = None

# Import _build_faiss_index and _embedder from analyzer for use in tests and functionality
try:
    from analyzer import _build_faiss_index, _embedder
except ImportError:
    # Fallback definitions if analyzer is not available
    def _build_faiss_index(text: str):
        return None, None
    _embedder = None

# Import voice services for STT/TTS capabilities
try:
    from voice_services import speech_to_text, text_to_speech
    VOICE_SERVICES_AVAILABLE = True
except ImportError:
    VOICE_SERVICES_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.debug("Voice services not available - continuing in text-only mode")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenAI-compatible client (Grok, OpenAI, Groq, Ollama, or Custom)
def init_llm_client():
    """
    Auto-detect configured LLM provider and model.
    Checks:
    1. Custom endpoint (LLM_BASE_URL + LLM_API_KEY)
    2. Groq (GROQ_API_KEY) -> llama-3.3-70b-versatile
    3. OpenAI (OPENAI_API_KEY) -> gpt-4o-mini
    4. Grok / xAI (GROK_API_KEY) -> grok-beta
    5. Ollama (OLLAMA_BASE_URL) -> llama3
    """
    custom_base = os.getenv("LLM_BASE_URL")
    custom_key = os.getenv("LLM_API_KEY")
    if custom_base and custom_key:
        return OpenAI(base_url=custom_base, api_key=custom_key), os.getenv("LLM_MODEL", "gpt-4o-mini")

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key), os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return OpenAI(api_key=openai_key), os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    grok_key = os.getenv("GROK_API_KEY")
    if grok_key:
        return OpenAI(base_url="https://api.x.ai/v1", api_key=grok_key), os.getenv("GROK_MODEL", "grok-beta")

    ollama_url = os.getenv("OLLAMA_BASE_URL")
    if ollama_url:
        return OpenAI(base_url=ollama_url, api_key="ollama"), os.getenv("OLLAMA_MODEL", "llama3")

    return None, "grok-beta"

try:
    client, default_llm_model = init_llm_client()
    if client:
        logger.info(f"Initialized LLM client using model: {default_llm_model}")
    else:
        logger.info("No external LLM key provided. Dynamic fallback will be used.")
except Exception as e:
    logger.warning(f"Failed to initialize LLM client: {str(e)}")
    client, default_llm_model = None, "grok-beta"

# ---------------------------------------------------------------------------
# Behavioral question bank (non-technical) — assesses growth mindset,
# learning agility, adaptability, and team management.
# Used by top companies: Google (Googleyness), Amazon (Leadership Principles),
# Microsoft (Growth Mindset), Meta, Adobe.
# ---------------------------------------------------------------------------
BEHAVIORAL_QUESTIONS = {
    "growth_mindset": [
        "Tell me about a time you had to learn a completely new technology or skill quickly to deliver a project. How did you approach it?",
        "Describe a situation where a project failed or didn't go as planned. What did you learn from it, and what would you do differently?",
        "Tell me about a time you received feedback that was hard to hear. How did you react, and what changed afterwards?",
    ],
    "adaptability": [
        "Tell me about a time when priorities shifted suddenly midway through a project. How did you manage the change and still deliver?",
        "Describe a situation where you had to make an important decision without having all the information you needed.",
        "Give me an example of a time you were asked to do something you had never done before. How did you handle the uncertainty?",
    ],
    "ownership_and_impact": [
        "Tell me about a project where you took ownership beyond your defined role. What drove you to do that and what was the outcome?",
        "Describe a time you identified a problem that wasn't your responsibility but you solved it anyway.",
    ],
    "collaboration_and_conflict": [
        "Tell me about a technical disagreement you had with a teammate or manager. How did you resolve it?",
        "Describe a time when you had to work with someone whose working style was very different from yours.",
    ],
    "self_awareness": [
        "What's one technical area where you know you're still weak, and what are you actively doing about it?",
        "If I asked your last team lead to describe both your biggest strength and your biggest blind spot, what would they say?",
    ],
}

class AdaptiveQuestionSelector:
    """
    IRT-inspired (Item Response Theory) adaptive difficulty tracker.

    Seeds starting difficulty from the candidate's resume experience level,
    then adjusts up or down live after each turn based on performance signals.
    The difficulty label is injected into the LLM prompt so questions
    naturally calibrate — no separate question bank needed.

    Difficulty scale:
      0.0–0.35  → easy      (surface concepts, fresher-appropriate)
      0.35–0.65 → medium    (trade-offs, design choices, alternatives)
      0.65–1.0  → hard      (failure modes, scale, architectural weaknesses)

    Starting levels by resume experience:
      fresher (0-1 yr)  → 0.20
      junior  (1-3 yr)  → 0.40
      mid     (3-6 yr)  → 0.60
      senior  (6+ yr)   → 0.80
    """
    _LEVEL_START: Dict[str, float] = {
        "fresher": 0.20,
        "junior":  0.40,
        "mid":     0.60,
        "senior":  0.80,
    }
    _STEP_UP   = 0.15   # how much to increase when candidate is doing well
    _STEP_DOWN = 0.15   # how much to decrease when candidate is struggling

    def __init__(self, level: str):
        """
        Args:
            level: resume experience level ('fresher', 'junior', 'mid', 'senior')
        """
        self.difficulty: float = self._LEVEL_START.get(level, 0.50)
        self._history: List[float] = [self.difficulty]

    def update(self, consistency: float, perplexity: float,
               stress_score: float = 0.0, knowledge_gap: bool = False) -> None:
        """
        Adjust difficulty after one turn.

        Push harder if:  candidate knows the material AND is not stressed
        Back off if:     candidate shows a knowledge gap OR is stressed
        Hold if:         ambiguous signals
        """
        candidate_strong = (consistency > 0.70 and perplexity > 45
                            and stress_score < 0.40 and not knowledge_gap)
        candidate_struggling = (consistency < 0.40 or stress_score > 0.60
                                or knowledge_gap)

        if candidate_strong:
            self.difficulty = round(min(1.0, self.difficulty + self._STEP_UP), 2)
        elif candidate_struggling:
            self.difficulty = round(max(0.10, self.difficulty - self._STEP_DOWN), 2)

        self._history.append(round(self.difficulty, 3))

    def label(self) -> str:
        """Return a plain-English label to inject into the LLM prompt."""
        if self.difficulty < 0.35:
            return "easy — ask surface-level questions suitable for freshers"
        if self.difficulty < 0.65:
            return "medium — probe trade-offs, design choices, and alternatives"
        return "hard (senior/principal level) — probe failure modes, scale assumptions, architectural weaknesses, and edge cases"

    @property
    def history(self) -> List[float]:
        """Difficulty value at each turn, for the HR report log."""
        return list(self._history)


def load_job_description(jd_text: str, resume_text: str) -> str:
    """
    Merge Job Description context into resume text before FAISS indexing.

    By appending the JD to the resume text we get hybrid RRF retrieval
    across BOTH sources for free — no second index needed.
    JD requirements become part of the retrieval context so Sasha can probe
    whether the candidate meets them specifically.

    Args:
        jd_text:     Raw text of the job description.
        resume_text: Raw text of the candidate's resume.

    Returns:
        Combined context string tagged with source headers.
    """
    if not jd_text or not jd_text.strip():
        return resume_text
    return resume_text + "\n\n--- JOB DESCRIPTION REQUIREMENTS ---\n" + jd_text.strip()


def load_competency_bank(yaml_path: str) -> Dict[str, Any]:
    """
    Load competency bank from YAML file.

    Args:
        yaml_path: Path to the YAML file

    Returns:
        Competency bank dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
        Exception: For other file errors
    """
    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Competency bank file not found: {yaml_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in competency bank {yaml_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading competency bank {yaml_path}: {str(e)}")
        raise

def load_competency_selector_model(model_path: str = 'competency_selector.pkl') -> Optional[Any]:
    """
    Load competency selector model from pickle file.

    Args:
        model_path: Path to the model file

    Returns:
        Loaded model or None if file doesn't exist
    """
    if not os.path.exists(model_path):
        logger.debug(f"Competency selector model not found: {model_path}")
        return None
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Error loading competency selector model {model_path}: {str(e)}")
        return None

def extract_features(resume_profile: Dict[str, Any], history: list, analysis: Dict[str, Any],
                    competency_bank: Dict[str, Any] = None, model: Any = None) -> np.ndarray:
    """
    Extract a feature vector for competency selection.

    Args:
        resume_profile: Parsed resume data
        history: Interview history
        analysis: Linguistic analysis results
        competency_bank: Competency bank dictionary (optional, for compatibility)
        model: ML model (optional, for compatibility)

    Returns:
        Feature vector as numpy array
    """
    try:
        # Role one-hot
        roles = ['SDE', 'ML Engineer', 'Gen-AI Focus', 'Data Science']
        role_vec = [1 if resume_profile.get('role') == r else 0 for r in roles]
        # Level one-hot
        levels = ['fresher', 'junior', 'mid', 'senior']
        level_vec = [1 if resume_profile.get('level') == l else 0 for l in levels]
        # Years experience (normalize by 10)
        years_norm = min(resume_profile.get('years_experience', 0) / 10.0, 1.0)
        # History averages
        if history:
            avg_ownership = np.mean([t.get('assessment', {}).get('ownership', 0) for t in history])
            avg_depth = np.mean([t.get('assessment', {}).get('depth', 0) for t in history])
            avg_impact = np.mean([t.get('assessment', {}).get('impact', 0) for t in history])
            avg_learning = np.mean([t.get('assessment', {}).get('learning', 0) for t in history])
            avg_communication = np.mean([t.get('assessment', {}).get('communication', 0) for t in history])
            history_len = len(history) / 10.0  # normalize
        else:
            avg_ownership = avg_depth = avg_impact = avg_learning = avg_communication = 0.0
            history_len = 0.0
        # Current analysis
        perplexity = analysis.get('perplexity', 50.0)
        # Normalize perplexity: assume typical range 0-200, we want lower is better? We'll just use raw.
        # We'll normalize to 0-1 assuming max 200
        perp_norm = min(perplexity / 200.0, 1.0)
        disflu = analysis.get('disfluency_rate', 0.1)
        # disfluency typical 0-0.5, we'll clip
        disflu_norm = min(disflu / 0.5, 1.0)
        consist = analysis.get('consistency', 0.5)  # already 0-1
        # Build feature vector
        feature_vec = [
            years_norm,
        ] + role_vec + level_vec + [
            avg_ownership, avg_depth, avg_impact, avg_learning, avg_communication,
            history_len,
            perp_norm, disflu_norm, consist
        ]
        # Normalize each feature to 0-1 range? We'll assume they are already scaled.
        return np.array(feature_vec).reshape(1, -1)
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        # Return a default feature vector
        default_features = [0.5] * 15  # 1 (years) + 4 (roles) + 4 (levels) + 5 (history) + 1 (perp) + 1 (disflu) + 1 (consist)
        return np.array(default_features).reshape(1, -1)

def select_competency(resume_profile: Dict[str, Any], history: list, analysis: Dict[str, Any],
                     competency_bank: Dict[str, Any], model: Any) -> str:
    """
    Select a competency to focus on using the ML model or rule-based fallback.

    Args:
        resume_profile: Parsed resume data
        history: Interview history
        analysis: Linguistic analysis results
        competency_bank: Competency bank dictionary
        model: Trained ML model for competency selection (can be None)

    Returns:
        Name of selected competency
    """
    try:
        if model is None:
            # Fallback to rule-based: pick the first competency of the detected role
            role_bank = competency_bank.get(resume_profile.get('role', ''), {})
            comps = role_bank.get('competencies', [])
            if comps:
                return comps[0]['name']
            return "Communication"  # default

        # Extract features
        features = extract_features(resume_profile, history, analysis)

        # Build flattened list of all competencies for model prediction
        all_competencies = []
        for role_key, role_data in competency_bank.items():
            for comp in role_data.get('competencies', []):
                all_competencies.append({
                    'role': role_key,
                    'name': comp['name'],
                    'signals': comp.get('signals', []),
                    'follow_up_templates': comp.get('follow_up_templates', [])
                })

        # The model was trained to predict the index of the competency to focus on.
        # We'll assume the model outputs a single integer class.
        try:
            pred_idx = model.predict(features)[0]
            # Ensure pred_idx is within bounds
            if pred_idx < 0 or pred_idx >= len(all_competencies):
                pred_idx = 0
            selected = all_competencies[pred_idx]
            return selected['name']
        except Exception as e:
            logger.warning(f"Error in ML competency selection: {str(e)}. Falling back to rule-based.")
            # Fallback
            role_bank = competency_bank.get(resume_profile.get('role', ''), {})
            comps = role_bank.get('competencies', [])
            if comps:
                return comps[0]['name']
            return "Communication"
    except Exception as e:
        logger.error(f"Error in select_competency: {str(e)}")
        return "Communication"

def format_prompt_basic(resume_profile: Dict[str, Any], history: list, analysis: Dict[str, Any],
                 selected_competency: str, competency_bank: Dict[str, Any]) -> str:
    """
    Format the basic prompt for the interviewer agent LLM (without Agentic RAG tool instructions).

    Args:
        resume_profile: Parsed resume data
        history: Interview history
        analysis: Linguistic analysis results
        selected_competency: Competency to focus on
        competency_bank: Competency bank dictionary

    Returns:
        Formatted prompt string for LLM
    """
    try:
        years = resume_profile.get('years_experience', 0)
        level = resume_profile.get('level', 'unknown')
        skills = ', '.join(resume_profile.get('skills', [])[:10])
        projects = '; '.join(resume_profile.get('projects', [])[:5])

        history_lines = []
        for i, turn in enumerate(history[-3:]):  # last 3 turns
            history_lines.append(f"Turn {i+1}:")
            history_lines.append(f"  Q: {turn.get('question', '')}")
            history_lines.append(f"  A: {turn.get('answer', '')}")
            history_lines.append(f"  Analysis: {turn.get('analysis', {})}")
        history_str = '\n'.join(history_lines) if history_lines else "No previous turns."

        # Get the selected competency details
        comp_details = None
        for role_key, role_data in competency_bank.items():
            for comp in role_data.get('competencies', []):
                if comp.get('name') == selected_competency:
                    comp_details = comp
                    break
            if comp_details:
                break

        prompt = f"""You are a senior technical interviewer at a top tech company (Google/Meta/Adobe) hiring for a {resume_profile.get('role', 'unknown').replace('_', ' ')} role with {years} years of experience ({level}).
Your goal: Accurately assess the candidate's ownership, depth, impact, learning, and communication from their resume-listed experience.
Be supportive but probing—start with acknowledgment before digging deeper.

RESUME SUMMARY:
- Experience: {years} years ({level})
- Top skills: {skills}
- Key projects: {projects}

INTERVIEW HISTORY:
{history_str}

CURRENT ANSWER ANALYSIS:
- Transcript: "{analysis.get('transcript', '')}"
- Perplexity score: {analysis.get('perplexity', 0)} (human-like: 50-150; <30 = likely AI-generated)
- Disfluency rate: {analysis.get('disfluency_rate', 0)} (human-like: 0.1-0.2; <0.05 = suspiciously smooth)
- Consistency with resume: {analysis.get('consistency', 0)} (0-1; <0.4 = poor match)

SELECTED COMPETENCY TO FOCUS ON: {selected_competency}
"""
        if comp_details:
            prompt += f"Competency signals: {', '.join(comp_details.get('signals', []))}\n"
            prompt += "Example follow-up questions for this competency:\n"
            for tmpl in comp_details.get('follow_up_templates', []):
                prompt += f"- \"{tmpl}\"\n"

        prompt += """

YOUR TASK:
1. Assess the answer on:
   - Ownership (0-10): Did they describe THEIR specific contributions?
   - Depth (0-10): Did they explain trade-offs, alternatives, or technical details?
   - Impact (0-10): Did they mention measurables (%, $, time saved)?
   - Learning (0-10): Did they share what they’d do differently or what they learned?
   - Communication (0-10): Was the answer clear, structured, and engaging?
2. Decide next action:
   - "follow_up": If answer lacks ownership/depth/impact/communication (score <6 on any), ask a targeted probe.
   - "move_on": If sufficient signal gathered (≥2 questions on this topic, or scores ≥7 on key competencies).
   - "end_early": ONLY if integrity flags are severe (perplexity <20 AND disfluency <0.03 AND consistency <0.2) — suggesting AI-assisted cheating.
3. Generate a natural, supportive follow-up question (if action is follow_up or move_on) that is relevant to the selected competency: {selected_competency}.

OUTPUT MUST BE VALID JSON:
{
  "assessment": {
    "ownership": <0-10>,
    "depth": <0-10>,
    "impact": <0-10>,
    "learning": <0-10>,
    "communication": <0-10>,
    "notes": "<brief explanation of scores>"
  },
  "decision": {
    "action": "<follow_up|move_on|end_early>",
    "reasoning": "<why you chose this action>",
    "follow_up_topic": "<if action=follow_up: what to probe e.g., 'specific metrics improved', 'trade-offs considered'>"
  },
  "next_question": "<natural, conversational question to ask - ONLY if action is follow_up or move_on>"
}
"""
        return prompt
    except Exception as e:
        logger.error(f"Error formatting prompt: {str(e)}")
        # Return a basic fallback prompt
        return f"""You are a technical interviewer. Assess the candidate's response and provide JSON feedback.

Candidate resume indicates: {resume_profile.get('role', 'unknown')} role with {resume_profile.get('years_experience', 0)} years experience.

Recent answer: {history[-1].get('answer', 'No answer') if history else 'No answers yet'}

Provide assessment in JSON format with ownership, depth, impact, learning, communication scores (0-10) and decision on next action.
"""

def format_prompt_for_agentic_rag(resume_profile: Dict[str, Any], history: list, analysis: Dict[str, Any],
                                 selected_competency: str, competency_bank: Dict[str, Any], resume_text: str,
                                 difficulty_label: str = "medium — probe trade-offs, design choices, and alternatives",
                                 jd_summary: str = "") -> str:
    """
    Format the prompt for the interviewer agent LLM with Agentic RAG tool instructions.

    Args:
        resume_profile: Parsed resume data
        history: Interview history
        analysis: Linguistic analysis results
        selected_competency: Competency to focus on
        competency_bank: Competency bank dictionary
        resume_text: Raw resume text for search tool

    Returns:
        Formatted prompt string for LLM with tool usage instructions
    """
    try:
        years = resume_profile.get('years_experience', 0)
        level = resume_profile.get('level', 'unknown')
        skills = ', '.join(resume_profile.get('skills', [])[:10])
        projects = '; '.join(resume_profile.get('projects', [])[:5])

        history_lines = []
        for i, turn in enumerate(history[-3:]):  # last 3 turns
            history_lines.append(f"Turn {i+1}:")
            history_lines.append(f"  Q: {turn.get('question', '')}")
            history_lines.append(f"  A: {turn.get('answer', '')}")
            history_lines.append(f"  Analysis: {turn.get('analysis', {})}")
        history_str = '\n'.join(history_lines) if history_lines else "No previous turns."

        # Get the selected competency details
        comp_details = None
        for role_key, role_data in competency_bank.items():
            for comp in role_data.get('competencies', []):
                if comp.get('name') == selected_competency:
                    comp_details = comp
                    break
            if comp_details:
                break

        prompt = f"""You are a senior technical interviewer at a top tech company (Google/Meta/Adobe) hiring for a {resume_profile.get('role', 'unknown').replace('_', ' ')} role with {years} years of experience ({level}).
Your goal: Accurately assess the candidate's ownership, depth, impact, learning, and communication from their resume-listed experience.
Be supportive but probing—start with acknowledgment before digging deeper.

You have access to a search_resume tool that lets you search the candidate's resume for specific information. Use this tool when you need to:
- Verify specific claims made by the candidate
- Look for details about projects, technologies, or experiences mentioned
- Check for inconsistencies between what they say and their resume
- Find relevant background to ask targeted follow-up questions

To use the tool, include in your response:
TOOL_USE: search_resume
TOOL_INPUT: "your natural language search query here"

After receiving tool results, continue your reasoning and provide your final JSON response.

RESUME SUMMARY:
- Experience: {years} years ({level})
- Top skills: {skills}
- Key projects: {projects}

INTERVIEW HISTORY:
{history_str}

CURRENT ANSWER ANALYSIS:
- Transcript: "{analysis.get('transcript', '')}"
- Perplexity score: {analysis.get('perplexity', 0)} (human-like: 50-150; <30 = likely AI-generated)
- Disfluency rate: {analysis.get('disfluency_rate', 0)} (human-like: 0.1-0.2; <0.05 = suspiciously smooth)
- Consistency with resume: {analysis.get('consistency', 0)} (0-1; <0.4 = poor match)

SELECTED COMPETENCY TO FOCUS ON: {selected_competency}
QUESTION DIFFICULTY TO TARGET: {difficulty_label}
  - easy: surface-level, clarify basic concepts, suitable for freshers
  - medium: ask about trade-offs, design choices, and alternatives
  - hard: probe failure modes, scale assumptions, architectural weaknesses, edge cases
"""
        if jd_summary:
            prompt += f"\nJOB DESCRIPTION KEY REQUIREMENTS:\n{jd_summary}\n→ Where relevant, probe whether the candidate's experience meets these specific requirements.\n"

        if comp_details:
            prompt += f"Competency signals: {', '.join(comp_details.get('signals', []))}\n"
            prompt += "Example follow-up questions for this competency:\n"
            for tmpl in comp_details.get('follow_up_templates', []):
                prompt += f"- \"{tmpl}\"\n"

        prompt += """

YOUR TASK:
1. Assess the answer on:
   - Ownership (0-10): Did they describe THEIR specific contributions?
   - Depth (0-10): Did they explain trade-offs, alternatives, or technical details?
   - Impact (0-10): Did they mention measurables (%, $, time saved)?
   - Learning (0-10): Did they share what they’d do differently or what they learned?
   - Communication (0-10): Was the answer clear, structured, and engaging?
2. Decide next action:
   - "follow_up": If answer lacks ownership/depth/impact/communication (score <6 on any), ask a targeted probe.
   - "move_on": If sufficient signal gathered (≥2 questions on this topic, or scores ≥7 on key competencies).
   - "end_early": ONLY if integrity flags are severe (perplexity <20 AND disfluency <0.03 AND consistency <0.2) — suggesting AI-assisted cheating.
3. Generate a natural, supportive follow-up question (if action is follow_up or move_on) that is relevant to the selected competency: {selected_competency}.

If you need to use the search_resume tool to gather information before completing your assessment, do so now. Otherwise, provide your final assessment in the JSON format below.

OUTPUT MUST BE VALID JSON (if using tool, include TOOL_USE and TOOL_INPUT lines before the JSON):
{
  "assessment": {
    "ownership": <0-10>,
    "depth": <0-10>,
    "impact": <0-10>,
    "learning": <0-10>,
    "communication": <0-10>,
    "notes": "<brief explanation of scores>"
  },
  "decision": {
    "action": "<follow_up|move_on|end_early>",
    "reasoning": "<why you chose this action>",
    "follow_up_topic": "<if action=follow_up: what to probe e.g., 'specific metrics improved', 'trade-offs considered'>"
  },
  "next_question": "<natural, conversational question to ask - ONLY if action is follow_up or move_on>"
}
"""
        return prompt
    except Exception as e:
        logger.error(f"Error formatting Agentic RAG prompt: {str(e)}")
        # Fallback to basic prompt
        return format_prompt_basic(resume_profile, history, analysis, selected_competency, competency_bank)

def get_interviewer_decision(resume_profile: Dict[str, Any], history: list, analysis: Dict[str, Any],
                           competency_bank_path: str = "competency_bank.yaml", model_path: str = "competency_selector.pkl",
                           difficulty_label: str = "medium — probe trade-offs, design choices, and alternatives",
                           jd_summary: str = "") -> Dict[str, Any]:
    """
    Get interviewer decision from the LLM based on resume profile, history, and analysis.
    Supports Agentic RAG - the LLM can call tools to search the resume for specific information.

    Args:
        resume_profile: Parsed resume data
        history: Interview history
        analysis: Linguistic analysis results
        competency_bank_path: Path to competency bank YAML file
        model_path: Path to competency selector model pickle file
        difficulty_label: Human-readable difficulty instruction for the LLM
                          (from AdaptiveQuestionSelector.label()). Controls
                          whether the next question probes easy, medium, or hard.
        jd_summary: Optional job description text. If provided, Sasha probes
                    whether the candidate meets the JD's specific requirements.

    Returns:
        Dictionary containing assessment, decision, and next_question
    """
    try:
        bank = load_competency_bank(competency_bank_path)
        model = load_competency_selector_model(model_path)
        selected_competency = select_competency(resume_profile, history, analysis, bank, model)

        # Get resume text for Agentic RAG tool
        resume_text = resume_profile.get('raw_text', '')

        # NEW: Process voice input if available in history
        # Check if we have audio data in the history that needs transcription
        answer_text = history[-1].get('answer', '') if history else ''
        voice_enabled = os.getenv('ENABLE_VOICE_INPUT', 'false').lower() == 'true'

        if voice_enabled and VOICE_SERVICES_AVAILABLE and not answer_text:
            # Try to get voice input if no text answer provided
            try:
                # In a real implementation, audio data would come from microphone/audio input
                # For now, we'll check if there's audio data in the history
                audio_data = history[-1].get('audio_data') if history else None
                if audio_data:
                    answer_text = speech_to_text(audio_data)
                    logger.info(f"Using voice input: {answer_text[:50]}...")
            except Exception as e:
                logger.warning(f"Voice input failed, falling back to text: {e}")
                # Keep existing answer_text (may be empty)

        # Update the history with the (potentially) transcribed answer
        if history and answer_text != history[-1].get('answer', ''):
            history[-1] = history[-1].copy()
            history[-1]['answer'] = answer_text

        # Continue with existing analysis using (possibly) updated answer text
        analysis = {
            'perplexity': get_perplexity(answer_text),
            'disfluency_rate': get_disfluency_rate(answer_text),
            'consistency': get_consistency(resume_text, answer_text),
            'transcript': answer_text
        }

        # Initial prompt — now includes adaptive difficulty and JD context
        prompt = format_prompt_for_agentic_rag(
            resume_profile, history, analysis, selected_competency, bank, resume_text,
            difficulty_label=difficulty_label,
            jd_summary=jd_summary
        )

        # Use Grok (OpenAI compatible) to generate
        if client is None:
            logger.warning("OpenAI client not available, using fallback response")
            return _get_fallback_response(resume_profile, history, analysis)

        # First call to LLM - might request tool usage
        response = client.chat.completions.create(
            model=default_llm_model,
            messages=[
                {"role": "system", "content": "You are an expert technical interviewer. You have access to a search_resume tool to find specific information in the candidate's resume. When you need to verify claims, check details, or gather specific information for follow-up questions, use the tool. If you need to use the tool, include TOOL_USE: search_resume and TOOL_INPUT: \"your query\" lines BEFORE your JSON response. Otherwise, provide direct JSON output."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000,  # Increased for tool usage
        )

        # Extract the response content
        response_content = response.choices[0].message.content
        logger.debug(f"LLM raw response: {response_content}")

        # Check if LLM wants to use a tool
        if "TOOL_USE: search_resume" in response_content and "TOOL_INPUT:" in response_content:
            logger.info("LLM requested to use search_resume tool")

            # Extract the tool input
            try:
                # Find the TOOL_INPUT line
                lines = response_content.split('\n')
                tool_input_line = None
                json_start_idx = None

                for i, line in enumerate(lines):
                    if line.startswith("TOOL_INPUT:"):
                        tool_input_line = line
                        json_start_idx = i + 1
                        break

                if tool_input_line is None:
                    # Try to extract from the content directly
                    import re
                    match = re.search(r'TOOL_INPUT:\s*"([^"]*)"', response_content)
                    if match:
                        tool_query = match.group(1)
                    else:
                        # Fallback: extract between quotes after TOOL_INPUT:
                        tool_query = tool_input_line.replace("TOOL_INPUT:", "").strip().strip('"')
                else:
                    # Extract the query from the line
                    tool_query = tool_input_line.replace("TOOL_INPUT:", "").strip().strip('"')

                logger.info(f"Searching resume with query: {tool_query}")

                # Use the tool
                search_results = search_resume_tool(tool_query, resume_text)
                logger.info(f"Search results: {search_results}")

                # Create follow-up prompt with tool results
                follow_up_prompt = f"""Based on the resume search results for query "{tool_query}", here is what I found:
{chr(10).join(['- ' + result for result in search_results]) if search_results else 'No relevant information found.'}

Now continue with your assessment and provide your final JSON response based on all available information including the resume search results above.

{format_prompt_basic(resume_profile, history, analysis, selected_competency, bank)}"""

                # Second call to LLM with tool results
                final_response = client.chat.completions.create(
                    model=default_llm_model,
                    messages=[
                        {"role": "system", "content": "You are an expert technical interviewer. You have already used the search_resume tool to gather information from the candidate's resume. Now provide your final assessment in JSON format."},
                        {"role": "user", "content": follow_up_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=800,
                    response_format={"type": "json_object"}
                )

                # Extract the final JSON
                try:
                    result = json.loads(final_response.choices[0].message.content)
                    logger.info("Successfully received and parsed LLM response after tool usage")
                    return result
                except (json.JSONDecodeError, KeyError, AttributeError) as e:
                    logger.error(f"Failed to parse LLM output after tool usage: {str(e)}")
                    # fallback
                    pass

            except Exception as e:
                logger.error(f"Error processing tool usage: {str(e)}")
                # fallback to direct JSON parsing attempt
                pass

        # If no tool usage was requested, try to parse direct JSON
        try:
            # Find JSON in the response (handle cases where LLM might add extra text)
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1

            if json_start != -1 and json_end != 0:
                json_str = response_content[json_start:json_end]
                result = json.loads(json_str)
                logger.info("Successfully received and parsed LLM response (direct JSON)")
                return result
            else:
                # Try parsing the whole content
                result = json.loads(response_content)
                logger.info("Successfully received and parsed LLM response (full content)")
                return result
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.error(f"Failed to parse LLM output: {str(e)}")
            logger.debug(f"LLM response was: {response_content}")
            # fallback
            return _get_fallback_response(resume_profile, history, analysis)

    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return _get_fallback_response(resume_profile, history, analysis)
    except Exception as e:
        logger.error(f"Error in get_interviewer_decision: {str(e)}")
        return _get_fallback_response(resume_profile, history, analysis)


def _get_fallback_response(resume_profile: Dict[str, Any], history: list, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a fallback response when LLM is not available or fails.

    Args:
        resume_profile: Parsed resume data
        history: Interview history
        analysis: Linguistic analysis results

    Returns:
        Fallback response dictionary
    """
    logger.warning("Using fallback response due to LLM unavailability")

    # Simple heuristic-based assessment
    perplexity = analysis.get('perplexity', 50.0)
    disfluency = analysis.get('disfluency_rate', 0.1)
    consistency = analysis.get('consistency', 0.5)

    # Basic scoring based on heuristics
    ownership = min(5 + (consistency * 3), 10)  # Higher consistency -> higher ownership
    depth = min(5 + (consistency * 2), 10)
    impact = 5  # Neutral impact without specific metrics
    learning = min(5 + ((100 - perplexity) / 20), 10) if perplexity < 100 else 5  # Lower perplexity -> higher learning
    communication = min(5 + ((0.5 - disfluency) * 10), 10) if disfluency < 0.5 else max(5 - ((disfluency - 0.5) * 10), 0)  # Optimal disfluency around 0.1-0.2

    # Ensure scores are in valid range
    ownership = max(0, min(10, ownership))
    depth = max(0, min(10, depth))
    impact = max(0, min(10, impact))
    learning = max(0, min(10, learning))
    communication = max(0, min(10, communication))

    # Determine action based on integrity flags
    if perplexity < 20 and disfluency < 0.03 and consistency < 0.2:
        action = "end_early"
        reasoning = "Severe integrity flags detected (very low perplexity, disfluency, and consistency)"
        next_question = None
    elif ownership < 5 or depth < 5 or communication < 4:
        action = "follow_up"
        reasoning = "Low scores in key areas (ownership, depth, or communication)"
        next_question = "Could you elaborate more on your specific role and contributions in this project?"
    else:
        action = "move_on"
        reasoning = "Sufficient signal gathered for this competency area"
        next_question = "Thank you for your answer. Let's move on to the next topic."

    return {
        "assessment": {
            "ownership": round(ownership, 1),
            "depth": round(depth, 1),
            "impact": round(impact, 1),
            "learning": round(learning, 1),
            "communication": round(communication, 1),
            "notes": "Fallback assessment used due to LLM unavailability"
        },
        "decision": {
            "action": action,
            "reasoning": reasoning,
            "follow_up_topic": "specific contributions" if action == "follow_up" else ""
        },
        "next_question": next_question
    }


def _bm25_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())

def _search_resume_bm25(query: str, sentences: List[str], k1: float = 1.5, b: float = 0.75) -> List[Tuple[int, float]]:
    """
    Score sentences using Okapi BM25 for sparse lexical retrieval.
    Returns list of (sentence_index, bm25_score) sorted descending.
    """
    query_tokens = _bm25_tokenize(query)
    if not query_tokens or not sentences:
        return []

    doc_tokens = [_bm25_tokenize(s) for s in sentences]
    N = len(sentences)
    avgdl = sum(len(d) for d in doc_tokens) / max(N, 1)

    scores = []
    for idx, d_tokens in enumerate(doc_tokens):
        score = 0.0
        d_len = len(d_tokens)
        token_counts = {}
        for t in d_tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

        for q in query_tokens:
            # Document frequency
            df = sum(1 for d in doc_tokens if q in d)
            # Standard smoothed IDF
            idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
            tf = token_counts.get(q, 0)
            denom = tf + k1 * (1.0 - b + b * (d_len / max(avgdl, 1e-6)))
            if denom > 0:
                score += idf * (tf * (k1 + 1.0)) / denom
        scores.append((idx, float(score)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def _search_resume_hybrid_rrf(query: str, resume_text: str, k: int = 5, rrf_k: int = 60) -> List[str]:
    """
    Hybrid RAG retrieval using Reciprocal Rank Fusion (RRF).
    Combines FAISS dense semantic embeddings with BM25 sparse keyword search.
    RRF Score(d) = 1 / (rrf_k + dense_rank) + 1 / (rrf_k + sparse_rank)
    """
    try:
        sentences = [s.strip() for s in resume_text.split('.') if len(s.strip()) > 10]
        if not sentences:
            sentences = [resume_text.strip()] if resume_text.strip() else []
        if not sentences:
            return []

        # 1. Sparse Retrieval (BM25)
        bm25_ranked = _search_resume_bm25(query, sentences)
        sparse_ranks = {doc_idx: rank + 1 for rank, (doc_idx, score) in enumerate(bm25_ranked) if score > 0}

        # 2. Dense Retrieval (FAISS)
        dense_ranks = {}
        index, faiss_sentences = _build_faiss_index(resume_text)
        if index is not None and _embedder is not None and faiss_sentences:
            query_emb = _embedder.encode([query], normalize_embeddings=True)
            query_emb = np.array(query_emb).astype('float32')
            top_k = min(len(faiss_sentences), len(sentences))
            D, I = index.search(query_emb, k=top_k)
            for rank, idx in enumerate(I[0]):
                if idx < len(sentences):
                    dense_ranks[idx] = rank + 1

        # 3. Reciprocal Rank Fusion
        all_doc_indices = set(sparse_ranks.keys()).union(set(dense_ranks.keys()))
        if not all_doc_indices:
            # Fallback to direct substring matching if neither engine produced hits
            return _search_resume_fallback(query, resume_text)

        rrf_scores = {}
        for doc_idx in all_doc_indices:
            score = 0.0
            if doc_idx in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[doc_idx])
            if doc_idx in sparse_ranks:
                score += 1.0 / (rrf_k + sparse_ranks[doc_idx])
            rrf_scores[doc_idx] = score

        # Sort by RRF score descending
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        return [sentences[i] for i in sorted_indices[:k]]
    except Exception as e:
        logger.warning(f"Hybrid RRF search error: {e}, falling back to keyword search")
def _search_resume_with_faiss(query: str, resume_text: str, embedder, index, sentences: List[str]) -> List[str]:
    """
    Search resume using FAISS index for semantic matching (backwards compatible helper).
    """
    try:
        if index is None or not sentences or embedder is None:
            return _search_resume_fallback(query, resume_text)
        query_emb = embedder.encode([query], normalize_embeddings=True)
        query_emb = np.array(query_emb).astype('float32')
        k = min(5, len(sentences))
        D, I = index.search(query_emb, k=k)
        results = []
        for idx in I[0]:
            if idx < len(sentences):
                results.append(sentences[idx].strip())
        return results
    except Exception as e:
        logger.error(f"Error in FAISS search: {str(e)}")
        return _search_resume_fallback(query, resume_text)

def search_resume_tool(query: str, resume_text: str) -> List[str]:
    """
    Search resume for relevant passages using True Hybrid RAG (FAISS + BM25 with RRF).
    """
    if not resume_text or not resume_text.strip():
        return []
    return _search_resume_hybrid_rrf(query, resume_text, k=5)

def _search_resume_fallback(query: str, resume_text: str) -> List[str]:
    """
    Fallback search method using simple keyword matching.
    """
    try:
        if not resume_text or not resume_text.strip():
            return []
        sentences = [s.strip() for s in resume_text.split('.') if len(s.strip()) > 10]
        if not sentences:
            sentences = [resume_text]
        query_words = query.lower().split()
        results = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in query_words):
                results.append(sentence)
        if not results:
            for sentence in sentences:
                if query.lower() in sentence.lower():
                    results.append(sentence)
        return results[:5]
    except Exception as e:
        logger.error(f"Error in fallback search: {str(e)}")
        return []


if __name__ == "__main__":
    # simple test
    resume_profile = {
        "years_experience": 2.5,
        "level": "junior",
        "skills": ["Python", "TensorFlow", "APIs"],
        "projects": ["Built a recommendation engine"],
        "role": "ML Engineer"
    }
    history = []
    analysis = {
        "transcript": "I built a recommendation engine using TensorFlow and deployed it on AWS.",
        "perplexity": 80,
        "disfluency_rate": 0.12,
        "consistency": 0.7
    }
    decision = get_interviewer_decision(resume_profile, history, analysis, "competency_bank.yaml", "competency_selector.pkl")
    print(json.dumps(decision, indent=2))