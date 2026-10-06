import pdfminer.high_level
import docx
import re
import spacy
from datetime import datetime
import logging
import os
from typing import Dict, List, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")
    # Create a blank English model as fallback
    nlp = spacy.blank("en")

def extract_text_from_pdf(path: str) -> str:
    """
    Extract text from a PDF file.

    Args:
        path: Path to the PDF file

    Returns:
        Extracted text as string

    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: For other PDF parsing errors
    """
    try:
        return pdfminer.high_level.extract_text(path)
    except FileNotFoundError:
        logger.error(f"PDF file not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error extracting text from PDF {path}: {str(e)}")
        raise

def extract_text_from_docx(path: str) -> str:
    """
    Extract text from a DOCX file.

    Args:
        path: Path to the DOCX file

    Returns:
        Extracted text as string

    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: For other DOCX parsing errors
    """
    try:
        doc = docx.Document(path)
        return "\n".join([para.text for para in doc.paragraphs])
    except FileNotFoundError:
        logger.error(f"DOCX file not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {path}: {str(e)}")
        raise

def extract_candidate_name(text: str, doc: Any = None) -> str:
    """
    Extracts the candidate's name from resume text.
    Combines header layout heuristic with spaCy entity recognition.
    Filters out contact information, emails, phone numbers, and section headings.
    """
    if not text or not text.strip():
        return "Candidate"

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return "Candidate"

    disallowed = {
        "resume", "curriculum", "vitae", "cv", "page", "email", "phone",
        "mobile", "contact", "address", "github", "linkedin", "portfolio",
        "summary", "objective", "experience", "education", "skills", "projects",
        "software", "engineer", "developer", "profile", "architect", "manager"
    }

    # 1. Check top 5 non-empty lines for candidate name
    for line in lines[:5]:
        clean = re.sub(r'[^a-zA-Z\s\.]', '', line).strip()
        words = clean.split()
        lower_line = line.lower()

        # Skip lines with contact info, links, or digits
        if any(d in lower_line.split() for d in disallowed):
            continue
        if "@" in line or "http" in lower_line or any(char.isdigit() for char in line):
            continue
        if "|" in line or "/" in line or "\\" in line:
            continue

        if 2 <= len(words) <= 4 and 3 <= len(clean) <= 35:
            if all(w.istitle() or w.isupper() for w in words):
                return clean.title()

    # 2. Check spaCy PERSON entities in the document
    if doc is not None:
        try:
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    clean_ent = ent.text.strip()
                    words = clean_ent.split()
                    if 2 <= len(words) <= 4 and not any(d in clean_ent.lower().split() for d in disallowed):
                        if not any(char.isdigit() for char in clean_ent) and "@" not in clean_ent:
                            return clean_ent.title()
        except Exception:
            pass

    # 3. Fallback: first non-empty line if it looks like a clean name
    if lines:
        first = re.sub(r'[^a-zA-Z\s]', '', lines[0]).strip()
        words = first.split()
        if 1 <= len(words) <= 3 and len(first) <= 25 and not any(d in lines[0].lower().split() for d in disallowed):
            return first.title()

    return "Candidate"

def parse_resume(filepath: str) -> Dict[str, Any]:
    """
    Parse a resume file (PDF or DOCX) and extract structured information.

    Args:
        filepath: Path to the resume file

    Returns:
        Dictionary containing parsed resume data:
        - raw_text: Full text content
        - years_experience: Years of experience (float)
        - level: Experience level (fresher/junior/mid/senior)
        - skills: List of extracted skills
        - projects: List of project descriptions

    Raises:
        ValueError: If file type is unsupported
        FileNotFoundError: If file doesn't exist
        Exception: For other parsing errors
    """
    # Check file extension first
    if not (filepath.lower().endswith('.pdf') or filepath.lower().endswith('.docx') or filepath.lower().endswith('.doc')):
        logger.error(f"Unsupported file type: {filepath}")
        raise ValueError("Unsupported file type. Use PDF or DOCX.")

    # Validate file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Resume file not found: {filepath}")

    # Extract text based on file extension
    if filepath.lower().endswith('.pdf'):
        logger.info(f"Parsing PDF resume: {filepath}")
        text = extract_text_from_pdf(filepath)
    elif filepath.lower().endswith('.docx') or filepath.lower().endswith('.doc'):
        logger.info(f"Parsing DOCX resume: {filepath}")
        text = extract_text_from_docx(filepath)

    if not text or not text.strip():
        logger.warning(f"Extracted text is empty from {filepath}")
        text = ""

    # Experience years extraction
    # Robust multi-format experience years extraction
    years = 0.0

    # 1. Direct explicit mention check (e.g. "5+ years of experience", "3.5 yrs experience")
    explicit_match = re.search(r'\b(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s+(?:of\s+)?(?:hands-on\s+)?(?:experience|exp)\b', text, re.IGNORECASE)
    if explicit_match:
        try:
            years = float(explicit_match.group(1))
        except ValueError:
            pass

    # 2. Date range extraction if explicit wasn't found or for confirmation
    if years <= 0.0:
        # Check standard 4-digit year ranges like "2019 - 2023", "2021 – Present", "2018 to Current"
        year_ranges = re.findall(r'\b(20\d{2}|19\d{2})\s*(?:-|–|—|to)\s*(20\d{2}|present|current|now)\b', text, re.IGNORECASE)
        if year_ranges:
            try:
                earliest_start = min(int(yr[0]) for yr in year_ranges)
                current_year = datetime.now().year
                latest_end = max(
                    current_year if yr[1].lower() in ['present', 'current', 'now'] else int(yr[1])
                    for yr in year_ranges
                )
                if latest_end >= earliest_start:
                    years = float(latest_end - earliest_start)
            except Exception as e:
                logger.debug(f"Year range parse exception: {e}")

    # 3. Fallback Month-Year parsing
    if years <= 0.0:
        date_pattern = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        if len(dates) >= 2:
            try:
                start = datetime.strptime(re.sub(r'[^a-zA-Z0-9 ]', '', dates[0]), "%b %Y")
                if 'present' in text.lower() or 'current' in text.lower():
                    end = datetime.now()
                else:
                    end = datetime.strptime(re.sub(r'[^a-zA-Z0-9 ]', '', dates[-1]), "%b %Y")
                years = max(0.0, (end - start).days / 365.25)
            except Exception as e:
                years = len(dates) * 0.75
        elif len(dates) == 1:
            years = 1.0

    # Experience level assignment
    if years < 1:
        level = "fresher"
    elif years < 3:
        level = "junior"
    elif years < 6:
        level = "mid"
    else:
        level = "senior"

    logger.debug(f"Experience level determined: {level} ({years} years)")

    # Skills extraction via clean filtering
    disallowed_skills = {
        "team", "client", "work", "project", "approach", "experience",
        "responsibility", "responsibilities", "role", "system", "systems",
        "application", "applications", "process", "service", "services",
        "time", "year", "years", "month", "months", "day", "days"
    }
    try:
        doc = nlp(text)
        raw_skills = [chunk.text.lower().strip() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG"]:
                raw_skills.append(ent.text.lower().strip())
        skills = [s for s in set(raw_skills) if len(s) >= 2 and s not in disallowed_skills and not any(char.isdigit() for char in s)]
    except Exception as e:
        logger.warning(f"Error during skills extraction: {str(e)}")
        skills = []

    # Extract candidate name from header
    name = extract_candidate_name(text, doc if 'doc' in locals() else None)

    # Robust Project extraction (strips markdown bullets, numbering, dashes)
    action_verbs = [
        "built", "developed", "designed", "implemented", "led", "optimized",
        "reduced", "increased", "created", "architected", "engineered",
        "spearheaded", "automated", "deployed", "refactored", "configured",
        "integrated", "authored", "orchestrated", "migrated", "scaled"
    ]
    projects = []
    for raw_line in text.split('\n'):
        line = raw_line.strip()
        # Clean leading bullets, numbers, dashes: "• ", "- ", "1. ", "* ", "· "
        cleaned = re.sub(r'^[\s\•\-\*\d\.\)\:\·\u00b7]+', '', line).strip()
        cleaned_lower = cleaned.lower()
        if any(cleaned_lower.startswith(verb) for verb in action_verbs) and len(cleaned) > 22:
            projects.append(cleaned)

    # Fallback projects if no bullet matched action verbs: find lines mentioning key technical metrics or tools
    if not projects:
        for raw_line in text.split('\n'):
            cleaned = re.sub(r'^[\s\•\-\*\d\.\)\:\·\u00b7]+', '', raw_line).strip()
            if len(cleaned) > 35 and any(kw in cleaned.lower() for kw in ["api", "pipeline", "database", "service", "model", "cloud", "docker", "redis", "kafka"]):
                projects.append(cleaned)
                if len(projects) >= 6:
                    break

    result = {
        "name": name,
        "raw_text": text,
        "years_experience": round(years, 1),
        "level": level,
        "skills": list(set(skills))[:50],
        "projects": projects[:10],
    }

    logger.info(f"Successfully parsed resume for {name}: {years} yrs ({level}), {len(result['skills'])} skills, {len(result['projects'])} projects")
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python resume_parser.py <resume.pdf|resume.docx>")
        sys.exit(1)
    profile = parse_resume(sys.argv[1])
    import json
    print(json.dumps(profile, indent=2))