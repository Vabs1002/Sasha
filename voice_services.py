import os
import logging
import requests
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# RUMIK API Configuration
RUMIK_API_KEY = os.getenv("RUMIK_API_KEY")
RUMIK_BASE_URL = os.getenv("RUMIK_BASE_URL", "https://silk-api.rumik.ai/v1")

class RumikVoiceError(Exception):
    """Custom exception for RUMIK voice service errors"""
    pass

# Alias for backwards compatibility
VoiceServiceError = RumikVoiceError

def speech_to_text(audio_data: bytes, language: str = "en") -> str:
    """
    Convert audio to text using RUMIK STT API.
    """
    if not RUMIK_API_KEY:
        raise RumikVoiceError("RUMIK_API_KEY environment variable not set")

    if not audio_data:
        raise RumikVoiceError("No audio data provided")

    try:
        headers = {
            "Authorization": f"Bearer {RUMIK_API_KEY}",
            "Content-Type": "audio/wav"
        }

        response = requests.post(
            f"{RUMIK_BASE_URL}/stt",
            headers=headers,
            data=audio_data,
            params={"language": language},
            timeout=30
        )

        if response.status_code != 200:
            raise RumikVoiceError(f"RUMIK STT API error: {response.status_code} - {response.text}")

        result = response.json()
        transcript = result.get("text", "").strip()

        if not transcript:
            raise RumikVoiceError("Empty transcription returned from RUMIK STT")

        return transcript

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error calling RUMIK STT: {str(e)}")
        raise RumikVoiceError(f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in RUMIK STT processing: {str(e)}")
        raise RumikVoiceError(f"Processing error: {str(e)}")

def text_to_speech(text: str, voice: str = "muga", speed: float = 1.0) -> bytes:
    """
    Convert text to audio using RUMIK Silk TTS API.

    Args:
        text: Text to convert to speech
        voice: Voice model identifier (default: "muga")
        speed: Speech rate parameter

    Returns:
        Audio bytes (WAV format)
    """
    if not RUMIK_API_KEY:
        raise RumikVoiceError("RUMIK_API_KEY environment variable not set")

    if not text or not text.strip():
        raise RumikVoiceError("No text provided for speech synthesis")

    try:
        headers = {
            "Authorization": f"Bearer {RUMIK_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "muga",
            "text": text.strip(),
        }

        response = requests.post(
            f"{RUMIK_BASE_URL}/tts",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            raise RumikVoiceError(f"RUMIK TTS API error: {response.status_code} - {response.text}")

        audio_bytes = response.content

        if not audio_bytes:
            raise RumikVoiceError("Empty audio returned from RUMIK TTS")

        return audio_bytes

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error calling RUMIK TTS: {str(e)}")
        raise RumikVoiceError(f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in RUMIK TTS processing: {str(e)}")
        raise RumikVoiceError(f"Processing error: {str(e)}")