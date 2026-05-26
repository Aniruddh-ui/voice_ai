"""
stt.py — Speech-to-Text

Transcribes audio using Groq Whisper Large v3.
Audio is preprocessed by preprocess.py before being sent to the API.
"""

import os
from groq import Groq

import preprocess
from config import GROQ_API_KEY, WHISPER_MODEL

client = Groq(api_key=GROQ_API_KEY)


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe an audio file using Groq Whisper Large v3.

    Audio is automatically preprocessed (format conversion, resampling,
    volume normalization) before transcription via preprocess.prepare().

    Args:
        audio_path: Path to any audio file (WAV, MP3, OGG, M4A, WebM, etc.)

    Returns:
        dict:
            text     (str) — transcribed text
            language (str) — detected language code (e.g. "en", "hi", "fr")

    Raises:
        FileNotFoundError: If audio_path does not exist.
        groq.APIError: If transcription fails on the API side.
    """
    # Normalize audio for best Whisper accuracy
    processed = preprocess.prepare(audio_path)

    with open(processed, "rb") as f:
        response = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=f,
            response_format="verbose_json",   # includes language detection
        )

    # Clean up the temp preprocessed file
    try:
        os.remove(processed)
    except OSError:
        pass

    return {
        "text":     response.text.strip(),
        "language": getattr(response, "language", "unknown"),
    }
