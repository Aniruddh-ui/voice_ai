"""
tts.py — Text-to-Speech Module

Uses Deepgram Aura API to synthesize natural, low-latency voice responses.
Returns a path to the generated audio file.

Note: Deepgram streams audio in chunks — we must use streaming mode to avoid
      httpx's "Response content shorter than Content-Length" error.
"""

import os
import uuid
import httpx

from config import DEEPGRAM_API_KEY, DEEPGRAM_TTS_URL, DEEPGRAM_TTS_MODEL, AUDIO_DIR


# Available Deepgram Aura voice models
AVAILABLE_VOICES = [
    "aura-asteria-en",   # Female — warm (default)
    "aura-luna-en",      # Female — soft
    "aura-stella-en",    # Female — bright
    "aura-athena-en",    # Female — authoritative
    "aura-hera-en",      # Female — calm
    "aura-orion-en",     # Male   — deep
    "aura-arcas-en",     # Male   — friendly
    "aura-perseus-en",   # Male   — professional
    "aura-orpheus-en",   # Male   — warm
    "aura-helios-en",    # Male   — British accent
    "aura-zeus-en",      # Male   — authoritative
]


def synthesize(text: str, voice: str = DEEPGRAM_TTS_MODEL) -> str:
    """
    Convert text to speech using the Deepgram Aura API.

    Deepgram returns audio as a chunked/streaming response, so we use
    httpx's streaming context manager to avoid Content-Length mismatches.

    Args:
        text:  The text to synthesize
        voice: Deepgram Aura voice model name (default: aura-asteria-en)

    Returns:
        Absolute file path to the generated audio file (WAV)

    Raises:
        RuntimeError: If the Deepgram API returns an error or empty audio
    """
    os.makedirs(AUDIO_DIR, exist_ok=True)

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type":  "application/json",
    }
    params  = {"model": voice}
    payload = {"text": text}

    filename    = f"tts_{uuid.uuid4().hex[:8]}.wav"
    output_path = os.path.join(AUDIO_DIR, filename)

    # Use streaming to handle Deepgram's chunked transfer encoding correctly
    with httpx.stream(
        "POST",
        DEEPGRAM_TTS_URL,
        headers=headers,
        params=params,
        json=payload,
        timeout=60.0,
    ) as response:
        response.raise_for_status()

        audio_bytes = b""
        for chunk in response.iter_bytes(chunk_size=4096):
            if chunk:
                audio_bytes += chunk

    if not audio_bytes:
        raise RuntimeError("Deepgram TTS returned empty audio — check your API key or voice model.")

    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    print(f"[TTS] Synthesized {len(audio_bytes):,} bytes → {output_path}")
    return output_path


def get_available_voices() -> list[str]:
    """Return the list of supported Deepgram Aura voice model names."""
    return AVAILABLE_VOICES
