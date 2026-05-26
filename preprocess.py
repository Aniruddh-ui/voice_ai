"""
preprocess.py — Audio Preprocessing

Normalizes any incoming audio to a clean 16kHz mono WAV before
sending to Groq Whisper. Uses pydub (which wraps FFmpeg) under the hood.

Why 16kHz mono WAV?
  - Whisper was trained on 16kHz audio
  - Mono reduces file size without quality loss for speech
  - WAV avoids any codec decoding overhead on Groq's side

Supports: WAV, MP3, OGG, M4A, FLAC, WebM, MP4 (audio track), and more.
"""

import os
import uuid

from pydub import AudioSegment
from pydub.effects import normalize

from config import AUDIO_DIR


# Target format for Whisper
_SAMPLE_RATE = 16_000   # Hz
_CHANNELS    = 1        # mono
_FORMAT      = "wav"


def prepare(audio_path: str) -> str:
    """
    Convert, normalize, and resample an audio file for Whisper transcription.

    Steps:
        1. Load the file (any format pydub supports via FFmpeg)
        2. Convert to mono
        3. Resample to 16kHz
        4. Normalize volume (peak normalization to -1 dBFS)
        5. Export as WAV to a unique temp path

    Args:
        audio_path: Path to the raw input audio file.

    Returns:
        Path to the preprocessed WAV file ready for Whisper.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        Exception: If pydub/FFmpeg cannot decode the file format.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    os.makedirs(AUDIO_DIR, exist_ok=True)

    # Load — pydub auto-detects format via FFmpeg
    audio = AudioSegment.from_file(audio_path)

    # Convert to mono + 16kHz
    audio = audio.set_channels(_CHANNELS).set_frame_rate(_SAMPLE_RATE)

    # Normalize volume so quiet recordings aren't misheard by Whisper
    audio = normalize(audio)

    # Write to a unique file so concurrent requests don't overwrite each other
    out_name = f"preprocessed_{uuid.uuid4().hex[:8]}.{_FORMAT}"
    out_path = os.path.join(AUDIO_DIR, out_name)
    audio.export(out_path, format=_FORMAT)

    print(f"[Preprocess] {os.path.basename(audio_path)} → {out_name}  "
          f"({len(audio) / 1000:.1f}s, {_SAMPLE_RATE}Hz mono)")

    return out_path
