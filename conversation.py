"""
conversation.py — Conversation Pipeline Orchestrator

The primary interaction module for the voice AI system.

Pipeline:
    audio file  →  STT (Groq Whisper)
                →  LLM (Groq Llama, with conversation memory)
                →  TTS (Deepgram Aura)
                →  audio file path

This module is intentionally simple — it handles the core voice loop
without any RAG or web-search complexity. Those remain available as
separate optional modules for future use.
"""

import stt
import llm as llm_module
import tts
from memory import session_memory


def process_voice(audio_path: str, voice: str) -> dict:
    """
    Run the full voice pipeline on an audio file.

    Args:
        audio_path: Path to the input audio file (from Gradio mic or upload).
        voice:      Deepgram Aura voice model name for TTS output.

    Returns:
        dict with keys:
            transcript  (str)       — what the user said
            language    (str)       — detected language code (e.g. "en")
            response    (str)       — the AI's text response
            audio_path  (str|None)  — path to synthesized audio, or None on failure
            error       (str|None)  — error message if any stage failed
    """
    result = {
        "transcript": "",
        "language":   "",
        "response":   "",
        "audio_path": None,
        "error":      None,
    }

    # ── Step 1: Speech → Text ─────────────────────────────────────────────────
    try:
        stt_result          = stt.transcribe_audio(audio_path)
        result["transcript"] = stt_result["text"].strip()
        result["language"]   = stt_result.get("language", "")
        print(f"[Conv] STT ({result['language']}): {result['transcript'][:80]}")
    except Exception as e:
        result["error"] = f"Transcription failed: {e}"
        return result

    if not result["transcript"]:
        result["error"] = "Could not understand the audio — please try again."
        return result

    # ── Step 2: Text → Response (LLM with memory) ────────────────────────────
    try:
        result["response"] = llm_module.generate_response(
            query=result["transcript"],
            context="",                        # no RAG context in primary flow
            history=session_memory.get_window(),
        )
        # Persist turn to session memory after successful response
        session_memory.add_turn(result["transcript"], result["response"])
        print(f"[Conv] LLM: {result['response'][:80]}...")
    except Exception as e:
        result["error"] = f"Response generation failed: {e}"
        return result

    # ── Step 3: Text → Speech ─────────────────────────────────────────────────
    try:
        result["audio_path"] = tts.synthesize(result["response"], voice=voice)
        print(f"[Conv] TTS → {result['audio_path']}")
    except Exception as e:
        # TTS failure is non-fatal — the text response is still returned
        result["error"] = f"Voice synthesis failed: {e}"

    return result


def process_text(message: str) -> dict:
    """
    Text-only pipeline (no audio I/O). Used by the Chat tab.

    Args:
        message: The user's typed message.

    Returns:
        dict with keys: response (str), error (str|None)
    """
    result = {"response": "", "error": None}

    try:
        result["response"] = llm_module.generate_response(
            query=message,
            context="",
            history=session_memory.get_window(),
        )
        session_memory.add_turn(message, result["response"])
        print(f"[Conv] Chat: {result['response'][:80]}...")
    except Exception as e:
        result["error"] = f"Response failed: {e}"

    return result
