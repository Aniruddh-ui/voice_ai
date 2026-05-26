"""
main.py — FastAPI Backend

REST API for the Voice AI platform. Provides programmatic access
to the same pipeline used by the Gradio UI.

Endpoints:
    GET  /health          → service health check
    POST /transcribe      → audio file → transcribed text (STT)
    POST /chat            → text message → AI response (with memory)
    POST /speak           → text → speech audio file (TTS)
    DELETE /memory        → clear conversation memory

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import stt
import tts
import conversation
from memory import session_memory
from config import UPLOADS_DIR, AUDIO_DIR

app = FastAPI(
    title="Voice AI",
    description="Conversational voice AI — STT, LLM, TTS, memory.",
    version="2.0.0",
)

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR,   exist_ok=True)


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class SpeakRequest(BaseModel):
    text:  str
    voice: str = "aura-orpheus-en"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Service health check."""
    return {
        "status":        "ok",
        "memory_turns":  session_memory.turn_count(),
    }


@app.post("/transcribe", tags=["Voice"])
async def transcribe(audio: UploadFile = File(...)):
    """
    Upload an audio file and receive its transcription.
    Accepts any format FFmpeg can decode (WAV, MP3, OGG, M4A, WebM…).
    """
    temp_path = os.path.join(UPLOADS_DIR, audio.filename or "upload.wav")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    try:
        result = stt.transcribe_audio(temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return result


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(req: ChatRequest):
    """Send a text message and receive a conversational response."""
    result = conversation.process_text(req.message)
    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return ChatResponse(response=result["response"])


@app.post("/speak", tags=["Voice"])
def speak(req: SpeakRequest):
    """Convert text to speech and return the audio file."""
    try:
        audio_path = tts.synthesize(req.text, voice=req.voice)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        filename=os.path.basename(audio_path),
    )


@app.delete("/memory", tags=["Chat"])
def clear_memory():
    """Clear the in-session conversation memory."""
    session_memory.clear()
    return {"status": "cleared"}
