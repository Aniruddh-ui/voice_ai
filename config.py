"""
config.py — Centralized Configuration
All environment variables, model settings, and path constants live here.
"""

import os

# Fix: TensorFlow 2.10 requires protobuf<=3.20 but newer protobuf is installed.
# Must be set BEFORE any langchain/tensorflow imports.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY     = os.getenv("groq_api_key")
DEEPGRAM_API_KEY = os.getenv("deepgram_api_key")
TAVILY_API_KEY   = os.getenv("Tavily_api_key")   # optional: web search fallback
JINA_API_KEY     = os.getenv("jina_api_key")      # optional: RAG embeddings

# ── STT — Groq Whisper ────────────────────────────────────────────────────────
WHISPER_MODEL = "whisper-large-v3"

# ── LLM — Groq Llama ─────────────────────────────────────────────────────────
LLM_MODEL       = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS  = 1024

# ── TTS — Deepgram Aura ───────────────────────────────────────────────────────
DEEPGRAM_TTS_MODEL = "aura-asteria-en"
DEEPGRAM_TTS_URL   = "https://api.deepgram.com/v1/speak"

# ── Conversation Memory ───────────────────────────────────────────────────────
MEMORY_WINDOW_SIZE = 10   # last N turns passed to LLM as context

# ── Data Paths ────────────────────────────────────────────────────────────────
AUDIO_DIR         = "data/audio"
CONVERSATIONS_DIR = "data/conversations"

# Keep for optional RAG module (not used in primary conversation flow)
UPLOADS_DIR      = "data/uploads"
VECTOR_STORE_DIR = "data/vector_store"
CHUNK_SIZE               = 500
CHUNK_OVERLAP            = 50
TOP_K_RESULTS            = 4
RAG_CONFIDENCE_THRESHOLD = 0.40
