"""
rag.py — RAG Pipeline Module

Full document pipeline: load → chunk → embed → FAISS index → retrieve.

Supported formats: PDF, TXT, Markdown
Embedding model:   Jina Embeddings v3 API (pure HTTP, no local binary needed)
Vector store:      FAISS (file-based, no server needed)
"""

import os
import pickle
import numpy as np
from typing import Tuple

import faiss
import httpx
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    JINA_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K_RESULTS, VECTOR_STORE_DIR, UPLOADS_DIR,
)

# ─────────────────────────────────────────────
# Jina Embeddings v3 API  (pure HTTP, no local binary)
# ─────────────────────────────────────────────
_JINA_URL   = "https://api.jina.ai/v1/embeddings"
_JINA_MODEL = "jina-embeddings-v3"

def _embed(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using Jina Embeddings v3 API.
    Returns a float32 numpy array of shape (len(texts), 1024).
    """
    response = httpx.post(
        _JINA_URL,
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={"model": _JINA_MODEL, "input": texts},
        timeout=60.0,
    )
    response.raise_for_status()
    data = sorted(response.json()["data"], key=lambda x: x["index"])
    return np.array([d["embedding"] for d in data], dtype="float32")

# In-memory FAISS index and chunk registry
_faiss_index: faiss.Index | None = None
_chunks: list[tuple[str, str]] = []   # list of (chunk_text, source_filename)


# ─────────────────────────────────────────────
# Persistence Helpers
# ─────────────────────────────────────────────

def _index_path()  -> str: return os.path.join(VECTOR_STORE_DIR, "index.faiss")
def _chunks_path() -> str: return os.path.join(VECTOR_STORE_DIR, "chunks.pkl")


def load_vector_store() -> None:
    """Load FAISS index and chunk list from disk if they exist."""
    global _faiss_index, _chunks

    if os.path.exists(_index_path()) and os.path.exists(_chunks_path()):
        _faiss_index = faiss.read_index(_index_path())
        with open(_chunks_path(), "rb") as f:
            _chunks = pickle.load(f)
        print(f"[RAG] Loaded {len(_chunks)} chunks from vector store.")
    else:
        print("[RAG] No existing vector store found — starting fresh.")


def _save_vector_store() -> None:
    """Persist FAISS index and chunk list to disk."""
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    faiss.write_index(_faiss_index, _index_path())
    with open(_chunks_path(), "wb") as f:
        pickle.dump(_chunks, f)


# ─────────────────────────────────────────────
# Document Loading
# ─────────────────────────────────────────────

def _load_document(file_path: str) -> str:
    """Load a document and return its full text content."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".txt", ".text"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext in (".md", ".markdown"):
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF, TXT, or Markdown.")

    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)


# ─────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────

def ingest_documents(file_path: str) -> dict:
    """
    Load, chunk, embed, and store a document in the FAISS vector store.

    Args:
        file_path: Absolute path to a PDF, TXT, or Markdown file

    Returns:
        dict with status, source filename, chunks added, and total chunks
    """
    global _faiss_index, _chunks

    os.makedirs(UPLOADS_DIR, exist_ok=True)

    # 1. Load document text
    full_text = _load_document(file_path)
    source    = os.path.basename(file_path)

    # 2. Chunk text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(full_text)

    if not chunks:
        return {"status": "error", "message": "Document appears to be empty."}

    # 3. Generate embeddings via Jina API
    vectors = _embed(chunks)
    faiss.normalize_L2(vectors)   # normalize for cosine similarity via inner product

    # 4. Build or extend FAISS index
    dim = vectors.shape[1]
    if _faiss_index is None:
        _faiss_index = faiss.IndexFlatIP(dim)

    _faiss_index.add(vectors)

    # 5. Register chunks
    _chunks.extend((chunk, source) for chunk in chunks)

    # 6. Persist to disk
    _save_vector_store()

    return {
        "status":       "success",
        "source":       source,
        "chunks_added": len(chunks),
        "total_chunks": len(_chunks),
    }


# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K_RESULTS) -> Tuple[str, float]:
    """
    Retrieve the most semantically relevant chunks for a query.

    Args:
        query: User's question
        top_k: Number of top chunks to return

    Returns:
        Tuple of:
            - context (str): Formatted retrieved text with source labels
            - score   (float): Average similarity score (0.0–1.0)
                               0.0 means no documents are stored.
    """
    if _faiss_index is None or not _chunks:
        return "", 0.0

    # Embed query via Jina API
    q_vec = _embed([query])
    faiss.normalize_L2(q_vec)

    # Search
    k = min(top_k, len(_chunks))
    scores, indices = _faiss_index.search(q_vec, k)

    results = [
        (_chunks[idx][0], float(score), _chunks[idx][1])
        for score, idx in zip(scores[0], indices[0])
        if idx >= 0
    ]

    if not results:
        return "", 0.0

    avg_score = sum(r[1] for r in results) / len(results)

    # Format context with source labels
    context_parts = [f"[Source: {r[2]}]\n{r[0]}" for r in results]
    context = "\n\n---\n\n".join(context_parts)

    return context, avg_score


def get_stats() -> dict:
    """Return current vector store statistics."""
    return {
        "total_chunks": len(_chunks),
        "sources":      list({c[1] for c in _chunks}),
        "index_ready":  _faiss_index is not None,
    }


def get_sources() -> list[str]:
    """Return sorted list of unique source document names in the store."""
    return sorted({c[1] for c in _chunks})


def clear_vector_store() -> dict:
    """
    Remove ALL documents from the vector store (memory + disk).
    Use this to start fresh without any RAG context.
    """
    global _faiss_index, _chunks

    _faiss_index = None
    _chunks      = []

    for path in [_index_path(), _chunks_path()]:
        if os.path.exists(path):
            os.remove(path)

    print("[RAG] Vector store cleared.")
    return {"status": "cleared", "message": "All documents removed from the knowledge base."}


def remove_source(source_name: str) -> dict:
    """
    Remove all chunks belonging to a specific source document and
    rebuild the FAISS index from the remaining chunks.

    Args:
        source_name: The filename (basename) of the document to remove.

    Returns:
        dict with status, chunks_removed, and chunks_remaining.
    """
    global _faiss_index, _chunks

    original_count = len(_chunks)
    remaining      = [(chunk, src) for chunk, src in _chunks if src != source_name]

    if len(remaining) == original_count:
        return {"status": "not_found", "message": f"'{source_name}' was not found in the store."}

    removed  = original_count - len(remaining)
    _chunks  = remaining

    if remaining:
        # Rebuild index from remaining chunks
        texts   = [c[0] for c in remaining]
        vectors = _embed(texts)
        faiss.normalize_L2(vectors)
        dim          = vectors.shape[1]
        _faiss_index = faiss.IndexFlatIP(dim)
        _faiss_index.add(vectors)
        _save_vector_store()
    else:
        # Store is now empty — clear files too
        _faiss_index = None
        for path in [_index_path(), _chunks_path()]:
            if os.path.exists(path):
                os.remove(path)

    print(f"[RAG] Removed '{source_name}': {removed} chunks removed, {len(remaining)} remaining.")
    return {
        "status":           "removed",
        "source":           source_name,
        "chunks_removed":   removed,
        "chunks_remaining": len(remaining),
    }


# Auto-load existing vector store when module is imported
load_vector_store()
