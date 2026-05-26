"""
memory.py — Conversation Memory

Manages in-session conversation history with optional persistence.

- Keeps a sliding window of recent turns passed to the LLM as context.
- Persistence is fully opt-in — the user must explicitly request a save.
- Structured to support future embedding/chunking of saved conversations.
"""

import os
import json
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from config import MEMORY_WINDOW_SIZE, CONVERSATIONS_DIR


class ConversationMemory:
    """
    Lightweight in-session conversation memory.

    Internally stores messages as LangChain message objects so they
    can be passed directly to ChatGroq without conversion overhead.
    """

    def __init__(self):
        self._messages: list[BaseMessage] = []

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_window(self) -> list[BaseMessage]:
        """Return the last MEMORY_WINDOW_SIZE turns for LLM context."""
        limit = MEMORY_WINDOW_SIZE * 2          # 2 messages per turn
        return self._messages[-limit:] if len(self._messages) > limit else list(self._messages)

    def get_all(self) -> list[BaseMessage]:
        """Return the complete conversation history."""
        return list(self._messages)

    def as_tuples(self) -> list[tuple[str, str]]:
        """
        Return conversation as (user_text, ai_text) pairs.
        Used by Gradio's Chatbot component.
        """
        pairs = []
        msgs  = self._messages
        for i in range(0, len(msgs) - 1, 2):
            if isinstance(msgs[i], HumanMessage) and isinstance(msgs[i + 1], AIMessage):
                pairs.append((msgs[i].content, msgs[i + 1].content))
        return pairs

    def turn_count(self) -> int:
        """Number of complete conversation turns (user + AI pairs)."""
        return len(self._messages) // 2

    def is_empty(self) -> bool:
        return len(self._messages) == 0

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_turn(self, user_text: str, ai_text: str) -> None:
        """Append a completed conversation turn to memory."""
        self._messages.append(HumanMessage(content=user_text))
        self._messages.append(AIMessage(content=ai_text))

    def clear(self) -> None:
        """Clear all messages. Does NOT touch saved files."""
        self._messages.clear()

    # ── Persistence (opt-in only) ─────────────────────────────────────────────

    def save(self, custom_path: str | None = None) -> str:
        """
        Save the current conversation to a JSON file.

        This is called ONLY when the user explicitly clicks "Save conversation".
        Saved files are never loaded automatically.

        Args:
            custom_path: Optional file path. Auto-generates a timestamped
                         filename inside CONVERSATIONS_DIR if not provided.

        Returns:
            The absolute path of the saved file.
        """
        if self.is_empty():
            raise ValueError("Nothing to save — conversation is empty.")

        os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

        if custom_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            custom_path = os.path.join(CONVERSATIONS_DIR, f"conversation_{stamp}.json")

        payload = {
            "saved_at":  datetime.now().isoformat(),
            "turns":     self.turn_count(),
            "messages": [
                {
                    "role":    "user" if isinstance(m, HumanMessage) else "assistant",
                    "content": m.content,
                }
                for m in self._messages
            ],
        }

        with open(custom_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[Memory] Saved {self.turn_count()} turns → {custom_path}")
        return custom_path


# ── Shared session instance ───────────────────────────────────────────────────
# One memory object per server process — resets when the server restarts.
session_memory = ConversationMemory()
