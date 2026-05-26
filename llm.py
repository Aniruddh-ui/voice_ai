"""
llm.py — LLM Response Generator

Uses Groq's Llama 3.3 via LangChain's ChatGroq.
Focused on natural, conversational voice responses.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage

from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


# ── System Prompt ─────────────────────────────────────────────────────────────
# Tuned for voice interaction: concise, natural, multilingual, no markdown.

SYSTEM_PROMPT = """You are a warm, intelligent voice AI assistant.

Speak naturally and conversationally — your responses will be read aloud.
Keep answers concise and to the point. Avoid bullet points, lists, or markdown.
If the user speaks a language other than English, reply in that same language.
Match the user's tone — casual, serious, curious, or playful.
If you don't know something, say so honestly.
Never mention that you're an AI unless directly asked."""


# ── LLM Chain ─────────────────────────────────────────────────────────────────

def _build_chain():
    """Build and return a LangChain prompt | LLM chain."""
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    return prompt | llm


def generate_response(
    query:   str,
    context: str = "",
    history: list[BaseMessage] = [],
) -> str:
    """
    Generate a conversational response.

    Args:
        query:   The user's current message or transcribed speech.
        context: Optional grounding context (from RAG or web search).
                 Pass an empty string for the primary conversation flow.
        history: Recent conversation turns as LangChain message objects.

    Returns:
        The AI's response as a plain string.
    """
    chain = _build_chain()

    # Prepend context to the user input if available
    user_input = query
    if context.strip():
        user_input = f"Context:\n{context}\n\nQuestion: {query}"

    response = chain.invoke({"input": user_input, "history": history})
    return response.content
