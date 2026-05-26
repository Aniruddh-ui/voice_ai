"""
agent.py — LangGraph Conversational Agent (The Brain)

This module defines a stateful LangGraph StateGraph that orchestrates
the full conversation pipeline:

    [user input]
         ↓
    [retrieve from RAG]
         ↓
    [check confidence]
       ↙         ↘
 [confident]   [not confident]
 [respond]     [web search → respond]

Conversation memory is retained across turns using LangChain's
ChatMessageHistory (windowed to the last N turns).
"""

import operator
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langgraph.graph import StateGraph, END

import rag
import search
import llm as llm_module
from config import MEMORY_WINDOW_SIZE


# ─────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Shared state that flows through every node in the graph.
    Each node reads from and writes to this state.
    """
    query:          str                                      # User's input text
    messages:       Annotated[list[BaseMessage], operator.add]  # Full message history
    rag_context:    str                                      # Context from RAG retrieval
    rag_score:      float                                    # RAG confidence score (0.0–1.0)
    search_context: str                                      # Context from web search
    final_response: str                                      # Final LLM-generated response


# ─────────────────────────────────────────────
# Conversation Memory  (windowed message history)
# ─────────────────────────────────────────────

# Stores the last MEMORY_WINDOW_SIZE * 2 messages (human + AI pairs)
_chat_history = ChatMessageHistory()


def _get_windowed_history() -> list[BaseMessage]:
    """Return the most recent k turns (2 messages per turn) from history."""
    msgs = _chat_history.messages
    max_msgs = MEMORY_WINDOW_SIZE * 2
    return msgs[-max_msgs:] if len(msgs) > max_msgs else msgs


# ─────────────────────────────────────────────
# Graph Nodes
# ─────────────────────────────────────────────

def retrieve_node(state: AgentState) -> dict:
    """
    Node: Query the RAG vector store for relevant context.
    Sets rag_context and rag_score in the state.
    """
    context, score = rag.retrieve(state["query"])
    print(f"[Agent] RAG score={score:.3f} | context={'yes' if context else 'none'}")
    return {
        "rag_context": context,
        "rag_score":   score,
    }


def search_node(state: AgentState) -> dict:
    """
    Node: Run a real-time web search (triggered when RAG confidence is low).
    Sets search_context in the state.
    """
    print(f"[Agent] Low RAG confidence ({state['rag_score']:.3f}) → triggering web search")
    results = search.web_search(state["query"])
    return {"search_context": results}


def respond_node(state: AgentState) -> dict:
    """
    Node: Generate the final conversational response using the LLM.

    Combines whichever context is available (search > RAG > none),
    then saves the turn to conversation memory.
    """
    # Prefer search context if web search ran, otherwise use RAG
    context = state.get("search_context") or state.get("rag_context") or ""

    # Load windowed conversation history
    history = _get_windowed_history()

    # Generate response
    response = llm_module.generate_response(
        query=state["query"],
        context=context,
        history=history,
    )

    # Save this turn to memory
    _chat_history.add_user_message(state["query"])
    _chat_history.add_ai_message(response)

    print(f"[Agent] Response: {response[:80]}...")
    return {"final_response": response}


# ─────────────────────────────────────────────
# Routing Logic (Conditional Edge)
# ─────────────────────────────────────────────

def route_after_retrieval(state: AgentState) -> str:
    """
    Decide the next node after RAG retrieval.

    Returns "search" if confidence is too low, otherwise "respond".
    This is used as a conditional edge in the graph.
    """
    if search.should_search(state["rag_score"]):
        return "search"
    return "respond"


# ─────────────────────────────────────────────
# Build the LangGraph StateGraph
# ─────────────────────────────────────────────

def _build_graph() -> StateGraph:
    """Construct and compile the agent's LangGraph state graph."""

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("search",   search_node)
    graph.add_node("respond",  respond_node)

    # Entry point
    graph.set_entry_point("retrieve")

    # Conditional routing after retrieval
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "respond": "respond",
            "search":  "search",
        },
    )

    # Linear edges
    graph.add_edge("search",  "respond")
    graph.add_edge("respond", END)

    return graph.compile()


# Compile once at import time
agent_graph = _build_graph()


# ─────────────────────────────────────────────
# Public Interface
# ─────────────────────────────────────────────

def run_agent(user_input: str) -> str:
    """
    Main entry point: send a user message to the agent and get a response.

    Args:
        user_input: The user's text query

    Returns:
        The agent's response as a plain string
    """
    initial_state: AgentState = {
        "query":          user_input,
        "messages":       [HumanMessage(content=user_input)],
        "rag_context":    "",
        "rag_score":      0.0,
        "search_context": "",
        "final_response": "",
    }

    result = agent_graph.invoke(initial_state)
    return result["final_response"]


def clear_memory() -> None:
    """Reset conversation memory — use this to start a fresh session."""
    _chat_history.clear()
    print("[Agent] Conversation memory cleared.")
