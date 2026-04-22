"""
LangGraph Agent Orchestration — EmpathAI Pipeline.

Flow:
  START -> router
        |-- CASUAL -> casual_response -> END
        |-- INQUIRY -> retrieve -> grade -> inquiry_writer -> END
        |-- COMPLAINT -> sentiment_analyzer -> retrieve -> grade_documents
                                                |-- GOOD -> empathy_writer -> reviewer -> END
                                                |-- BAD (retries < 2) -> rewrite -> retrieve (loop)
                                                |-- BAD (retries >= 2) -> empathy_writer -> reviewer -> END

Entry point: run_streaming(question, history, stream_callback)
"""
import asyncio
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Callable, Awaitable, Optional
from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.router import classify
from agents.sentiment_analyzer import sentiment_analyzer_node
from agents.empathy_writer import (
    generate_empathy_streaming, generate_casual, generate_inquiry,
)
from agents.reviewer import needs_review, review_with_retry
from agents.grader import grade_documents_node
from agents.rewriter import rewrite_query_node
from agents.llm_client import observe
from indexing.query_engine import retrieve_and_rerank, format_evidence

from config import MAX_REWRITE_RETRIES
from utils.console import console


# ================================================================
# Graph Nodes
# ================================================================

def router_node(state: AgentState) -> dict:
    """Node 1: Classify intent (COMPLAINT / INQUIRY / CASUAL)."""
    t0 = time.time()
    question = state["question"]

    history = state.get("history", [])
    contextualized_q = _build_contextualized_question(question, history)

    intent = classify(contextualized_q)

    elapsed = int((time.time() - t0) * 1000)
    console.print(f"[dim]  Router: {intent} ({elapsed}ms)[/]")

    return {
        "intent": intent,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            "router_decision": intent,
            "router_ms": elapsed,
        },
    }


async def casual_node(state: AgentState) -> dict:
    """Node: Casual response (không cần RAG)."""
    answer = await generate_casual(state["question"])
    return {
        "answer": answer,
        "reviewer_triggered": False,
        "reviewer_result": {"is_approved": True, "issues": [], "retry_count": 0},
    }


def retrieve_node(state: AgentState) -> dict:
    """Node: Hybrid Search + Rerank tren policy DB."""
    t0 = time.time()
    # Use rewritten query if available, otherwise use original question
    query = state.get("translated_query", state["question"])

    documents = retrieve_and_rerank(query)
    evidence_text = format_evidence(documents)

    elapsed = int((time.time() - t0) * 1000)
    console.print(
        f"[dim]  Retrieved: {len(documents)} docs, "
        f"{len(evidence_text)} chars ({elapsed}ms)[/]"
    )

    return {
        "evidence": documents,
        "evidence_text": evidence_text,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            "retrieved_count": len(documents),
            "retrieve_ms": elapsed,
        },
    }


async def empathy_writer_node(state: AgentState) -> dict:
    """Node: Generate empathetic response with streaming."""
    t0 = time.time()
    question = state["question"]
    evidence_text = state.get("evidence_text", "")
    sentiment = state.get("sentiment", "")
    sentiment_score = state.get("sentiment_score", 0)
    stream_callback = state.get("stream_callback")

    answer = await generate_empathy_streaming(
        question=question,
        evidence_text=evidence_text,
        sentiment=sentiment,
        score=sentiment_score,
        stream_callback=stream_callback,
    )

    elapsed = int((time.time() - t0) * 1000)
    console.print(f"[dim]  EmpathyWriter: {len(answer)} chars ({elapsed}ms)[/]")

    return {
        "answer": answer,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            "writer_answer": answer[:500],
            "writer_ms": elapsed,
        },
    }


async def inquiry_writer_node(state: AgentState) -> dict:
    """Node: Answer inquiry based on policy (no sentiment needed)."""
    t0 = time.time()
    question = state["question"]
    evidence_text = state.get("evidence_text", "")

    answer = await generate_inquiry(question, evidence_text)

    elapsed = int((time.time() - t0) * 1000)
    console.print(f"[dim]  InquiryWriter: {len(answer)} chars ({elapsed}ms)[/]")

    return {
        "answer": answer,
        "reviewer_triggered": False,
        "reviewer_result": {"is_approved": True, "issues": [], "retry_count": 0},
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            "inquiry_answer": answer[:500],
            "inquiry_ms": elapsed,
        },
    }


async def reviewer_node(state: AgentState) -> dict:
    """Node: Empathy quality check."""
    t0 = time.time()
    question = state["question"]
    answer = state.get("answer", "")
    evidence_text = state.get("evidence_text", "")
    sentiment = state.get("sentiment", "")

    # Always review for complaints (especially toxic/frustrated)
    reviewer_triggered = sentiment in ("toxic", "frustrated") or needs_review(question)

    if reviewer_triggered:
        console.print("[dim]  Reviewer triggered[/]")
        final_answer, reviewer_result = await review_with_retry(
            question, answer, evidence_text
        )
    else:
        console.print("[dim]  Reviewer skipped[/]")
        final_answer = answer
        reviewer_result = {"is_approved": True, "issues": [], "retry_count": 0}

    elapsed = int((time.time() - t0) * 1000)

    return {
        "answer": final_answer,
        "reviewer_triggered": reviewer_triggered,
        "reviewer_result": reviewer_result,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            "reviewer_triggered": reviewer_triggered,
            "reviewer_result": reviewer_result,
            "reviewer_ms": elapsed,
        },
    }


# ================================================================
# Conditional Edges
# ================================================================

def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "")
    if intent == "CASUAL":
        return "casual"
    elif intent == "INQUIRY":
        return "inquiry"
    else:
        return "complaint"


def route_by_grade(state: AgentState) -> str:
    if state.get("is_evidence_sufficient", True):
        return "good"
    if state.get("rewrite_count", 0) >= MAX_REWRITE_RETRIES:
        return "give_up"
    return "rewrite"


# ================================================================
# Graph Builder
# ================================================================

def build_graph() -> StateGraph:
    """Build LangGraph StateGraph cho EmpathAI pipeline."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("casual", casual_node)
    graph.add_node("sentiment", sentiment_analyzer_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_documents_node)
    graph.add_node("rewrite", rewrite_query_node)
    graph.add_node("empathy_writer", empathy_writer_node)
    graph.add_node("inquiry_writer", inquiry_writer_node)
    graph.add_node("reviewer", reviewer_node)

    # Entry point
    graph.set_entry_point("router")

    # Router -> 3 branches
    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "casual": "casual",
            "inquiry": "retrieve",
            "complaint": "sentiment",
        },
    )

    # Casual -> END
    graph.add_edge("casual", END)

    # Complaint: sentiment -> retrieve
    graph.add_edge("sentiment", "retrieve")

    # Both INQUIRY and COMPLAINT share: retrieve -> grade
    graph.add_edge("retrieve", "grade")

    # Combined routing after grade:
    # - INQUIRY intent -> inquiry_writer
    # - COMPLAINT + good evidence -> empathy_writer
    # - COMPLAINT + bad evidence + retries left -> rewrite
    # - COMPLAINT + bad evidence + no retries -> empathy_writer (give up)
    def route_after_grade(state):
        intent = state.get("intent", "")
        if intent == "INQUIRY":
            return "inquiry_writer"
        # For COMPLAINT, check evidence quality
        if state.get("is_evidence_sufficient", True):
            return "good"
        if state.get("rewrite_count", 0) >= MAX_REWRITE_RETRIES:
            return "give_up"
        return "rewrite"

    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "inquiry_writer": "inquiry_writer",
            "good": "empathy_writer",
            "rewrite": "rewrite",
            "give_up": "empathy_writer",
        },
    )

    # Rewrite -> loop back to retrieve
    graph.add_edge("rewrite", "retrieve")

    # Writers -> reviewers / END
    graph.add_edge("empathy_writer", "reviewer")
    graph.add_edge("reviewer", END)
    graph.add_edge("inquiry_writer", END)

    return graph.compile()


# ================================================================
# Entry Point
# ================================================================

_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


@observe(name="empathAI_pipeline", as_type="generation")
async def run_streaming(
    question: str,
    history: list[dict] = None,
    session_id: str = "",
    stream_callback: Optional[Callable[[str], Awaitable[None]]] = None,
) -> dict:
    """Run full EmpathAI pipeline with streaming."""
    start_time = time.time()
    console.print(f"[cyan]Incoming: '{question[:60]}...'[/]")

    graph = _get_graph()

    initial_state: AgentState = {
        "session_id": session_id,
        "question": question,
        "history": history or [],
        "intent": "",
        "sentiment": "",
        "sentiment_score": 0.0,
        "translated_query": "",
        "evidence": [],
        "evidence_text": "",
        "policy_context": "",
        "compensation": "",
        "rewrite_count": 0,
        "is_evidence_sufficient": True,
        "answer": "",
        "reviewer_triggered": False,
        "reviewer_result": {},
        "agent_trace": {},
        "processing_time_ms": 0,
        "stream_callback": stream_callback,
    }

    final_state = await graph.ainvoke(initial_state)

    processing_time = int((time.time() - start_time) * 1000)
    final_state["processing_time_ms"] = processing_time

    console.print(f"[green]Done in {processing_time}ms[/]")
    return final_state


# ================================================================
# Utility
# ================================================================

def _build_contextualized_question(question, history):
    if not history:
        return question

    recent = history[-6:]
    context = "Lịch sử hội thoại:\n"
    for msg in recent:
        role = "Khách" if msg.get("role") == "user" else "Bot"
        content = msg.get("content", "")[:200]
        context += f"- {role}: {content}\n"

    context += f"\nTin nhắn hiện tại: {question}"
    return context
