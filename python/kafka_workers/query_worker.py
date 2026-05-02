"""
Query Processing Worker — Kafka Consumer cho topic "query.request".

Sử dụng LangGraph pipeline với Groq Streaming:
1. Router (Embedding-based, no LLM)
2. Translate VN → EN (Groq Fast)
3. Hybrid Search + Rerank (BGE-M3 + BGE-Reranker)
4. Writer (Groq Streaming — yield từng token)
5. Reviewer (Conditional fact-check)

Streaming tokens được publish vào "query.response" với is_final=False.
Final answer được publish với is_final=True.
"""
import asyncio
import signal
import json
import sys
import os
import time
from pathlib import Path

# Fix Windows console encoding for Unicode emoji
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.append(str(Path(__file__).parent.parent))

from confluent_kafka import Consumer, Producer, KafkaError
from rich.console import Console

from kafka_workers.kafka_config import (
    BROKERS, GROUP_QUERY,
    TOPIC_QUERY_REQUEST, TOPIC_QUERY_RESPONSE,
    serialize, deserialize,
)
from agents.graph import run_streaming
from retrieval.cache import get_cached_answer, set_cached_answer

PIPELINE_TIMEOUT_SECONDS = 120  # Max 2 phút cho một request

console = Console(force_terminal=True, safe_box=True)
running = True


def signal_handler(sig, frame):
    global running
    console.print("[yellow]🛑 Shutting down query worker...[/]")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def create_consumer() -> Consumer:
    return Consumer({
        "bootstrap.servers": BROKERS,
        "group.id": GROUP_QUERY,
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    })


def create_producer() -> Producer:
    return Producer({
        "bootstrap.servers": BROKERS,
        "acks": "all",
    })


def run_worker():
    """Main loop cho EmpathAI query processing worker."""
    console.print("[bold cyan]Starting EmpathAI Query Worker (LangGraph + Sentiment + Cache)...[/]")
    console.print(f"[dim]  Listening on: {TOPIC_QUERY_REQUEST}[/]")
    console.print(f"[dim]  Publishing to: {TOPIC_QUERY_RESPONSE}[/]")
    console.print("[dim]  Pipeline: [Cache?] -> Router -> Sentiment -> Retrieve -> Grade/Rewrite -> EmpathyWriter -> Reviewer[/]")

    # Pre-load models trước khi nhận query (tránh cold start timeout)
    from agents.model_registry import warmup
    warmup()

    consumer = create_consumer()
    producer = create_producer()
    consumer.subscribe([TOPIC_QUERY_REQUEST])

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    query_count = 0

    while running:
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            continue

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            console.print(f"[red]❌ Kafka error: {msg.error()}[/]")
            continue

        try:
            query_event = deserialize(msg.value())
            session_id = query_event.get("session_id", "unknown")
            question = query_event.get("question", "")
            history = query_event.get("history", [])

            console.print(
                f"[cyan]💬 Query: '{question[:60]}...' (session: {session_id[:8]})[/]"
            )

            # ── Check Redis Cache FIRST ──
            cached = get_cached_answer(question)
            if cached:
                # CACHE HIT → bypass toàn bộ LangGraph, trả kết quả ngay (<50ms)
                cache_response = {
                    "session_id": session_id,
                    "answer": cached["answer"],
                    "sources": cached.get("sources", []),
                    "agent_trace": {
                        **cached.get("agent_trace", {}),
                        "from_cache": True,
                    },
                    "processing_time_ms": 0,
                    "is_final": True,
                    "chunk_type": None,
                }
                producer.produce(
                    TOPIC_QUERY_RESPONSE,
                    key=session_id.encode("utf-8"),
                    value=serialize(cache_response),
                )
                producer.flush()
                query_count += 1
                console.print(
                    f"[bold green]⚡ Cache HIT — response sent for session {session_id[:8]}... "
                    f"(query #{query_count})[/]"
                )
                continue  # Skip LangGraph entirely

            # ── CACHE MISS → Run full LangGraph pipeline ──

            # Create stream callback that publishes tokens to Kafka
            def make_stream_callback(sid: str, prod: Producer):
                async def stream_callback(token_chunk: str):
                    """Push streaming tokens vào Kafka."""
                    stream_event = {
                        "session_id": sid,
                        "answer": token_chunk,
                        "sources": [],
                        "agent_trace": {},
                        "processing_time_ms": 0,
                        "is_final": False,
                        "chunk_type": "token",
                    }
                    prod.produce(
                        TOPIC_QUERY_RESPONSE,
                        key=sid.encode("utf-8"),
                        value=serialize(stream_event),
                    )
                    prod.poll(0)  # Trigger delivery callbacks
                return stream_callback

            stream_cb = make_stream_callback(session_id, producer)

            # Run LangGraph pipeline (timeout 2 phút)
            final_state = loop.run_until_complete(
                asyncio.wait_for(
                    run_streaming(
                        question=question,
                        history=history,
                        session_id=session_id,
                        stream_callback=stream_cb,
                    ),
                    timeout=PIPELINE_TIMEOUT_SECONDS,
                )
            )

            # Build sources from evidence (policy docs)
            evidence = final_state.get("evidence", [])
            sources = []
            for doc in evidence:
                sources.append({
                    "text": doc.get("text", "")[:200],
                    "doc_title": doc.get("doc_title", ""),
                    "category": doc.get("category", ""),
                    "policy_id": doc.get("policy_id", ""),
                    "relevance_score": doc.get("rerank_score", doc.get("rrf_score", 0)),
                })

            # Publish final response
            final_answer = final_state.get("answer", "")
            final_trace = final_state.get("agent_trace", {})
            final_response = {
                "session_id": session_id,
                "answer": final_answer,
                "sources": sources,
                "sentiment": final_state.get("sentiment", ""),
                "sentiment_score": final_state.get("sentiment_score", 0),
                "agent_trace": final_trace,
                "processing_time_ms": final_state.get("processing_time_ms", 0),
                "is_final": True,
                "chunk_type": None,
            }
            producer.produce(
                TOPIC_QUERY_RESPONSE,
                key=session_id.encode("utf-8"),
                value=serialize(final_response),
            )
            producer.flush()

            # ── Write to Redis Cache ──
            set_cached_answer(
                question=question,
                answer=final_answer,
                sources=sources,
                agent_trace=final_trace,
            )

            # ── Flush Langfuse immediately for real-time dashboard ──
            try:
                from utils.observability import flush_langfuse
                flush_langfuse()
            except Exception:
                pass

            query_count += 1
            console.print(
                f"[bold green]📤 Response sent for session {session_id[:8]}... "
                f"(query #{query_count})[/]"
            )

        except Exception as e:
            console.print(f"[red]❌ Query processing error: {e}[/]")
            import traceback
            traceback.print_exc()

            try:
                error_response = {
                    "session_id": query_event.get("session_id", ""),
                    "answer": f"❌ Lỗi xử lý: {str(e)}",
                    "sources": [],
                    "agent_trace": {},
                    "processing_time_ms": 0,
                    "is_final": True,
                    "chunk_type": None,
                }
                producer.produce(
                    TOPIC_QUERY_RESPONSE,
                    key=query_event.get("session_id", "error").encode("utf-8"),
                    value=serialize(error_response),
                )
                producer.flush()
            except Exception:
                pass

    loop.close()
    consumer.close()
    producer.flush()

    # ── Langfuse: flush pending traces ──
    try:
        from utils.observability import flush_langfuse
        flush_langfuse()
    except Exception:
        pass

    console.print(f"[yellow]👋 Query worker stopped. Total queries: {query_count}[/]")


if __name__ == "__main__":
    run_worker()
