"""
Redis Cache Layer — Upstash Serverless Redis (HTTP REST).

Caching câu trả lời RAG để bypass LangGraph pipeline cho các query đã từng trả lời.
Sử dụng Upstash (serverless) → không cần cài Redis local, chỉ gọi HTTP.

Cache Strategy:
  - Key: SHA-256 hash của câu hỏi (normalized lowercase + strip)
  - Value: JSON {answer, sources, agent_trace, cached_at}
  - TTL: Mặc định 7 ngày (cấu hình qua REDIS_CACHE_TTL)
"""
import hashlib
import json
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN, REDIS_CACHE_TTL
from utils.console import console


# ─── Singleton Redis Client ──────────────────────────────

_redis_client = None
_redis_available = False


def _get_redis():
    """Lazy init Upstash Redis client. Returns None if not configured."""
    global _redis_client, _redis_available

    if _redis_client is not None:
        return _redis_client if _redis_available else None

    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        console.print("[dim]  Cache: Upstash Redis not configured, skipping[/]")
        _redis_available = False
        return None

    try:
        from upstash_redis import Redis
        _redis_client = Redis(
            url=UPSTASH_REDIS_REST_URL,
            token=UPSTASH_REDIS_REST_TOKEN,
        )
        _redis_available = True
        console.print("[green]  Cache: Upstash Redis connected ✓[/]")
        return _redis_client
    except Exception as e:
        console.print(f"[yellow]  Cache: Redis init failed: {e}[/]")
        _redis_available = False
        return None


# ─── Key Generation ──────────────────────────────────────

def _make_cache_key(question: str) -> str:
    """
    Tạo Redis key từ câu hỏi.
    Normalize: lowercase + strip + SHA-256 (16 chars).
    """
    normalized = question.strip().lower()
    hash_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"rag:cache:{hash_digest}"


# ─── Public API ──────────────────────────────────────────

def get_cached_answer(question: str) -> dict | None:
    """
    Kiểm tra cache cho câu hỏi.

    Args:
        question: Câu hỏi tiếng Việt gốc

    Returns:
        dict {answer, sources, agent_trace, cached_at} nếu HIT, None nếu MISS.
    """
    redis = _get_redis()
    if redis is None:
        return None

    try:
        key = _make_cache_key(question)
        data = redis.get(key)

        if data is None:
            return None

        # Upstash trả về string hoặc bytes
        if isinstance(data, bytes):
            data = data.decode("utf-8")

        result = json.loads(data) if isinstance(data, str) else data
        console.print(f"[green]  ⚡ Cache HIT: {key}[/]")
        return result

    except Exception as e:
        console.print(f"[yellow]  Cache GET error: {e}[/]")
        return None


def set_cached_answer(
    question: str,
    answer: str,
    sources: list,
    agent_trace: dict,
    ttl: int = REDIS_CACHE_TTL,
) -> bool:
    """
    Ghi cache cho câu trả lời.

    Args:
        question: Câu hỏi gốc
        answer: Câu trả lời đầy đủ
        sources: Danh sách sources
        agent_trace: Trace metadata
        ttl: Thời gian sống (giây), mặc định 7 ngày

    Returns:
        True nếu ghi thành công
    """
    redis = _get_redis()
    if redis is None:
        return False

    try:
        key = _make_cache_key(question)
        value = json.dumps({
            "answer": answer,
            "sources": sources,
            "agent_trace": {
                **(agent_trace or {}),
                "from_cache": True,
            },
            "cached_at": time.time(),
        }, ensure_ascii=False)

        redis.setex(key, ttl, value)
        console.print(f"[dim]  Cache SET: {key} (TTL={ttl}s)[/]")
        return True

    except Exception as e:
        console.print(f"[yellow]  Cache SET error: {e}[/]")
        return False
