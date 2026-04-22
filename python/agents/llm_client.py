"""
Groq LLM Client — Wrapper cho Groq API (stream + non-stream).

Extracted từ lightrag_setup/llm_wrapper.py và mở rộng thêm streaming.
Hỗ trợ:
  - Round-robin API key rotation
  - Rate limiting cho Free Tier
  - Auto-truncation prompt
  - Streaming completions (SSE)
"""
import asyncio
import aiohttp
import json
import time
import tiktoken
from typing import AsyncGenerator

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, GROQ_API_KEYS, KAGGLE_INFERENCE_URL

# Langfuse decorator (graceful fallback if not configured)
try:
    from langfuse import observe as _observe
    def observe(**kwargs):
        """Wrapper that silently degrades if Langfuse is not configured."""
        return _observe(**kwargs)
except ImportError:
    def observe(**kwargs):
        """No-op decorator when langfuse not installed."""
        def decorator(func):
            return func
        return decorator


# ─── Constants ───────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_SMART = "llama-3.3-70b-versatile"
GROQ_MODEL_FAST = "llama-3.1-8b-instant"

# Token limits
GROQ_FREE_TIER_MAX_INPUT_TOKENS = 4500
_NUM_KEYS = max(1, len(GROQ_API_KEYS) if GROQ_API_KEYS else 1)
GROQ_RATE_LIMIT_DELAY = 6.0 / _NUM_KEYS  # Dãn cách rộng hơn để Groq 70B không bị nghẽn TPM

# Tokenizer
_ENCODER = tiktoken.get_encoding("cl100k_base")

# Round-robin state
_groq_key_index = 0
_last_groq_call_time = 0.0
_rate_limit_lock = asyncio.Lock()
_BAD_GROQ_KEYS = set()


# ─── Token Utilities ─────────────────────────────────────

def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text, disallowed_special=()))


def _truncate_text(text: str, max_tokens: int) -> str:
    tokens = _ENCODER.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return _ENCODER.decode(tokens[:max_tokens]) + "\n[...truncated...]"


def _truncate_messages(
    messages: list[dict],
    max_total_tokens: int = GROQ_FREE_TIER_MAX_INPUT_TOKENS,
) -> list[dict]:
    """Cắt ngắn messages để vừa giới hạn TPM."""
    total = sum(_count_tokens(m.get("content", "")) for m in messages)
    if total <= max_total_tokens:
        return messages

    longest_idx = max(
        range(len(messages)),
        key=lambda i: _count_tokens(messages[i].get("content", "")),
    )

    overflow = total - max_total_tokens
    longest_content = messages[longest_idx]["content"]
    longest_tokens = _count_tokens(longest_content)
    new_max = max(500, longest_tokens - overflow)

    truncated = list(messages)
    truncated[longest_idx] = {
        **messages[longest_idx],
        "content": _truncate_text(longest_content, new_max),
    }
    return truncated


# ─── Key Management ──────────────────────────────────────

def _get_groq_key() -> str:
    global _groq_key_index
    all_keys = GROQ_API_KEYS if GROQ_API_KEYS else [GROQ_API_KEY]
    
    # Filter valid keys
    valid_keys = [k for k in all_keys if k not in _BAD_GROQ_KEYS]
    if not valid_keys:
        raise RuntimeError("All Groq API keys are invalid or restricted.")
    
    key = valid_keys[_groq_key_index % len(valid_keys)]
    _groq_key_index += 1
    return key


async def _respect_rate_limit():
    global _last_groq_call_time
    async with _rate_limit_lock:
        now = time.time()
        elapsed = now - _last_groq_call_time
        if elapsed < GROQ_RATE_LIMIT_DELAY and _last_groq_call_time > 0:
            wait = GROQ_RATE_LIMIT_DELAY - elapsed
            await asyncio.sleep(wait)
        _last_groq_call_time = time.time()


# ─── Non-Streaming Completion ────────────────────────────

async def groq_complete(
    prompt: str,
    system_prompt: str = "",
    model: str = GROQ_MODEL_SMART,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    """Groq completion (non-streaming)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    return await groq_chat_complete(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


@observe(name="groq_chat_complete", as_type="generation")
async def groq_chat_complete(
    messages: list[dict],
    model: str = GROQ_MODEL_SMART,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    """Groq chat completion với auto-truncation và rate limiting."""
    api_key = _get_groq_key()

    await _respect_rate_limit()
    
    if model == GROQ_MODEL_FAST:
        messages = _truncate_messages(messages)
        max_tokens = min(max_tokens, 1500)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    GROQ_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=900),
                ) as resp:
                    if resp.status == 429:
                        wait_time = 25 * (attempt + 1)
                        await asyncio.sleep(wait_time)
                        api_key = _get_groq_key()
                        headers["Authorization"] = f"Bearer {api_key}"
                        continue

                    if resp.status == 413:
                        new_limit = int(
                            GROQ_FREE_TIER_MAX_INPUT_TOKENS
                            * (0.7 ** (attempt + 1))
                        )
                        messages = _truncate_messages(
                            messages, max_total_tokens=max(800, new_limit)
                        )
                        payload["messages"] = messages
                        await asyncio.sleep(25)
                        continue

                    if resp.status in [400, 401, 403]:
                        error_text = await resp.text()
                        if "restricted" in error_text.lower() or resp.status in [401, 403]:
                            _BAD_GROQ_KEYS.add(api_key)
                            api_key = _get_groq_key()
                            headers["Authorization"] = f"Bearer {api_key}"
                            continue

                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"Groq API error ({resp.status}): {error_text[:300]}"
                        )

                    result = await resp.json()
                    choices = result.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "")
                    return ""
        except aiohttp.ClientError as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Groq connection error after {max_retries} retries: {e}"
                )
            await asyncio.sleep(5)

    return ""


# ─── Streaming Completion ────────────────────────────────

async def groq_stream_complete(
    prompt: str,
    system_prompt: str = "",
    model: str = GROQ_MODEL_SMART,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> AsyncGenerator[str, None]:
    """
    Groq streaming completion — yield từng token.
    Sử dụng SSE (Server-Sent Events) từ Groq API.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    async for token in groq_stream_chat_complete(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    ):
        yield token


@observe(name="groq_stream_chat_complete", as_type="generation")
async def groq_stream_chat_complete(
    messages: list[dict],
    model: str = GROQ_MODEL_SMART,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> AsyncGenerator[str, None]:
    """Groq streaming chat completion — yield từng token chunk."""
    api_key = _get_groq_key()

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    GROQ_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    if resp.status == 429:
                        wait_time = 25 * (attempt + 1)
                        await asyncio.sleep(wait_time)
                        api_key = _get_groq_key()
                        headers["Authorization"] = f"Bearer {api_key}"
                        continue

                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"Groq stream error ({resp.status}): {error_text[:300]}"
                        )

                    # Parse SSE stream
                    async for line in resp.content:
                        line_str = line.decode("utf-8").strip()
                        if not line_str or not line_str.startswith("data:"):
                            continue

                        data_str = line_str[5:].strip()
                        if data_str == "[DONE]":
                            return

                        try:
                            chunk = json.loads(data_str)
                            delta = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, IndexError):
                            continue

                    return  # Stream completed
        except aiohttp.ClientError as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Groq stream error after retries: {e}")
            await asyncio.sleep(5)


# ─── Kaggle Inference Client ────────────────────────────

async def kaggle_complete(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Call fine-tuned model on Kaggle via FastAPI + Cloudflare Tunnel."""
    if not KAGGLE_INFERENCE_URL:
        raise RuntimeError("KAGGLE_INFERENCE_URL not configured in .env")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
    }

    url = f"{KAGGLE_INFERENCE_URL.rstrip('/')}/api/chat"

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload,
            timeout=aiohttp.ClientTimeout(total=900),
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"Kaggle API error ({resp.status}): {error[:300]}")
            result = await resp.json()
            # Kaggle FastAPI returns {"reply": "..."}
            return result.get("reply", "")


async def kaggle_stream_complete(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    """Streaming call to Kaggle fine-tuned model (fallback to non-stream)."""
    # Kaggle FastAPI server doesn't support streaming yet,
    # so we simulate it by yielding the full response in chunks
    response = await kaggle_complete(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Simulate streaming: yield word by word
    words = response.split(" ")
    for i, word in enumerate(words):
        if i > 0:
            yield " " + word
        else:
            yield word
