from __future__ import annotations

import asyncio
import json
import os
import re
import argparse
from json import JSONDecodeError
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Load environment variables from .env
load_dotenv()

class LLMAPIError(RuntimeError):
    pass

def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()

def extract_json_payload(text: str) -> Any:
    # Remove code fences
    cleaned = strip_code_fences(text)
    
    # Try simple load first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Final attempt: try to find anything between { and } using regex and strip markdown
    # Some LLMs output: "Here is the JSON: ```json { ... } ```"
    # We strip common garbage first
    text_no_md = cleaned.replace('```json', '').replace('```', '').strip()
    
    # Try the scan logic on the cleaned text
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text_no_md):
        if ch in ('{', '['):
            try:
                obj, _ = decoder.raw_decode(text_no_md, i)
                return obj
            except json.JSONDecodeError:
                continue

    # One last desperate regex for the largest possible block
    match = re.search(r'(\{.*\})', text_no_md, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"LLM output does not contain valid JSON. Raw output: {text[:200]}...")

def parse_http_response_payload(text: str) -> tuple[str, str]:
    raw = text.strip()
    if not raw:
        return "", "empty"

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            message = data.get("message")
            if isinstance(message, dict) and message.get("content") is not None:
                return str(message["content"]), "json_message"
            return str(data.get("reply") or data.get("content") or data.get("text") or ""), "json"
        if isinstance(data, str):
            return data, "json_string"
    except JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(raw)
        trailing = raw[end:].strip()
        if isinstance(obj, dict):
            message = obj.get("message")
            if isinstance(message, dict) and message.get("content") is not None:
                payload = str(message["content"])
            else:
                payload = str(obj.get("reply") or obj.get("content") or obj.get("text") or "")
            return payload, "json_prefix_with_trailing" if trailing else "json_prefix"
        if isinstance(obj, str):
            return obj, "json_string_prefix"
    except JSONDecodeError:
        pass

    json_lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("data:"):
            stripped = stripped[5:].strip()
        if stripped == "[DONE]":
            continue
        try:
            item = json.loads(stripped)
        except JSONDecodeError:
            continue
        json_lines.append(item)

    if json_lines:
        last = json_lines[-1]
        if isinstance(last, dict):
            message = last.get("message")
            if isinstance(message, dict) and message.get("content") is not None:
                return str(message["content"]), "json_lines_message"
            return str(last.get("reply") or last.get("content") or last.get("text") or ""), "json_lines"
        if isinstance(last, str):
            return last, "json_lines_string"

    return raw, "plain_text"

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return records

def load_jsonl_best_effort(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Best-effort mode for crash recovery: skip truncated tail lines.
                print(f"[WARN] Skipping invalid JSONL line {line_no} in {path}")
    return records

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def normalize_role(role: str) -> str:
    role_lower = role.lower()
    if role_lower in {"human", "customer", "client"}:
        return "user"
    if role_lower in {"bot", "model"}:
        return "assistant"
    return role_lower

def normalize_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in record and isinstance(record["messages"], list):
        items = record["messages"]
    elif "chosen" in record and isinstance(record["chosen"], list):
        items = record["chosen"]
    else:
        items = None

    if items is not None:
        return [
            {
                "role": str(message.get("role", "")).strip(),
                "content": str(message.get("content", "")).strip(),
            }
            for message in items
            if (message.get("role") or message.get("from")) and (message.get("content") is not None or message.get("value") is not None)
        ]

    if "conversation" in record and isinstance(record["conversation"], list):
        normalized = []
        for turn in record["conversation"]:
            role = turn.get("role") or turn.get("from") or turn.get("speaker")
            content = turn.get("content") or turn.get("value") or turn.get("text")
            if role and content is not None:
                normalized.append({"role": str(role).strip(), "content": str(content).strip()})
        if normalized:
            return normalized

    if "prompt" in record and "completion" in record:
        return [
            {"role": "user", "content": str(record["prompt"]).strip()},
            {"role": "assistant", "content": str(record["completion"]).strip()},
        ]

    if "user" in record and "assistant" in record:
        return [
            {"role": "user", "content": str(record["user"]).strip()},
            {"role": "assistant", "content": str(record["assistant"]).strip()},
        ]

    raise ValueError("Unsupported conversation schema.")

def build_chatml_record(user_text: str, assistant_text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    record = {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }
    if metadata:
        record.update(metadata)
    return record

@dataclass
class RateLimiter:
    requests_per_minute: int

    def __post_init__(self) -> None:
        self._interval = 0.0 if self.requests_per_minute <= 0 else 60.0 / self.requests_per_minute
        self._lock = asyncio.Lock()
        self._last_ts = 0.0

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            now = asyncio.get_running_loop().time()
            wait_for = self._interval - (now - self._last_ts)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_ts = asyncio.get_running_loop().time()

class AsyncInferenceClient:
    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str,
        inference_type: str,
        timeout_seconds: int,
        max_concurrency: int,
        requests_per_minute: int,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.inference_type = inference_type
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)
        self.vertex_project = vertex_project
        self.vertex_location = vertex_location
        self.session: aiohttp.ClientSession | None = None
        self._vertex_model: Any | None = None

    async def __aenter__(self) -> "AsyncInferenceClient":
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        if self.inference_type == "vertex":
            import vertexai
            from vertexai.generative_models import GenerativeModel
            vertexai.init(project=self.vertex_project, location=self.vertex_location)
            self._vertex_model = GenerativeModel(self.model)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.session:
            await self.session.close()

    @retry(
        reraise=True,
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, LLMAPIError)),
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
        response_format: dict[str, str] | None = None,
    ) -> str:
        if not self.session:
            raise RuntimeError("ClientSession is not initialized.")

        await self.rate_limiter.acquire()
        async with self.semaphore:
            if self.inference_type == "kaggle":
                url = f"{self.base_url}/api/chat"
                print(f"[DEBUG] Calling Kaggle: {url}")
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                }
                headers = {"Content-Type": "application/json"}
                
                async with self.session.post(url, json=payload, headers=headers) as response:
                    text = await response.text()
                    if response.status in {429, 500, 502, 503, 504}:
                        raise LLMAPIError(f"API {response.status}: {text[:200]}")
                    if response.status >= 400:
                        raise RuntimeError(f"API FATAL {response.status}: {text[:200]}")
                    parsed_text, response_kind = parse_http_response_payload(text)
                    if response_kind != "json":
                        snippet = text[:200].replace("\n", "\\n")
                        print(f"[DEBUG] Kaggle response parsed as {response_kind}: {snippet}")
                    return parsed_text
            elif self.inference_type == "vertex":
                from vertexai.generative_models import Content, Part
                # Format messages for Vertex AI
                contents = []
                system_instruction = None
                for msg in messages:
                    role = normalize_role(msg["role"])
                    if role == "system":
                        system_instruction = msg["content"]
                    else:
                        # Vertex roles are 'user' and 'model'
                        v_role = "user" if role == "assistant" else "user" 
                        if role == "assistant": v_role = "model"
                        contents.append(Content(role=v_role, parts=[Part.from_text(msg["content"])]))
                
                model = self._vertex_model
                if system_instruction:
                    from vertexai.generative_models import GenerativeModel
                    model = GenerativeModel(self.model, system_instruction=system_instruction)

                gen_config = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                if response_format and response_format.get("type") == "json_object":
                    gen_config["response_mime_type"] = "application/json"

                from vertexai.generative_models import HarmCategory, HarmBlockThreshold
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                }

                response = await model.generate_content_async(
                    contents,
                    generation_config=gen_config,
                    safety_settings=safety_settings
                )
                return response.text
            else:
                payload: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    payload["response_format"] = response_format
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                async with self.session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                    text = await response.text()
                    if response.status in {429, 500, 502, 503, 504}:
                        raise LLMAPIError(f"API {response.status}: {text[:200]}")
                    if response.status >= 400:
                        raise RuntimeError(f"API FATAL {response.status}: {text[:200]}")
                    data = json.loads(text)
                    choices = data.get("choices") or []
                    if not choices:
                        raise LLMAPIError("No choices.")
                    return choices[0]["message"]["content"]

def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inference-type", type=str, choices=["openai", "kaggle", "google", "vertex"], default="openai")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--requests-per-minute", type=int, default=120)
    parser.add_argument("--timeout-seconds", type=int, default=180)

def resolve_api_settings(args: argparse.Namespace) -> tuple[str | None, str, str, str]:
    inference_type = args.inference_type
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    base_url = args.base_url or os.getenv("GOOGLE_API_BASE") or os.getenv("OPENAI_API_BASE") or os.getenv("GROQ_API_BASE") or "https://api.groq.com/openai/v1"
    model = args.model or os.getenv("LLM_MODEL") or "llama-3.3-70b-versatile"

    if inference_type == "google":
        api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
        base_url = args.base_url or os.getenv("GOOGLE_API_BASE") or "https://generativelanguage.googleapis.com/v1beta/openai"
        model = args.model or os.getenv("EMPATHY_MODEL_ID") or "gemini-2.0-flash-exp"
        return api_key, base_url, model, inference_type, None, None
    elif inference_type == "vertex":
        model = args.model or os.getenv("EMPATHY_MODEL_ID") or "gemini-2.5-flash"
        v_project = os.getenv("VERTEX_PROJECT_ID")
        v_location = os.getenv("VERTEX_REGION")
        return None, "", model, inference_type, v_project, v_location
    elif inference_type == "kaggle":
        base_url = args.base_url or os.getenv("KAGGLE_INFERENCE_URL")
        if not base_url:
            raise RuntimeError(
                "LỖI: Bạn chọn chế độ Kaggle nhưng chưa cấu hình KAGGLE_INFERENCE_URL trong file .env hoặc tham số --base-url"
            )
        return None, base_url, model, inference_type, None, None
    
    return api_key, base_url, model, inference_type, None, None
