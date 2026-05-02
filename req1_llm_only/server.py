"""
FastAPI server for req1_llm_only — LLM Only (No RAG)
Endpoint: POST /chat
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Add req1 to path so chatbot.py imports work
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio

from chatbot import llm_chat, SYSTEM_PROMPT

app = FastAPI(title="EmpathAI — LLM Only", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    answer: str
    model: str = "llama-3.1-8b-instant"
    processing_time_ms: int = 0


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    import time
    t0 = time.time()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in (req.history or [])[-10:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": req.question})

    answer = await llm_chat(messages)

    elapsed = int((time.time() - t0) * 1000)
    return ChatResponse(answer=answer, processing_time_ms=elapsed)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "llm_only"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("REQ1_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
