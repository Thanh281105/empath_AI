"""
FastAPI server for req2_llm_finetune — Fine-tuned LLM (or Groq fallback)
Endpoint: POST /chat
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time

from chatbot import finetuned_chat, SYSTEM_PROMPT, EMPATHY_MODE

app = FastAPI(title="EmpathAI — LLM Fine-tune", version="1.0")

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
    model: str
    processing_time_ms: int = 0


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    t0 = time.time()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in (req.history or [])[-10:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": req.question})

    answer = await finetuned_chat(messages)

    elapsed = int((time.time() - t0) * 1000)
    model_label = "vertex-finetuned" if EMPATHY_MODE == "vertex" else "groq-llama-3.1-8b"
    return ChatResponse(answer=answer, model=model_label, processing_time_ms=elapsed)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "llm_finetune", "mode": EMPATHY_MODE}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("REQ2_PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
