"""
FastAPI server for req3_llm_rag — LLM + RAG (Qdrant Hybrid Search)
Endpoint: POST /chat
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Add both req3 and python to path for retrieval imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time

from chatbot import rag_answer

app = FastAPI(title="EmpathAI — LLM + RAG", version="1.0")

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
    model: str = "llama-3.1-8b-instant + RAG"
    processing_time_ms: int = 0


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    t0 = time.time()

    history = []
    for msg in (req.history or [])[-6:]:
        history.append({"role": msg.role, "content": msg.content})

    answer = await rag_answer(req.question, history)

    elapsed = int((time.time() - t0) * 1000)
    return ChatResponse(answer=answer, processing_time_ms=elapsed)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "llm_rag"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("REQ3_PORT", "8003"))
    uvicorn.run(app, host="0.0.0.0", port=port)
