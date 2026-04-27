"""
Yêu cầu 3: LLM + RAG để hỏi đáp trên tài liệu chính sách MyKingdom.

Kiến trúc:
  Query --> Qdrant Hybrid Search (Dense BGE-M3 + Sparse BM25 + RRF)
         --> Cross-encoder Reranker (BGE-Reranker-v2-M3)
         --> Top-K policy chunks
         --> Groq LLM (base model, không fine-tune)
         --> Câu trả lời

Ưu điểm : Chỉ inject chunks liên quan → không bị giới hạn context window,
           scale tốt với tài liệu lớn, base LLM không cần fine-tune.
Nhược điểm: Phụ thuộc chất lượng retrieval, cần Qdrant đã được index.

Yêu cầu: Qdrant collection "empathAI_policies" phải đã được index
         (chạy indexing trong thư mục python/ trước).
"""
import os
import sys
import asyncio
import aiohttp
import time
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Thêm ../python vào sys.path để tái sử dụng retrieval modules
# ---------------------------------------------------------------------------
PYTHON_ROOT  = Path(__file__).parent.parent / "python"
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PYTHON_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", GROQ_API_KEY).split(",") if k.strip()]
GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.1-8b-instant"

TOP_K_SEARCH  = int(os.getenv("TOP_K_RETRIEVAL", "8"))
TOP_K_RERANK  = int(os.getenv("TOP_K_RERANK",    "3"))

_key_idx = 0


def _get_groq_key() -> str:
    global _key_idx
    keys = GROQ_API_KEYS if GROQ_API_KEYS else [GROQ_API_KEY]
    key  = keys[_key_idx % len(keys)]
    _key_idx += 1
    return key


# ---------------------------------------------------------------------------
# System prompt (không nhúng toàn bộ document — RAG sẽ cung cấp context)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
Bạn là EmpathAI - trợ lý chăm sóc khách hàng của MyKingdom \
(chuỗi cửa hàng đồ chơi trẻ em hàng đầu Việt Nam).

THÔNG TIN LIÊN HỆ:
- Hotline: 1900 1208 (Thứ 2-7: 08:00-17:00, CN: 08:00-12:00)
- Email: hotro@mykingdom.com.vn
- Website: https://www.mykingdom.com.vn

QUY TẮC TRẢ LỜI:
1. Chỉ dùng thông tin trong [CHÍNH SÁCH LIÊN QUAN] được cung cấp.
2. Nếu không tìm thấy thông tin, hướng dẫn liên hệ hotline.
3. Thân thiện, dùng "mình/bạn" thay vì "chúng tôi/quý khách".
4. Ngắn gọn, rõ ràng, dẫn chiếu tên chính sách khi có thể.
"""


# ---------------------------------------------------------------------------
# Retrieval (import từ ../python/)
# ---------------------------------------------------------------------------
_retrieval_ready = False
_retrieve_fn     = None


def _init_retrieval():
    """Lazy-load retrieval modules lần đầu tiên khi cần."""
    global _retrieval_ready, _retrieve_fn
    if _retrieval_ready:
        return

    print("  [RAG] Đang khởi tạo embedding model & Qdrant client...")
    try:
        from indexing.query_engine import retrieve_and_rerank, format_evidence
        _retrieve_fn    = (retrieve_and_rerank, format_evidence)
        _retrieval_ready = True
        print("  [RAG] Sẵn sàng.")
    except ImportError as e:
        print(f"  [RAG] Lỗi import: {e}")
        print("  [RAG] Đảm bảo đã cài requirements trong python/ và Qdrant đang chạy.")
        raise


def retrieve_context(query: str) -> str:
    """Lấy top-K policy chunks liên quan đến query."""
    _init_retrieval()
    retrieve_and_rerank, format_evidence = _retrieve_fn

    docs = retrieve_and_rerank(
        query,
        top_k_search=TOP_K_SEARCH,
        top_k_rerank=TOP_K_RERANK,
    )

    if not docs:
        return ""

    return format_evidence(docs)


# ---------------------------------------------------------------------------
# LLM call (Groq base model)
# ---------------------------------------------------------------------------
async def llm_chat(messages: list[dict]) -> str:
    """Gọi Groq base LLM và trả về nội dung phản hồi."""
    if not GROQ_API_KEY and not GROQ_API_KEYS:
        raise RuntimeError("GROQ_API_KEY chưa được cấu hình trong file .env")

    api_key = _get_groq_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    max_retries = 3
    for attempt in range(max_retries):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                GROQ_API_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 429:
                    wait = 20 * (attempt + 1)
                    print(f"  [Rate limit] Chờ {wait}s...")
                    await asyncio.sleep(wait)
                    api_key = _get_groq_key()
                    headers["Authorization"] = f"Bearer {api_key}"
                    continue
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"Groq API lỗi ({resp.status}): {error[:300]}")
                result = await resp.json()
                return result["choices"][0]["message"]["content"]
    return ""


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------
async def rag_answer(question: str, history: list[dict]) -> str:
    """
    RAG pipeline:
      1. Retrieve top-K chunks từ Qdrant (Hybrid Search + Rerank)
      2. Build prompt với retrieved context
      3. Groq base LLM sinh câu trả lời
    """
    t0 = time.time()

    # Step 1: Retrieval (sync, chạy trong thread để không block event loop)
    evidence_text = await asyncio.to_thread(retrieve_context, question)

    elapsed_retrieve = int((time.time() - t0) * 1000)
    if evidence_text:
        print(f"  [Retrieve] {len(evidence_text)} ký tự context ({elapsed_retrieve}ms)")
    else:
        print(f"  [Retrieve] Không tìm thấy chunk liên quan ({elapsed_retrieve}ms)")

    # Step 2: Build RAG prompt
    if evidence_text:
        rag_user_message = (
            f"[CHÍNH SÁCH LIÊN QUAN]\n{evidence_text}\n\n"
            f"[CÂU HỎI KHÁCH HÀNG]\n{question}"
        )
    else:
        rag_user_message = (
            f"[CHÍNH SÁCH LIÊN QUAN]\nKhông tìm thấy chính sách liên quan.\n\n"
            f"[CÂU HỎI KHÁCH HÀNG]\n{question}"
        )

    # Step 3: Build messages (system + history + rag_message)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history[-(6):]:          # Giữ 6 lượt gần nhất
        messages.append(msg)
    messages.append({"role": "user", "content": rag_user_message})

    # Step 4: LLM generation
    response = await llm_chat(messages)

    elapsed_total = int((time.time() - t0) * 1000)
    print(f"  [Total] {elapsed_total}ms")
    return response


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------
async def main():
    print("=" * 65)
    print("  EmpathAI — Yêu cầu 3: LLM + RAG (Không fine-tune)")
    print("  Kiến trúc: Qdrant Hybrid Search + Rerank --> Groq LLM")
    print("=" * 65)
    print(f"  Model    : {GROQ_MODEL} (base, không fine-tune)")
    print(f"  Retrieval: Top-{TOP_K_SEARCH} search --> Top-{TOP_K_RERANK} rerank")
    print("  Gõ 'quit' để thoát\n")

    history: list[dict] = []

    while True:
        try:
            user_input = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĐã thoát.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "thoát", "q"):
            print("Tạm biệt!")
            break

        try:
            response = await rag_answer(user_input, history)
            print(f"\nBot: {response}\n")
            # Lưu lịch sử với câu hỏi gốc (không có context RAG)
            history.append({"role": "user",      "content": user_input})
            history.append({"role": "assistant",  "content": response})
        except Exception as e:
            print(f"\n[Lỗi] {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
