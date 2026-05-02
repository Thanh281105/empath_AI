"""
Yêu cầu 1: Chỉ dùng LLM để hỏi đáp trên tài liệu chính sách MyKingdom.

Kiến trúc:
  Policy JSON --> Format thành context string --> LLM (Groq) --> Câu trả lời

Ưu điểm : Đơn giản, không cần vector DB, không cần infrastructure phức tạp.
Nhược điểm: Bị giới hạn context window, không scale với tài liệu lớn.
"""
import os
import json
import asyncio
import aiohttp
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", GROQ_API_KEY).split(",") if k.strip()]
GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.1-8b-instant"

DATA_PATH = PROJECT_ROOT / "data" / "mykingdom_policies.json"

# Round-robin key index
_key_idx = 0


def _get_key() -> str:
    global _key_idx
    keys = GROQ_API_KEYS if GROQ_API_KEYS else [GROQ_API_KEY]
    key  = keys[_key_idx % len(keys)]
    _key_idx += 1
    return key


# ---------------------------------------------------------------------------
# Load & format toàn bộ tài liệu JSON thành chuỗi context
# ---------------------------------------------------------------------------
def load_policy_context() -> str:
    """Đọc mykingdom_policies.json và chuyển thành plain-text cho system prompt."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    contact  = metadata.get("contact", {})
    policies = data.get("policies", [])

    lines = [
        f"=== TÀI LIỆU CHÍNH SÁCH {metadata.get('brand', 'MYKINGDOM').upper()} ===",
        f"Hotline      : {contact.get('hotline', '1900 1208')}",
        f"Email        : {contact.get('email', 'hotro@mykingdom.com.vn')}",
        f"Website      : {contact.get('website', 'https://www.mykingdom.com.vn')}",
        f"Giờ làm việc : {contact.get('working_hours', '')}",
        "",
    ]

    for policy in policies:
        lines.append(f"## {policy['title']}")
        if policy.get("summary"):
            lines.append(f"Tóm tắt: {policy['summary']}")
        if policy.get("keywords"):
            lines.append(f"Từ khóa: {', '.join(policy['keywords'])}")
        lines.append("")

        for section in policy.get("sections", []):
            heading = section.get("heading", "")
            if heading:
                lines.append(f"### {heading}")
            content = section.get("content", "")
            if content:
                lines.append(content)
            for item in section.get("items", []):
                lines.append(f"- {item}")
            lines.append("")

    return "\n".join(lines)


POLICY_CONTEXT = load_policy_context()

SYSTEM_PROMPT = f"""Bạn là EmpathAI - trợ lý chăm sóc khách hàng của MyKingdom \
(chuỗi cửa hàng đồ chơi trẻ em hàng đầu Việt Nam).

Hãy trả lời câu hỏi của khách hàng DỰA TRÊN tài liệu chính sách dưới đây.
Nếu thông tin không có trong tài liệu, hãy nói rõ và hướng dẫn liên hệ hotline.

{POLICY_CONTEXT}

QUY TẮC TRẢ LỜI:
- Thân thiện, dùng "mình/bạn" thay vì "chúng tôi/quý khách"
- Ngắn gọn, đi thẳng vào vấn đề
- Dẫn chiếu tên chính sách cụ thể khi có thể
- Luôn cung cấp thông tin liên hệ nếu cần hỗ trợ thêm
"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
async def llm_chat(messages: list[dict]) -> str:
    """Gọi Groq LLM và trả về nội dung phản hồi."""
    if not GROQ_API_KEY and not GROQ_API_KEYS:
        raise RuntimeError("GROQ_API_KEY chưa được cấu hình trong file .env")

    api_key = _get_key()
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
                    api_key = _get_key()
                    headers["Authorization"] = f"Bearer {api_key}"
                    continue
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"Groq API lỗi ({resp.status}): {error[:300]}")
                result = await resp.json()
                return result["choices"][0]["message"]["content"]
    return ""


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------
async def main():
    print("=" * 60)
    print("  EmpathAI — Yêu cầu 1: Chỉ dùng LLM (Không RAG)")
    print("  Kiến trúc: Full Document Context --> Groq LLM")
    print("=" * 60)
    print(f"  Model    : {GROQ_MODEL}")
    print(f"  Context  : {len(POLICY_CONTEXT):,} ký tự đã nạp vào system prompt")
    print("  Gõ 'quit' để thoát\n")

    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

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

        history.append({"role": "user", "content": user_input})
        try:
            response = await llm_chat(history)
            print(f"\nBot: {response}\n")
            history.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"\n[Lỗi] {e}\n")
            history.pop()


if __name__ == "__main__":
    asyncio.run(main())
