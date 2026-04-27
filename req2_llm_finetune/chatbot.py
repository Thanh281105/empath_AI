"""
Yêu cầu 2: LLM fine-tune để hỏi đáp trên tài liệu chính sách MyKingdom.

Kiến trúc:
  Policy JSON --> Format thành context string --> Fine-tuned LLM --> Câu trả lời

Backend: Vertex AI Custom Endpoint (fine-tuned Llama 3.1-8B trên GCP/HuggingFace)
         Groq llama-3.1-8b-instant là fallback nếu Vertex endpoint chưa sẵn sàng.

Ưu điểm : Model đã được học đặc thù trên dữ liệu MyKingdom → phong cách & nội dung tốt hơn.
Nhược điểm: Tốn công fine-tune, vẫn bị giới hạn context window.
"""
import os
import json
import asyncio
import aiohttp
import time
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

EMPATHY_MODE      = os.getenv("EMPATHY_MODE", "vertex")  # "vertex" | "groq"
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEYS     = [k.strip() for k in os.getenv("GROQ_API_KEYS", GROQ_API_KEY).split(",") if k.strip()]
GROQ_API_URL      = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL        = "llama-3.1-8b-instant"   # same family as fine-tuned model
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "")
VERTEX_REGION        = os.getenv("VERTEX_REGION", "asia-southeast1")
VERTEX_ENDPOINT_ID   = os.getenv("VERTEX_ENDPOINT_ID", "")
VERTEX_ENDPOINT_URL  = os.getenv("VERTEX_ENDPOINT_URL", "")

DATA_PATH = PROJECT_ROOT / "data" / "mykingdom_policies.json"

_key_idx = 0
_vertex_access_token  = ""
_vertex_token_expiry  = 0.0
_VERTEX_TOKEN_TTL     = 3000


def _get_groq_key() -> str:
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
# Vertex AI Custom Endpoint (fine-tuned Llama 3.1 8B)
# ---------------------------------------------------------------------------
def _get_vertex_endpoint_url() -> str:
    if VERTEX_ENDPOINT_URL:
        return VERTEX_ENDPOINT_URL
    if VERTEX_ENDPOINT_ID and VERTEX_PROJECT_ID:
        return (
            f"https://{VERTEX_REGION}-aiplatform.googleapis.com/v1/"
            f"projects/{VERTEX_PROJECT_ID}/locations/{VERTEX_REGION}/"
            f"endpoints/{VERTEX_ENDPOINT_ID}"
        )
    return ""


async def _get_vertex_token() -> str:
    global _vertex_access_token, _vertex_token_expiry
    now = time.time()
    if _vertex_access_token and now < _vertex_token_expiry:
        return _vertex_access_token

    token = os.getenv("VERTEX_ACCESS_TOKEN", "")
    if not token:
        try:
            import google.auth
            import google.auth.transport.requests
            credentials, _ = google.auth.default()
            credentials.refresh(google.auth.transport.requests.Request())
            token = credentials.token
        except Exception:
            try:
                import subprocess
                result = subprocess.run(
                    ["gcloud", "auth", "print-access-token"],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    token = result.stdout.strip()
            except Exception:
                pass

    if not token:
        raise RuntimeError(
            "Không lấy được GCP access token. "
            "Chạy: gcloud auth application-default login  hoặc set VERTEX_ACCESS_TOKEN"
        )

    _vertex_access_token = token
    _vertex_token_expiry = now + _VERTEX_TOKEN_TTL
    return token


async def vertex_chat(messages: list[dict]) -> str:
    """Gọi Vertex AI Custom Endpoint (fine-tuned model)."""
    endpoint_url = _get_vertex_endpoint_url()
    if not endpoint_url:
        raise RuntimeError(
            "Vertex AI Endpoint chưa cấu hình. Set VERTEX_ENDPOINT_ID hoặc VERTEX_ENDPOINT_URL trong .env"
        )

    access_token = await _get_vertex_token()
    payload = {
        "instances": [{
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7,
        }]
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{endpoint_url}:predict",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"Vertex AI lỗi ({resp.status}): {error[:300]}")
            result = await resp.json()
            predictions = result.get("predictions", [])
            if predictions:
                pred = predictions[0]
                if isinstance(pred, dict):
                    return pred.get("reply") or pred.get("choices", [{}])[0].get("message", {}).get("content", "")
                return str(pred)
            return ""


# ---------------------------------------------------------------------------
# Groq fallback — Llama 3.1-8B (cùng family với fine-tuned model)
# ---------------------------------------------------------------------------
async def groq_chat(messages: list[dict]) -> str:
    """Gọi Groq base model (fallback)."""
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
        "max_tokens": 512,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            GROQ_API_URL,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"Groq API lỗi ({resp.status}): {error[:300]}")
            result = await resp.json()
            return result["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Unified chat dispatcher
# ---------------------------------------------------------------------------
async def finetuned_chat(messages: list[dict]) -> str:
    """
    Gọi fine-tuned model:
      EMPATHY_MODE=vertex -> Vertex AI Custom Endpoint -> (fallback Groq Llama 3.1-8B)
      EMPATHY_MODE=groq   -> Groq Llama 3.1-8B trực tiếp
    """
    if EMPATHY_MODE == "vertex":
        try:
            return await vertex_chat(messages)
        except Exception as e:
            print(f"  [Vertex AI lỗi] {e} — chuyển sang Groq fallback")

    return await groq_chat(messages)


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------
async def main():
    backend_label = {
        "vertex": "Vertex AI Custom Endpoint (fine-tuned Llama 3.1-8B)",
        "groq"  : f"Groq — {GROQ_MODEL} (base model / fallback)",
    }.get(EMPATHY_MODE, EMPATHY_MODE)

    print("=" * 65)
    print("  EmpathAI — Yêu cầu 2: LLM Fine-tune (Không RAG)")
    print("  Kiến trúc: Full Document Context --> Fine-tuned LLM")
    print("=" * 65)
    print(f"  Backend  : {backend_label}")
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
            response = await finetuned_chat(history)
            print(f"\nBot: {response}\n")
            history.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"\n[Lỗi] {e}\n")
            history.pop()


if __name__ == "__main__":
    asyncio.run(main())
