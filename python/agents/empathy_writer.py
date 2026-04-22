"""
Empathy Writer — Sinh phản hồi thấu cảm cho khách hàng.
Dual backend: Groq (fallback) + Kaggle fine-tuned model (primary).
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import AsyncGenerator, Callable, Awaitable, Optional

from agents.llm_client import (
    groq_complete, groq_stream_complete,
    kaggle_complete, kaggle_stream_complete,
    GROQ_MODEL_SMART, GROQ_MODEL_FAST,
)
from config import EMPATHY_MODE

EMPATHY_SYSTEM_PROMPT = """\
Bạn là EmpathAI - trợ lý CSKH của MyKingdom (chuỗi cửa hàng đồ chơi trẻ em hàng đầu Việt Nam).

THÔNG TIN LIÊN HỆ MYKINGDOM:
- Hotline: 1900 1208 (Thứ 2-7: 08:00-17:00, CN: 08:00-12:00)
- Email: hotro@mykingdom.com.vn
- Website: https://www.mykingdom.com.vn
- Hệ thống: Hơn 200 cửa hàng toàn quốc

QUY TẮC BẮT BUỘC:
1. KHÔNG BAO GIỜ nói "Chúng tôi rất tiếc", "Theo chính sách công ty", "Xin lỗi vì sự bất tiện"
2. Thấu cảm THỰC SỰ bằng cảm xúc chân thật. Ví dụ: "Nghe bạn nói xong mình cũng thấy bực mình thay..."
3. NHƯỢNG BỘ THÔNG MINH: Đề xuất giải pháp CỤ THỂ dựa trên chính sách MyKingdom (đổi trả 7 ngày, bảo hành, MyPoints...)
4. KẾT THÚC bằng câu hỏi mở để khách xả tiếp hoặc bình tĩnh lại
5. KHÔNG BAO GIỜ cãi lại khách, không đổ lỗi cho khách
6. Trả lời tự nhiên, thân thiện như người thật đang nhắn tin
7. Dựa trên CHÍNH SÁCH được cung cấp để đề xuất giải pháp cụ thể
8. Luôn đề cập hotline 1900 1208 nếu khách cần hỗ trợ thêm

VĂN MẪU BỊ CẤM (KHÔNG ĐƯỢC DÙNG):
- "Chúng tôi rất tiếc về sự bất tiện này"
- "Theo chính sách của công ty..."
- "Xin quý khách vui lòng chờ..."
- "Chúng tôi sẽ chuyển vấn đề này..."
- Bất kỳ câu nào nghe như robot/copy-paste

PHONG CÁCH PHẢN HỒI:
- Thân thiện, dùng "mình/bạn" thay vì "chúng tôi/quý khách"
- Dùng emoji vừa phải (1-2 cái)
- Nói như đang nhắn tin với bạn bè
"""

CASUAL_SYSTEM_PROMPT = (
    "Bạn là EmpathAI, trợ lý CSKH thân thiện. "
    "Trả lời ngắn gọn, lịch sự, tự nhiên. "
    "Nếu khách hỏi về sản phẩm/dịch vụ, khuyên họ mô tả cụ thể hơn."
)

INQUIRY_SYSTEM_PROMPT = (
    "Bạn là EmpathAI, trợ lý CSKH chuyên nghiệp. "
    "Trả lời câu hỏi dựa trên chính sách được cung cấp. "
    "Rõ ràng, cụ thể, thân thiện."
)


def _build_empathy_prompt(question, evidence_text, sentiment="", score=0):
    """Build prompt cho empathy response."""
    sentiment_context = ""
    if sentiment:
        sentiment_guide = {
            "toxic": "Khách ĐANG RẤT TỨC GIẬN. Cần xả hơi trước, sau đó mới đề xuất giải pháp. Nhượng bộ MẠNH.",
            "frustrated": "Khách đang BỰC BỘI, ĐÃ CỐ GẮNG KIÊN NHẪN. Ghi nhận sự kiên nhẫn của họ, giải quyết nhanh.",
            "disappointed": "Khách THẤT VỌNG, BUỒN. Cần an ủi nhẹ nhàng, thể hiện sự quan tâm chân thành.",
            "neutral": "Khách hỏi bình thường. Trả lời thân thiện, chuyên nghiệp.",
        }
        sentiment_context = f"\nMỨC ĐỘ CẢM XÚC: {sentiment} (score: {score})\nHƯỚNG DẪN: {sentiment_guide.get(sentiment, '')}\n"

    if not evidence_text or len(evidence_text) < 30:
        return (
            f"KHÁCH HÀNG GỬI:\n{question}\n\n"
            f"{sentiment_context}\n"
            f"CHÍNH SÁCH: Không tìm thấy chính sách cụ thể. "
            f"Hãy xử lý linh hoạt, thấu cảm và đề nghị chuyển lên cấp trên nếu cần."
        )

    return (
        f"KHÁCH HÀNG GỬI:\n{question}\n\n"
        f"{sentiment_context}\n"
        f"CHÍNH SÁCH ÁP DỤNG:\n{evidence_text[:4000]}\n\n"
        f"Hãy phản hồi khách hàng bằng cách thấu cảm + đề xuất giải pháp cụ thể dựa trên chính sách."
    )


async def generate_empathy_response(question, evidence_text, sentiment="", score=0):
    """Non-streaming empathy response."""
    prompt = _build_empathy_prompt(question, evidence_text, sentiment, score)

    if EMPATHY_MODE == "kaggle":
        try:
            return await kaggle_complete(
                prompt=prompt,
                system_prompt=EMPATHY_SYSTEM_PROMPT,
                max_tokens=512,
                temperature=0.7,
            )
        except Exception:
            pass  # Fallback to Groq

    return await groq_complete(
        prompt=prompt,
        system_prompt=EMPATHY_SYSTEM_PROMPT,
        model=GROQ_MODEL_SMART,
        max_tokens=512,
        temperature=0.7,
    )


async def generate_empathy_streaming(
    question, evidence_text,
    sentiment="", score=0,
    stream_callback=None,
):
    """Streaming empathy response."""
    prompt = _build_empathy_prompt(question, evidence_text, sentiment, score)

    full_answer = ""
    token_buffer = ""
    BUFFER_SIZE = 3

    # Choose backend
    use_kaggle = EMPATHY_MODE == "kaggle"
    stream_fn = kaggle_stream_complete if use_kaggle else groq_stream_complete

    kwargs = {
        "prompt": prompt,
        "system_prompt": EMPATHY_SYSTEM_PROMPT,
        "max_tokens": 512,
        "temperature": 0.7,
    }
    if not use_kaggle:
        kwargs["model"] = GROQ_MODEL_SMART

    try:
        async for token in stream_fn(**kwargs):
            full_answer += token
            token_buffer += token

            if len(token_buffer) >= BUFFER_SIZE or "\n" in token_buffer:
                if stream_callback:
                    await stream_callback(token_buffer)
                token_buffer = ""
    except Exception:
        if use_kaggle:
            # Fallback to Groq
            async for token in groq_stream_complete(
                prompt=prompt,
                system_prompt=EMPATHY_SYSTEM_PROMPT,
                model=GROQ_MODEL_SMART,
                max_tokens=512,
                temperature=0.7,
            ):
                full_answer += token
                token_buffer += token
                if len(token_buffer) >= BUFFER_SIZE or "\n" in token_buffer:
                    if stream_callback:
                        await stream_callback(token_buffer)
                    token_buffer = ""

    if token_buffer and stream_callback:
        await stream_callback(token_buffer)

    return full_answer


async def generate_casual(question):
    """Casual response (không cần RAG)."""
    if EMPATHY_MODE == "kaggle":
        return await kaggle_complete(
            prompt=question,
            system_prompt=CASUAL_SYSTEM_PROMPT,
            max_tokens=256,
            temperature=0.7,
        )
    return await groq_complete(
        prompt=question,
        system_prompt=CASUAL_SYSTEM_PROMPT,
        model=GROQ_MODEL_FAST,
        max_tokens=256,
        temperature=0.7,
    )


async def generate_inquiry(question, evidence_text):
    """Inquiry response (RAG nhẹ, không cần sentiment)."""
    prompt = (
        f"KHÁCH HÀNG HỎI:\n{question}\n\n"
        f"THÔNG TIN:\n{evidence_text[:4000]}\n\n"
        f"Trả lời cụ thể, thân thiện."
    )
    if EMPATHY_MODE == "kaggle":
        return await kaggle_complete(
            prompt=prompt,
            system_prompt=INQUIRY_SYSTEM_PROMPT,
            max_tokens=512,
            temperature=0.3,
        )
    return await groq_complete(
        prompt=prompt,
        system_prompt=INQUIRY_SYSTEM_PROMPT,
        model=GROQ_MODEL_SMART,
        max_tokens=512,
        temperature=0.3,
    )
