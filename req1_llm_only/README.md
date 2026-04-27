# Yêu cầu 1 — Chỉ dùng LLM

## Kiến trúc

```
data/mykingdom_policies.json
        |
        v
  Load & format JSON
        |
        v
  System Prompt (Full Document Context)
        |
        v
  Groq LLM (llama-3.3-70b-versatile)
        |
        v
     Câu trả lời
```

## Mô tả

Toàn bộ nội dung `data/mykingdom_policies.json` (9 chính sách) được đọc, định dạng thành
plain-text và nhúng trực tiếp vào **system prompt** của LLM.  
LLM phải tự tra cứu trong context đó để trả lời câu hỏi — không có vector DB, không có retrieval.

| Tiêu chí        | Chi tiết |
|-----------------|----------|
| **Model**       | Groq — `llama-3.3-70b-versatile` |
| **RAG**         | Không |
| **Fine-tune**   | Không |
| **Infrastructure** | Groq API + `.env` |

## Cài đặt & chạy

```bash
pip install -r requirements.txt
python chatbot.py
```

## Ưu / Nhược điểm

**Ưu điểm**
- Cực kỳ đơn giản, không cần vector DB hay embedding model
- Dễ triển khai, chỉ cần API key

**Nhược điểm**
- Bị giới hạn bởi context window của LLM
- Không scale khi tài liệu lớn (hàng trăm trang)
- Toàn bộ document phải gửi lên mỗi lần gọi API → tốn token
