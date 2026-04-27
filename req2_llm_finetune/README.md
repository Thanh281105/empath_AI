# Yêu cầu 2 — LLM Fine-tune (Không RAG)

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
  Fine-tuned LLM
  (Vertex AI / Kaggle / Groq fallback)
        |
        v
     Câu trả lời
```

## Mô tả

Tương tự Yêu cầu 1 nhưng thay model gốc bằng **model đã fine-tune** trên dữ liệu hội thoại
CSKH của MyKingdom (tập dữ liệu trong `fine-tune-llm.ipynb`).  
Toàn bộ chính sách vẫn được nhúng trực tiếp vào system prompt — không có RAG.

| Tiêu chí        | Chi tiết |
|-----------------|----------|
| **Model**       | Fine-tuned Llama 3.1 8B (Vertex AI / Kaggle) |
| **RAG**         | Không |
| **Fine-tune**   | Có |
| **Infrastructure** | Vertex AI Endpoint hoặc Kaggle FastAPI |

## Cài đặt & chạy

```bash
pip install -r requirements.txt
python chatbot.py
```

## Cấu hình backend (`.env`)

| Biến môi trường       | Giá trị          | Ý nghĩa |
|-----------------------|------------------|---------|
| `EMPATHY_MODE`        | `vertex`         | Dùng Vertex AI Custom Endpoint |
| `EMPATHY_MODE`        | `kaggle`         | Dùng Kaggle + Cloudflare Tunnel |
| `EMPATHY_MODE`        | `groq`           | Groq base model (fallback/test) |
| `VERTEX_ENDPOINT_ID`  | ID endpoint      | ID của deployed model trên Vertex |
| `KAGGLE_INFERENCE_URL`| URL ngrok/cloudflare | URL FastAPI trên Kaggle |

## Ưu / Nhược điểm

**Ưu điểm**
- Model hiểu sâu phong cách giao tiếp CSKH MyKingdom
- Phản hồi tự nhiên, phù hợp domain hơn base model
- Vẫn đơn giản: không cần vector DB

**Nhược điểm**
- Tốn công và tài nguyên để fine-tune
- Vẫn bị giới hạn context window
- Cần deploy và duy trì model endpoint
