<div align="center">
  <img src="assets/empathai_logo.png" alt="EmpathAI Logo" width="200" />
  
# 🧠 EmpathAI

### Customer Service AI: RAG + Fine-Tuning + Emotion Intelligence
  
  [![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
  [![Status](https://img.shields.io/badge/Status-Production--Ready-success?style=for-the-badge)](https://github.com/)

  **EmpathAI** là hệ thống AI Chăm sóc khách hàng (CSKH) tiếng Việt, chuyên giải quyết khiếu nại bằng sự **thấu cảm thực thụ** thay vì những câu trả lời rập khuôn.
</div>

---

## 📖 Tổng quan

Dự án là sự kết hợp giữa **Agentic RAG (LangGraph)**, **Phân tích Cảm xúc (Sentiment Analysis)**, và **Supervised Fine-Tuning** trên **Vertex AI / HuggingFace**. Chúng tôi biến một mô hình ngôn ngữ thành một trợ lý CSKH tận tâm, biết lắng nghe và phản hồi bằng ngôn từ xoa dịu.

### 🔄 Luồng xử lý chính (Yêu cầu 4)

```mermaid
graph TD
    A[Khách hàng khiếu nại] --> B[LangGraph AI Worker]
    
    subgraph "AI Core Pipeline (Fine-tuned LLM + RAG)"
        B --> C[Sentiment Analysis]
        C --> D[Hybrid Search: Qdrant]
        D --> E{Grade Docs}
        E -->|Bad| F[Rewrite Query]
        F --> D
        E -->|Good| G[Empathy Writer: Fine-tuned Llama-3.1-8B]
        G --> H[Empathy Reviewer]
    end
    
    H --> I[Streaming Response]
```

---

## 🏗️ Kiến trúc & Công nghệ

| Thành phần | Công nghệ | Vai trò |
|:--- |:--- |:--- |
| **Agentic RAG** | **LangGraph / Python** | Điều phối workflow phức tạp và Self-Reflective RAG. |
| **LLM Backends** | **Vertex AI Custom Endpoint**, **Groq** | Fine-tuned Llama-3.1-8B và fallback base model. |
| **Vector DB** | **Qdrant** | Hybrid Search (Dense + Sparse) với RRF fusion. |
| **Reranker** | **BGE-Reranker-v2-M3** | Cross-encoder reranking để tăng độ chính xác. |
| **Observability** | **Langfuse** | Tracing, giám sát và đánh giá chất lượng hội thoại. |
| **Infrastructure** | **Docker** | Dễ dàng triển khai Qdrant và các dịch vụ phụ trợ. |

---

## ✨ Điểm nổi bật

- 🎭 **Empathy-First Learning**: Fine-tuned trên dữ liệu CSKH MyKingdom giúp AI phân biệt văn phong thấu cảm và văn mẫu máy móc.
- 🔍 **Hybrid & Self-Reflective RAG**: Kết hợp Dense + Sparse search trên Qdrant, tự động sửa truy vấn (Rewriter) nếu thông tin không đạt yêu cầu.
- 🛡️ **Empathy Quality Checker**: Agent độc lập kiểm duyệt lần cuối để đảm bảo AI không dùng từ ngữ "robot".
- 📦 **4 Yêu cầu rõ ràng**: Cung cấp 4 kiến trúc khác nhau để so sánh hiệu quả: LLM only, Fine-tuned only, RAG only, và Fine-tuned + RAG.

---

## 🚀 Hướng dẫn cài đặt

### 1. Chuẩn bị Hạ tầng

```bash
# Khởi động Qdrant (vector database)
docker-compose up -d qdrant

# Cài đặt thư viện Python
cd python && pip install -r requirements.txt
```

### 2. Cấu hình Environment

Sao chép và chỉnh sửa file `.env`:

```bash
# Vertex AI (fine-tuned model)
VERTEX_PROJECT_ID=your-project-id
VERTEX_REGION=asia-southeast1
VERTEX_ENDPOINT_ID=your-endpoint-id

# Groq (fallback)
GROQ_API_KEY=your-groq-key

# Chế độ backend: "vertex" hoặc "groq"
EMPATHY_MODE=vertex
```

### 3. Chạy để Demo (4 Model + Giao diện So sánh)

Khởi động toàn bộ hệ thống để demo so sánh 4 kiến trúc trên giao diện web:

**Bước 1 — Hạ tầng (Qdrant + Kafka):**

```bash
docker-compose up -d qdrant redpanda
```

**Bước 2 — Backend 1: EmpathAI Full (Rust + LangGraph):**

```bash
cd python/kafka_workers
python query_worker.py

# New terminal
cd rust_backend
cargo run
# WebSocket: ws://localhost:8083/ws/chat
```

**Bước 3 — Backend 2: LLM Only (FastAPI):**

```bash
cd req1_llm_only
uvicorn server:app --host 0.0.0.0 --port 8001
# API: http://localhost:8001/chat
```

**Bước 4 — Backend 3: LLM Fine-tune (FastAPI):**

```bash
cd req2_llm_finetune
uvicorn server:app --host 0.0.0.0 --port 8002
# API: http://localhost:8002/chat
```

**Bước 5 — Backend 4: LLM + RAG (FastAPI):**

```bash
cd req3_llm_rag
uvicorn server:app --host 0.0.0.0 --port 8003
# API: http://localhost:8003/chat
```

**Bước 6 — Mở giao diện:**

```
http://localhost:8083
```

**Chế độ So sánh 4 Model:**

- Chọn **"So sánh cả 4"** trong dropdown bên trái thanh input
- Nhập câu hỏi → giao diện hiển thị **grid 2×2** so sánh kết quả từng model song song
- Mỗi panel hiển thị: tên model, thời gian phản hồi (ms), và nội dung trả lời

**Chế độ Chat đơn:**

- Chọn 1 model trong dropdown → chat bình thường với model đó
- Badge trên tin nhắn AI hiển thị tên model đang dùng

> 💡 *Có thể chạy tất cả backend Python bằng Docker Compose:* `docker-compose up -d req1_llm_only req2_llm_finetune req3_llm_rag`

---

## 📈 Giám sát (Monitoring)

Hệ thống tích hợp **Langfuse** để theo dõi từng bước suy nghĩ của AI:

- Xem chi tiết Retrieval docs.
- Theo dõi độ trễ (Latency) và chi phí (Tokens).
- Đánh giá Feedback của khách hàng trên Dashboard.

## 📁 Cấu trúc thư mục

```
vibe coding/
├── frontend/               # Giao diện web (HTML/JS/Tailwind)
├── rust_backend/           # Gateway: WebSocket + REST API + Kafka
├── python/                 # Yêu cầu 4: EmpathAI Full (LangGraph + Agentic RAG)
│   ├── agents/             # LangGraph pipeline
│   ├── retrieval/          # Hybrid Search + Rerank
│   └── data_processing/    # Indexing
├── req1_llm_only/          # Yêu cầu 1: LLM only (FastAPI server.py)
├── req2_llm_finetune/      # Yêu cầu 2: Fine-tuned LLM (FastAPI server.py)
├── req3_llm_rag/           # Yêu cầu 3: LLM + RAG (FastAPI server.py)
├── data/
│   └── mykingdom_policies.json
├── docker-compose.yml
└── .env
```

---

<div align="center">
  <sub>Built with ❤️ by Hoàng Xuân Thành. Powered by Python, Vertex AI & Llama 3.1.</sub>
</div>
