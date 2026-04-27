# Yêu cầu 3 — LLM + RAG (Không fine-tune)

## Kiến trúc

```
       Câu hỏi
          |
          v
  BGE-M3 Embedding
          |
          v
  Qdrant Hybrid Search
  (Dense + Sparse BM25 + RRF)
          |
          v
  BGE-Reranker-v2-M3
  (Cross-encoder rerank)
          |
          v
  Top-K Policy Chunks
          |
          v
  System Prompt + Context
          |
          v
  Groq LLM (llama-3.3-70b)
  BASE model — không fine-tune
          |
          v
       Câu trả lời
```

## Mô tả

RAG (Retrieval-Augmented Generation) pipeline:  
1. **Encode** câu hỏi bằng BGE-M3 → dense vector  
2. **Hybrid Search** trên Qdrant: dense (semantic) + sparse (BM25-like) kết hợp bằng RRF  
3. **Rerank** top-K candidates bằng BGE-Reranker cross-encoder  
4. **Inject** các chunks liên quan vào prompt  
5. **Groq base LLM** sinh câu trả lời dựa trên context được cung cấp  

Khác biệt với Yêu cầu 4: LLM là **base model** (chưa fine-tune), chỉ RAG phụ trách chất lượng context.

| Tiêu chí           | Chi tiết |
|--------------------|----------|
| **Model**          | Groq — `llama-3.3-70b-versatile` (base) |
| **Embedding**      | `BAAI/bge-m3` (1024D) |
| **Reranker**       | `BAAI/bge-reranker-v2-m3` |
| **Vector DB**      | Qdrant (Hybrid: Dense + Sparse) |
| **RAG**            | Có |
| **Fine-tune**      | Không |

## Yêu cầu trước khi chạy

Collection `empathAI_policies` trong Qdrant phải đã được index.  
Nếu chưa, chạy indexing từ thư mục `python/`:

```bash
# Khởi động Qdrant
docker-compose up -d qdrant

# Index tài liệu (từ python/)
cd ../python
python data_processing/indexer.py
```

## Cài đặt & chạy

```bash
# Cài dependencies của req3 (retrieval cần torch + sentence-transformers)
pip install -r requirements.txt

# Chạy chatbot
python chatbot.py
```

> **Lưu ý:** `chatbot.py` tự động import retrieval modules từ `../python/`.  
> Đảm bảo `../python/requirements.txt` cũng đã được cài đặt.

## Ưu / Nhược điểm

**Ưu điểm**
- Scale tốt: chỉ inject chunks liên quan, không bị giới hạn context window
- Không cần fine-tune model → tiết kiệm thời gian & tài nguyên
- Hybrid Search (dense + sparse) bắt được cả câu hỏi ngữ nghĩa lẫn từ khóa

**Nhược điểm**
- Phụ thuộc chất lượng retrieval (garbage in → garbage out)
- Cần infrastructure: Qdrant + embedding model nạp vào RAM
- Base LLM không có phong cách CSKH đặc thù như fine-tuned model
