# Vertex AI Custom Endpoint Deployment

Deploy EmpathAI fine-tuned model lên Vertex AI Endpoint với L4 GPU.

## Kiến trúc

```
Local (Q1000 4GB)          │         Cloud (Vertex AI)
───────────────────────────┼────────────────────────────────
Embedding (BGE-M3 fp16)    │    Fine-tuned Llama 3.1 8B
Vector DB (Qdrant)         │    L4 GPU 24GB VRAM
RAG Logic (LangGraph)      │    Auto-scaling 0-1 replica
         │                 │
         ▼                 │         ▲
    ┌──────────────┐       │    ┌──────────────┐
    │  gRPC/REST   │───────┼───▶│   Endpoint   │
    └──────────────┘       │    └──────────────┘
```

## Prerequisites

1. **GCP Project** với billing enabled
2. **Free Credit**: $300 (~7.5 triệu VND)
3. **gcloud CLI** installed: https://cloud.google.com/sdk/docs/install
4. **HuggingFace Token** (để download model)

## Cost Estimate (L4 GPU)

| Thành phần | Chi phí |
|---|---|
| NVIDIA L4 GPU | ~$0.80-1.00/giờ |
| n1-standard-4 (CPU + RAM) | ~$0.15/giờ |
| **Tổng** | **~$0.95-1.15/giờ** |
| **Với $300 credit** | **~260-315 giờ** |

### Tiết kiệm chi phí

**Option 1: Auto-scaling (scale to 0)**
```bash
gcloud ai endpoints deploy-model ... \
  --min-replica-count=0 \
  --max-replica-count=1
```
→ Chỉ trả khi có request, cold start ~30-60 giây

**Option 2: Scheduled shutdown**
```bash
# Tự động undeploy sau giờ làm việc
0 18 * * * gcloud ai endpoints undeploy-model ...
```

## Deployment Steps

### 1. Set up gcloud

```bash
gcloud auth login
gcloud config set project empathai-494308  # Thay bằng project ID của bạn
gcloud config set compute/region asia-southeast1
```

### 2. Edit deploy.sh

```bash
# Mở file và sửa:
PROJECT_ID="your-actual-project-id"
HF_TOKEN="your-huggingface-token"
```

### 3. Run deployment

```bash
cd vertex_inference
chmod +x deploy.sh
./deploy.sh
```

Quá trình này mất **15-20 phút** lần đầu (build image + download model).

### 4. Get Endpoint ID

Sau khi deploy xong:

```bash
gcloud ai endpoints list --region=asia-southeast1
```

Copy `ENDPOINT_ID` (dạng: `1234567890123456789`)

### 5. Update .env trong dự án chính

```
# .env file
echo "VERTEX_PROJECT_ID=empathai-494308" >> .env
echo "VERTEX_REGION=asia-southeast1" >> .env
echo "VERTEX_ENDPOINT_ID=your-endpoint-id-here" >> .env
# Hoặc dùng API key nếu public endpoint
echo "VERTEX_API_KEY=your-vertex-api-key" >> .env

# Chuyển sang dùng Vertex cho empathy mode
echo "EMPATHY_MODE=vertex" >> .env
```

## API Usage

### Python Client (llm_client.py đã support)

```python
from agents.llm_client import vertex_chat_complete

messages = [
    {"role": "system", "content": "Bạn là EmpathAI..."},
    {"role": "user", "content": "Sản phẩm bị lỗi"}
]

response = await vertex_chat_complete(
    messages=messages,
    model="projects/xxx/locations/asia-southeast1/endpoints/xxx",
    max_tokens=512,
    temperature=0.7
)
```

### REST API trực tiếp

```bash
# Lấy access token
TOKEN=$(gcloud auth print-access-token)

# Gọi endpoint
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  https://asia-southeast1-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/asia-southeast1/endpoints/ENDPOINT_ID:predict \
  -d '{
    "instances": [{
      "messages": [{"role": "user", "content": "Hello"}],
      "max_tokens": 512
    }]
  }'
```

## Troubleshooting

### Model loading timeout

Vertex AI có 30 phút để download model. Nếu lâu hơn, dùng pre-built image:

```bash
# Download model local trước, bundle vào image
docker build --build-arg HF_TOKEN=$HF_TOKEN -t $IMAGE_NAME .
```

### Out of memory

L4 24GB vẫn OOM? Thử:
- Giảm `max_new_tokens` (512 → 256)
- Dùng 8-bit quantization thay vì 4-bit
- Chuyển sang A100 (đắt hơn ~2x)

### Cold start quá chậm

Nếu dùng `--min-replica-count=0`, cold start ~30-60s để load model từ GCS.
→ Dùng `--min-replica-count=1` để keep warm (đắt hơn nhưng nhanh)

## Monitoring

```bash
# Xem logs
gcloud ai endpoints logs-stream ENDPOINT_ID --region=asia-southeast1

# Xem metrics (latency, requests/sec)
gcloud monitoring metrics list --filter="metric.type:aiplatform.googleapis.com"
```

## So sánh với Kaggle

| | Kaggle | Vertex AI (L4) |
|---|---|---|
| **Chi phí** | Free | ~$1/giờ |
| **Uptime** | 9h max/session | 24/7 |
| **GPU** | P100 16GB | L4 24GB (nhanh hơn 2x) |
| **Cold start** | ~2 phút | ~30 giây (nếu warm) |
| **Credit** | Không cần | $300 free |
| **Tổng thời gian** | 12h/24h | ~300 giờ với $300 |

→ **Vertex AI phù hợp** nếu bạn cần production-stable inference.

## Next Steps

1. Chạy `./deploy.sh` để deploy
2. Test endpoint với curl/Python
3. Update `llm_client.py` để wire vào `EMPATHY_MODE=vertex`
