"""
Cau hinh chung cho he thong EmpathAI — CSKH thau cam.
Doc bien moi truong tu file .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
POLICY_DIR = DATA_DIR

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Load .env ---
load_dotenv(ENV_FILE)

# --- HuggingFace Login ---
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
    except ImportError:
        pass

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", GROQ_API_KEY).split(",") if k.strip()]
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "empathai-494308")
VERTEX_REGION = os.getenv("VERTEX_REGION", "asia-southeast1")  # Singapore, gần VN nhất

# --- Model Configuration ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# --- EmpathAI Configuration ---
EMPATHY_MODE = os.getenv("EMPATHY_MODE", "vertex")  # "vertex" | "groq"

# Sentiment labels
SENTIMENT_LABELS = ["toxic", "frustrated", "disappointed", "neutral"]

# --- Kafka Configuration ---
KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "localhost:9092")

# --- Qdrant Configuration ---
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "empathAI_policies")
EMBEDDING_DIM = 1024  # bge-m3 output dimension

# --- Retrieval Configuration ---
TOP_K_RETRIEVAL = 8
TOP_K_RERANK = 3

# --- RRF Configuration ---
RRF_DENSE_WEIGHT = float(os.getenv("RRF_DENSE_WEIGHT", "0.6"))
RRF_SPARSE_WEIGHT = float(os.getenv("RRF_SPARSE_WEIGHT", "0.4"))
RRF_K = int(os.getenv("RRF_K", "60"))

# --- Upstash Redis ---
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", str(7 * 24 * 3600)))

# --- Langfuse ---
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# --- Rewrite / Self-Reflective RAG ---
MAX_REWRITE_RETRIES = 2
MIN_GOOD_DOCS = 1  # Data nho, 1 doc chat luong cao la du
GRADE_SCORE_THRESHOLD = 0.15  # Noi long cho tieng Viet
