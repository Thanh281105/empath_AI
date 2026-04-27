# Dockerfile cho Vertex AI Custom Prediction
# Build và push lên Google Container Registry

FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Install Python và dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    torch==2.1.2 \
    transformers==4.41.0 \
    accelerate>=0.25.0 \
    bitsandbytes>=0.41.0 \
    flask>=3.0.0 \
    gunicorn>=21.0.0 \
    google-cloud-storage \
    huggingface-hub

# Copy model serving code
COPY serve.py /app/serve.py
WORKDIR /app

# Environment variables
ENV MODEL_ID=Thanh28105/empathAI-llama3.1-8b
ENV PORT=8080
ENV HF_HOME=/tmp/hf_cache

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run server
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 serve:app
