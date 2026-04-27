#!/bin/bash
# Deploy fine-tuned model lên Vertex AI Endpoint
# Chạy từ Cloud Shell hoặc local với gcloud CLI

# Configuration
PROJECT_ID="empathai-494308"  # Thay bằng project ID của bạn
REGION="asia-southeast1"       # Singapore (gần VN nhất, giảm latency)
MODEL_NAME="empathai-llama8b"
ENDPOINT_NAME="empathai-endpoint"
IMAGE_NAME="gcr.io/$PROJECT_ID/empathai-llama8b:latest"
HF_TOKEN="your_huggingface_token_here"  # Thay bằng token thật

# Step 1: Enable APIs
echo "Enabling Vertex AI API..."
gcloud services enable aiplatform.googleapis.com

# Step 2: Build và push Docker image
echo "Building Docker image..."
cd "$(dirname "$0")"
gcloud builds submit --tag $IMAGE_NAME

# Step 3: Upload model lên Vertex AI Model Registry
echo "Uploading model to Vertex AI..."
gcloud ai models upload \
  --region=$REGION \
  --display-name=$MODEL_NAME \
  --container-image-uri=$IMAGE_NAME \
  --container-env-vars=HF_TOKEN=$HF_TOKEN,MODEL_ID=Thanh28105/empathAI-llama3.1-8b \
  --container-health-route=/health \
  --container-predict-route=/v1/chat/completions \
  --container-ports=8080

MODEL_ID=$(gcloud ai models list --region=$REGION --filter="display_name:$MODEL_NAME" --format="value(model_id)" | head -1)
echo "Model ID: $MODEL_ID"

# Step 4: Create Endpoint
echo "Creating endpoint..."
gcloud ai endpoints create \
  --region=$REGION \
  --display-name=$ENDPOINT_NAME

ENDPOINT_ID=$(gcloud ai endpoints list --region=$REGION --filter="display_name:$ENDPOINT_NAME" --format="value(endpoint_id)" | head -1)
echo "Endpoint ID: $ENDPOINT_ID"

# Step 5: Deploy model to endpoint
echo "Deploying model to endpoint..."
gcloud ai endpoints deploy-model $ENDPOINT_ID \
  --region=$REGION \
  --model=$MODEL_ID \
  --display-name=$MODEL_NAME \
  --machine-type=n1-standard-4 \
  --accelerator-type=NVIDIA_L4 \
  --accelerator-count=1 \
  --min-replica-count=1 \
  --max-replica-count=1 \
  --traffic-split=0=100

echo "✅ Deployment complete!"
echo "Endpoint URL: https://$REGION-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/$REGION/endpoints/$ENDPOINT_ID"
