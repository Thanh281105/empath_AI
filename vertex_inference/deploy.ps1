$ErrorActionPreference = "Stop"

# Configuration
$PROJECT_ID = "empathai-494308"
$REGION = "us-central1"
$MODEL_NAME = "empathai-llama8b"
$ENDPOINT_NAME = "empathai-endpoint"
$IMAGE_NAME = "gcr.io/$PROJECT_ID/empathai-llama8b:latest"
$HF_TOKEN = "YOUR_HF_TOKEN"

function Run-Gcloud {
  param([scriptblock]$Command)
  & $Command
  if ($LASTEXITCODE -ne 0) {
    Write-Error "Command failed with exit code $LASTEXITCODE. Stopping script."
    exit $LASTEXITCODE
  }
}

# Step 1: Enable APIs
Write-Host "Enabling Vertex AI API..."
Run-Gcloud { gcloud services enable aiplatform.googleapis.com --project=$PROJECT_ID }
Run-Gcloud { gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID }
Run-Gcloud { gcloud services enable containerregistry.googleapis.com --project=$PROJECT_ID }

# Step 2: Build and push Docker image
Write-Host "Building Docker image..."
Set-Location $PSScriptRoot
Run-Gcloud { gcloud builds submit --tag $IMAGE_NAME --project=$PROJECT_ID }

# Step 3: Upload model to Vertex AI Model Registry
Write-Host "Uploading model to Vertex AI..."
Run-Gcloud { gcloud ai models upload `
    --project=$PROJECT_ID `
    --region=$REGION `
    --display-name=$MODEL_NAME `
    --container-image-uri=$IMAGE_NAME `
    --container-env-vars="HF_TOKEN=$HF_TOKEN,MODEL_ID=thanhhoangnvbg/empathAI-llama3.1-8b" `
    --container-health-route=/health `
    --container-predict-route=/v1/chat/completions `
    --container-ports=8080 }

$MODEL_ID_VAR = (gcloud ai models list --project=$PROJECT_ID --region=$REGION --filter="display_name:$MODEL_NAME" --format="value(name)" | Select-Object -First 1)

if ([string]::IsNullOrWhiteSpace($MODEL_ID_VAR)) {
  Write-Error "Failed to retrieve Model ID"
  exit 1
}
Write-Host "Model ID: $MODEL_ID_VAR"

# Step 4: Create Endpoint (Check if exists first)
$ENDPOINT_ID = (gcloud ai endpoints list --project=$PROJECT_ID --region=$REGION --filter="display_name:$ENDPOINT_NAME" --format="value(name)" | Select-Object -First 1)

if ([string]::IsNullOrWhiteSpace($ENDPOINT_ID)) {
  Write-Host "Creating endpoint..."
  Run-Gcloud { gcloud ai endpoints create `
      --project=$PROJECT_ID `
      --region=$REGION `
      --display-name=$ENDPOINT_NAME }
    
  # Wait a few seconds for endpoint to be fully registered
  Start-Sleep -Seconds 5
  $ENDPOINT_ID = (gcloud ai endpoints list --project=$PROJECT_ID --region=$REGION --filter="display_name:$ENDPOINT_NAME" --format="value(name)" | Select-Object -First 1)
}
else {
  Write-Host "Endpoint already exists."
}

if ([string]::IsNullOrWhiteSpace($ENDPOINT_ID)) {
  Write-Error "Failed to retrieve Endpoint ID"
  exit 1
}
Write-Host "Endpoint ID: $ENDPOINT_ID"

# Step 5: Deploy model to endpoint
Write-Host "Deploying model to endpoint..."
Run-Gcloud { gcloud ai endpoints deploy-model $ENDPOINT_ID `
    --project=$PROJECT_ID `
    --region=$REGION `
    --model=$MODEL_ID_VAR `
    --display-name=$MODEL_NAME `
    --machine-type=g2-standard-8 `
    --accelerator="type=nvidia-l4,count=1" `
    --min-replica-count=1 `
    --max-replica-count=1 `
    --traffic-split=0=100 }

Write-Host "✅ Deployment complete!"
Write-Host "Endpoint URL: https://$REGION-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/$REGION/endpoints/$ENDPOINT_ID"
