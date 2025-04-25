#!/bin/bash

set -e

# Load environment variables
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
  echo "Loading variables from $ENV_FILE"
  set -o allexport
  source "$ENV_FILE"
  set +o allexport
else
  echo "Error: $ENV_FILE not found"
  exit 1
fi

echo "Starting NVIDIA RAG Docker Compose deployment..."

echo "Authenticating with nvcr.io..."
echo "${NVIDIA_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin

# --- Setup ---
echo "Creating model cache directory..."
mkdir -p "${MODEL_DIRECTORY}"
export MODEL_DIRECTORY

if [ "$USE_ON_PREM" = true ]; then
  echo "Deploying on-prem NIMs..."
  export LLM_MS_GPU_ID
  USERID=$(id -u) docker compose -f deploy/compose/nims.yaml up -d

  echo "Waiting for 'nim-llm-ms' container to be healthy..."
  while true; do
    STATUS=$(docker ps --format '{{.Names}}\t{{.Status}}' | grep '^nim-llm-ms' | awk -F '\t' '{print $2}')
    if echo "$STATUS" | grep -q "(healthy)"; then
      echo "'nim-llm-ms' is healthy."
      break
    else
      echo "Current status of 'nim-llm-ms': $STATUS"
      sleep 10
    fi
  done
else
  echo "Using NVIDIA hosted models..."
  export APP_EMBEDDINGS_SERVERURL
  export APP_LLM_SERVERURL
  export APP_RANKING_SERVERURL
  export EMBEDDING_NIM_ENDPOINT
  export PADDLE_HTTP_ENDPOINT
  export YOLOX_HTTP_ENDPOINT
  export YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT
  export YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT
fi

# --- Deploy services ---
echo "Starting vector DB containers..."
docker compose -f deploy/compose/vectordb.yaml up -d

echo "Starting ingestion services..."
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d

echo "Starting RAG services..."
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d

sleep 10

echo "Checking RAG service health..."
curl -X 'GET' 'http://localhost:8081/v1/health?check_dependencies=true' -H 'accept: application/json'

echo "Deployment complete."
docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"

echo "Access RAG Playground at: http://localhost:8090"