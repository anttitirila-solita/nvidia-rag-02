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

echo "Checking for NVIDIA GPU..."

# Detect GPU and set Compose profile accordingly
if docker info --format '{{index .Runtimes "nvidia"}}' &>/dev/null \
   && command -v nvidia-smi &>/dev/null \
   && nvidia-smi -L | grep -q 'GPU'; then
    export COMPOSE_PROFILES=gpu
    echo "NVIDIA GPU found — using GPU profile"
else
    export COMPOSE_PROFILES=cpu
    echo "No NVIDIA GPU found — using CPU profile"
fi

echo "Authenticating with nvcr.io..."
echo "${NVIDIA_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin

echo "Starting Open WebUI RAG Docker Compose deployment..."

# --- Setup ---
echo "Creating model cache directory..."
mkdir -p "${MODEL_DIRECTORY}"
export MODEL_DIRECTORY

echo "Deploying on-prem NIMs..."

USERID=$(id -u) docker compose -f deploy/compose/nims_openwebui.yaml up -d --build

# --- Deploy services ---

echo "Starting Ollama..."
docker compose -f deploy/compose/ollama.yaml up -d --build

echo "Starting vector DB containers..."
docker compose -f deploy/compose/vectordb.yaml up -d

echo "Starting Docling..."
docker compose -f deploy/compose/docling.yaml up -d --build

echo "Starting Open WebUI..."
docker compose -f deploy/compose/openwebui.yaml up "$@" -d --build

# Optional: Check if model is already pulled before pulling
MODEL_NAME="gemma3:27b-it-qat"
if ! curl -s http://localhost:11434/api/tags | grep -q "$MODEL_NAME"; then
  echo "Pulling model $MODEL_NAME..."
  curl -X POST http://localhost:11434/api/pull -d "{\"name\": \"$MODEL_NAME\"}"
else
  echo "Model $MODEL_NAME already present."
fi

echo "Deployment complete."
docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"


  # echo "Access RAG Playground at: http://localhost:8090"
  # echo "Access Open WebUI at: http://localhost:8999"

  #     entrypoint: ["sh", "-c"]
  #   command: >
  #     "ollama pull qwen3:14b-q8_0 &&
  #      ollama pull llama4:17b-maverick-128e-instruct-q8_0 &&
  #      ollama pull gemma3:12b-it-q8_0 &&
  #      ollama serve"

