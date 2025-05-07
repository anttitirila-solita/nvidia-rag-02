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

echo "Starting Open WebUI RAG Docker Compose deployment..."

  # --- Deploy services ---
  
  # echo "Starting Ollaam..."
  docker compose -f deploy/compose/ollama.yaml up -d --build

  echo "Starting vector DB containers..."
  docker compose -f deploy/compose/vectordb.yaml up -d

  echo "Starting Open WebUI..."
  docker compose -f deploy/compose/openwebui.yaml up -d

  sleep 10

  # echo "Checking RAG service health..."
  # curl -X 'GET' 'http://localhost:8081/v1/health?check_dependencies=true' -H 'accept: application/json'

  curl -X POST http://localhost:11434/api/pull -d '{"name": "qwen3:14b-q8_0"}'

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

