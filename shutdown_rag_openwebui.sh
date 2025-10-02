#!/bin/bash

set -e

export USERID=$(id -u) 

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

echo "Shutting down Open WebUI..."
docker compose -f deploy/compose/openwebui.yaml down

echo "Shutting down Docling..."
docker compose -f deploy/compose/docling.yaml down

echo "Shutting down Ollama..."
docker compose -f deploy/compose/ollama.yaml down

echo "Shutting down vector DB containers..."
docker compose -f deploy/compose/vectordb.yaml down

echo "Shutting down NIMs..."
docker compose -f deploy/compose/nims_openwebui.yaml down