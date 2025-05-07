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

echo "Shutting down Open WebUI..."
docker compose -f deploy/compose/openwebui.yaml down

echo "Shutting down Ollama..."
docker compose -f deploy/compose/ollama.yaml down

echo "Shutting down vector DB containers..."
docker compose -f deploy/compose/vectordb.yaml down