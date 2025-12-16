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

echo "Token: $HUGGING_FACE_HUB_TOKEN"
docker compose -f deploy/compose/vLLM.yaml up -d --build