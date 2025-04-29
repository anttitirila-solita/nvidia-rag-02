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

echo "Shutting down RAG services..."
docker compose -f deploy/compose/docker-compose-rag-server.yaml down

echo "Shutting down ingestion services..."
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down

echo "Shutting down  vector DB containers..."
docker compose -f deploy/compose/vectordb.yaml down

echo "Shutting down language model and other NIMs services..."
docker compose -f deploy/compose/nims.yaml down