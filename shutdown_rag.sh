#!/bin/bash

set -e

export USERID=$(id -u) 

echo "Shutting down RAG services..."
docker compose -f deploy/compose/docker-compose-rag-server.yaml down

echo "Shutting down ingestion services..."
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down

echo "Shutting down  vector DB containers..."
docker compose -f deploy/compose/vectordb.yaml down

echo "Shutting down language model and other NIMs services..."
docker compose -f deploy/compose/nims.yaml down