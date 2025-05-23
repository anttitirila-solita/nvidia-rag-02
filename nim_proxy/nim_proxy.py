from fastapi import FastAPI, Request
import httpx
import os

app = FastAPI()

NIM_BACKEND_URL = os.getenv("NIM_BACKEND_URL", "http://nemoretriever-embedding-ms:8000/v1/embeddings")

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    payload = await request.json()

    # Add input_type for asymmetric models
    payload.setdefault("input_type", "query")

    async with httpx.AsyncClient() as client:
        resp = await client.post(NIM_BACKEND_URL, json=payload)
        return resp.json()

