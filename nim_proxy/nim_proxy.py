import os
import logging
import math
from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST
import httpx

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "CRITICAL").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
log = logging.getLogger("nim-proxy")

# URLs for NIM services
NIM_EMBEDDING_URL = os.getenv("NIM_EMBEDDING_URL", "http://nemoretriever-embedding-ms:8000/v1/embeddings")
NIM_RERANKER_URL = os.getenv("NIM_RERANKER_URL", "http://nemoretriever-ranking-ms:8000/v1/ranking")
DEFAULT_RERANK_MODEL = os.getenv("NIM_RANKING_MODELNAME", "nvidia/nv-rerankqa-mistral-4b-v3")

app = FastAPI()


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    payload = await request.json()
    payload.setdefault("input_type", "query")

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(NIM_EMBEDDING_URL, json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            log.error(f"NIM embedding error: {e.response.text}")
            raise
        return resp.json()


@app.post("/v1/ranking")
async def rerank(request: Request, authorization: str = Header(None)):
    raw = await request.json()

    if not isinstance(raw, dict) or "query" not in raw or "documents" not in raw:
        return JSONResponse(
            status_code=HTTP_400_BAD_REQUEST,
            content={"error": "Missing required fields: 'query' and 'documents'"}
        )

    query_text = raw["query"]
    passages_raw = raw["documents"]

    payload = {
        "model": raw.get("model", DEFAULT_RERANK_MODEL),
        "query": {"text": query_text},
        "passages": [{"text": doc} for doc in passages_raw],
        "truncate": "END"
    }

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"Query length: {len(query_text)}")
        log.debug(f"Query text: {query_text}")
        lengths = [len(p) for p in passages_raw]
        log.debug(f"Passage count: {len(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}, Max: {max(lengths)}")

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(NIM_RERANKER_URL, json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            log.error(f"NIM reranker error: {e.response.text}")
            raise

        nim_response = resp.json()

    if "rankings" in nim_response:
        rankings = nim_response["rankings"]
        probs = [math.exp(-r["logit"]) for r in rankings]

        results = [
            {
                "index": r["index"],
                "relevance_score": prob / (1 + prob)
            }
            for r, prob in zip(rankings, probs)
        ]

        return {"results": sorted(results, key=lambda x: x["index"])}

    return JSONResponse(
        status_code=500,
        content={"error": "NIM reranker did not return expected 'rankings'"}
    )
