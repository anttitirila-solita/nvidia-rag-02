import os
import logging
from fastapi import FastAPI, Request, Header
import httpx

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
log = logging.getLogger("rerank-proxy")

# URLs for NIM services
NIM_EMBEDDING_URL = os.getenv("NIM_EMBEDDING_URL", "http://nemoretriever-embedding-ms:8000/v1/embeddings")
NIM_RERANKER_URL = os.getenv("NIM_RERANKER_URL", "http://nemoretriever-ranking-ms:8000/v1/ranking")
DEFAULT_RERANK_MODEL = os.getenv("APP_RANKING_MODELNAME", "nvidia/nv-rerankqa-mistral-4b-v3")

app = FastAPI()


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    payload = await request.json()
    payload.setdefault("input_type", "query")

    async with httpx.AsyncClient() as client:
        resp = await client.post(NIM_EMBEDDING_URL, json=payload)
        resp.raise_for_status()
        return resp.json()


@app.post("/v1/ranking")
async def rerank(request: Request, authorization: str = Header(default=None)):
    raw = await request.json()

    if not isinstance(raw, dict) or "query" not in raw or "documents" not in raw:
        return {"error": "Missing required fields: 'query' and 'documents'"}

    query_text = raw["query"]
    passages_raw = raw["documents"]

    payload = {
        "model": raw.get("model", DEFAULT_RERANK_MODEL),
        "query": { "text": query_text },
        "passages": [{ "text": doc } for doc in passages_raw],
        "truncate": "END"
    }

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"Query length: {len(query_text)}")
        lengths = [len(p) for p in passages_raw]
        log.debug(f"Passage count: {len(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}, Max: {max(lengths)}")

    async with httpx.AsyncClient() as client:
        resp = await client.post(NIM_RERANKER_URL, json=payload)
        resp.raise_for_status()
        nim_response = resp.json()

    # Convert NIM output to Open WebUI-compatible format
    if "rankings" in nim_response:
        rankings = nim_response["rankings"]

        # Normalize logits to a positive scale (optional)
        min_logit = min(r["logit"] for r in rankings)
        max_logit = max(r["logit"] for r in rankings)
        range_logit = max_logit - min_logit or 1e-5  # avoid division by zero

        results = [
            {
                "index": r["index"],
                "relevance_score": (r["logit"] - min_logit) / range_logit
            }
            for r in rankings
        ]

        return {"results": sorted(results, key=lambda x: x["index"])}

    return {"error": "NIM reranker did not return expected 'rankings'"}
