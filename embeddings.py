"""Voyage AI embeddings — used instead of a local sentence-transformers model
so the deployed app doesn't need PyTorch in memory (that was likely causing
out-of-memory kills on Streamlit Cloud's free tier)."""
import time

import httpx

VOYAGE_MODEL = "voyage-4-large"
VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
BATCH_SIZE = 128
MAX_RETRIES = 6


def embed_texts(texts: list, input_type: str, api_key: str) -> list:
    """Returns one embedding vector per input text, in the same order.
    input_type must be "document" (for indexing content) or "query"
    (for a user's search query) — Voyage optimizes each differently.
    Retries with backoff on 429 (rate limit) responses."""
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        for attempt in range(MAX_RETRIES):
            resp = httpx.post(
                VOYAGE_API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"input": batch, "model": VOYAGE_MODEL, "input_type": input_type},
                timeout=60,
            )
            if resp.status_code == 429 and attempt < MAX_RETRIES - 1:
                wait = float(resp.headers.get("Retry-After", 2 ** attempt * 5))
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        all_embeddings.extend(d["embedding"] for d in data)
    return all_embeddings
