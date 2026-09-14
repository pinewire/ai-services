"""Embedding provider abstraction.

Default is a deterministic local feature-hashing embedder, so retrieval works
with zero external dependencies, no API key, and no model download — good
enough for dev/CI. Swap in a hosted API or sentence-transformers model in
production by adding a provider branch to `get_embedding_client`.
"""

import hashlib
import math
import os
import re
from typing import Any

EMBEDDING_DIM = 1536
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _hash_embed(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Feature-hashing embedding: bag-of-tokens hashed into a fixed-size,
    signed vector. No model, no network call, fully deterministic."""
    vec = [0.0] * dim
    for token in _tokenize(text):
        digest = hashlib.sha256(token.encode()).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class EmbeddingClient:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashingEmbeddingClient(EmbeddingClient):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embed(t) for t in texts]


class OpenAIEmbeddingClient(EmbeddingClient):
    """OpenAI embeddings sized for the vector(1536) column."""

    def __init__(self) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI()
        self._model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response: Any = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=EMBEDDING_DIM,
        )
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def get_embedding_client() -> EmbeddingClient:
    provider = os.getenv("EMBEDDING_PROVIDER", "hashing")
    if provider == "hashing":
        return HashingEmbeddingClient()
    if provider == "openai":
        return OpenAIEmbeddingClient()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")
