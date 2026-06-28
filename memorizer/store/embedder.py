"""Dense embeddings via a Privatemode (OpenAI-compatible) ``/embeddings`` endpoint.

The default model is ``qwen3-embedding-4b`` served through the same Privatemode
proxy used for chat completions, so embedding requests stay on the same local /
private path. The model's native output is 2560-dim; we request a reduced
``dimensions`` (1024 by default) to keep vectors small and recall fast.

``Embedder`` is the minimal interface ``MemoryStore`` depends on; tests inject a
deterministic fake instead of calling the network.
"""

from __future__ import annotations

import logging
from typing import Protocol

import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3-embedding-4b"
DEFAULT_DIMENSIONS = 1024
_BATCH_SIZE = 32
_MAX_TEXT_CHARS = 24000  # ~8k tokens, well within the model's limit
# Qwen3 embedding is instruction-tuned; queries get a retrieval instruction.
_QUERY_INSTRUCT = (
    "Instruct: Given a query, retrieve relevant memories that answer it\nQuery: "
)


class Embedder(Protocol):
    """Minimal embedding interface required by :class:`MemoryStore`."""

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class PrivatemodeEmbedder:
    """Calls an OpenAI-compatible ``/embeddings`` endpoint (Privatemode proxy)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "dummy",
        model: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout

    @property
    def dim(self) -> int:
        return self.dimensions

    def _call(self, inputs: list[str]) -> list[list[float]]:
        truncated = [t[:_MAX_TEXT_CHARS] for t in inputs]
        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": truncated,
                "dimensions": self.dimensions,
                "encoding_format": "float",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            out.extend(self._call(texts[i : i + _BATCH_SIZE]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._call([f"{_QUERY_INSTRUCT}{text}"])[0]
