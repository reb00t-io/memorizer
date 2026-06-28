"""Deterministic, network-free embedder for tests.

Produces a hashed bag-of-words vector so cosine similarity tracks token overlap —
enough for the store's hybrid search to return sensible orderings in tests.
"""

from __future__ import annotations

import math
import re

_DIM = 64


def _vec(text: str) -> list[float]:
    v = [0.0] * _DIM
    for tok in re.findall(r"\w+", text.lower()):
        v[hash(tok) % _DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class FakeEmbedder:
    @property
    def dim(self) -> int:
        return _DIM

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _vec(text)
