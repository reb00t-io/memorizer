"""Retrieval-backed memory store for Memorizer.

Provides a Qdrant-backed :class:`MemoryStore` with hybrid (dense + BM25) search,
short agent-friendly identifiers, and the :data:`RECALL_TOOL` the model uses to
fetch its own memory. Requires the ``store`` extra (``pip install memorizer[store]``).
"""

from .bm25 import BM25Encoder
from .embedder import (
    DEFAULT_DIMENSIONS,
    DEFAULT_MODEL,
    Embedder,
    PrivatemodeEmbedder,
)
from .qdrant_store import MemoryHit, MemoryStore, build_stores
from .recall import RECALL_TOOL, execute_recall

__all__ = [
    "BM25Encoder",
    "Embedder",
    "PrivatemodeEmbedder",
    "DEFAULT_MODEL",
    "DEFAULT_DIMENSIONS",
    "MemoryStore",
    "MemoryHit",
    "build_stores",
    "RECALL_TOOL",
    "execute_recall",
]
