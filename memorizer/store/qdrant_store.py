"""Qdrant-backed memory store with short, agent-friendly identifiers.

Each stored unit (an archived raw exchange, an episodic summary, an org fact, …)
gets a **short id** such as ``m12`` or ``o3``. These ids are deliberately tiny so
the model reproduces them correctly when it calls the ``recall`` tool. A short id
maps to a stable Qdrant point id (UUIDv5), so re-adding is idempotent per id.

Units carry dense + BM25 sparse vectors and are retrieved with Qdrant's built-in
RRF fusion (hybrid search). Provenance is kept via ``source_ids`` so a compressed
unit points back to the fuller objects it was derived from.
"""

from __future__ import annotations

import json
import logging
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient, models

from .bm25 import BM25Encoder
from .embedder import Embedder

logger = logging.getLogger(__name__)

_UUID_NS = uuid.NAMESPACE_URL


@dataclass(slots=True)
class MemoryHit:
    """A retrieved (or fetched) memory unit."""

    short_id: str
    kind: str
    scope: str
    text: str
    timestamp: Optional[str] = None
    source_ids: list[str] = field(default_factory=list)
    score: Optional[float] = None

    def render(self) -> str:
        ts = f"{self.timestamp} " if self.timestamp else ""
        src = f" (source: {', '.join(self.source_ids)})" if self.source_ids else ""
        return f"[{self.short_id}] {ts}({self.kind}){src}\n{self.text}"


class MemoryStore:
    """One Qdrant collection of memory units sharing a scope and id prefix.

    A single :class:`qdrant_client.QdrantClient` can back several stores (e.g. an
    agent store and an org store) via distinct collection names. Pass the client
    and an :class:`Embedder` in; use :meth:`create` to build local defaults.
    """

    def __init__(
        self,
        client: QdrantClient,
        *,
        collection: str,
        embedder: Embedder,
        scope: str = "agent",
        id_prefix: str = "m",
        bm25: Optional[BM25Encoder] = None,
        state_dir: str | Path | None = None,
    ) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.scope = scope
        self.id_prefix = id_prefix
        self.bm25 = bm25 or BM25Encoder()
        self._state_dir = Path(state_dir).expanduser() if state_dir is not None else None
        self._next = 1

        if self._state_dir is not None:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self.bm25.load(self._bm25_path)
            self._load_counter()

        self._ensure_collection()

    # -- construction -------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        embedder: Embedder,
        data_dir: str | Path,
        collection: str = "agent_memory",
        scope: str = "agent",
        id_prefix: str = "m",
        location: str | None = None,
    ) -> "MemoryStore":
        """Build a store with a local (on-disk) Qdrant unless ``location`` is a URL."""
        data_dir = Path(data_dir).expanduser()
        if location and location.startswith(("http://", "https://")):
            client = QdrantClient(url=location)
        else:
            client = QdrantClient(path=str(data_dir / "qdrant"))
        return cls(
            client,
            collection=collection,
            embedder=embedder,
            scope=scope,
            id_prefix=id_prefix,
            state_dir=data_dir / "store_state",
        )

    # -- collection / state -------------------------------------------------

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.embedder.dim, distance=models.Distance.COSINE
                ),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )
        for field_name in ("short_id", "scope", "kind"):
            # Payload indexes are a no-op (and warn) in local mode; they matter
            # only for a real server, so suppress the local warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                self.client.create_payload_index(
                    self.collection, field_name, models.PayloadSchemaType.KEYWORD
                )
        logger.info(
            "Created Qdrant collection %r (dense=%d + bm25)",
            self.collection,
            self.embedder.dim,
        )

    @property
    def _counter_path(self) -> Path:
        assert self._state_dir is not None
        return self._state_dir / f"{self.collection}.counter.json"

    @property
    def _bm25_path(self) -> Path:
        assert self._state_dir is not None
        return self._state_dir / f"{self.collection}.bm25.json"

    def _load_counter(self) -> None:
        try:
            self._next = int(json.loads(self._counter_path.read_text())["next"])
        except (FileNotFoundError, ValueError, KeyError):
            self._next = 1

    def _save_state(self) -> None:
        if self._state_dir is None:
            return
        self._counter_path.write_text(json.dumps({"next": self._next}))
        self.bm25.save(self._bm25_path)

    def _alloc_id(self) -> str:
        short_id = f"{self.id_prefix}{self._next}"
        self._next += 1
        return short_id

    def _point_id(self, short_id: str) -> str:
        return str(uuid.uuid5(_UUID_NS, f"{self.collection}:{short_id}"))

    # -- writes -------------------------------------------------------------

    def add(
        self,
        text: str,
        *,
        kind: str,
        timestamp: Optional[str] = None,
        source_ids: Optional[list[str]] = None,
        short_id: Optional[str] = None,
    ) -> str:
        """Embed and upsert one unit. Returns its short id."""
        text = text.strip()
        sid = short_id or self._alloc_id()
        dense = self.embedder.embed_documents([text])[0]
        indices, values = self.bm25.encode_document(text)

        vector: dict[str, object] = {"dense": dense}
        if indices:
            vector["bm25"] = models.SparseVector(indices=indices, values=values)

        payload = {
            "short_id": sid,
            "scope": self.scope,
            "kind": kind,
            "text": text,
            "timestamp": timestamp,
            "source_ids": source_ids or [],
        }
        self.client.upsert(
            collection_name=self.collection,
            points=[models.PointStruct(id=self._point_id(sid), vector=vector, payload=payload)],
        )
        self._save_state()
        return sid

    # -- reads --------------------------------------------------------------

    def get(self, short_id: str) -> Optional[MemoryHit]:
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="short_id", match=models.MatchValue(value=short_id))]
            ),
            limit=1,
            with_payload=True,
        )
        if not points:
            return None
        return self._to_hit(points[0].payload, score=None)

    def search(self, query: str, *, limit: int = 5, kinds: Optional[list[str]] = None) -> list[MemoryHit]:
        """Hybrid (dense + BM25, RRF-fused) search over this store."""
        query = query.strip()
        if not query:
            return []
        query_filter = None
        if kinds:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="kind", match=models.MatchAny(any=list(kinds)))]
            )

        dense = self.embedder.embed_query(query)
        indices, values = self.bm25.encode_query(query)

        prefetch = [models.Prefetch(query=dense, using="dense", limit=limit * 4, filter=query_filter)]
        if indices:
            prefetch.insert(
                0,
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using="bm25",
                    limit=limit * 4,
                    filter=query_filter,
                ),
            )

        points = self.client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points
        return [self._to_hit(p.payload, score=p.score) for p in points]

    def all_units(self, *, kind: Optional[str] = None, limit: int = 1000) -> list[MemoryHit]:
        """Scroll every unit (optionally of one kind), ordered by short-id number."""
        query_filter = None
        if kind:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="kind", match=models.MatchValue(value=kind))]
            )
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        hits = [self._to_hit(p.payload, score=None) for p in points]
        hits.sort(key=lambda h: _id_num(h.short_id))
        return hits

    def count(self) -> int:
        return self.client.count(self.collection).count

    def close(self) -> None:
        """Release the underlying Qdrant client (local mode holds a path lock)."""
        try:
            self.client.close()
        except Exception:
            pass

    def _to_hit(self, payload: Optional[dict], *, score: Optional[float]) -> MemoryHit:
        payload = payload or {}
        return MemoryHit(
            short_id=payload.get("short_id", ""),
            kind=payload.get("kind", ""),
            scope=payload.get("scope", self.scope),
            text=payload.get("text", ""),
            timestamp=payload.get("timestamp"),
            source_ids=list(payload.get("source_ids") or []),
            score=score,
        )


def _id_num(short_id: str) -> int:
    digits = "".join(ch for ch in short_id if ch.isdigit())
    return int(digits) if digits else 0


def build_stores(
    *,
    embedder: Embedder,
    data_dir: str | Path,
    location: str | None = None,
    agent: bool = True,
    org: bool = True,
    agent_collection: str = "agent_memory",
    org_collection: str = "org_memory",
) -> tuple[Optional[MemoryStore], Optional[MemoryStore]]:
    """Build agent and/or org stores sharing one Qdrant client.

    Local (on-disk) Qdrant allows only one client per path, so the agent and org
    collections must share a single client; pass a ``location`` URL to use a
    shared server instead (the natural choice for org-wide memory).
    """
    data_dir = Path(data_dir).expanduser()
    if location and location.startswith(("http://", "https://")):
        client = QdrantClient(url=location)
    else:
        client = QdrantClient(path=str(data_dir / "qdrant"))
    state_dir = data_dir / "store_state"

    agent_store = (
        MemoryStore(client, collection=agent_collection, embedder=embedder,
                    scope="agent", id_prefix="m", state_dir=state_dir)
        if agent else None
    )
    org_store = (
        MemoryStore(client, collection=org_collection, embedder=embedder,
                    scope="org", id_prefix="o", state_dir=state_dir)
        if org else None
    )
    return agent_store, org_store
