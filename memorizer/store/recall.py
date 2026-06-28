"""The ``recall`` tool: lets the model fetch its own memory on demand.

The tool takes either a free-text ``query`` (hybrid search) or an exact ``id``
(fetch one unit and the fuller objects it was compressed from). Ids are the short
ids the model sees inline in its memory (e.g. ``[m12]``), so following a pointer
back to detail is just ``recall(id="m12")``.
"""

from __future__ import annotations

from typing import Optional

from .qdrant_store import MemoryHit, MemoryStore

RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": (
            "Search or fetch your long-term memory. Use `query` to find relevant "
            "past information by meaning or keyword. Use `id` (a short id like "
            "'m12' or 'o3' shown in brackets next to a memory) to fetch that exact "
            "memory and the original detail it was compressed from."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language or keyword query to search memory.",
                },
                "id": {
                    "type": "string",
                    "description": "Exact short id of a memory to fetch (e.g. 'm12').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results for a query search (default 5).",
                },
            },
        },
    },
}


def execute_recall(
    args: dict,
    *,
    agent_store: Optional[MemoryStore],
    org_store: Optional[MemoryStore] = None,
    member_id: Optional[str] = None,
    role: Optional[str] = None,
    default_limit: int = 5,
) -> str:
    """Run a recall tool call, scoped to the caller's member id and role.

    Agent (personal) memory is restricted to ``member_id``; org memory is
    filtered to facts visible to ``role`` (or to everyone).
    """
    if agent_store is None and org_store is None:
        return "Memory is not available."

    raw_id = (args.get("id") or "").strip()
    if raw_id:
        return _fetch_by_id(raw_id, agent_store, org_store, member_id, role)

    query = (args.get("query") or "").strip()
    if not query:
        return "Provide either `query` or `id`."

    limit = int(args.get("limit") or default_limit)
    hits: list[MemoryHit] = []
    if agent_store is not None:
        hits.extend(agent_store.search(query, limit=limit, member_id=member_id))
    if org_store is not None:
        hits.extend(org_store.search(query, limit=limit, role=role))
    if not hits:
        return f"No memories found for {query!r}."
    # Best-scoring first across stores; keep it short.
    hits.sort(key=lambda h: (h.score is not None, h.score or 0.0), reverse=True)
    rendered = "\n\n".join(h.render() for h in hits[:limit])
    return f"Found {min(len(hits), limit)} memory result(s):\n\n{rendered}"


def _fetch_by_id(
    short_id: str,
    agent_store: Optional[MemoryStore],
    org_store: Optional[MemoryStore],
    member_id: Optional[str],
    role: Optional[str],
) -> str:
    candidates = [
        (agent_store, {"member_id": member_id}),
        (org_store, {"role": role}),
    ]
    for store, access in candidates:
        if store is None:
            continue
        hit = store.get(short_id, **access)
        if hit is None:
            continue
        parts = [hit.render()]
        for src_id in hit.source_ids:
            src = store.get(src_id, **access)
            if src is not None:
                parts.append("\nOriginal detail:\n" + src.render())
        return "\n".join(parts)
    return f"No memory with id {short_id!r}."
