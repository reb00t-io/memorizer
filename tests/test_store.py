from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient

from memorizer.store import MemoryStore, RECALL_TOOL, execute_recall
from tests.fake_embedder import FakeEmbedder


def _store(tmp_path, scope="agent", prefix="m", collection="agent_memory"):
    client = QdrantClient(":memory:")
    return MemoryStore(
        client,
        collection=collection,
        embedder=FakeEmbedder(),
        scope=scope,
        id_prefix=prefix,
        state_dir=tmp_path / "state",
    )


def test_short_ids_increment_and_prefix(tmp_path):
    store = _store(tmp_path)
    a = store.add("the cat sat on the mat", kind="raw")
    b = store.add("dogs are loyal animals", kind="raw")
    assert a == "m1" and b == "m2"
    assert store.count() == 2


def test_get_by_id_returns_unit(tmp_path):
    store = _store(tmp_path)
    sid = store.add("Marko prefers concise answers", kind="fact", timestamp="2026-06-28 10:00")
    hit = store.get(sid)
    assert hit is not None
    assert hit.short_id == sid
    assert hit.kind == "fact"
    assert "concise" in hit.text


def test_search_finds_relevant_unit(tmp_path):
    store = _store(tmp_path)
    store.add("the user lives in Berlin and likes hiking", kind="fact")
    store.add("python is a programming language", kind="fact")
    hits = store.search("where does the user live", limit=2)
    assert hits
    assert "Berlin" in hits[0].text


def test_provenance_source_ids(tmp_path):
    store = _store(tmp_path)
    raw = store.add("long raw conversation about deployment", kind="raw")
    summ = store.add("discussed deployment", kind="episode", source_ids=[raw])
    hit = store.get(summ)
    assert hit.source_ids == [raw]


def test_execute_recall_by_query_and_id(tmp_path):
    store = _store(tmp_path)
    raw = store.add("raw detail: the API key rotates weekly", kind="raw")
    summ = store.add("API keys rotate", kind="episode", source_ids=[raw])

    out = execute_recall({"query": "API key rotation"}, agent_store=store)
    assert "rotate" in out.lower()

    out_id = execute_recall({"id": summ}, agent_store=store)
    assert "Original detail" in out_id
    assert "weekly" in out_id


def test_execute_recall_searches_org_too(tmp_path):
    agent = _store(tmp_path, scope="agent", prefix="m", collection="agent_memory")
    org = _store(tmp_path, scope="org", prefix="o", collection="org_memory")
    org.add("Company VPN must be on for all internal tools", kind="org_fact")
    out = execute_recall({"query": "do I need the VPN"}, agent_store=agent, org_store=org)
    assert "VPN" in out


def test_recall_tool_schema_shape():
    fn = RECALL_TOOL["function"]
    assert fn["name"] == "recall"
    assert set(fn["parameters"]["properties"]) == {"query", "id", "limit"}
