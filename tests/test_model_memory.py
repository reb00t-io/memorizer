from __future__ import annotations

import json

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient

from memorizer.model.context import Context
from memorizer.model.message import Message
from memorizer.model.model import Model
from memorizer.store import MemoryStore
from tests.fake_embedder import FakeEmbedder


def _make_model(tmp_path, *, org=False):
    client = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    agent = MemoryStore(client, collection="agent_memory", embedder=embedder,
                        scope="agent", id_prefix="m", state_dir=tmp_path / "s")
    org_store = None
    profile = None
    if org:
        org_store = MemoryStore(client, collection="org_memory", embedder=embedder,
                                scope="org", id_prefix="o", state_dir=tmp_path / "s")
        profile = "ORG RULES: extract generic org-wide facts only."
    ctx = Context.create(persist=False)
    model = Model(
        ctx,
        model_id="test-model",
        max_completion_tokens=100,
        memory=agent,
        org_memory=org_store,
        org_profile=profile,
    )
    return model


def _stub_nostream(model, mapping, default=""):
    """Replace model.nostream with a router keyed on substrings of the prompt."""
    def fake(messages):
        prompt = messages[-1]["content"]
        for needle, value in mapping.items():
            if needle in prompt:
                return value
        return default
    model.nostream = fake  # type: ignore[assignment]


def test_recall_tools_gated_on_memory(tmp_path):
    model = _make_model(tmp_path)
    tools = model.recall_tools()
    assert tools and tools[0]["function"]["name"] == "recall"

    model.memory = None
    assert model.recall_tools() is None


def test_episodic_consolidation_writes_store_and_tags_summary(tmp_path):
    model = _make_model(tmp_path)
    _stub_nostream(model, {
        "Compress the following": "Discussed weekly DB password rotation.",
        "factual memory": "- DB password rotates weekly",
        "long-term goals": "Be a helpful assistant.",
    })

    model.context.long_term_episodic.extend([
        Message(role="user", content="When does the staging DB password rotate?"),
        Message(role="assistant", content="Every Monday, done by the on-call engineer."),
    ])

    model._compress_long_term_memory()

    # Store now holds a raw unit + an episode summary.
    kinds = {u.kind for u in model.memory.all_units()}
    assert {"raw", "episode"} <= kinds

    # The in-context episodic summary is tagged with its short id so the model
    # can recall the original detail.
    mem_msgs = [m for m in model.context.long_term_episodic.messages() if m.role == "memory"]
    assert mem_msgs and mem_msgs[-1].content.startswith("[m2]")

    # Following the pointer back returns the original raw exchange.
    summary = model.memory.get("m2")
    assert summary.source_ids == ["m1"]
    assert "on-call" in model.memory.get("m1").text


def test_execute_tool_calls_resolves_recall(tmp_path):
    model = _make_model(tmp_path)
    model.memory.add("the deploy key is stored in vault", kind="fact", short_id="m1")

    tool_calls = [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "recall", "arguments": json.dumps({"id": "m1"})},
    }]
    msgs = model.execute_tool_calls(tool_calls)
    assert msgs[0]["role"] == "tool"
    assert msgs[0]["tool_call_id"] == "call_1"
    assert "vault" in msgs[0]["content"]


def test_generate_runs_tool_loop_then_answers(tmp_path):
    model = _make_model(tmp_path)
    model.memory.add("Project Zephyr ships in Q3", kind="fact", short_id="m1")

    calls = {"n": 0}

    def fake_nostream_int(messages, *, tools=None, max_completion_tokens, reasoning_effort="low"):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"choices": [{"message": {
                "content": "",
                "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "recall", "arguments": json.dumps({"query": "Zephyr"})},
                }],
            }}]}
        # Second round: the recalled fact must be present in the messages.
        assert any(m.get("role") == "tool" and "Zephyr" in m.get("content", "") for m in messages)
        return {"choices": [{"message": {"content": "Project Zephyr ships in Q3."}}]}

    model._nostream_int = fake_nostream_int  # type: ignore[assignment]

    answer = model.generate(messages=[{"role": "user", "content": "When does Zephyr ship?"}])
    assert answer == "Project Zephyr ships in Q3."
    assert calls["n"] == 2


def test_org_extraction_populates_org_store_and_block(tmp_path):
    model = _make_model(tmp_path, org=True)
    _stub_nostream(model, {
        "Compress the following": "Talked about deploy policy.",
        "factual memory": "- deploy policy noted",
        "long-term goals": "Help the team.",
        "ORG RULES": '["Production deploys require two approvals."]',
    })

    model.context.long_term_episodic.extend([
        Message(role="user", content="What's our deploy policy?"),
        Message(role="assistant", content="Production deploys require two approvals."),
    ])

    model._compress_long_term_memory()

    org_units = model.org_memory.all_units(kind="org_fact")
    assert any("two approvals" in u.text for u in org_units)

    # Org knowledge is rendered into the in-context org block (cacheable prefix).
    org_block = model.context.org.messages()
    assert org_block and "two approvals" in org_block[0].content
    assert "[o1]" in org_block[0].content


def test_org_extraction_dedupes_near_duplicates(tmp_path):
    model = _make_model(tmp_path, org=True)
    model.org_memory.add("Production deploys require two approvals.", kind="org_fact")
    _stub_nostream(model, {
        "factual memory": "-",
        "Compress the following": "x",
        "long-term goals": "g",
        # Near-identical to the existing fact -> should be dropped by the guard.
        "ORG RULES": '["Production deploys require two approvals"]',
    })
    model.context.long_term_episodic.extend([
        Message(role="user", content="deploy policy?"),
        Message(role="assistant", content="two approvals"),
    ])
    model._compress_long_term_memory()
    assert len(model.org_memory.all_units(kind="org_fact")) == 1
