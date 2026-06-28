from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient

from memorizer.model.context import Context
from memorizer.model.message import Message
from memorizer.model.model import Model
from memorizer.org import OrgPolicy
from memorizer.store import MemoryStore, execute_recall
from tests.fake_embedder import FakeEmbedder


def _stores(tmp_path):
    client = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    agent = MemoryStore(client, collection="agent_memory", embedder=embedder,
                        scope="agent", id_prefix="m", state_dir=tmp_path / "s")
    org = MemoryStore(client, collection="org_memory", embedder=embedder,
                      scope="org", id_prefix="o", state_dir=tmp_path / "s")
    return agent, org


# -- store-level access control -----------------------------------------------

def test_agent_memory_scoped_by_member(tmp_path):
    agent, _ = _stores(tmp_path)
    agent.add("alice's calendar is full on Fridays", kind="fact", member_id="alice")
    agent.add("bob prefers morning standups", kind="fact", member_id="bob")

    alice_hits = agent.search("calendar standup preferences", limit=5, member_id="alice")
    assert alice_hits and all(h.member_id == "alice" for h in alice_hits)
    assert not any("bob" in h.text for h in alice_hits)


def test_org_read_visibility_filtered_by_role(tmp_path):
    _, org = _stores(tmp_path)
    org.add("All-hands is every Monday.", kind="org_fact", visible_to=["*"])
    org.add("Comp bands are confidential to managers.", kind="org_fact", visible_to=["manager"])

    eng = org.search("monday comp confidential", limit=5, role="engineer")
    texts = " ".join(h.text for h in eng)
    assert "All-hands" in texts
    assert "Comp bands" not in texts  # restricted to managers

    mgr = org.search("monday comp confidential", limit=5, role="manager")
    mtexts = " ".join(h.text for h in mgr)
    assert "All-hands" in mtexts and "Comp bands" in mtexts


def test_get_by_id_enforces_visibility(tmp_path):
    _, org = _stores(tmp_path)
    sid = org.add("Restricted: layoffs planned Q4.", kind="org_fact", visible_to=["admin"])
    assert org.get(sid, role="engineer") is None      # hidden
    assert org.get(sid, role="admin") is not None      # allowed


def test_get_by_id_enforces_member_ownership(tmp_path):
    agent, _ = _stores(tmp_path)
    sid = agent.add("alice secret note", kind="fact", member_id="alice")
    assert agent.get(sid, member_id="bob") is None
    assert agent.get(sid, member_id="alice") is not None


def test_execute_recall_respects_member_and_role(tmp_path):
    agent, org = _stores(tmp_path)
    agent.add("alice deploy token is in her vault", kind="fact", member_id="alice")
    agent.add("bob deploy token is in his vault", kind="fact", member_id="bob")
    org.add("Managers approve budget over 10k.", kind="org_fact", visible_to=["manager"])

    out = execute_recall(
        {"query": "deploy token budget"},
        agent_store=agent, org_store=org, member_id="alice", role="engineer",
    )
    assert "alice" in out
    assert "bob" not in out
    assert "budget" not in out  # manager-only org fact hidden from engineer


# -- model-level write gating -------------------------------------------------

def _make_model(tmp_path, *, role, policy):
    client = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    agent = MemoryStore(client, collection="agent_memory", embedder=embedder,
                        scope="agent", id_prefix="m", state_dir=tmp_path / "s")
    org = MemoryStore(client, collection="org_memory", embedder=embedder,
                      scope="org", id_prefix="o", state_dir=tmp_path / "s")
    ctx = Context.create(persist=False)
    return Model(
        ctx, model_id="t", max_completion_tokens=100,
        memory=agent, org_memory=org,
        org_profile="RULES: extract generic org facts.",
        org_policy=policy, role=role, member_id="u1",
    )


def _stub(model, value):
    def fake(messages):
        prompt = messages[-1]["content"]
        if "RULES" in prompt:
            return value
        return "x"
    model.nostream = fake  # type: ignore[assignment]


def _seed_episode(model):
    model.context.long_term_episodic.extend([
        Message(role="user", content="what's the deploy policy"),
        Message(role="assistant", content="two approvals required"),
    ])


def test_org_write_blocked_for_non_writer_role(tmp_path):
    policy = OrgPolicy.create(roles={"engineer", "manager"}, writer_roles={"manager"})
    model = _make_model(tmp_path, role="engineer", policy=policy)
    _stub(model, '[{"fact": "Deploys need two approvals.", "visible_to": "all"}]')
    _seed_episode(model)
    model._compress_long_term_memory()
    assert model.org_memory.all_units(kind="org_fact") == []  # engineer can't write


def test_org_write_allowed_and_visibility_stored(tmp_path):
    policy = OrgPolicy.create(roles={"engineer", "manager"}, writer_roles={"manager"})
    model = _make_model(tmp_path, role="manager", policy=policy)
    _stub(model, '[{"fact": "Comp review is in March.", "visible_to": ["manager"]}]')
    _seed_episode(model)
    model._compress_long_term_memory()

    units = model.org_memory.all_units(kind="org_fact")
    assert len(units) == 1
    assert units[0].visible_to == ["manager"]

    # The manager who wrote it sees it in their org block; an engineer would not.
    assert "Comp review" in model.context.org.messages()[0].content
    eng_visible = model.org_memory.all_units(kind="org_fact", role="engineer")
    assert eng_visible == []


def test_org_unknown_visibility_role_defaults_to_everyone(tmp_path):
    policy = OrgPolicy.create(roles={"engineer", "manager"}, writer_roles={"manager"})
    model = _make_model(tmp_path, role="manager", policy=policy)
    # 'intern' is not a known role -> dropped -> falls back to everyone.
    _stub(model, '[{"fact": "Coffee is free.", "visible_to": ["intern"]}]')
    _seed_episode(model)
    model._compress_long_term_memory()
    units = model.org_memory.all_units(kind="org_fact")
    assert units and units[0].visible_to == ["*"]
