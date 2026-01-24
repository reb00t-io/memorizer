import pytest

from src.context import Context


def test_context_to_messages_order_and_roles() -> None:
    ctx = Context.create(system_prompt="You are helpful.", persist=False)

    ctx.long_term.append("user", "LT user")
    ctx.long_term.append("assistant", "LT assistant")

    ctx.short_term.append("user", "ST user")
    ctx.short_term.append("assistant", "ST assistant")

    # Recall intentionally empty for now

    ctx.append("user", "WK user")
    ctx.append("assistant", "WK assistant")

    messages = ctx.to_messages()
    simplified = [{"role": m["role"], "content": m["content"]} for m in messages]
    assert simplified[0] == {"role": "system", "content": "You are helpful."}

    # Ensure fixed section order by checking subsequences
    assert {"role": "user", "content": "LT user"} in simplified
    assert {"role": "user", "content": "ST user"} in simplified
    assert {"role": "user", "content": "WK user"} in simplified

    idx_lt = simplified.index({"role": "user", "content": "LT user"})
    idx_st = simplified.index({"role": "user", "content": "ST user"})
    idx_wk = simplified.index({"role": "user", "content": "WK user"})
    assert idx_lt < idx_st < idx_wk


def test_system_memory_is_single_message() -> None:
    ctx = Context.create(system_prompt="First", persist=False)
    ctx.system.append("system", "Second")

    system_messages = ctx.system.to_messages()
    assert system_messages == [{"role": "system", "content": "Second"}]


def test_non_system_memory_rejects_invalid_roles() -> None:
    ctx = Context.create(system_prompt="Hi", persist=False)

    with pytest.raises(ValueError):
        ctx.short_term.append("system", "not allowed here")


def test_to_string_contains_sections_in_order() -> None:
    ctx = Context.create(system_prompt="S", persist=False)
    ctx.long_term.append("user", "LT")
    ctx.short_term.append("user", "ST")
    ctx.append("user", "WK")

    s = ctx.to_string()

    i_system = s.index("# System")
    i_lt = s.index("# Long-term")
    i_st = s.index("# Short-term")
    i_recall = s.index("# Recall")
    i_working = s.index("# Working")

    assert i_system < i_lt < i_st < i_recall < i_working


def test_context_append_writes_to_working_memory() -> None:
    ctx = Context.create(system_prompt="S", persist=False)
    ctx.append("user", "hello")
    messages = ctx.working.to_messages()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"].endswith("\n\nhello")


def test_context_compress_appends_to_long_term_and_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.model import Model

    def fake_nostream(self, messages):
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        return "- compressed summary"

    monkeypatch.setattr(Model, "nostream", fake_nostream)

    ctx = Context.create(system_prompt="S", persist=False)
    ctx.short_term.append("user", "ST1")
    ctx.short_term.append("assistant", "ST2")
    ctx.append("user", "WK1")

    model = Model(ctx)
    summary = model.compress()
    assert summary == "- compressed summary"

    # short-term and working cleared
    assert ctx.short_term.to_messages() == []
    assert ctx.working.to_messages() == []

    # long-term contains a single system message with explanation
    lt = ctx.long_term.to_messages()
    assert len(lt) == 1
    assert lt[0]["role"] == "system"
    assert "compressed from short-term + working" in lt[0]["content"]
