import pytest

from src.model.context import Context


def test_context_to_messages_order_and_roles() -> None:
    ctx = Context.create(system_prompt="You are helpful.", persist=False)

    ctx.long_term_episodic.append("user", "LT user")
    ctx.long_term_episodic.append("assistant", "LT assistant")

    ctx.short_term.append("user", "ST user")
    ctx.short_term.append("assistant", "ST assistant")

    # Recall intentionally empty for now

    ctx.append("user", "WK user")
    ctx.append("assistant", "WK assistant")

    messages = ctx.to_messages()
    simplified = [{"role": m["role"], "content": m["content"]} for m in messages]
    assert simplified[0] == {"role": "system", "content": "You are helpful."}

    # Ensure fixed section order by checking subsequences
    assert any(m["role"] == "user" and m["content"].endswith("LT user") for m in simplified)
    assert any(m["role"] == "user" and m["content"].endswith("ST user") for m in simplified)
    assert any(m["role"] == "user" and m["content"].endswith("WK user") for m in simplified)

    idx_lt = next(i for i, m in enumerate(simplified) if m["role"] == "user" and m["content"].endswith("LT user"))
    idx_st = next(i for i, m in enumerate(simplified) if m["role"] == "user" and m["content"].endswith("ST user"))
    idx_wk = next(i for i, m in enumerate(simplified) if m["role"] == "user" and m["content"].endswith("WK user"))
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
    ctx.long_term_episodic.append("user", "LT")
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
