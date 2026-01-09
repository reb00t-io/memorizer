import json

from src.memory import Memory


def test_memory_loads_from_file_on_init(tmp_path) -> None:
    p = tmp_path / "mem.json"
    p.write_text(json.dumps([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]))

    m = Memory(allowed_roles={"user", "assistant"}, persist_path=p)
    assert m.to_messages() == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_memory_persists_on_modify(tmp_path) -> None:
    p = tmp_path / "mem.json"
    m = Memory(allowed_roles={"user", "assistant"}, persist_path=p)

    m.append("user", "hi")
    assert json.loads(p.read_text()) == [{"role": "user", "content": "hi"}]

    m.append("assistant", "hello")
    assert json.loads(p.read_text()) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
